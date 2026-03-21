from __future__ import annotations
# pylint: disable=duplicate-code

import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints

from algo_trader.domain import ConfigError
from ..batch_utils import BatchShape, resolve_batch_shape
from ..protocols import ModelBatch, PyroGuide
from ..registry_core import register_guide

_MIN_SCALE = 1e-4
_INIT_LOGSCALE = 0.30
_INIT_TAU0_LOC = -4.76
_INIT_SIGMA_LOC = math.log(0.03)
_INIT_W_SCALE = 0.02
_INIT_POSITIVE_LOC = math.log(0.25)
_INIT_S_U_LOC = math.log(0.02)
_MAX_H_LOC_STEP = 0.35
_MAX_H_LOG_SCALE_STEP = 0.40
_MAX_H_SCALE = 0.25


@dataclass(frozen=True)
class FilteringState:
    h_loc: torch.Tensor
    h_scale: torch.Tensor
    steps_seen: int = 0


@dataclass(frozen=True)
class Level10RuntimeBatch:
    X_asset: torch.Tensor
    X_global: torch.Tensor
    y_input: torch.Tensor
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None
    filtering_state: FilteringState | None = None


def build_level10_runtime_batch(batch: ModelBatch) -> Level10RuntimeBatch:
    shape = resolve_batch_shape(batch)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is None:
        raise ConfigError(
            "Level 10 online-filtering runtime requires batch.X_asset"
        )
    if X_asset.ndim != 3:
        raise ConfigError("batch.X_asset must have shape [T, A, F]")
    if batch.X_global is None:
        raise ConfigError(
            "Level 10 online-filtering runtime requires batch.X_global"
        )
    if batch.X_global.ndim != 2:
        raise ConfigError("batch.X_global must have shape [T, G]")
    if int(batch.X_global.shape[0]) != shape.T:
        raise ConfigError("batch.X_global and targets must align on T")
    y_input = _resolve_y_input(shape)
    time_mask = _resolve_time_mask(batch, shape.T, shape.A)
    return Level10RuntimeBatch(
        X_asset=X_asset.to(device=shape.device, dtype=shape.dtype),
        X_global=batch.X_global.to(device=shape.device, dtype=shape.dtype),
        y_input=y_input,
        y_obs=shape.y_obs,
        time_mask=time_mask,
        obs_scale=batch.obs_scale,
        filtering_state=_resolve_filtering_state(
            batch.filtering_state,
            device=shape.device,
            dtype=shape.dtype,
        ),
    )


@dataclass(frozen=True)
class StructuralPosteriorMeans:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    beta: torch.Tensor
    B: torch.Tensor
    s_u_mean: torch.Tensor

    def to_mapping(self) -> Mapping[str, torch.Tensor]:
        return {
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "beta": self.beta.detach(),
            "B": self.B.detach(),
            "s_u_mean": self.s_u_mean.detach(),
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeans":
        required = (
            "alpha",
            "sigma_idio",
            "w",
            "beta",
            "B",
            "s_u_mean",
        )
        values: dict[str, torch.Tensor] = {}
        for key in required:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ConfigError(
                    "structural_posterior_means must include tensor entries",
                    context={"field": key},
                )
            values[key] = value.detach()
        return cls(
            alpha=values["alpha"],
            sigma_idio=values["sigma_idio"],
            w=values["w"],
            beta=values["beta"],
            B=values["B"],
            s_u_mean=values["s_u_mean"],
        )


@dataclass(frozen=True)
class Level10GuideConfig:
    factor_count: int = 3
    phi: float = 0.97
    eps: float = 1e-12
    hidden_dim: int = 64


@dataclass(frozen=True)
class _GuideContext:
    batch: Level10RuntimeBatch
    device: torch.device
    dtype: torch.dtype

    @property
    def T(self) -> int:
        return int(self.batch.X_asset.shape[0])

    @property
    def A(self) -> int:
        return int(self.batch.X_asset.shape[1])

    @property
    def F(self) -> int:
        return int(self.batch.X_asset.shape[2])

    @property
    def G(self) -> int:
        return int(self.batch.X_global.shape[1])

    @property
    def encoder_input_dim(self) -> int:
        return 2 * self.F + self.G + 6


@dataclass(frozen=True)
class _GlobalGuideSites:
    s_u_mean: torch.Tensor


@dataclass(frozen=True)
class _LocalGuideSites:
    h_loc: torch.Tensor
    h_scale: torch.Tensor
    v_loc: torch.Tensor
    v_scale: torch.Tensor


class RegimeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._network(inputs)


@dataclass
class FactorGuideL10OnlineFiltering(PyroGuide):
    config: Level10GuideConfig
    _encoder: RegimeEncoder | None = None
    _encoder_input_dim: int | None = None

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_level10_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        pyro.module("factor_l10_online_filtering_encoder", encoder)
        structural = _sample_global_sites(
            context=context, factor_count=self.config.factor_count
        )
        local = _encode_local_sites(
            context=context,
            encoder=encoder,
            s_u_mean=structural.s_u_mean,
            phi=self.config.phi,
            eps=self.config.eps,
        )
        _sample_regime_sites(local)
        _sample_scale_sites(local)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch = build_level10_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        structural = self.structural_posterior_means()
        local = _encode_local_sites(
            context=context,
            encoder=encoder,
            s_u_mean=structural.s_u_mean,
            phi=self.config.phi,
            eps=self.config.eps,
        )
        return FilteringState(
            h_loc=local.h_loc[-1].detach(),
            h_scale=local.h_scale[-1].detach(),
            steps_seen=_next_steps_seen(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=_lognormal_mean(
                store.get_param("sigma_loc"),
                store.get_param("sigma_scale"),
            )
            .squeeze(-1)
            .detach(),
            w=store.get_param("w_loc").detach(),
            beta=store.get_param("beta_loc").detach(),
            B=store.get_param("B_loc").detach(),
            s_u_mean=_lognormal_mean(
                store.get_param("s_u_loc"),
                store.get_param("s_u_scale"),
            ).detach(),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=_lognormal_median(store.get_param("sigma_loc"))
            .squeeze(-1)
            .detach(),
            w=store.get_param("w_loc").detach(),
            beta=store.get_param("beta_loc").detach(),
            B=store.get_param("B_loc").detach(),
            s_u_mean=_lognormal_median(store.get_param("s_u_loc")).detach(),
        )

    def _require_encoder(self, context: _GuideContext) -> RegimeEncoder:
        if self._encoder is None:
            self._encoder = RegimeEncoder(
                input_dim=context.encoder_input_dim,
                hidden_dim=self.config.hidden_dim,
            ).to(device=context.device, dtype=context.dtype)
            self._encoder_input_dim = context.encoder_input_dim
        if self._encoder_input_dim != context.encoder_input_dim:
            raise ConfigError(
                "Level 10 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)


@register_guide("factor_guide_l10_online_filtering")
def build_factor_guide_l10_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FactorGuideL10OnlineFiltering(config=_build_guide_config(params))


def _build_guide_config(params: Mapping[str, Any]) -> Level10GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return Level10GuideConfig()
    extra = set(values) - {"factor_count", "phi", "eps", "hidden_dim"}
    if extra:
        raise ConfigError(
            "Unknown factor_guide_l10_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = Level10GuideConfig()
    try:
        updated = Level10GuideConfig(
            factor_count=int(values.get("factor_count", base.factor_count)),
            phi=float(values.get("phi", base.phi)),
            eps=float(values.get("eps", base.eps)),
            hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid factor_guide_l10_online_filtering params",
            context={"params": str(dict(values))},
        ) from exc
    if updated.factor_count <= 0:
        raise ConfigError("factor_count must be positive")
    if updated.hidden_dim <= 0:
        raise ConfigError("hidden_dim must be positive")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError("phi must be in (0, 1)")
    return updated


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(value)


def _build_context(batch: Level10RuntimeBatch) -> _GuideContext:
    return _GuideContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )


def _next_steps_seen(batch: Level10RuntimeBatch) -> int:
    previous = 0
    if batch.filtering_state is not None:
        previous = int(batch.filtering_state.steps_seen)
    return previous + int(batch.y_input.shape[0])


def _sample_global_sites(
    *, context: _GuideContext, factor_count: int
) -> _GlobalGuideSites:
    _sample_tau0(context)
    _sample_lambda(context)
    _sample_c(context)
    _sample_beta_hyperpriors(context)
    _sample_factor_loading_scale(context, factor_count)
    _sample_asset_sites(context, factor_count)
    return _GlobalGuideSites(s_u_mean=_sample_s_u(context))


def _sample_tau0(context: _GuideContext) -> None:
    loc = pyro.param(
        "tau0_loc",
        torch.tensor(_INIT_TAU0_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "tau0_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("tau0", dist.LogNormal(loc, scale))


def _sample_lambda(context: _GuideContext) -> None:
    with pyro.plate("feature", context.F, dim=-1):
        loc = pyro.param(
            "lambda_loc",
            torch.zeros(context.F, device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            "lambda_scale",
            torch.full(
                (context.F,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("lambda", dist.LogNormal(loc, scale))


def _sample_c(context: _GuideContext) -> None:
    loc = pyro.param(
        "c_loc",
        torch.tensor(math.log(0.5), device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "c_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("c", dist.LogNormal(loc, scale))


def _sample_beta_hyperpriors(context: _GuideContext) -> None:
    with pyro.plate("global_feature", context.G, dim=-1):
        beta0_loc = pyro.param(
            "beta0_loc",
            torch.zeros(context.G, device=context.device, dtype=context.dtype),
        )
        beta0_scale = pyro.param(
            "beta0_scale",
            torch.full((context.G,), 0.05, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("beta0", dist.Normal(beta0_loc, beta0_scale))
        tau_loc = pyro.param(
            "tau_beta_loc",
            torch.full(
                (context.G,),
                _INIT_POSITIVE_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        tau_scale = pyro.param(
            "tau_beta_scale",
            torch.full(
                (context.G,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("tau_beta", dist.LogNormal(tau_loc, tau_scale))


def _sample_factor_loading_scale(
    context: _GuideContext, factor_count: int
) -> None:
    with pyro.plate("factor_loading_col", factor_count, dim=-1):
        loc = pyro.param(
            "b_col_loc",
            torch.full(
                (factor_count,),
                _INIT_POSITIVE_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        scale = pyro.param(
            "b_col_scale",
            torch.full(
                (factor_count,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("b_col", dist.LogNormal(loc, scale))


def _sample_asset_sites(context: _GuideContext, factor_count: int) -> None:
    with pyro.plate("asset", context.A, dim=-2):
        _sample_alpha(context)
        _sample_sigma(context)
        _sample_feature_sites(context)
        _sample_beta(context)
        _sample_B(context, factor_count)


def _sample_alpha(context: _GuideContext) -> None:
    shape = (context.A, 1)
    loc = pyro.param(
        "alpha_loc",
        torch.zeros(shape, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "alpha_scale",
        torch.full(shape, 0.10, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("alpha", dist.Normal(loc, scale))


def _sample_sigma(context: _GuideContext) -> None:
    shape = (context.A, 1)
    loc = pyro.param(
        "sigma_loc",
        torch.full(shape, _INIT_SIGMA_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "sigma_scale",
        torch.full(shape, _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("sigma_idio", dist.LogNormal(loc, scale))


def _sample_feature_sites(context: _GuideContext) -> None:
    shape = (context.A, context.F)
    with pyro.plate("feature_w", context.F, dim=-1):
        kappa_loc = pyro.param(
            "kappa_loc",
            torch.zeros(shape, device=context.device, dtype=context.dtype),
        )
        kappa_scale = pyro.param(
            "kappa_scale",
            torch.full(shape, _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("kappa", dist.LogNormal(kappa_loc, kappa_scale))
        w_loc = pyro.param(
            "w_loc",
            torch.zeros(shape, device=context.device, dtype=context.dtype),
        )
        w_scale = pyro.param(
            "w_scale",
            torch.full(shape, _INIT_W_SCALE, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("w", dist.Normal(w_loc, w_scale))


def _sample_beta(context: _GuideContext) -> None:
    shape = (context.A, context.G)
    with pyro.plate("global_loading", context.G, dim=-1):
        loc = pyro.param(
            "beta_loc",
            torch.zeros(shape, device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            "beta_scale",
            torch.full(shape, 0.05, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("beta", dist.Normal(loc, scale))


def _sample_B(context: _GuideContext, factor_count: int) -> None:
    shape = (context.A, factor_count)
    with pyro.plate("factor_loading_k", factor_count, dim=-1):
        loc = pyro.param(
            "B_loc",
            torch.zeros(shape, device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            "B_scale",
            torch.full(shape, 0.05, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("B", dist.Normal(loc, scale))


def _sample_s_u(context: _GuideContext) -> torch.Tensor:
    loc = pyro.param(
        "s_u_loc",
        torch.tensor(_INIT_S_U_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "s_u_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("s_u", dist.LogNormal(loc, scale))
    return _lognormal_mean(loc, scale)


def _encode_local_sites(
    *,
    context: _GuideContext,
    encoder: RegimeEncoder,
    s_u_mean: torch.Tensor,
    phi: float,
    eps: float,
) -> _LocalGuideSites:
    h_loc: list[torch.Tensor] = []
    h_scale: list[torch.Tensor] = []
    v_loc: list[torch.Tensor] = []
    v_scale: list[torch.Tensor] = []
    prior_loc, prior_scale = _initial_prior_message(context, s_u_mean, phi, eps)
    for time_index in range(context.T):
        encoded = encoder(
            _encoder_features(
                context=context,
                time_index=time_index,
                prior_loc=prior_loc,
                prior_scale=prior_scale,
            )
        )
        current = _decode_local_parameters(
            encoded,
            prior_loc=prior_loc,
            prior_scale=prior_scale,
        )
        h_loc.append(current[0])
        h_scale.append(current[1])
        v_loc.append(current[2])
        v_scale.append(current[3])
        prior_loc, prior_scale = _propagated_message(
            current_loc=current[0],
            current_scale=current[1],
            s_u_mean=s_u_mean,
            phi=phi,
            eps=eps,
        )
    return _LocalGuideSites(
        h_loc=torch.stack(h_loc),
        h_scale=torch.stack(h_scale),
        v_loc=torch.stack(v_loc),
        v_scale=torch.stack(v_scale),
    )


def _initial_prior_message(
    context: _GuideContext,
    s_u_mean: torch.Tensor,
    phi: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi_t = torch.tensor(phi, device=context.device, dtype=context.dtype)
    if context.batch.filtering_state is None:
        denom = (
            torch.tensor(1.0, device=context.device, dtype=context.dtype)
            - phi_t.pow(2)
            + eps
        )
        zero = torch.tensor(0.0, device=context.device, dtype=context.dtype)
        return zero, s_u_mean / torch.sqrt(denom)
    h_loc = context.batch.filtering_state.h_loc.to(
        device=context.device, dtype=context.dtype
    )
    h_scale = context.batch.filtering_state.h_scale.to(
        device=context.device, dtype=context.dtype
    )
    prior_var = phi_t.pow(2) * h_scale.pow(2) + s_u_mean.pow(2) + eps
    return phi_t * h_loc, torch.sqrt(prior_var)


def _encoder_features(
    *,
    context: _GuideContext,
    time_index: int,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> torch.Tensor:
    x_asset_t = context.batch.X_asset[time_index]
    x_global_t = context.batch.X_global[time_index]
    y_t = context.batch.y_input[time_index]
    y_summary = torch.stack([y_t.mean(), y_t.std(unbiased=False), y_t.min(), y_t.max()])
    prior_summary = torch.stack(
        [prior_loc, torch.log(prior_scale + _MIN_SCALE)]
    )
    return torch.cat(
        [
            x_asset_t.mean(dim=0),
            x_asset_t.std(dim=0, unbiased=False),
            x_global_t,
            y_summary,
            prior_summary,
        ]
    )


def _decode_local_parameters(
    encoded: torch.Tensor,
    *,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep the amortized local state close to the propagated AR(1) message.
    # Without a bounded residual step, the encoder can drive h_t too negative
    # in a single update, which collapses u_t and inflates predictive variance.
    loc_step = (
        prior_scale * _MAX_H_LOC_STEP * torch.tanh(encoded[0])
    )
    current_h_loc = prior_loc + loc_step
    scale_step = torch.exp(
        _MAX_H_LOG_SCALE_STEP * torch.tanh(encoded[1])
    )
    current_h_scale = torch.clamp(
        prior_scale * scale_step,
        min=_MIN_SCALE,
        max=_MAX_H_SCALE,
    )
    current_v_scale = torch.nn.functional.softplus(encoded[3]) + _MIN_SCALE
    return current_h_loc, current_h_scale, encoded[2], current_v_scale


def _propagated_message(
    *,
    current_loc: torch.Tensor,
    current_scale: torch.Tensor,
    s_u_mean: torch.Tensor,
    phi: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi_t = torch.tensor(phi, device=current_loc.device, dtype=current_loc.dtype)
    prior_var = phi_t.pow(2) * current_scale.pow(2) + s_u_mean.pow(2) + eps
    return phi_t * current_loc, torch.sqrt(prior_var)


def _sample_regime_sites(local: _LocalGuideSites) -> None:
    for index, (h_loc_t, h_scale_t) in enumerate(
        zip(local.h_loc, local.h_scale), start=1
    ):
        pyro.sample(f"h_{index}", dist.Normal(h_loc_t, h_scale_t))


def _sample_scale_sites(local: _LocalGuideSites) -> None:
    with pyro.plate("time_v", int(local.v_loc.shape[0]), dim=-1):
        pyro.sample("v", dist.LogNormal(local.v_loc, local.v_scale))


def _lognormal_mean(loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(loc + 0.5 * scale.pow(2))


def _lognormal_median(loc: torch.Tensor) -> torch.Tensor:
    return torch.exp(loc)


def _resolve_y_input(shape: BatchShape) -> torch.Tensor:
    if shape.y_obs is not None:
        return shape.y_obs
    return torch.zeros(
        (shape.T, shape.A),
        device=shape.device,
        dtype=shape.dtype,
    )


def _resolve_time_mask(
    batch: ModelBatch, T: int, A: int
) -> torch.BoolTensor | None:
    if batch.M is None:
        return None
    if batch.M.ndim != 2 or tuple(batch.M.shape) != (T, A):
        raise ConfigError("batch.M must have shape [T, A]")
    return cast(torch.BoolTensor, batch.M.all(dim=-1))


def _resolve_filtering_state(
    raw_state: object | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FilteringState | None:
    if raw_state is None:
        return None
    if isinstance(raw_state, FilteringState):
        return _move_filtering_state(
            raw_state, device=device, dtype=dtype
        )
    if isinstance(raw_state, Mapping):
        return _mapping_to_filtering_state(
            raw_state, device=device, dtype=dtype
        )
    if _looks_like_filtering_state(raw_state):
        return FilteringState(
            h_loc=cast(Any, raw_state).h_loc.to(device=device, dtype=dtype),
            h_scale=cast(Any, raw_state).h_scale.to(
                device=device, dtype=dtype
            ),
            steps_seen=_resolve_steps_seen(cast(Any, raw_state).steps_seen),
        )
    raise ConfigError(
        "batch.filtering_state must be a FilteringState or mapping"
    )


def _move_filtering_state(
    state: FilteringState, *, device: torch.device, dtype: torch.dtype
) -> FilteringState:
    return FilteringState(
        h_loc=state.h_loc.to(device=device, dtype=dtype),
        h_scale=state.h_scale.to(device=device, dtype=dtype),
        steps_seen=_resolve_steps_seen(state.steps_seen),
    )


def _mapping_to_filtering_state(
    raw_state: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FilteringState:
    h_loc = raw_state.get("h_loc")
    h_scale = raw_state.get("h_scale")
    steps_seen = raw_state.get("steps_seen", 0)
    if not isinstance(h_loc, torch.Tensor) or not isinstance(
        h_scale, torch.Tensor
    ):
        raise ConfigError(
            "batch.filtering_state must include tensor h_loc and h_scale"
        )
    return FilteringState(
        h_loc=h_loc.to(device=device, dtype=dtype),
        h_scale=h_scale.to(device=device, dtype=dtype),
        steps_seen=_resolve_steps_seen(steps_seen),
    )


def _looks_like_filtering_state(raw_state: object) -> bool:
    return all(
        hasattr(raw_state, attr)
        for attr in ("h_loc", "h_scale", "steps_seen")
    )


def _resolve_steps_seen(value: object) -> int:
    if isinstance(value, bool):
        raise ConfigError("batch.filtering_state.steps_seen must be an integer")
    try:
        steps_seen = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "batch.filtering_state.steps_seen must be an integer"
        ) from exc
    if steps_seen < 0:
        raise ConfigError(
            "batch.filtering_state.steps_seen must be non-negative"
        )
    return steps_seen
