from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.batch_utils import resolve_batch_shape
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from algo_trader.pipeline.stages.modeling.factor.guide_l11 import (
    _coerce_mapping,
    _lognormal_mean,
    _lognormal_median,
    _resolve_filtering_state,
    _resolve_time_mask,
    _resolve_y_input,
)
from algo_trader.pipeline.stages.modeling.factor.guide_l13 import (
    _build_gain_inputs,
    _build_gain_local_sites,
    _decode_gain_components,
    _GainLocalEncodingPlan,
    _split_gain_slices,
)
from .guide_v2_l2 import (
    build_currency_structure,
    _validated_fx_asset_names,
)
from .shared_v2_l6 import (
    CurrencyPosteriorMetadata,
    FilteringState,
    RegimePosteriorMeans,
    RuntimeCurrencyMetadata,
    RuntimeObservations,
    StructuralPosteriorMeans,
    StructuralTensorMeans,
    V2L6RuntimeBatch,
    coerce_two_state_tensor,
)

_MIN_SCALE = 1e-4
_MAX_H_GAIN = 0.35
_MAX_H_LOG_SCALE_STEP = 0.40
_MAX_H_SCALE = 0.25
_INIT_LOGSCALE = 0.30
_INIT_TAU0_LOC = -4.76
_INIT_SIGMA_LOC = math.log(0.03)
_INIT_W_SCALE = 0.02
_INIT_POSITIVE_LOC = math.log(0.25)
_INIT_S_U_BROAD_LOC = math.log(0.02)
_INIT_S_U_CROSS_LOC = math.log(0.01)


@dataclass(frozen=True)
class V2L6GuideConfig:
    broad_factor_count: int = 1
    cross_factor_count: int = 1
    phi_broad: float = 0.95
    phi_cross: float = 0.985
    eps: float = 1e-12
    hidden_dim: int = 64


@dataclass(frozen=True)
class _GuideContext:
    batch: V2L6RuntimeBatch
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
    def C(self) -> int:
        return int(self.batch.currency_count)

    @property
    def encoder_input_dim(self) -> int:
        return 2 * self.F + self.G + 8


@dataclass(frozen=True)
class _GlobalGuideSites:
    s_u_broad_mean: torch.Tensor
    s_u_cross_mean: torch.Tensor


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
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._network(inputs)


@dataclass
class FXCurrencyFactorGuideV2L6OnlineFiltering(PyroGuide):
    config: V2L6GuideConfig
    _encoder: RegimeEncoder | None = None
    _encoder_input_dim: int | None = None
    _currency_names: tuple[str, ...] | None = None
    _anchor_currency: str | None = None
    _sigma_abs_exposure: torch.Tensor | None = None

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch, context, encoder = self._build_runtime_context(batch)
        pyro.module("fx_currency_factor_v2_l6_online_filtering_encoder", encoder)
        structural = _sample_global_sites(context=context, config=self.config)
        local = _encode_local_sites(
            context=context,
            encoder=encoder,
            s_u_mean=torch.stack(
                [structural.s_u_broad_mean, structural.s_u_cross_mean]
            ),
            phi=_phi_vector(self.config, context),
            eps=self.config.eps,
        )
        _sample_regime_sites(local)
        _sample_scale_sites(local)
        self._set_currency_metadata(runtime_batch)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch, context, encoder = self._build_runtime_context(batch)
        structural = self.structural_posterior_means()
        local = _encode_local_sites(
            context=context,
            encoder=encoder,
            s_u_mean=torch.stack(
                [structural.s_u_broad_mean, structural.s_u_cross_mean]
            ),
            phi=_phi_vector(self.config, context),
            eps=self.config.eps,
        )
        self._set_currency_metadata(runtime_batch)
        return FilteringState(
            h_loc=local.h_loc[-1].detach(),
            h_scale=local.h_scale[-1].detach(),
            steps_seen=_next_steps_seen_v2_l6(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            tensors=StructuralTensorMeans(
                alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
                sigma_idio=self._sigma_idio_summary(use_median=False),
                w=store.get_param("w_loc").detach(),
                gamma_currency=store.get_param("gamma_currency_loc").detach(),
                B_currency_broad=store.get_param("B_currency_broad_loc").detach(),
                B_currency_cross=store.get_param("B_currency_cross_loc").detach(),
            ),
            regime=RegimePosteriorMeans(
                s_u_broad_mean=_lognormal_mean(
                    store.get_param("s_u_broad_loc"),
                    store.get_param("s_u_broad_scale"),
                ).detach(),
                s_u_cross_mean=_lognormal_mean(
                    store.get_param("s_u_cross_loc"),
                    store.get_param("s_u_cross_scale"),
                ).detach(),
            ),
            metadata=CurrencyPosteriorMetadata(
                currency_names=self._require_currency_names(),
                anchor_currency=self._require_anchor_currency(),
            ),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            tensors=StructuralTensorMeans(
                alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
                sigma_idio=self._sigma_idio_summary(use_median=True),
                w=store.get_param("w_loc").detach(),
                gamma_currency=store.get_param("gamma_currency_loc").detach(),
                B_currency_broad=store.get_param("B_currency_broad_loc").detach(),
                B_currency_cross=store.get_param("B_currency_cross_loc").detach(),
            ),
            regime=RegimePosteriorMeans(
                s_u_broad_mean=_lognormal_median(
                    store.get_param("s_u_broad_loc")
                ).detach(),
                s_u_cross_mean=_lognormal_median(
                    store.get_param("s_u_cross_loc")
                ).detach(),
            ),
            metadata=CurrencyPosteriorMetadata(
                currency_names=self._require_currency_names(),
                anchor_currency=self._require_anchor_currency(),
            ),
        )

    def _build_runtime_context(
        self, batch: ModelBatch
    ) -> tuple[V2L6RuntimeBatch, _GuideContext, RegimeEncoder]:
        runtime_batch = build_v2_l6_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        return runtime_batch, context, encoder

    def _require_encoder(self, context: _GuideContext) -> RegimeEncoder:
        if self._encoder is None:
            self._encoder = self._build_encoder(context)
            self._encoder_input_dim = context.encoder_input_dim
        if self._encoder_input_dim != context.encoder_input_dim:
            raise ConfigError(
                "V2 L6 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)

    def _build_encoder(self, context: _GuideContext) -> RegimeEncoder:
        return RegimeEncoder(
            input_dim=context.encoder_input_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(device=context.device, dtype=context.dtype)

    def _set_currency_metadata(self, batch: V2L6RuntimeBatch) -> None:
        self._currency_names = batch.currency_names
        self._anchor_currency = batch.anchor_currency
        self._sigma_abs_exposure = batch.exposure_matrix.detach().abs()

    def _require_currency_names(self) -> tuple[str, ...]:
        if self._currency_names is None:
            raise ConfigError("Currency metadata is not initialized in the guide")
        return self._currency_names

    def _require_anchor_currency(self) -> str:
        if self._anchor_currency is None:
            raise ConfigError("Currency metadata is not initialized in the guide")
        return self._anchor_currency

    def _require_sigma_abs_exposure(self) -> torch.Tensor:
        if self._sigma_abs_exposure is None:
            raise ConfigError("Sigma exposure metadata is not initialized in the guide")
        return self._sigma_abs_exposure

    def _sigma_idio_summary(self, *, use_median: bool) -> torch.Tensor:
        store = pyro.get_param_store()
        sigma0_loc = store.get_param("sigma0_loc")
        sigma_currency_loc = store.get_param("sigma_currency_loc")
        sigma_pair_delta_loc = store.get_param("sigma_pair_delta_loc")
        abs_exposure = self._require_sigma_abs_exposure().to(
            device=sigma_currency_loc.device,
            dtype=sigma_currency_loc.dtype,
        )
        log_loc = (
            sigma0_loc
            + abs_exposure @ sigma_currency_loc
            + sigma_pair_delta_loc.squeeze(-1)
        )
        if use_median:
            return torch.exp(log_loc).detach()
        sigma0_var = store.get_param("sigma0_scale").pow(2)
        sigma_currency_var = abs_exposure.pow(2) @ store.get_param(
            "sigma_currency_scale"
        ).pow(2)
        sigma_pair_delta_var = store.get_param("sigma_pair_delta_scale").pow(2).squeeze(
            -1
        )
        total_var = sigma0_var + sigma_currency_var + sigma_pair_delta_var
        return torch.exp(log_loc + 0.5 * total_var).detach()


@register_guide("fx_currency_factor_guide_v2_l6_online_filtering")
def build_fx_currency_factor_guide_v2_l6_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FXCurrencyFactorGuideV2L6OnlineFiltering(
        config=_build_guide_config(params)
    )


def build_v2_l6_runtime_batch(
    batch: ModelBatch,
    *,
    currency_names: Sequence[str] | None = None,
    anchor_currency: str | None = None,
) -> V2L6RuntimeBatch:
    shape = resolve_batch_shape(batch)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is None:
        raise ConfigError("V2 L6 runtime requires batch.X_asset")
    if X_asset.ndim != 3:
        raise ConfigError("batch.X_asset must have shape [T, A, F]")
    if batch.X_global is None:
        raise ConfigError("V2 L6 runtime requires batch.X_global")
    if batch.X_global.ndim != 2:
        raise ConfigError("batch.X_global must have shape [T, G]")
    if int(batch.X_global.shape[0]) != shape.T:
        raise ConfigError("batch.X_global and targets must align on T")
    asset_names = _validated_fx_asset_names(batch.asset_names, expected=shape.A)
    structure = build_currency_structure(
        asset_names=asset_names,
        device=shape.device,
        dtype=shape.dtype,
        currency_names=currency_names,
        anchor_currency=anchor_currency,
    )
    return V2L6RuntimeBatch(
        X_asset=X_asset.to(device=shape.device, dtype=shape.dtype),
        X_global=batch.X_global.to(device=shape.device, dtype=shape.dtype),
        observations=RuntimeObservations(
            y_input=_resolve_y_input(shape),
            y_obs=shape.y_obs,
            time_mask=_resolve_time_mask(batch, shape.T, shape.A),
            obs_scale=batch.obs_scale,
        ),
        currency=RuntimeCurrencyMetadata(
            exposure_matrix=structure.exposure_matrix,
            currency_names=structure.currency_names,
            anchor_currency=structure.anchor_currency,
            pair_components=structure.pair_components,
        ),
        filtering_state=_resolve_filtering_state(
            batch.filtering_state,
            device=shape.device,
            dtype=shape.dtype,
        ),
    )


def _build_guide_config(params: Mapping[str, Any]) -> V2L6GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V2L6GuideConfig()
    extra = set(values) - {
        "broad_factor_count",
        "cross_factor_count",
        "phi_broad",
        "phi_cross",
        "eps",
        "hidden_dim",
    }
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_guide_v2_l6_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = V2L6GuideConfig()
    try:
        updated = V2L6GuideConfig(
            broad_factor_count=int(
                values.get("broad_factor_count", base.broad_factor_count)
            ),
            cross_factor_count=int(
                values.get("cross_factor_count", base.cross_factor_count)
            ),
            phi_broad=float(values.get("phi_broad", base.phi_broad)),
            phi_cross=float(values.get("phi_cross", base.phi_cross)),
            eps=float(values.get("eps", base.eps)),
            hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid fx_currency_factor_guide_v2_l6_online_filtering params",
            context={"params": str(dict(values))},
        ) from exc
    if updated.broad_factor_count <= 0 or updated.cross_factor_count <= 0:
        raise ConfigError("v2_l6 guide factor counts must be positive")
    if updated.hidden_dim <= 0:
        raise ConfigError("hidden_dim must be positive")
    if not 0.0 < updated.phi_broad < 1.0:
        raise ConfigError("phi_broad must be in (0, 1)")
    if not 0.0 < updated.phi_cross < 1.0:
        raise ConfigError("phi_cross must be in (0, 1)")
    return updated


def _build_context(batch: V2L6RuntimeBatch) -> _GuideContext:
    return _GuideContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )


def _next_steps_seen_v2_l6(batch: V2L6RuntimeBatch) -> int:
    previous = 0
    if batch.filtering_state is not None:
        previous = int(batch.filtering_state.steps_seen)
    return previous + int(batch.y_input.shape[0])


def _sample_global_sites(
    *, context: _GuideContext, config: V2L6GuideConfig
) -> _GlobalGuideSites:
    _sample_tau0(context)
    _sample_lambda(context)
    _sample_c(context)
    _sample_asset_sites(context)
    _sample_currency_macro_hyperpriors(context)
    _sample_factor_loading_scale(context, config)
    _sample_currency_sites(context, config)
    return _GlobalGuideSites(
        s_u_broad_mean=_sample_s_u_broad(context),
        s_u_cross_mean=_sample_s_u_cross(context),
    )


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


def _sample_asset_sites(context: _GuideContext) -> None:
    with pyro.plate("asset", context.A, dim=-2):
        _sample_alpha(context)
        _sample_sigma_pair_delta(context)
        _sample_feature_sites(context)
    _sample_sigma_hierarchy(context)


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


def _sample_sigma_pair_delta(context: _GuideContext) -> None:
    shape = (context.A, 1)
    loc = pyro.param(
        "sigma_pair_delta_loc",
        torch.zeros(shape, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "sigma_pair_delta_scale",
        torch.full(shape, _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("sigma_pair_delta", dist.Normal(loc, scale))


def _sample_sigma_hierarchy(context: _GuideContext) -> None:
    sigma0_loc = pyro.param(
        "sigma0_loc",
        torch.tensor(_INIT_SIGMA_LOC, device=context.device, dtype=context.dtype),
    )
    sigma0_scale = pyro.param(
        "sigma0_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("sigma0", dist.Normal(sigma0_loc, sigma0_scale))
    tau_loc = pyro.param(
        "tau_sigma_pair_loc",
        torch.tensor(_INIT_POSITIVE_LOC, device=context.device, dtype=context.dtype),
    )
    tau_scale = pyro.param(
        "tau_sigma_pair_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("tau_sigma_pair", dist.LogNormal(tau_loc, tau_scale))
    with pyro.plate("currency_sigma", context.C, dim=-1):
        loc = pyro.param(
            "sigma_currency_loc",
            torch.zeros(context.C, device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            "sigma_currency_scale",
            torch.full(
                (context.C,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("sigma_currency", dist.Normal(loc, scale))


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


def _sample_currency_macro_hyperpriors(context: _GuideContext) -> None:
    with pyro.plate("global_feature_currency", context.G, dim=-1):
        gamma0_loc = pyro.param(
            "gamma0_loc",
            torch.zeros(context.G, device=context.device, dtype=context.dtype),
        )
        gamma0_scale = pyro.param(
            "gamma0_scale",
            torch.full((context.G,), 0.05, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("gamma0", dist.Normal(gamma0_loc, gamma0_scale))
        tau_loc = pyro.param(
            "tau_gamma_loc",
            torch.full(
                (context.G,),
                _INIT_POSITIVE_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        tau_scale = pyro.param(
            "tau_gamma_scale",
            torch.full(
                (context.G,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("tau_gamma", dist.LogNormal(tau_loc, tau_scale))


def _sample_factor_loading_scale(
    context: _GuideContext, config: V2L6GuideConfig
) -> None:
    with pyro.plate(
        "currency_factor_loading_col_broad", config.broad_factor_count, dim=-1
    ):
        loc = pyro.param(
            "b_col_broad_loc",
            torch.full(
                (config.broad_factor_count,),
                _INIT_POSITIVE_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        scale = pyro.param(
            "b_col_broad_scale",
            torch.full(
                (config.broad_factor_count,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("b_col_broad", dist.LogNormal(loc, scale))
    with pyro.plate(
        "currency_factor_loading_col_cross", config.cross_factor_count, dim=-1
    ):
        loc = pyro.param(
            "b_col_cross_loc",
            torch.full(
                (config.cross_factor_count,),
                _INIT_POSITIVE_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        scale = pyro.param(
            "b_col_cross_scale",
            torch.full(
                (config.cross_factor_count,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("b_col_cross", dist.LogNormal(loc, scale))


def _sample_currency_sites(
    context: _GuideContext, config: V2L6GuideConfig
) -> None:
    shape_gamma = (context.C, context.G)
    shape_broad = (context.C, config.broad_factor_count)
    shape_cross = (context.C, config.cross_factor_count)
    with pyro.plate("currency", context.C, dim=-2):
        with pyro.plate("currency_global_loading", context.G, dim=-1):
            loc = pyro.param(
                "gamma_currency_loc",
                torch.zeros(shape_gamma, device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "gamma_currency_scale",
                torch.full(shape_gamma, 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample("gamma_currency", dist.Normal(loc, scale))
        with pyro.plate(
            "currency_factor_loading_k_broad", config.broad_factor_count, dim=-1
        ):
            loc = pyro.param(
                "B_currency_broad_loc",
                torch.zeros(shape_broad, device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "B_currency_broad_scale",
                torch.full(shape_broad, 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample("B_currency_broad", dist.Normal(loc, scale))
        with pyro.plate(
            "currency_factor_loading_k_cross", config.cross_factor_count, dim=-1
        ):
            loc = pyro.param(
                "B_currency_cross_loc",
                torch.zeros(shape_cross, device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "B_currency_cross_scale",
                torch.full(shape_cross, 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample("B_currency_cross", dist.Normal(loc, scale))


def _sample_s_u_broad(context: _GuideContext) -> torch.Tensor:
    loc = pyro.param(
        "s_u_broad_loc",
        torch.tensor(_INIT_S_U_BROAD_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "s_u_broad_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("s_u_broad", dist.LogNormal(loc, scale))
    return _lognormal_mean(loc, scale)


def _sample_s_u_cross(context: _GuideContext) -> torch.Tensor:
    loc = pyro.param(
        "s_u_cross_loc",
        torch.tensor(_INIT_S_U_CROSS_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "s_u_cross_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("s_u_cross", dist.LogNormal(loc, scale))
    return _lognormal_mean(loc, scale)


def _encode_local_sites(
    *,
    context: _GuideContext,
    encoder: RegimeEncoder,
    s_u_mean: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
) -> _LocalGuideSites:
    return cast(
        _LocalGuideSites,
        _build_gain_local_sites(
            inputs=_gain_local_inputs((context, encoder, s_u_mean, phi, eps)),
            plan=_gain_local_plan(),
        ),
    )


def _gain_local_inputs(
    spec: tuple[
        _GuideContext, RegimeEncoder, torch.Tensor, torch.Tensor, float
    ],
) -> Any:
    context, encoder, s_u_mean, phi, eps = spec
    gain_kwargs = {
        "context": cast(Any, context),
        "encoder": cast(Any, encoder),
        "s_u_mean": s_u_mean,
        "phi": cast(Any, phi),
        "eps": eps,
    }
    return cast(Any, _build_gain_inputs(**gain_kwargs))


def _gain_local_plan() -> Any:
    return cast(
        Any,
        _GainLocalEncodingPlan(
            decode_step=_decode_local_parameters_args,
            initial_prior_message=cast(Any, _initial_prior_message_args),
            encoder_features=cast(Any, _encoder_features_args),
            propagated_message=cast(Any, _propagated_message_args),
        ),
    )


def _initial_prior_message_args(
    context: _GuideContext,
    s_u_mean: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _initial_prior_message(
        context=context,
        s_u_mean=s_u_mean,
        phi=phi,
        eps=eps,
    )


def _decode_local_parameters_args(
    encoded: torch.Tensor,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_local_parameters(
        encoded=encoded,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _encoder_features_args(
    context: _GuideContext,
    time_index: int,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> torch.Tensor:
    return _encoder_features(
        context=context,
        time_index=time_index,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _propagated_message_args(
    *,
    current_loc: torch.Tensor,
    current_scale: torch.Tensor,
    s_u_mean: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _propagated_message(
        current_loc=current_loc,
        current_scale=current_scale,
        s_u_mean=s_u_mean,
        phi=phi,
        eps=eps,
    )


def _initial_prior_message(
    *,
    context: _GuideContext,
    s_u_mean: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.batch.filtering_state is None:
        denom = torch.ones_like(phi) - phi.pow(2) + eps
        zero = torch.zeros_like(phi)
        return zero, s_u_mean / torch.sqrt(denom)
    h_loc = coerce_two_state_tensor(
        context.batch.filtering_state.h_loc,
        device=context.device,
        dtype=context.dtype,
    )
    h_scale = coerce_two_state_tensor(
        context.batch.filtering_state.h_scale,
        device=context.device,
        dtype=context.dtype,
    )
    prior_var = phi.pow(2) * h_scale.pow(2) + s_u_mean.pow(2) + eps
    return phi * h_loc, torch.sqrt(prior_var)


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
    y_summary = torch.stack(
        [y_t.mean(), y_t.std(unbiased=False), y_t.min(), y_t.max()]
    )
    return torch.cat(
        [
            x_asset_t.mean(dim=0),
            x_asset_t.std(dim=0, unbiased=False),
            x_global_t,
            y_summary,
            prior_loc,
            torch.log(prior_scale + _MIN_SCALE),
        ]
    )


def _decode_local_parameters(
    *,
    encoded: torch.Tensor,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_gain_components(
        slices=_split_gain_slices(encoded, class_count=int(prior_loc.numel())),
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _propagated_message(
    *,
    current_loc: torch.Tensor,
    current_scale: torch.Tensor,
    s_u_mean: torch.Tensor,
    phi: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_var = phi.pow(2) * current_scale.pow(2) + s_u_mean.pow(2) + eps
    return phi * current_loc, torch.sqrt(prior_var)


def _sample_regime_sites(local: _LocalGuideSites) -> None:
    for index, (h_loc_t, h_scale_t) in enumerate(
        zip(local.h_loc, local.h_scale), start=1
    ):
        pyro.sample(f"h_{index}", dist.Normal(h_loc_t, h_scale_t).to_event(1))


def _sample_scale_sites(local: _LocalGuideSites) -> None:
    pyro.sample("v", dist.LogNormal(local.v_loc, local.v_scale).to_event(2))


def _phi_vector(
    config: V2L6GuideConfig, context: _GuideContext
) -> torch.Tensor:
    return torch.tensor(
        [config.phi_broad, config.phi_cross],
        device=context.device,
        dtype=context.dtype,
    )
