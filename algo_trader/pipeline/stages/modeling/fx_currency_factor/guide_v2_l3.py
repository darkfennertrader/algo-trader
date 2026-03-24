from __future__ import annotations
# pylint: disable=duplicate-code

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.domain import ConfigError
from algo_trader.infrastructure.data import symbol_directory
from algo_trader.pipeline.stages.modeling.batch_utils import resolve_batch_shape
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from algo_trader.pipeline.stages.modeling.factor.guide_l11 import (
    FilteringState,
    _coerce_mapping,
    _encoder_features as _encoder_features_l11,
    _initial_prior_message as _initial_prior_message_l11,
    _lognormal_mean,
    _lognormal_median,
    _propagated_message as _propagated_message_l11,
    _resolve_filtering_state,
    _resolve_time_mask,
    _resolve_y_input,
)
from algo_trader.pipeline.stages.modeling.factor.guide_l13 import (
    _GainLocalEncodingPlan,
    _build_gain_inputs,
    _build_gain_local_sites,
    _decode_gain_components,
    _unbound_gain_slices,
)

_MIN_SCALE = 1e-4
_INIT_LOGSCALE = 0.30
_INIT_TAU0_LOC = -4.76
_INIT_SIGMA_LOC = math.log(0.03)
_INIT_W_SCALE = 0.02
_INIT_POSITIVE_LOC = math.log(0.25)
_INIT_S_U_LOC = math.log(0.02)
_REPO_ROOT = Path(__file__).resolve().parents[5]
_TICKERS_CONFIG_PATH = _REPO_ROOT / "config" / "tickers.yml"


@dataclass(frozen=True)
class CurrencyStructure:
    asset_names: tuple[str, ...]
    pair_components: tuple[tuple[str, str], ...]
    currency_names: tuple[str, ...]
    anchor_currency: str
    exposure_matrix: torch.Tensor


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class V2L3RuntimeBatch:
    X_asset: torch.Tensor
    X_global: torch.Tensor
    y_input: torch.Tensor
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None
    exposure_matrix: torch.Tensor
    currency_names: tuple[str, ...]
    anchor_currency: str
    pair_components: tuple[tuple[str, str], ...]
    filtering_state: FilteringState | None = None

    @property
    def currency_count(self) -> int:
        return int(self.exposure_matrix.shape[1])


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class StructuralPosteriorMeans:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    gamma_currency: torch.Tensor
    omega_currency: torch.Tensor
    s_u_mean: torch.Tensor
    currency_names: tuple[str, ...]
    anchor_currency: str

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "gamma_currency": self.gamma_currency.detach(),
            "omega_currency": self.omega_currency.detach(),
            "s_u_mean": self.s_u_mean.detach(),
            "currency_names": tuple(self.currency_names),
            "anchor_currency": self.anchor_currency,
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeans":
        tensor_keys = (
            "alpha",
            "sigma_idio",
            "w",
            "gamma_currency",
            "omega_currency",
            "s_u_mean",
        )
        values: dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ConfigError(
                    "structural_posterior_means must include tensor entries",
                    context={"field": key},
                )
            values[key] = value.detach()
        currency_names = _resolve_currency_names(payload.get("currency_names"))
        anchor_currency = _resolve_anchor_currency(
            payload.get("anchor_currency"),
            currency_names=currency_names,
        )
        return cls(
            alpha=values["alpha"],
            sigma_idio=values["sigma_idio"],
            w=values["w"],
            gamma_currency=values["gamma_currency"],
            omega_currency=values["omega_currency"],
            s_u_mean=values["s_u_mean"],
            currency_names=currency_names,
            anchor_currency=anchor_currency,
        )


@dataclass(frozen=True)
class V2L3GuideConfig:
    phi: float = 0.97
    eps: float = 1e-12
    hidden_dim: int = 64


@dataclass(frozen=True)
class _GuideContext:
    batch: V2L3RuntimeBatch
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
        return 2 * self.F + self.G + 6


@dataclass(frozen=True)
class _GlobalGuideSites:
    s_u_mean: torch.Tensor


class RegimeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._network(inputs)


def build_v2_l3_runtime_batch(
    batch: ModelBatch,
    *,
    currency_names: Sequence[str] | None = None,
    anchor_currency: str | None = None,
) -> V2L3RuntimeBatch:
    shape = resolve_batch_shape(batch)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is None:
        raise ConfigError("V2 L3 runtime requires batch.X_asset")
    if X_asset.ndim != 3:
        raise ConfigError("batch.X_asset must have shape [T, A, F]")
    if batch.X_global is None:
        raise ConfigError("V2 L3 runtime requires batch.X_global")
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
    return V2L3RuntimeBatch(
        X_asset=X_asset.to(device=shape.device, dtype=shape.dtype),
        X_global=batch.X_global.to(device=shape.device, dtype=shape.dtype),
        y_input=_resolve_y_input(shape),
        y_obs=shape.y_obs,
        time_mask=_resolve_time_mask(batch, shape.T, shape.A),
        obs_scale=batch.obs_scale,
        exposure_matrix=structure.exposure_matrix,
        currency_names=structure.currency_names,
        anchor_currency=structure.anchor_currency,
        pair_components=structure.pair_components,
        filtering_state=_resolve_filtering_state(
            batch.filtering_state,
            device=shape.device,
            dtype=shape.dtype,
        ),
    )


def build_currency_structure(
    *,
    asset_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    currency_names: Sequence[str] | None = None,
    anchor_currency: str | None = None,
) -> CurrencyStructure:
    pair_components = tuple(_split_pair_name(name) for name in asset_names)
    all_currencies = _all_currencies(pair_components)
    chosen_anchor = _choose_anchor_currency(all_currencies, requested=anchor_currency)
    names = _ordered_currency_names(
        all_currencies=all_currencies,
        anchor_currency=chosen_anchor,
        requested=currency_names,
    )
    index_by_currency = {name: index for index, name in enumerate(names)}
    exposure = torch.zeros(
        (len(pair_components), len(names)),
        device=device,
        dtype=dtype,
    )
    for asset_index, (base, quote) in enumerate(pair_components):
        if base != chosen_anchor:
            exposure[asset_index, index_by_currency[base]] = 1.0
        if quote != chosen_anchor:
            exposure[asset_index, index_by_currency[quote]] = -1.0
    return CurrencyStructure(
        asset_names=tuple(str(name) for name in asset_names),
        pair_components=pair_components,
        currency_names=names,
        anchor_currency=chosen_anchor,
        exposure_matrix=exposure,
    )


@dataclass
class FXCurrencyFactorGuideV2L3OnlineFiltering(PyroGuide):
    config: V2L3GuideConfig
    _encoder: RegimeEncoder | None = None
    _encoder_input_dim: int | None = None
    _currency_names: tuple[str, ...] | None = None
    _anchor_currency: str | None = None
    _sigma_abs_exposure: torch.Tensor | None = None

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch, context, encoder = self._build_runtime_context(batch)
        pyro.module("fx_currency_factor_v2_l3_online_filtering_encoder", encoder)
        structural = _sample_global_sites(context=context)
        local = _build_gain_local_sites(
            inputs=cast(
                Any,
                _build_gain_inputs(
                    context=cast(Any, context),
                    encoder=cast(Any, encoder),
                    s_u_mean=structural.s_u_mean,
                    phi=self.config.phi,
                    eps=self.config.eps,
                ),
            ),
            plan=cast(
                Any,
                _GainLocalEncodingPlan(
                    decode_step=lambda encoded, prior_loc, prior_scale: _decode_local_parameters(
                        encoded=encoded,
                        prior_loc=prior_loc,
                        prior_scale=prior_scale,
                    ),
                    initial_prior_message=cast(
                        Any, _initial_prior_message_l11
                    ),
                    encoder_features=cast(Any, _encoder_features_l11),
                    propagated_message=cast(Any, _propagated_message_l11),
                ),
            ),
        )
        _sample_regime_sites_v2(local)
        _sample_scale_sites_v2(local)
        self._set_currency_metadata(runtime_batch)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch, context, encoder = self._build_runtime_context(batch)
        structural = self.structural_posterior_means()
        local = _build_gain_local_sites(
            inputs=cast(
                Any,
                _build_gain_inputs(
                    context=cast(Any, context),
                    encoder=cast(Any, encoder),
                    s_u_mean=structural.s_u_mean,
                    phi=self.config.phi,
                    eps=self.config.eps,
                ),
            ),
            plan=cast(
                Any,
                _GainLocalEncodingPlan(
                    decode_step=lambda encoded, prior_loc, prior_scale: _decode_local_parameters(
                        encoded=encoded,
                        prior_loc=prior_loc,
                        prior_scale=prior_scale,
                    ),
                    initial_prior_message=cast(
                        Any, _initial_prior_message_l11
                    ),
                    encoder_features=cast(Any, _encoder_features_l11),
                    propagated_message=cast(Any, _propagated_message_l11),
                ),
            ),
        )
        self._set_currency_metadata(runtime_batch)
        return FilteringState(
            h_loc=local.h_loc[-1].detach(),
            h_scale=local.h_scale[-1].detach(),
            steps_seen=_next_steps_seen_v2_l3(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=self._sigma_idio_summary(use_median=False),
            w=store.get_param("w_loc").detach(),
            gamma_currency=store.get_param("gamma_currency_loc").detach(),
            omega_currency=self._omega_currency_summary(use_median=False),
            s_u_mean=_lognormal_mean(
                store.get_param("s_u_loc"),
                store.get_param("s_u_scale"),
            ).detach(),
            currency_names=self._require_currency_names(),
            anchor_currency=self._require_anchor_currency(),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=self._sigma_idio_summary(use_median=True),
            w=store.get_param("w_loc").detach(),
            gamma_currency=store.get_param("gamma_currency_loc").detach(),
            omega_currency=self._omega_currency_summary(use_median=True),
            s_u_mean=_lognormal_median(store.get_param("s_u_loc")).detach(),
            currency_names=self._require_currency_names(),
            anchor_currency=self._require_anchor_currency(),
        )

    def _build_runtime_context(
        self, batch: ModelBatch
    ) -> tuple[V2L3RuntimeBatch, _GuideContext, RegimeEncoder]:
        runtime_batch = build_v2_l3_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        return runtime_batch, context, encoder

    def _require_encoder(self, context: _GuideContext) -> RegimeEncoder:
        if self._encoder is None:
            self._encoder = RegimeEncoder(
                input_dim=context.encoder_input_dim,
                hidden_dim=self.config.hidden_dim,
            ).to(device=context.device, dtype=context.dtype)
            self._encoder_input_dim = context.encoder_input_dim
        if self._encoder_input_dim != context.encoder_input_dim:
            raise ConfigError(
                "V2 L3 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)

    def _set_currency_metadata(self, batch: V2L3RuntimeBatch) -> None:
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

    def _omega_currency_summary(self, *, use_median: bool) -> torch.Tensor:
        store = pyro.get_param_store()
        omega_loc = store.get_param("omega_currency_loc")
        if use_median:
            return _lognormal_median(omega_loc).detach()
        return _lognormal_mean(
            omega_loc,
            store.get_param("omega_currency_scale"),
        ).detach()

@register_guide("fx_currency_factor_guide_v2_l3_online_filtering")
def build_fx_currency_factor_guide_v2_l3_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FXCurrencyFactorGuideV2L3OnlineFiltering(
        config=_build_guide_config(params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V2L3GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V2L3GuideConfig()
    extra = set(values) - {"phi", "eps", "hidden_dim"}
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_guide_v2_l3_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = V2L3GuideConfig()
    try:
        updated = V2L3GuideConfig(
            phi=float(values.get("phi", base.phi)),
            eps=float(values.get("eps", base.eps)),
            hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid fx_currency_factor_guide_v2_l3_online_filtering params",
            context={"params": str(dict(values))},
        ) from exc
    if updated.hidden_dim <= 0:
        raise ConfigError("hidden_dim must be positive")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError("phi must be in (0, 1)")
    return updated


def _build_context(batch: V2L3RuntimeBatch) -> _GuideContext:
    return _GuideContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )


def _next_steps_seen_v2_l3(batch: V2L3RuntimeBatch) -> int:
    previous = 0
    if batch.filtering_state is not None:
        previous = int(batch.filtering_state.steps_seen)
    return previous + int(batch.y_input.shape[0])


def _sample_global_sites(*, context: _GuideContext) -> _GlobalGuideSites:
    _sample_tau0(context)
    _sample_lambda(context)
    _sample_c(context)
    _sample_asset_sites(context)
    _sample_currency_macro_hyperpriors(context)
    _sample_currency_sites(context)
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


def _sample_currency_sites(context: _GuideContext) -> None:
    shape = (context.C, context.G)
    with pyro.plate("currency", context.C, dim=-2):
        with pyro.plate("currency_global_loading", context.G, dim=-1):
            loc = pyro.param(
                "gamma_currency_loc",
                torch.zeros(shape, device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "gamma_currency_scale",
                torch.full(
                    shape,
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample("gamma_currency", dist.Normal(loc, scale))
    loc = pyro.param(
        "omega_currency_loc",
        torch.full(
            (context.C,),
            _INIT_POSITIVE_LOC,
            device=context.device,
            dtype=context.dtype,
        ),
    )
    scale = pyro.param(
        "omega_currency_scale",
        torch.full(
            (context.C,),
            _INIT_LOGSCALE,
            device=context.device,
            dtype=context.dtype,
        ),
        constraint=constraints.positive,
    )
    pyro.sample("omega_currency", dist.LogNormal(loc, scale).to_event(1))


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


def _decode_local_parameters(
    *,
    encoded: torch.Tensor,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_gain_components(
        slices=_unbound_gain_slices(encoded),
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _sample_regime_sites_v2(local: Any) -> None:
    for index, (h_loc_t, h_scale_t) in enumerate(
        zip(local.h_loc, local.h_scale), start=1
    ):
        pyro.sample(f"h_{index}", dist.Normal(h_loc_t, h_scale_t))


def _sample_scale_sites_v2(local: Any) -> None:
    with pyro.plate("time_v", int(local.v_loc.shape[0]), dim=-1):
        pyro.sample("v", dist.LogNormal(local.v_loc, local.v_scale))


def _validated_fx_asset_names(
    asset_names: Sequence[str] | None, *, expected: int
) -> tuple[str, ...]:
    if asset_names is None:
        raise ConfigError("V2 L3 requires batch.asset_names for FX validation")
    names = tuple(str(name) for name in asset_names)
    if len(names) != expected:
        raise ConfigError("batch.asset_names must align with the asset dimension")
    fx_names = _configured_fx_asset_names()
    invalid = [name for name in names if name not in fx_names]
    if invalid:
        raise ConfigError(
            "V2 L3 is FX-only and received non-FX assets",
            context={"assets": ", ".join(sorted(invalid))},
        )
    return names


def _split_pair_name(name: str) -> tuple[str, str]:
    parts = tuple(str(part).strip().upper() for part in name.split("."))
    if len(parts) != 2 or any(not part for part in parts):
        raise ConfigError(
            "FX asset names must use BASE.QUOTE format",
            context={"asset": name},
        )
    return parts[0], parts[1]


def _all_currencies(
    pair_components: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    currencies = {currency for pair in pair_components for currency in pair}
    if len(currencies) < 2:
        raise ConfigError("FX runtime requires at least two distinct currencies")
    return tuple(sorted(currencies))


def _choose_anchor_currency(
    all_currencies: Sequence[str], *, requested: str | None
) -> str:
    if requested is not None:
        anchor = str(requested).strip().upper()
        if anchor not in all_currencies:
            raise ConfigError(
                "Requested anchor_currency is not present in the FX universe",
                context={"anchor_currency": anchor},
            )
        return anchor
    if "USD" in all_currencies:
        return "USD"
    return str(sorted(all_currencies)[0])


def _ordered_currency_names(
    *,
    all_currencies: Sequence[str],
    anchor_currency: str,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    non_anchor = tuple(
        currency for currency in sorted(all_currencies) if currency != anchor_currency
    )
    if requested is None:
        return non_anchor
    names = tuple(str(name).strip().upper() for name in requested)
    if anchor_currency in names:
        raise ConfigError("currency_names must exclude the anchor currency")
    if set(names) != set(non_anchor):
        raise ConfigError(
            "currency_names must match the non-anchor FX currencies exactly",
            context={
                "expected": ", ".join(non_anchor),
                "received": ", ".join(names),
            },
        )
    return names


def _resolve_currency_names(raw_value: object) -> tuple[str, ...]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ConfigError("currency_names must be a sequence")
    names = tuple(str(name).strip().upper() for name in raw_value)
    if not names:
        raise ConfigError("currency_names must not be empty")
    return names


def _resolve_anchor_currency(
    raw_value: object, *, currency_names: Sequence[str]
) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError("anchor_currency must be a non-empty string")
    anchor = raw_value.strip().upper()
    if anchor in currency_names:
        raise ConfigError("anchor_currency must not appear in currency_names")
    return anchor


@lru_cache(maxsize=1)
def _configured_fx_asset_names() -> frozenset[str]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG_PATH)
    return frozenset(symbol_directory(ticker) for ticker in config.tickers)
