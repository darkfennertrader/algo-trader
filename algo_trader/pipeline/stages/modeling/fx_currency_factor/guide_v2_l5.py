from __future__ import annotations
# pylint: disable=duplicate-code

import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from algo_trader.domain import ConfigError
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
)
from algo_trader.pipeline.stages.modeling.factor.guide_l13 import (
    _GainLocalEncodingPlan,
    _build_gain_inputs,
    _build_gain_local_sites,
)

from .guide_v2_l2 import (
    RegimeEncoder,
    V2L2RuntimeBatch,
    _GuideContext,
    _decode_local_parameters,
    _next_steps_seen_v2_l2,
    _resolve_anchor_currency,
    _resolve_currency_names,
    _sample_alpha,
    _sample_c,
    _sample_currency_macro_hyperpriors,
    _sample_factor_loading_scale,
    _sample_feature_sites,
    _sample_regime_sites_v2,
    _sample_s_u,
    _sample_scale_sites_v2,
    _sample_sigma_hierarchy,
    _sample_sigma_pair_delta,
    _sample_tau0,
    _sample_lambda,
    build_v2_l2_runtime_batch,
)

_INIT_LOGSCALE = 0.30
_INIT_ALPHA_CENTER_LOC = math.log(0.05)
_INIT_TAU_THETA_LOC = math.log(0.05)

V2L5RuntimeBatch = V2L2RuntimeBatch


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class StructuralPosteriorMeans:
    alpha_currency: torch.Tensor
    theta_currency: torch.Tensor
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    gamma_currency: torch.Tensor
    B_currency: torch.Tensor
    s_u_mean: torch.Tensor
    currency_names: tuple[str, ...]
    anchor_currency: str

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "alpha_currency": self.alpha_currency.detach(),
            "theta_currency": self.theta_currency.detach(),
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "gamma_currency": self.gamma_currency.detach(),
            "B_currency": self.B_currency.detach(),
            "s_u_mean": self.s_u_mean.detach(),
            "currency_names": tuple(self.currency_names),
            "anchor_currency": self.anchor_currency,
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeans":
        tensor_keys = (
            "alpha_currency",
            "theta_currency",
            "alpha",
            "sigma_idio",
            "w",
            "gamma_currency",
            "B_currency",
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
            alpha_currency=values["alpha_currency"],
            theta_currency=values["theta_currency"],
            alpha=values["alpha"],
            sigma_idio=values["sigma_idio"],
            w=values["w"],
            gamma_currency=values["gamma_currency"],
            B_currency=values["B_currency"],
            s_u_mean=values["s_u_mean"],
            currency_names=currency_names,
            anchor_currency=anchor_currency,
        )


@dataclass(frozen=True)
class V2L5GuideConfig:
    factor_count: int = 2
    phi: float = 0.97
    eps: float = 1e-12
    hidden_dim: int = 64


def build_v2_l5_runtime_batch(
    batch: ModelBatch,
    *,
    currency_names: tuple[str, ...] | None = None,
    anchor_currency: str | None = None,
) -> V2L5RuntimeBatch:
    return build_v2_l2_runtime_batch(
        batch,
        currency_names=currency_names,
        anchor_currency=anchor_currency,
    )


@dataclass
class FXCurrencyFactorGuideV2L5OnlineFiltering(PyroGuide):
    config: V2L5GuideConfig
    _encoder: RegimeEncoder | None = None
    _encoder_input_dim: int | None = None
    _currency_names: tuple[str, ...] | None = None
    _anchor_currency: str | None = None
    _sigma_abs_exposure: torch.Tensor | None = None

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch, context, encoder = self._build_runtime_context(batch)
        pyro.module("fx_currency_factor_v2_l5_online_filtering_encoder", encoder)
        structural = _sample_global_sites(
            context=context,
            factor_count=self.config.factor_count,
        )
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
            steps_seen=_next_steps_seen_v2_l5(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
            alpha_currency=store.get_param("alpha_currency_loc").detach(),
            theta_currency=store.get_param("theta_currency_loc").detach(),
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=self._sigma_idio_summary(use_median=False),
            w=store.get_param("w_loc").detach(),
            gamma_currency=store.get_param("gamma_currency_loc").detach(),
            B_currency=store.get_param("B_currency_loc").detach(),
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
            alpha_currency=store.get_param("alpha_currency_loc").detach(),
            theta_currency=store.get_param("theta_currency_loc").detach(),
            alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
            sigma_idio=self._sigma_idio_summary(use_median=True),
            w=store.get_param("w_loc").detach(),
            gamma_currency=store.get_param("gamma_currency_loc").detach(),
            B_currency=store.get_param("B_currency_loc").detach(),
            s_u_mean=_lognormal_median(store.get_param("s_u_loc")).detach(),
            currency_names=self._require_currency_names(),
            anchor_currency=self._require_anchor_currency(),
        )

    def _build_runtime_context(
        self, batch: ModelBatch
    ) -> tuple[V2L5RuntimeBatch, _GuideContext, RegimeEncoder]:
        runtime_batch = build_v2_l5_runtime_batch(batch)
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
                "V2 L5 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)

    def _set_currency_metadata(self, batch: V2L5RuntimeBatch) -> None:
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


@register_guide("fx_currency_factor_guide_v2_l5_online_filtering")
def build_fx_currency_factor_guide_v2_l5_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FXCurrencyFactorGuideV2L5OnlineFiltering(
        config=_build_guide_config(params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V2L5GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V2L5GuideConfig()
    extra = set(values) - {"factor_count", "phi", "eps", "hidden_dim"}
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_guide_v2_l5_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = V2L5GuideConfig()
    try:
        updated = V2L5GuideConfig(
            factor_count=int(values.get("factor_count", base.factor_count)),
            phi=float(values.get("phi", base.phi)),
            eps=float(values.get("eps", base.eps)),
            hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid fx_currency_factor_guide_v2_l5_online_filtering params",
            context={"params": str(dict(values))},
        ) from exc
    if updated.factor_count <= 0:
        raise ConfigError("factor_count must be positive")
    if updated.hidden_dim <= 0:
        raise ConfigError("hidden_dim must be positive")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError("phi must be in (0, 1)")
    return updated


def _build_context(batch: V2L5RuntimeBatch) -> _GuideContext:
    return _GuideContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )


def _next_steps_seen_v2_l5(batch: V2L5RuntimeBatch) -> int:
    return _next_steps_seen_v2_l2(batch)


def _sample_global_sites(
    *, context: _GuideContext, factor_count: int
) -> Any:
    _sample_tau0(context)
    _sample_lambda(context)
    _sample_c(context)
    _sample_mean_hierarchy_sites(context)
    _sample_sigma_hierarchy(context)
    _sample_currency_macro_hyperpriors(context)
    _sample_factor_loading_scale(context, factor_count)
    _sample_currency_sites(context, factor_count)
    return _GlobalGuideSites(s_u_mean=_sample_s_u(context))


@dataclass(frozen=True)
class _GlobalGuideSites:
    s_u_mean: torch.Tensor


def _sample_mean_hierarchy_sites(context: _GuideContext) -> None:
    _sample_alpha_currency(context)
    _sample_theta_hierarchy(context)
    _sample_tau_alpha_pair(context)
    with pyro.plate("asset", context.A, dim=-2):
        _sample_alpha(context)
        _sample_sigma_pair_delta(context)
        _sample_feature_sites(context)


def _sample_alpha_currency(context: _GuideContext) -> None:
    loc = pyro.param(
        "alpha_currency_loc",
        torch.zeros(context.C, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "alpha_currency_scale",
        torch.full(
            (context.C,),
            0.05,
            device=context.device,
            dtype=context.dtype,
        ),
        constraint=constraints.positive,
    )
    with pyro.plate("currency_alpha", context.C, dim=-1):
        pyro.sample("alpha_currency", dist.Normal(loc, scale))


def _sample_theta_hierarchy(context: _GuideContext) -> None:
    with pyro.plate("feature_theta", context.F, dim=-1):
        theta0_loc = pyro.param(
            "theta0_loc",
            torch.zeros(context.F, device=context.device, dtype=context.dtype),
        )
        theta0_scale = pyro.param(
            "theta0_scale",
            torch.full(
                (context.F,),
                0.05,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("theta0", dist.Normal(theta0_loc, theta0_scale))
        tau_theta_loc = pyro.param(
            "tau_theta_loc",
            torch.full(
                (context.F,),
                _INIT_TAU_THETA_LOC,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        tau_theta_scale = pyro.param(
            "tau_theta_scale",
            torch.full(
                (context.F,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("tau_theta", dist.LogNormal(tau_theta_loc, tau_theta_scale))
    shape = (context.C, context.F)
    with pyro.plate("currency_theta", context.C, dim=-2):
        with pyro.plate("feature_theta_currency", context.F, dim=-1):
            loc = pyro.param(
                "theta_currency_loc",
                torch.zeros(shape, device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "theta_currency_scale",
                torch.full(
                    shape,
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample("theta_currency", dist.Normal(loc, scale))


def _sample_tau_alpha_pair(context: _GuideContext) -> None:
    tau_loc = pyro.param(
        "tau_alpha_pair_loc",
        torch.tensor(
            _INIT_ALPHA_CENTER_LOC,
            device=context.device,
            dtype=context.dtype,
        ),
    )
    tau_scale = pyro.param(
        "tau_alpha_pair_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("tau_alpha_pair", dist.LogNormal(tau_loc, tau_scale))


def _sample_currency_sites(
    context: _GuideContext, factor_count: int
) -> None:
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
        shape_factor = (context.C, factor_count)
        with pyro.plate("currency_factor_loading_k", factor_count, dim=-1):
            loc = pyro.param(
                "B_currency_loc",
                torch.zeros(
                    shape_factor,
                    device=context.device,
                    dtype=context.dtype,
                ),
            )
            scale = pyro.param(
                "B_currency_scale",
                torch.full(
                    shape_factor,
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample("B_currency", dist.Normal(loc, scale))
