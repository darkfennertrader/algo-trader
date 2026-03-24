from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.debug_utils import debug_log
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroGuide,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .guide_v2_l6 import (
    build_v2_l6_runtime_batch,
)
from .predict_v2_l6 import predict_fx_currency_factor_v2_l6
from .shared_v2_l6 import (
    FilteringState,
    StructuralTensorMeans,
    V2L6RuntimeBatch,
    coerce_two_state_tensor,
)


@dataclass(frozen=True)
class MeanPriors:
    alpha_scale: float = 0.02
    sigma_idio_scale: float = 0.05
    sigma_currency_scale: float = 0.20
    tau_sigma_pair_scale: float = 0.15
    gamma0_scale: float = 0.05
    tau_gamma_scale: float = 0.05


@dataclass(frozen=True)
class ShrinkagePriors:
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12


@dataclass(frozen=True)
class FactorPriors:
    broad_factor_count: int = 1
    cross_factor_count: int = 1
    broad_b_scale: float = 0.20
    broad_b_col_shrink_scale: float = 0.50
    cross_b_scale: float = 0.10
    cross_b_col_shrink_scale: float = 0.25


@dataclass(frozen=True)
class RegimeBlockPriors:
    nu: float = 10.0
    phi: float = 0.95
    s_u_df: float = 4.0
    s_u_scale: float = 0.03


@dataclass(frozen=True)
class RegimePriors:
    broad: RegimeBlockPriors = field(default_factory=RegimeBlockPriors)
    cross: RegimeBlockPriors = field(
        default_factory=lambda: RegimeBlockPriors(
            phi=0.985,
            s_u_scale=0.01,
        )
    )
    eps: float = 1e-12


def _default_mean_priors() -> MeanPriors:
    return MeanPriors()


def _default_shrinkage_priors() -> ShrinkagePriors:
    return ShrinkagePriors()


def _default_factor_priors() -> FactorPriors:
    return FactorPriors()


def _default_regime_priors() -> RegimePriors:
    return RegimePriors()


@dataclass(frozen=True)
class V2L6ModelPriors:
    mean: MeanPriors = field(default_factory=_default_mean_priors)
    shrinkage: ShrinkagePriors = field(default_factory=_default_shrinkage_priors)
    factors: FactorPriors = field(default_factory=_default_factor_priors)
    regime: RegimePriors = field(default_factory=_default_regime_priors)


@dataclass(frozen=True)
class _ModelContext:
    batch: V2L6RuntimeBatch
    device: torch.device
    dtype: torch.dtype
    priors: V2L6ModelPriors

    @property
    def T(self) -> int:
        return int(self.batch.X_asset.shape[0])

    @property
    def A(self) -> int:
        return int(self.batch.X_asset.shape[1])

    @property
    def C(self) -> int:
        return int(self.batch.exposure_matrix.shape[1])

    @property
    def F(self) -> int:
        return int(self.batch.X_asset.shape[2])

    @property
    def G(self) -> int:
        return int(self.batch.X_global.shape[1])

    @property
    def K_broad(self) -> int:
        return int(self.priors.factors.broad_factor_count)

    @property
    def K_cross(self) -> int:
        return int(self.priors.factors.cross_factor_count)


@dataclass(frozen=True)
class _ShrinkageSites:
    tau0: torch.Tensor
    lam: torch.Tensor
    c: torch.Tensor


@dataclass(frozen=True)
class _MacroHyperSites:
    gamma0: torch.Tensor
    tau_gamma: torch.Tensor


@dataclass(frozen=True)
class _LoadingScaleSites:
    b_col_broad: torch.Tensor
    b_col_cross: torch.Tensor


@dataclass(frozen=True)
class _RegimeScales:
    broad: torch.Tensor
    cross: torch.Tensor


@dataclass(frozen=True)
class _RegimePath:
    h_broad: torch.Tensor
    h_cross: torch.Tensor


@dataclass(frozen=True)
class _TotalScale:
    broad: torch.Tensor
    cross: torch.Tensor


@dataclass(frozen=True)
class FXCurrencyFactorModelV2L6OnlineFiltering(PyroModel):
    priors: V2L6ModelPriors = field(default_factory=V2L6ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v2_l6_runtime_batch(batch)
        context = _build_context(runtime_batch, self.priors)
        _log_inputs(batch, context)
        shrinkage = _sample_shrinkage(context)
        macro_hypers = _sample_macro_hyperpriors(context)
        loading_scales = _sample_loading_scales(context)
        structural = _sample_structural_sites(
            context,
            shrinkage=shrinkage,
            macro_hypers=macro_hypers,
            loading_scales=loading_scales,
        )
        regime_scales = _sample_regime_scales(context)
        regime_path = _sample_regime_path(context, regime_scales)
        total_scale = _sample_total_scale(context, regime_path, regime_scales)
        obs_dist = _build_observation_distribution(
            context, structural, total_scale
        )
        _sample_observations(context, obs_dist)

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        structural_summaries = getattr(
            guide, "structural_predictive_summaries", None
        )
        if not callable(structural_summaries):
            structural_summaries = getattr(
                guide, "structural_posterior_means", None
            )
        if not callable(structural_summaries):
            return None
        return predict_fx_currency_factor_v2_l6(
            model=self,
            guide=guide,  # type: ignore[arg-type]
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("fx_currency_factor_model_v2_l6_online_filtering")
def build_fx_currency_factor_model_v2_l6_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return FXCurrencyFactorModelV2L6OnlineFiltering(
        priors=_build_model_priors(params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V2L6ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V2L6ModelPriors()
    extra = set(values) - {"mean", "shrinkage", "factors", "regime"}
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_model_v2_l6_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return V2L6ModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        shrinkage=_build_shrinkage_priors(values.get("shrinkage")),
        factors=_build_factor_priors(values.get("factors")),
        regime=_build_regime_priors(values.get("regime")),
    )
def _build_mean_priors(raw: object) -> MeanPriors:
    values = _coerce_mapping(raw, label="model.params.mean")
    base = MeanPriors()
    extra = set(values) - _mean_prior_keys()
    if extra:
        raise ConfigError(
            "Unknown V2 L6 mean priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _mean_prior_keys() -> set[str]:
    return {
        "alpha_scale",
        "sigma_idio_scale",
        "sigma_currency_scale",
        "tau_sigma_pair_scale",
        "gamma0_scale",
        "tau_gamma_scale",
    }


def _build_shrinkage_priors(raw: object) -> ShrinkagePriors:
    values = _coerce_mapping(raw, label="model.params.shrinkage")
    base = ShrinkagePriors()
    extra = set(values) - {
        "tau0_scale",
        "lambda_scale",
        "kappa_scale",
        "c_scale",
        "eps",
    }
    if extra:
        raise ConfigError(
            "Unknown V2 L6 shrinkage priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _build_factor_priors(raw: object) -> FactorPriors:
    values = _coerce_mapping(raw, label="model.params.factors")
    base = FactorPriors()
    extra = set(values) - {
        "broad_factor_count",
        "cross_factor_count",
        "broad_b_scale",
        "broad_b_col_shrink_scale",
        "cross_b_scale",
        "cross_b_col_shrink_scale",
    }
    if extra:
        raise ConfigError(
            "Unknown V2 L6 factor priors",
            context={"params": ", ".join(sorted(extra))},
        )
    updated = _replace_dataclass(base, values)
    if updated.broad_factor_count <= 0 or updated.cross_factor_count <= 0:
        raise ConfigError("v2_l6 factor counts must be positive")
    return updated


def _build_regime_priors(raw: object) -> RegimePriors:
    values = _coerce_mapping(raw, label="model.params.regime")
    base = RegimePriors()
    extra = set(values) - {
        "nu_broad",
        "nu_cross",
        "phi_broad",
        "phi_cross",
        "s_u_broad_df",
        "s_u_cross_df",
        "s_u_broad_scale",
        "s_u_cross_scale",
        "eps",
    }
    if extra:
        raise ConfigError(
            "Unknown V2 L6 regime priors",
            context={"params": ", ".join(sorted(extra))},
        )
    broad = _build_regime_block(
        base=base.broad,
        values={
            "nu": values.get("nu_broad", base.broad.nu),
            "phi": values.get("phi_broad", base.broad.phi),
            "s_u_df": values.get("s_u_broad_df", base.broad.s_u_df),
            "s_u_scale": values.get("s_u_broad_scale", base.broad.s_u_scale),
        },
        label="broad",
    )
    cross = _build_regime_block(
        base=base.cross,
        values={
            "nu": values.get("nu_cross", base.cross.nu),
            "phi": values.get("phi_cross", base.cross.phi),
            "s_u_df": values.get("s_u_cross_df", base.cross.s_u_df),
            "s_u_scale": values.get("s_u_cross_scale", base.cross.s_u_scale),
        },
        label="cross",
    )
    updated = RegimePriors(
        broad=broad,
        cross=cross,
        eps=float(values.get("eps", base.eps)),
    )
    return updated


def _build_regime_block(
    *,
    base: RegimeBlockPriors,
    values: Mapping[str, Any],
    label: str,
) -> RegimeBlockPriors:
    updated = _replace_dataclass(base, values)
    if updated.nu <= 2.0:
        raise ConfigError(f"v2_l6 regime {label}.nu must be > 2")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError(f"v2_l6 regime {label}.phi must be in (0, 1)")
    return updated


def _replace_dataclass(base: Any, values: Mapping[str, Any]) -> Any:
    try:
        return replace(
            base,
            **{key: _coerce_number(value) for key, value in values.items()},
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid V2 L6 model params",
            context={"params": str(dict(values))},
        ) from exc


def _coerce_number(value: Any) -> float | int:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid here")
    if isinstance(value, int):
        return value
    return float(value)


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(value)


def _build_context(
    batch: V2L6RuntimeBatch, priors: V2L6ModelPriors
) -> _ModelContext:
    return _ModelContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        priors=priors,
    )


def _log_inputs(batch: ModelBatch, context: _ModelContext) -> None:
    debug_log(batch.debug, "Inputs:")
    debug_log(batch.debug, "X_asset shape: %s", tuple(context.batch.X_asset.shape))
    debug_log(batch.debug, "X_global shape: %s", tuple(context.batch.X_global.shape))
    debug_log(
        batch.debug,
        "exposure_matrix [A, C]: %s anchor=%s currencies=%s",
        tuple(context.batch.exposure_matrix.shape),
        context.batch.anchor_currency,
        context.batch.currency_names,
    )


def _sample_shrinkage(context: _ModelContext) -> _ShrinkageSites:
    tau0 = pyro.sample(
        "tau0",
        dist.HalfCauchy(
            torch.tensor(
                context.priors.shrinkage.tau0_scale,
                device=context.device,
                dtype=context.dtype,
            )
        ),
    )
    with pyro.plate("feature", context.F, dim=-1):
        lam = pyro.sample(
            "lambda",
            dist.HalfCauchy(
                torch.full(
                    (context.F,),
                    context.priors.shrinkage.lambda_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    c = pyro.sample(
        "c",
        dist.HalfCauchy(
            torch.tensor(
                context.priors.shrinkage.c_scale,
                device=context.device,
                dtype=context.dtype,
            )
        ),
    )
    return _ShrinkageSites(tau0=tau0, lam=lam, c=c)


def _sample_macro_hyperpriors(context: _ModelContext) -> _MacroHyperSites:
    with pyro.plate("global_feature_currency", context.G, dim=-1):
        gamma0 = pyro.sample(
            "gamma0",
            dist.Normal(
                torch.zeros(context.G, device=context.device, dtype=context.dtype),
                torch.full(
                    (context.G,),
                    context.priors.mean.gamma0_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
        tau_gamma = pyro.sample(
            "tau_gamma",
            dist.HalfNormal(
                torch.full(
                    (context.G,),
                    context.priors.mean.tau_gamma_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    return _MacroHyperSites(gamma0=gamma0, tau_gamma=tau_gamma)


def _sample_loading_scales(context: _ModelContext) -> _LoadingScaleSites:
    with pyro.plate("currency_factor_loading_col_broad", context.K_broad, dim=-1):
        b_col_broad = pyro.sample(
            "b_col_broad",
            dist.HalfNormal(
                torch.full(
                    (context.K_broad,),
                    context.priors.factors.broad_b_col_shrink_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    with pyro.plate("currency_factor_loading_col_cross", context.K_cross, dim=-1):
        b_col_cross = pyro.sample(
            "b_col_cross",
            dist.HalfNormal(
                torch.full(
                    (context.K_cross,),
                    context.priors.factors.cross_b_col_shrink_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    return _LoadingScaleSites(
        b_col_broad=b_col_broad,
        b_col_cross=b_col_cross,
    )


def _sample_structural_sites(
    context: _ModelContext,
    *,
    shrinkage: _ShrinkageSites,
    macro_hypers: _MacroHyperSites,
    loading_scales: _LoadingScaleSites,
) -> StructuralTensorMeans:
    sigma0 = pyro.sample(
        "sigma0",
        dist.Normal(
            torch.tensor(
                _sigma0_prior_loc(context),
                device=context.device,
                dtype=context.dtype,
            ),
            torch.tensor(0.50, device=context.device, dtype=context.dtype),
        ),
    )
    tau_sigma_pair = pyro.sample(
        "tau_sigma_pair",
        dist.HalfNormal(
            torch.tensor(
                context.priors.mean.tau_sigma_pair_scale,
                device=context.device,
                dtype=context.dtype,
            )
        ),
    )
    with pyro.plate("currency_sigma", context.C, dim=-1):
        sigma_currency = pyro.sample(
            "sigma_currency",
            dist.Normal(
                torch.zeros(context.C, device=context.device, dtype=context.dtype),
                torch.full(
                    (context.C,),
                    context.priors.mean.sigma_currency_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
    with pyro.plate("asset", context.A, dim=-2):
        alpha = _sample_alpha_site(context)
        sigma_pair_delta = pyro.sample(
            "sigma_pair_delta",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                tau_sigma_pair.unsqueeze(-1),
            ),
        )
        w = _sample_feature_weights(context, shrinkage)
    sigma_idio = _compose_sigma_idio(
        context=context,
        sigma0=sigma0,
        sigma_currency=sigma_currency,
        sigma_pair_delta=sigma_pair_delta.squeeze(-1),
    )
    with pyro.plate("currency", context.C, dim=-2):
        with pyro.plate("currency_global_loading", context.G, dim=-1):
            gamma_currency = pyro.sample(
                "gamma_currency",
                dist.Normal(macro_hypers.gamma0, macro_hypers.tau_gamma),
            )
        with pyro.plate("currency_factor_loading_k_broad", context.K_broad, dim=-1):
            B_currency_broad = pyro.sample(
                "B_currency_broad",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(
                        context.priors.factors.broad_b_scale,
                        device=context.device,
                        dtype=context.dtype,
                    )
                    * loading_scales.b_col_broad,
                ),
            )
        with pyro.plate("currency_factor_loading_k_cross", context.K_cross, dim=-1):
            B_currency_cross = pyro.sample(
                "B_currency_cross",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(
                        context.priors.factors.cross_b_scale,
                        device=context.device,
                        dtype=context.dtype,
                    )
                    * loading_scales.b_col_cross,
                ),
            )
    return StructuralTensorMeans(
        alpha=alpha,
        sigma_idio=sigma_idio,
        w=w,
        gamma_currency=gamma_currency,
        B_currency_broad=B_currency_broad,
        B_currency_cross=B_currency_cross,
    )


def _sample_alpha_site(context: _ModelContext) -> torch.Tensor:
    return pyro.sample(
        "alpha",
        dist.Normal(
            torch.tensor(0.0, device=context.device, dtype=context.dtype),
            torch.tensor(
                context.priors.mean.alpha_scale,
                device=context.device,
                dtype=context.dtype,
            ),
        ),
    )


def _sample_feature_weights(
    context: _ModelContext,
    shrinkage: _ShrinkageSites,
) -> torch.Tensor:
    with pyro.plate("feature_w", context.F, dim=-1):
        kappa = pyro.sample(
            "kappa",
            dist.HalfCauchy(
                torch.tensor(
                    context.priors.shrinkage.kappa_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
        w_scale = _regularized_horseshoe_scale(
            tau0=shrinkage.tau0,
            lam=shrinkage.lam,
            kappa=kappa,
            c=shrinkage.c,
            eps=context.priors.shrinkage.eps,
        )
        return pyro.sample(
            "w",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                w_scale,
            ),
        )


def _regularized_horseshoe_scale(
    *,
    tau0: torch.Tensor,
    lam: torch.Tensor,
    kappa: torch.Tensor,
    c: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    base = lam * kappa
    base_sq = base.pow(2)
    c_sq = c.pow(2)
    tau_sq = tau0.pow(2)
    lam_tilde = torch.sqrt((c_sq * base_sq) / (c_sq + tau_sq * base_sq + eps))
    return tau0 * lam_tilde


def _sigma0_prior_loc(context: _ModelContext) -> float:
    return float(
        torch.log(torch.tensor(context.priors.mean.sigma_idio_scale)).item()
    )


def _compose_sigma_idio(
    *,
    context: _ModelContext,
    sigma0: torch.Tensor,
    sigma_currency: torch.Tensor,
    sigma_pair_delta: torch.Tensor,
) -> torch.Tensor:
    log_sigma = (
        sigma0
        + context.batch.exposure_matrix.abs() @ sigma_currency
        + sigma_pair_delta
    )
    return torch.exp(log_sigma)


def _sample_regime_scales(context: _ModelContext) -> _RegimeScales:
    broad = pyro.sample(
        "s_u_broad",
        _half_student_t(
            df=context.priors.regime.broad.s_u_df,
            scale=context.priors.regime.broad.s_u_scale,
            device=context.device,
            dtype=context.dtype,
        ),
    )
    cross = pyro.sample(
        "s_u_cross",
        _half_student_t(
            df=context.priors.regime.cross.s_u_df,
            scale=context.priors.regime.cross.s_u_scale,
            device=context.device,
            dtype=context.dtype,
        ),
    )
    return _RegimeScales(broad=broad, cross=cross)


def _sample_regime_path(
    context: _ModelContext, regime_scales: _RegimeScales
) -> _RegimePath:
    initial = _sample_initial_regime(context, regime_scales)
    h_broad = [initial[0]]
    h_cross = [initial[1]]
    phi = _phi_vector(context)
    for index in range(1, context.T):
        prev = torch.stack([h_broad[-1], h_cross[-1]])
        current = pyro.sample(
            f"h_{index + 1}",
            dist.Normal(phi * prev, _scale_vector(context, regime_scales)).to_event(1),
        )
        h_broad.append(current[0])
        h_cross.append(current[1])
    return _RegimePath(
        h_broad=torch.stack(h_broad),
        h_cross=torch.stack(h_cross),
    )


def _sample_initial_regime(
    context: _ModelContext, regime_scales: _RegimeScales
) -> tuple[torch.Tensor, torch.Tensor]:
    phi = _phi_vector(context)
    scales = _scale_vector(context, regime_scales)
    if context.batch.filtering_state is None:
        denom = _stationary_denom(context, phi)
        current = pyro.sample(
            "h_1",
            dist.Normal(
                torch.zeros(2, device=context.device, dtype=context.dtype),
                scales / torch.sqrt(denom),
            ).to_event(1),
        )
        return current[0], current[1]
    loc, scale = _propagate_filtering_state(
        context=context,
        filtering_state=context.batch.filtering_state,
        phi=phi,
        regime_scales=scales,
    )
    current = pyro.sample("h_1", dist.Normal(loc, scale).to_event(1))
    return current[0], current[1]


def _propagate_filtering_state(
    *,
    context: _ModelContext,
    filtering_state: FilteringState,
    phi: torch.Tensor,
    regime_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_loc = coerce_two_state_tensor(
        filtering_state.h_loc,
        device=context.device,
        dtype=context.dtype,
    )
    h_scale = coerce_two_state_tensor(
        filtering_state.h_scale,
        device=context.device,
        dtype=context.dtype,
    )
    predicted_var = (
        phi.pow(2) * h_scale.pow(2)
        + regime_scales.pow(2)
        + context.priors.regime.eps
    )
    return phi * h_loc, torch.sqrt(predicted_var)


def _sample_total_scale(
    context: _ModelContext,
    regime_path: _RegimePath,
    regime_scales: _RegimeScales,
) -> _TotalScale:
    nu_half = _nu_half_vector(context)
    v = pyro.sample(
        "v",
        dist.Gamma(
            nu_half.unsqueeze(0).expand(context.T, -1),
            nu_half.unsqueeze(0).expand(context.T, -1),
        ).to_event(2),
    )
    phi = _phi_vector(context)
    scales = _scale_vector(context, regime_scales)
    denom = _stationary_denom(context, phi)
    var_h = scales.pow(2) / denom
    h = torch.stack([regime_path.h_broad, regime_path.h_cross], dim=-1)
    u = torch.exp(h - 0.5 * var_h.unsqueeze(0)) * v
    return _TotalScale(broad=u[:, 0], cross=u[:, 1])


def _stationary_denom(
    context: _ModelContext, phi: torch.Tensor
) -> torch.Tensor:
    return (
        torch.ones(2, device=context.device, dtype=context.dtype)
        - phi.pow(2)
        + context.priors.regime.eps
    )


def _build_observation_distribution(
    context: _ModelContext,
    structural: StructuralTensorMeans,
    total_scale: _TotalScale,
) -> dist.LowRankMultivariateNormal:
    alpha_vec = structural.alpha.squeeze(-1)
    sigma_vec = structural.sigma_idio
    mu_asset = (context.batch.X_asset * structural.w.unsqueeze(0)).sum(dim=-1)
    mu_currency = context.batch.X_global @ structural.gamma_currency.transpose(0, 1)
    mu_global = mu_currency @ context.batch.exposure_matrix.transpose(0, 1)
    mu = alpha_vec.unsqueeze(0) + mu_asset + mu_global
    pair_factor_broad = context.batch.exposure_matrix @ structural.B_currency_broad
    pair_factor_cross = context.batch.exposure_matrix @ structural.B_currency_cross
    broad_factor = pair_factor_broad.unsqueeze(0) * torch.rsqrt(
        total_scale.broad
    ).view(context.T, 1, 1)
    cross_factor = pair_factor_cross.unsqueeze(0) * torch.rsqrt(
        total_scale.cross
    ).view(context.T, 1, 1)
    cov_factor = torch.cat([broad_factor, cross_factor], dim=-1)
    cov_diag = sigma_vec.pow(2).unsqueeze(0).expand(context.T, context.A)
    return dist.LowRankMultivariateNormal(
        loc=mu,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    )


def _sample_observations(
    context: _ModelContext, obs_dist: dist.LowRankMultivariateNormal
) -> None:
    with pyro.plate("time", context.T, dim=-1):
        if context.batch.time_mask is None and context.batch.obs_scale is None:
            pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
            return
        if context.batch.time_mask is None:
            obs_scale = context.batch.obs_scale
            if obs_scale is None:
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
                return
            with _scale_context(obs_scale):
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
            return
        with _mask_context(context.batch.time_mask):
            if context.batch.obs_scale is None:
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
                return
            with _scale_context(context.batch.obs_scale):
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)


def _phi_vector(context: _ModelContext) -> torch.Tensor:
    return torch.tensor(
        [
            context.priors.regime.broad.phi,
            context.priors.regime.cross.phi,
        ],
        device=context.device,
        dtype=context.dtype,
    )


def _nu_half_vector(context: _ModelContext) -> torch.Tensor:
    return torch.tensor(
        [
            context.priors.regime.broad.nu / 2.0,
            context.priors.regime.cross.nu / 2.0,
        ],
        device=context.device,
        dtype=context.dtype,
    )


def _scale_vector(
    context: _ModelContext, regime_scales: _RegimeScales
) -> torch.Tensor:
    return torch.stack(
        [regime_scales.broad, regime_scales.cross]
    ).to(device=context.device, dtype=context.dtype)
def _half_student_t(
    *, df: float, scale: float, device: torch.device, dtype: torch.dtype
) -> dist.FoldedDistribution:
    base = dist.StudentT(
        torch.tensor(df, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(scale, device=device, dtype=dtype),
    )
    return dist.FoldedDistribution(base)


@contextmanager
def _scale_context(scale: float) -> Iterator[None]:
    with _managed_context(poutine.scale(scale=scale)):
        yield


@contextmanager
def _mask_context(mask: torch.BoolTensor) -> Iterator[None]:
    with _managed_context(poutine.mask(mask=mask)):
        yield


@contextmanager
def _managed_context(handler_obj: object) -> Iterator[None]:
    handler = cast(AbstractContextManager[None], handler_obj)
    enter = cast(Callable[[], object], getattr(handler, "__enter__"))
    exit_handler = cast(
        Callable[[object | None, object | None, object | None], object],
        getattr(handler, "__exit__"),
    )
    enter()
    try:
        yield
    finally:
        exit_handler(None, None, None)
