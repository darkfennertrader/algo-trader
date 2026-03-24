from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroGuide,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .guide_v2_l5 import (
    FilteringState,
    V2L5RuntimeBatch,
    build_v2_l5_runtime_batch,
)
from .model_v2_l2 import (
    FactorPriors,
    RegimePriors,
    ShrinkagePriors,
    _coerce_mapping,
    _compose_sigma_idio,
    _half_student_t,
    _log_inputs,
    _regularized_horseshoe_scale,
    _sample_loading_scales,
    _sample_macro_hyperpriors,
    _sample_observations,
    _sample_regime_path,
    _sample_shrinkage,
    _sample_total_scale,
)
from .predict_v2_l5 import predict_fx_currency_factor_v2_l5


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class MeanPriors:
    alpha_currency_scale: float = 0.03
    tau_alpha_pair_scale: float = 0.02
    theta0_scale: float = 0.03
    tau_theta_scale: float = 0.05
    sigma_idio_scale: float = 0.05
    sigma_currency_scale: float = 0.20
    tau_sigma_pair_scale: float = 0.15
    gamma0_scale: float = 0.05
    tau_gamma_scale: float = 0.05


@dataclass(frozen=True)
class V2L5ModelPriors:
    mean: MeanPriors = field(default_factory=MeanPriors)
    shrinkage: ShrinkagePriors = field(default_factory=ShrinkagePriors)
    factors: FactorPriors = field(default_factory=FactorPriors)
    regime: RegimePriors = field(default_factory=RegimePriors)


@dataclass(frozen=True)
class _ModelContext:
    batch: V2L5RuntimeBatch
    device: torch.device
    dtype: torch.dtype
    priors: V2L5ModelPriors

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
    def K(self) -> int:
        return int(self.priors.factors.factor_count)


@dataclass(frozen=True)
class _StructuralSites:
    alpha_currency: torch.Tensor
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    theta_currency: torch.Tensor
    w: torch.Tensor
    gamma_currency: torch.Tensor
    B_currency: torch.Tensor


@dataclass(frozen=True)
class _PairMeanCenters:
    alpha: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class FXCurrencyFactorModelV2L5OnlineFiltering(PyroModel):
    priors: V2L5ModelPriors = field(default_factory=V2L5ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v2_l5_runtime_batch(batch)
        context = _build_context(runtime_batch, self.priors)
        _log_inputs(batch, cast(Any, context))
        shrinkage = _sample_shrinkage(cast(Any, context))
        macro_hypers = _sample_macro_hyperpriors(cast(Any, context))
        loading_scales = _sample_loading_scales(cast(Any, context))
        structural = _sample_structural_sites(
            context,
            shrinkage=shrinkage,
            macro_hypers=macro_hypers,
            loading_scales=loading_scales,
        )
        s_u = pyro.sample(
            "s_u",
            _half_student_t(
                df=context.priors.regime.s_u_df,
                scale=context.priors.regime.s_u_scale,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        h = _sample_regime_path(cast(Any, context), s_u)
        u = _sample_total_scale(cast(Any, context), h, s_u)
        obs_dist = _build_observation_distribution(context, structural, u)
        _sample_observations(cast(Any, context), obs_dist)

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
        return predict_fx_currency_factor_v2_l5(
            model=self,
            guide=guide,  # type: ignore[arg-type]
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("fx_currency_factor_model_v2_l5_online_filtering")
def build_fx_currency_factor_model_v2_l5_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return FXCurrencyFactorModelV2L5OnlineFiltering(
        priors=_build_model_priors(params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V2L5ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V2L5ModelPriors()
    extra = set(values) - {"mean", "shrinkage", "factors", "regime"}
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_model_v2_l5_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return V2L5ModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        shrinkage=_build_shrinkage_priors(values.get("shrinkage")),
        factors=_build_factor_priors(values.get("factors")),
        regime=_build_regime_priors(values.get("regime")),
    )


def _build_mean_priors(raw: object) -> MeanPriors:
    values = _coerce_mapping(raw, label="model.params.mean")
    base = MeanPriors()
    extra = set(values) - {
        "alpha_currency_scale",
        "tau_alpha_pair_scale",
        "theta0_scale",
        "tau_theta_scale",
        "sigma_idio_scale",
        "sigma_currency_scale",
        "tau_sigma_pair_scale",
        "gamma0_scale",
        "tau_gamma_scale",
    }
    if extra:
        raise ConfigError(
            "Unknown V2 L5 mean priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


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
            "Unknown V2 L5 shrinkage priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _build_factor_priors(raw: object) -> FactorPriors:
    values = _coerce_mapping(raw, label="model.params.factors")
    base = FactorPriors()
    extra = set(values) - {"factor_count", "b_scale", "b_col_shrink_scale"}
    if extra:
        raise ConfigError(
            "Unknown V2 L5 factor priors",
            context={"params": ", ".join(sorted(extra))},
        )
    updated = _replace_dataclass(base, values)
    if updated.factor_count <= 0:
        raise ConfigError("factor_count must be positive")
    return updated


def _build_regime_priors(raw: object) -> RegimePriors:
    values = _coerce_mapping(raw, label="model.params.regime")
    base = RegimePriors()
    extra = set(values) - {"nu", "phi", "s_u_df", "s_u_scale", "eps"}
    if extra:
        raise ConfigError(
            "Unknown V2 L5 regime priors",
            context={"params": ", ".join(sorted(extra))},
        )
    updated = _replace_dataclass(base, values)
    if updated.nu <= 2.0:
        raise ConfigError("regime.nu must be > 2")
    if not 0.0 < updated.phi < 1.0:
        raise ConfigError("regime.phi must be in (0, 1)")
    return updated


def _replace_dataclass(base: Any, values: Mapping[str, Any]) -> Any:
    try:
        return replace(
            base,
            **{key: _coerce_number(value) for key, value in values.items()},
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid V2 L5 model params",
            context={"params": str(dict(values))},
        ) from exc


def _coerce_number(value: Any) -> float | int:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid here")
    if isinstance(value, int):
        return value
    return float(value)


def _build_context(
    batch: V2L5RuntimeBatch, priors: V2L5ModelPriors
) -> _ModelContext:
    return _ModelContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        priors=priors,
    )


def _sample_structural_sites(
    context: _ModelContext,
    *,
    shrinkage: Any,
    macro_hypers: Any,
    loading_scales: Any,
) -> _StructuralSites:
    # pylint: disable=too-many-locals
    sigma0, tau_sigma_pair, sigma_currency = _sample_sigma_sites(context)
    alpha_currency, theta_currency, tau_alpha_pair = _sample_currency_mean_sites(
        context
    )
    centers = _pair_mean_centers(
        context,
        alpha_currency=alpha_currency,
        theta_currency=theta_currency,
    )
    alpha, sigma_pair_delta, w = _sample_pair_mean_sites(
        context,
        centers=centers,
        shrinkage=shrinkage,
        tau_alpha_pair=tau_alpha_pair,
        tau_sigma_pair=tau_sigma_pair,
    )
    sigma_idio = _compose_sigma_idio(
        context=cast(Any, context),
        sigma0=sigma0,
        sigma_currency=sigma_currency,
        sigma_pair_delta=sigma_pair_delta.squeeze(-1),
    )
    gamma_currency, b_currency = _sample_currency_covariance_sites(
        context,
        macro_hypers=macro_hypers,
        loading_scales=loading_scales,
    )
    return _StructuralSites(
        alpha_currency=alpha_currency,
        alpha=alpha,
        sigma_idio=sigma_idio,
        theta_currency=theta_currency,
        w=w,
        gamma_currency=gamma_currency,
        B_currency=b_currency,
    )


def _sample_sigma_sites(
    context: _ModelContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return sigma0, tau_sigma_pair, sigma_currency


def _sample_currency_mean_sites(
    context: _ModelContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_alpha_pair = pyro.sample(
        "tau_alpha_pair",
        dist.HalfNormal(
            torch.tensor(
                context.priors.mean.tau_alpha_pair_scale,
                device=context.device,
                dtype=context.dtype,
            )
        ),
    )
    with pyro.plate("feature_theta", context.F, dim=-1):
        theta0 = pyro.sample(
            "theta0",
            dist.Normal(
                torch.zeros(context.F, device=context.device, dtype=context.dtype),
                torch.full(
                    (context.F,),
                    context.priors.mean.theta0_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
        tau_theta = pyro.sample(
            "tau_theta",
            dist.HalfNormal(
                torch.full(
                    (context.F,),
                    context.priors.mean.tau_theta_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    with pyro.plate("currency_alpha", context.C, dim=-1):
        alpha_currency = pyro.sample(
            "alpha_currency",
            dist.Normal(
                torch.zeros(context.C, device=context.device, dtype=context.dtype),
                torch.full(
                    (context.C,),
                    context.priors.mean.alpha_currency_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
    with pyro.plate("currency_theta", context.C, dim=-2):
        with pyro.plate("feature_theta_currency", context.F, dim=-1):
            theta_currency = pyro.sample(
                "theta_currency",
                dist.Normal(theta0, tau_theta),
            )
    return alpha_currency, theta_currency, tau_alpha_pair


def _pair_mean_centers(
    context: _ModelContext,
    *,
    alpha_currency: torch.Tensor,
    theta_currency: torch.Tensor,
) -> _PairMeanCenters:
    return _PairMeanCenters(
        alpha=context.batch.exposure_matrix @ alpha_currency,
        weight=context.batch.exposure_matrix @ theta_currency,
    )


def _sample_pair_mean_sites(
    context: _ModelContext,
    *,
    centers: _PairMeanCenters,
    shrinkage: Any,
    tau_alpha_pair: torch.Tensor,
    tau_sigma_pair: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with pyro.plate("asset", context.A, dim=-2):
        alpha = pyro.sample(
            "alpha",
            dist.Normal(centers.alpha.unsqueeze(-1), tau_alpha_pair.unsqueeze(-1)),
        )
        sigma_pair_delta = pyro.sample(
            "sigma_pair_delta",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                tau_sigma_pair.unsqueeze(-1),
            ),
        )
        w = _sample_feature_weights(
            context,
            shrinkage=shrinkage,
            weight_center=centers.weight,
        )
    return alpha, sigma_pair_delta, w


def _sample_currency_covariance_sites(
    context: _ModelContext,
    *,
    macro_hypers: Any,
    loading_scales: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    with pyro.plate("currency", context.C, dim=-2):
        with pyro.plate("currency_global_loading", context.G, dim=-1):
            gamma_currency = pyro.sample(
                "gamma_currency",
                dist.Normal(macro_hypers.gamma0, macro_hypers.tau_gamma),
            )
        with pyro.plate("currency_factor_loading_k", context.K, dim=-1):
            b_currency = pyro.sample(
                "B_currency",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(
                        context.priors.factors.b_scale,
                        device=context.device,
                        dtype=context.dtype,
                    )
                    * loading_scales.b_col,
                ),
            )
    return gamma_currency, b_currency


def _sample_feature_weights(
    context: _ModelContext,
    *,
    shrinkage: Any,
    weight_center: torch.Tensor,
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
        return pyro.sample("w", dist.Normal(weight_center, w_scale))


def _sigma0_prior_loc(context: _ModelContext) -> float:
    return float(torch.log(torch.tensor(context.priors.mean.sigma_idio_scale)).item())


def _build_observation_distribution(
    context: _ModelContext,
    structural: _StructuralSites,
    u: torch.Tensor,
) -> dist.LowRankMultivariateNormal:
    alpha_vec = structural.alpha.squeeze(-1)
    sigma_vec = structural.sigma_idio
    mu_asset = (context.batch.X_asset * structural.w.unsqueeze(0)).sum(dim=-1)
    mu_currency = context.batch.X_global @ structural.gamma_currency.transpose(0, 1)
    mu_global = mu_currency @ context.batch.exposure_matrix.transpose(0, 1)
    mu = alpha_vec.unsqueeze(0) + mu_asset + mu_global
    pair_factor = context.batch.exposure_matrix @ structural.B_currency
    cov_factor = pair_factor.unsqueeze(0) * torch.rsqrt(u).unsqueeze(-1)
    cov_diag = sigma_vec.pow(2).unsqueeze(0).expand(context.T, context.A)
    return dist.LowRankMultivariateNormal(
        loc=mu,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    )
