from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

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

from .guide_v2_l3 import (
    FilteringState,
    V2L3RuntimeBatch,
    build_v2_l3_runtime_batch,
)
from .predict_v2_l3 import predict_fx_currency_factor_v2_l3


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
class CurrencyShockPriors:
    omega_scale: float = 0.20


@dataclass(frozen=True)
class RegimePriors:
    nu: float = 10.0
    phi: float = 0.97
    s_u_df: float = 4.0
    s_u_scale: float = 0.03
    eps: float = 1e-12


@dataclass(frozen=True)
class V2L3ModelPriors:
    mean: MeanPriors = field(default_factory=MeanPriors)
    shrinkage: ShrinkagePriors = field(default_factory=ShrinkagePriors)
    currency_shocks: CurrencyShockPriors = field(default_factory=CurrencyShockPriors)
    regime: RegimePriors = field(default_factory=RegimePriors)


@dataclass(frozen=True)
class _ModelContext:
    batch: V2L3RuntimeBatch
    device: torch.device
    dtype: torch.dtype
    priors: V2L3ModelPriors

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
class _StructuralSites:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    gamma_currency: torch.Tensor
    omega_currency: torch.Tensor


@dataclass(frozen=True)
class FXCurrencyFactorModelV2L3OnlineFiltering(PyroModel):
    priors: V2L3ModelPriors = field(default_factory=V2L3ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v2_l3_runtime_batch(batch)
        context = _build_context(runtime_batch, self.priors)
        _log_inputs(batch, context)
        shrinkage = _sample_shrinkage(context)
        macro_hypers = _sample_macro_hyperpriors(context)
        structural = _sample_structural_sites(
            context,
            shrinkage=shrinkage,
            macro_hypers=macro_hypers,
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
        h = _sample_regime_path(context, s_u)
        u = _sample_total_scale(context, h, s_u)
        obs_dist = _build_observation_distribution(context, structural, u)
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
        return predict_fx_currency_factor_v2_l3(
            model=self,
            guide=guide,  # type: ignore[arg-type]
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("fx_currency_factor_model_v2_l3_online_filtering")
def build_fx_currency_factor_model_v2_l3_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return FXCurrencyFactorModelV2L3OnlineFiltering(
        priors=_build_model_priors(params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V2L3ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V2L3ModelPriors()
    extra = set(values) - {"mean", "shrinkage", "currency_shocks", "regime"}
    if extra:
        raise ConfigError(
            "Unknown fx_currency_factor_model_v2_l3_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return V2L3ModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        shrinkage=_build_shrinkage_priors(values.get("shrinkage")),
        currency_shocks=_build_currency_shock_priors(values.get("currency_shocks")),
        regime=_build_regime_priors(values.get("regime")),
    )


def _build_mean_priors(raw: object) -> MeanPriors:
    values = _coerce_mapping(raw, label="model.params.mean")
    base = MeanPriors()
    extra = set(values) - {
        "alpha_scale",
        "sigma_idio_scale",
        "sigma_currency_scale",
        "tau_sigma_pair_scale",
        "gamma0_scale",
        "tau_gamma_scale",
    }
    if extra:
        raise ConfigError(
            "Unknown V2 L3 mean priors",
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
            "Unknown V2 L3 shrinkage priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _build_currency_shock_priors(raw: object) -> CurrencyShockPriors:
    values = _coerce_mapping(raw, label="model.params.currency_shocks")
    base = CurrencyShockPriors()
    extra = set(values) - {"omega_scale"}
    if extra:
        raise ConfigError(
            "Unknown V2 L3 currency shock priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _build_regime_priors(raw: object) -> RegimePriors:
    values = _coerce_mapping(raw, label="model.params.regime")
    base = RegimePriors()
    extra = set(values) - {"nu", "phi", "s_u_df", "s_u_scale", "eps"}
    if extra:
        raise ConfigError(
            "Unknown V2 L3 regime priors",
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
            "Invalid V2 L3 model params",
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
    batch: V2L3RuntimeBatch, priors: V2L3ModelPriors
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


def _sample_structural_sites(
    context: _ModelContext,
    *,
    shrinkage: _ShrinkageSites,
    macro_hypers: _MacroHyperSites,
) -> _StructuralSites:
    sigma0 = pyro.sample(
        "sigma0",
        dist.Normal(
            torch.tensor(_sigma0_prior_loc(context), device=context.device, dtype=context.dtype),
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
        alpha = pyro.sample(
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
    omega_currency = pyro.sample(
        "omega_currency",
        dist.HalfNormal(
            torch.full(
                (context.C,),
                context.priors.currency_shocks.omega_scale,
                device=context.device,
                dtype=context.dtype,
            )
        ).to_event(1),
    )
    return _StructuralSites(
        alpha=alpha,
        sigma_idio=sigma_idio,
        w=w,
        gamma_currency=gamma_currency,
        omega_currency=omega_currency,
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
    return float(torch.log(torch.tensor(context.priors.mean.sigma_idio_scale)).item())


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


def _sample_regime_path(
    context: _ModelContext, s_u: torch.Tensor
) -> torch.Tensor:
    phi_t = torch.tensor(
        context.priors.regime.phi, device=context.device, dtype=context.dtype
    )
    values = [_sample_initial_regime(context, phi_t, s_u)]
    for index in range(1, context.T):
        values.append(
            pyro.sample(
                f"h_{index + 1}",
                dist.Normal(phi_t * values[-1], s_u),
            )
        )
    return torch.stack(values, dim=0)


def _sample_initial_regime(
    context: _ModelContext, phi_t: torch.Tensor, s_u: torch.Tensor
) -> torch.Tensor:
    filtering_state = context.batch.filtering_state
    if filtering_state is None:
        denom = _stationary_denom(context, phi_t)
        return pyro.sample(
            "h_1",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                s_u / torch.sqrt(denom),
            ),
        )
    loc, scale = _propagate_filtering_state(
        context=context,
        filtering_state=filtering_state,
        phi_t=phi_t,
        s_u=s_u,
    )
    return pyro.sample("h_1", dist.Normal(loc, scale))


def _propagate_filtering_state(
    *,
    context: _ModelContext,
    filtering_state: FilteringState,
    phi_t: torch.Tensor,
    s_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_loc = filtering_state.h_loc.to(device=context.device, dtype=context.dtype)
    h_scale = filtering_state.h_scale.to(
        device=context.device, dtype=context.dtype
    )
    predicted_var = (
        phi_t.pow(2) * h_scale.pow(2)
        + s_u.pow(2)
        + context.priors.regime.eps
    )
    return phi_t * h_loc, torch.sqrt(predicted_var)


def _sample_total_scale(
    context: _ModelContext,
    h: torch.Tensor,
    s_u: torch.Tensor,
) -> torch.Tensor:
    nu_half = torch.tensor(
        context.priors.regime.nu / 2.0, device=context.device, dtype=context.dtype
    )
    with pyro.plate("time_v", context.T, dim=-1):
        v = pyro.sample("v", dist.Gamma(nu_half, nu_half))
    denom = _stationary_denom(
        context,
        torch.tensor(
            context.priors.regime.phi,
            device=context.device,
            dtype=context.dtype,
        ),
    )
    var_h = s_u.pow(2) / denom
    u_scalar = torch.exp(h - 0.5 * var_h) * v
    return u_scalar.unsqueeze(-1).expand(-1, context.A)


def _stationary_denom(
    context: _ModelContext, phi_t: torch.Tensor
) -> torch.Tensor:
    return (
        torch.tensor(1.0, device=context.device, dtype=context.dtype)
        - phi_t.pow(2)
        + context.priors.regime.eps
    )


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
    pair_factor = context.batch.exposure_matrix * structural.omega_currency.unsqueeze(0)
    inv_sqrt_u = torch.rsqrt(u)
    cov_factor = pair_factor.unsqueeze(0) * inv_sqrt_u.unsqueeze(-1)
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
            with poutine.scale(scale=obs_scale):  # pylint: disable=not-context-manager
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
            return
        with poutine.mask(mask=context.batch.time_mask):  # pylint: disable=not-context-manager
            if context.batch.obs_scale is None:
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)
                return
            obs_scale = context.batch.obs_scale
            with poutine.scale(scale=obs_scale):  # pylint: disable=not-context-manager
                pyro.sample("obs", obs_dist, obs=context.batch.y_obs)


def _half_student_t(
    *, df: float, scale: float, device: torch.device, dtype: torch.dtype
) -> dist.FoldedDistribution:
    base = dist.StudentT(
        torch.tensor(df, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(scale, device=device, dtype=dtype),
    )
    return dist.FoldedDistribution(base)
