from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from algo_trader.domain import ConfigError
from ..debug_utils import debug_log
from ..protocols import ModelBatch, PyroGuide, PyroModel
from ..registry_core import register_model
from .guide_l11 import (
    FilteringState,
    Level11RuntimeBatch,
    build_level11_runtime_batch,
)
from .predict_l11 import predict_factor_l11


@dataclass(frozen=True)
class MeanPriors:
    alpha_scale: float = 0.02
    sigma_idio_scale: float = 0.05
    beta0_scale: float = 0.05
    tau_beta_scale: float = 0.05


@dataclass(frozen=True)
class ShrinkagePriors:
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12


@dataclass(frozen=True)
class FactorPriors:
    factor_count: int = 3
    b_scale: float = 0.20
    b_col_shrink_scale: float = 0.50


@dataclass(frozen=True)
class RegimePriors:
    nu: float = 10.0
    phi: float = 0.97
    s_u_df: float = 4.0
    s_u_scale: float = 0.20
    lambda_h_scale: float = 0.35
    eps: float = 1e-12


@dataclass(frozen=True)
class Level11ModelPriors:
    mean: MeanPriors = field(default_factory=MeanPriors)
    shrinkage: ShrinkagePriors = field(default_factory=ShrinkagePriors)
    factors: FactorPriors = field(default_factory=FactorPriors)
    regime: RegimePriors = field(default_factory=RegimePriors)


@dataclass(frozen=True)
class _ModelContext:
    batch: Level11RuntimeBatch
    device: torch.device
    dtype: torch.dtype
    priors: Level11ModelPriors

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
    def K(self) -> int:
        return int(self.priors.factors.factor_count)


@dataclass(frozen=True)
class _ShrinkageSites:
    tau0: torch.Tensor
    lam: torch.Tensor
    c: torch.Tensor


@dataclass(frozen=True)
class _GlobalLoadingSites:
    beta0: torch.Tensor
    tau_beta: torch.Tensor
    b_col: torch.Tensor


@dataclass(frozen=True)
class _AssetSites:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    beta: torch.Tensor
    B: torch.Tensor
    lambda_h: torch.Tensor


@dataclass(frozen=True)
class FactorModelL11OnlineFiltering(PyroModel):
    priors: Level11ModelPriors = field(default_factory=Level11ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_level11_runtime_batch(batch)
        context = _build_context(runtime_batch, self.priors)
        _log_inputs(batch, context)
        shrinkage = _sample_shrinkage(context)
        _log_shrinkage(batch, shrinkage)
        loading_sites = _sample_global_loadings(context)
        _log_global_loadings(batch, loading_sites)
        asset_sites = _sample_asset_sites(context, shrinkage, loading_sites)
        _log_asset_sites(batch, asset_sites)
        s_u = pyro.sample(
            "s_u",
            half_student_t(
                df=context.priors.regime.s_u_df,
                scale=context.priors.regime.s_u_scale,
                device=context.device,
                dtype=context.dtype,
            ),
        )
        h = _sample_regime_path(context, s_u)
        u = _sample_total_scale(context, h, s_u, asset_sites.lambda_h)
        _log_regime_and_scale(batch, s_u, h, u)
        obs_dist = _build_observation_distribution(context, asset_sites, u)
        _log_observation_distribution(batch, obs_dist)
        _sample_observations(context, obs_dist)

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, torch.Tensor] | None:
        structural_summaries = getattr(
            guide, "structural_predictive_summaries", None
        )
        if not callable(structural_summaries):
            structural_summaries = getattr(
                guide, "structural_posterior_means", None
            )
        if not callable(structural_summaries):
            return None
        return predict_factor_l11(
            model=self,
            guide=guide,  # type: ignore[arg-type]
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("factor_model_l11_online_filtering")
def build_factor_model_l11_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return FactorModelL11OnlineFiltering(priors=_build_model_priors(params))


def _build_model_priors(params: Mapping[str, Any]) -> Level11ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return Level11ModelPriors()
    extra = set(values) - {"mean", "shrinkage", "factors", "regime"}
    if extra:
        raise ConfigError(
            "Unknown factor_model_l11_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return Level11ModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        shrinkage=_build_shrinkage_priors(values.get("shrinkage")),
        factors=_build_factor_priors(values.get("factors")),
        regime=_build_regime_priors(values.get("regime")),
    )


def _build_mean_priors(raw: object) -> MeanPriors:
    values = _coerce_mapping(raw, label="model.params.mean")
    base = MeanPriors()
    extra = set(values) - {
        "alpha_scale",
        "sigma_idio_scale",
        "beta0_scale",
        "tau_beta_scale",
    }
    if extra:
        raise ConfigError(
            "Unknown Level 11 mean priors",
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
            "Unknown Level 11 shrinkage priors",
            context={"params": ", ".join(sorted(extra))},
        )
    return _replace_dataclass(base, values)


def _build_factor_priors(raw: object) -> FactorPriors:
    values = _coerce_mapping(raw, label="model.params.factors")
    base = FactorPriors()
    extra = set(values) - {"factor_count", "b_scale", "b_col_shrink_scale"}
    if extra:
        raise ConfigError(
            "Unknown Level 11 factor priors",
            context={"params": ", ".join(sorted(extra))},
        )
    updated = _replace_dataclass(base, values)
    if updated.factor_count <= 0:
        raise ConfigError("factor_count must be positive")
    return updated


def _build_regime_priors(raw: object) -> RegimePriors:
    values = _coerce_mapping(raw, label="model.params.regime")
    base = RegimePriors()
    extra = set(values) - {
        "nu",
        "phi",
        "s_u_df",
        "s_u_scale",
        "lambda_h_scale",
        "eps",
    }
    if extra:
        raise ConfigError(
            "Unknown Level 11 regime priors",
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
            "Invalid Level 11 model params",
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


def half_student_t(
    *, df: float, scale: float, device: torch.device, dtype: torch.dtype
) -> dist.FoldedDistribution:
    base = dist.StudentT(
        torch.tensor(df, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(scale, device=device, dtype=dtype),
    )
    return dist.FoldedDistribution(base)


def _build_context(
    batch: Level11RuntimeBatch, priors: Level11ModelPriors
) -> _ModelContext:
    return _ModelContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        priors=priors,
    )


def _log_inputs(batch: ModelBatch, context: _ModelContext) -> None:
    debug_log(batch.debug, "Inputs:")
    debug_log(
        batch.debug,
        "X_asset [T, A, F] = per-time, per-asset feature tensor: %s",
        tuple(context.batch.X_asset.shape),
    )
    debug_log(
        batch.debug,
        "X_global [T, G] = per-time global/exogenous feature block shared across assets: %s",
        tuple(context.batch.X_global.shape),
    )
    y_obs = context.batch.y_obs
    y_obs_shape = tuple(y_obs.shape) if y_obs is not None else None
    debug_log(
        batch.debug,
        "y_obs [T, A] = observed target returns used in the likelihood: %s",
        y_obs_shape,
    )
    time_mask = context.batch.time_mask
    time_mask_shape = tuple(time_mask.shape) if time_mask is not None else None
    debug_log(
        batch.debug,
        "time_mask [T] = valid-week mask; False drops the whole multivariate observation if any asset is missing: %s",
        time_mask_shape,
    )
    filtering_state = context.batch.filtering_state
    debug_log(batch.debug, "T = number of time steps in this batch: %s", context.T)
    debug_log(batch.debug, "A = number of assets: %s", context.A)
    debug_log(batch.debug, "F = number of asset-level features: %s", context.F)
    debug_log(batch.debug, "G = number of global features: %s", context.G)
    debug_log(batch.debug, "K = latent factor count: %s", context.K)
    debug_log(
        batch.debug,
        "filtering_state = incoming online state for the AR(1) regime latent h_t: %s",
        _render_filtering_state_shape(filtering_state),
    )
    debug_log(batch.debug, "")


def _log_shrinkage(batch: ModelBatch, shrinkage: _ShrinkageSites) -> None:
    debug_log(batch.debug, "Shrinkage sites:")
    debug_log(
        batch.debug,
        "tau0 = global horseshoe shrinkage scale shared across feature weights: %s",
        tuple(shrinkage.tau0.shape),
    )
    debug_log(
        batch.debug,
        "lambda [F] = feature-level horseshoe shrinkage scales: %s",
        tuple(shrinkage.lam.shape),
    )
    debug_log(
        batch.debug,
        "c = regularized horseshoe slab scale controlling shrinkage saturation: %s",
        tuple(shrinkage.c.shape),
    )
    debug_log(batch.debug, "")


def _log_global_loadings(
    batch: ModelBatch, loadings: _GlobalLoadingSites
) -> None:
    debug_log(batch.debug, "Global loading sites:")
    debug_log(
        batch.debug,
        "beta0 [G] = prior mean for asset sensitivities to global features: %s",
        tuple(loadings.beta0.shape),
    )
    debug_log(
        batch.debug,
        "tau_beta [G] = prior scales for those global-feature sensitivities: %s",
        tuple(loadings.tau_beta.shape),
    )
    debug_log(
        batch.debug,
        "b_col [K] = column-wise shrinkage for latent factor loading matrix B: %s",
        tuple(loadings.b_col.shape),
    )
    debug_log(batch.debug, "")


def _log_asset_sites(batch: ModelBatch, sites: _AssetSites) -> None:
    debug_log(batch.debug, "Asset-level sites:")
    debug_log(
        batch.debug,
        "alpha [A] = per-asset intercepts in the conditional mean: %s",
        tuple(sites.alpha.shape),
    )
    debug_log(
        batch.debug,
        "sigma_idio [A] = idiosyncratic residual scales per asset: %s",
        tuple(sites.sigma_idio.shape),
    )
    debug_log(
        batch.debug,
        "w [A, F] = asset-specific weights on asset-level features: %s",
        tuple(sites.w.shape),
    )
    debug_log(
        batch.debug,
        "beta [A, G] = asset exposures to global feature block: %s",
        tuple(sites.beta.shape),
    )
    debug_log(
        batch.debug,
        "B [A, K] = low-rank factor loading matrix driving cross-asset covariance: %s",
        tuple(sites.B.shape),
    )
    debug_log(
        batch.debug,
        "lambda_h [A] = asset-specific loading on the shared regime volatility state: %s",
        tuple(sites.lambda_h.shape),
    )
    debug_log(batch.debug, "")


def _log_regime_and_scale(
    batch: ModelBatch,
    s_u: torch.Tensor,
    h: torch.Tensor,
    u: torch.Tensor,
) -> None:
    debug_log(batch.debug, "Regime and scale sites:")
    debug_log(
        batch.debug,
        "s_u = innovation scale of the AR(1) regime process h_t: %s",
        tuple(s_u.shape),
    )
    debug_log(
        batch.debug,
        "h [T] = latent AR(1) regime path controlling common volatility state: %s",
        tuple(h.shape),
    )
    debug_log(
        batch.debug,
        "u [T, A] = asset-specific total volatility scale driven by shared regime and heavy-tail shock: %s",
        tuple(u.shape),
    )
    debug_log(batch.debug, "")


def _log_observation_distribution(
    batch: ModelBatch,
    obs_dist: dist.LowRankMultivariateNormal,
) -> None:
    debug_log(batch.debug, "Observation distribution:")
    debug_log(
        batch.debug,
        "obs loc [T, A] = conditional mean of returns before observation noise: %s",
        tuple(obs_dist.loc.shape),
    )
    debug_log(
        batch.debug,
        "obs cov_factor [T, A, K] = low-rank covariance factor after asset-specific regime scaling: %s",
        tuple(obs_dist.cov_factor.shape),
    )
    debug_log(
        batch.debug,
        "obs cov_diag [T, A] = diagonal idiosyncratic variance contribution: %s",
        tuple(obs_dist.cov_diag.shape),
    )
    debug_log(batch.debug, "")


def _render_filtering_state_shape(
    filtering_state: FilteringState | None,
) -> str:
    if filtering_state is None:
        return "None"
    return (
        "{"
        f"h_loc: {tuple(filtering_state.h_loc.shape)}, "
        f"h_scale: {tuple(filtering_state.h_scale.shape)}, "
        f"steps_seen: {filtering_state.steps_seen}"
        "}"
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


def _sample_global_loadings(context: _ModelContext) -> _GlobalLoadingSites:
    with pyro.plate("global_feature", context.G, dim=-1):
        beta0 = pyro.sample(
            "beta0",
            dist.Normal(
                torch.zeros(context.G, device=context.device, dtype=context.dtype),
                torch.full(
                    (context.G,),
                    context.priors.mean.beta0_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
        tau_beta = pyro.sample(
            "tau_beta",
            dist.HalfNormal(
                torch.full(
                    (context.G,),
                    context.priors.mean.tau_beta_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    with pyro.plate("factor_loading_col", context.K, dim=-1):
        b_col = pyro.sample(
            "b_col",
            dist.HalfNormal(
                torch.full(
                    (context.K,),
                    context.priors.factors.b_col_shrink_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    return _GlobalLoadingSites(beta0=beta0, tau_beta=tau_beta, b_col=b_col)


def _sample_asset_sites(
    context: _ModelContext,
    shrinkage: _ShrinkageSites,
    loadings: _GlobalLoadingSites,
) -> _AssetSites:
    mean_priors = context.priors.mean
    factor_priors = context.priors.factors
    shrink_priors = context.priors.shrinkage
    with pyro.plate("asset", context.A, dim=-2):
        alpha = pyro.sample(
            "alpha",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                torch.tensor(
                    mean_priors.alpha_scale,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        )
        sigma_idio = pyro.sample(
            "sigma_idio",
            dist.HalfNormal(
                torch.tensor(
                    mean_priors.sigma_idio_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
        w = _sample_feature_weights(context, shrinkage, shrink_priors)
        with pyro.plate("global_loading", context.G, dim=-1):
            beta = pyro.sample("beta", dist.Normal(loadings.beta0, loadings.tau_beta))
        with pyro.plate("factor_loading_k", context.K, dim=-1):
            B = pyro.sample(
                "B",
                dist.Normal(
                    torch.tensor(0.0, device=context.device, dtype=context.dtype),
                    torch.tensor(
                        factor_priors.b_scale,
                        device=context.device,
                        dtype=context.dtype,
                    )
                    * loadings.b_col,
                ),
            )
    lambda_h = pyro.sample(
        "lambda_h",
        dist.LogNormal(
            torch.zeros(context.A, device=context.device, dtype=context.dtype),
            torch.full(
                (context.A,),
                context.priors.regime.lambda_h_scale,
                device=context.device,
                dtype=context.dtype,
            ),
        ).to_event(1),
    )
    return _AssetSites(
        alpha=alpha,
        sigma_idio=sigma_idio,
        w=w,
        beta=beta,
        B=B,
        lambda_h=lambda_h,
    )


def _sample_feature_weights(
    context: _ModelContext,
    shrinkage: _ShrinkageSites,
    priors: ShrinkagePriors,
) -> torch.Tensor:
    with pyro.plate("feature_w", context.F, dim=-1):
        kappa = pyro.sample(
            "kappa",
            dist.HalfCauchy(
                torch.tensor(
                    priors.kappa_scale, device=context.device, dtype=context.dtype
                )
            ),
        )
        w_scale = _regularized_horseshoe_scale(
            tau0=shrinkage.tau0,
            lam=shrinkage.lam,
            kappa=kappa,
            c=shrinkage.c,
            eps=priors.eps,
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
    lambda_h: torch.Tensor,
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
    lambda_view = lambda_h.unsqueeze(0)
    log_u = (
        lambda_view * h.unsqueeze(-1)
        - 0.5 * lambda_view.pow(2) * var_h
    )
    return torch.exp(log_u) * v.unsqueeze(-1)


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
    asset_sites: _AssetSites,
    u: torch.Tensor,
) -> dist.LowRankMultivariateNormal:
    alpha_vec = asset_sites.alpha.squeeze(-1)
    sigma_vec = asset_sites.sigma_idio.squeeze(-1)
    mu_asset = (context.batch.X_asset * asset_sites.w.unsqueeze(0)).sum(dim=-1)
    mu_global = context.batch.X_global @ asset_sites.beta.transpose(0, 1)
    mu = alpha_vec.unsqueeze(0) + mu_asset + mu_global
    inv_sqrt_u = torch.rsqrt(u)
    cov_factor = asset_sites.B.unsqueeze(0) * inv_sqrt_u.unsqueeze(-1)
    cov_diag = sigma_vec.pow(2).unsqueeze(0) / u
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
