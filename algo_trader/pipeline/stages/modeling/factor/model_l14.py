from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .guide_l12 import build_level12_runtime_batch
from .model_l12 import (
    FactorPriors,
    RegimePriors,
    ShrinkagePriors,
    _AssetSites,
    _GlobalLoadingSites,
    _ModelContext,
    _ShrinkageSites,
    _build_context,
    _build_factor_priors,
    _build_observation_distribution,
    _build_regime_priors,
    _build_shrinkage_priors,
    _coerce_mapping,
    _log_asset_sites,
    _log_global_loadings,
    _log_inputs,
    _log_observation_distribution,
    _log_regime_and_scale,
    _log_shrinkage,
    _sample_feature_weights,
    _sample_global_loadings,
    _sample_observations,
    _sample_regime_path,
    _sample_regime_scale,
    _sample_shrinkage,
    _sample_total_scale,
)
from .model_l13 import FactorModelL13OnlineFiltering
from .model_l13 import _build_registered_model


@dataclass(frozen=True)
class MeanPriors:
    alpha_scale: float = 0.02
    sigma_idio_scale: float = 0.05
    sigma_idio_scale_fx: float = 0.03
    sigma_idio_scale_index: float = 0.05
    sigma_idio_scale_commodity: float = 0.06
    beta0_scale: float = 0.05
    tau_beta_scale: float = 0.05


@dataclass(frozen=True)
class Level14ModelPriors:
    mean: MeanPriors = field(default_factory=MeanPriors)
    shrinkage: ShrinkagePriors = field(default_factory=ShrinkagePriors)
    factors: FactorPriors = field(default_factory=FactorPriors)
    regime: RegimePriors = field(default_factory=RegimePriors)


@dataclass(frozen=True)
class _StructuralSitePlan:
    log_inputs: Any
    sample_shrinkage: Any
    log_shrinkage: Any
    sample_global_loadings: Any
    log_global_loadings: Any
    sample_asset_sites: Any
    log_asset_sites: Any


@dataclass(frozen=True)
class _StageHooks:
    sample: Any
    log: Any


@dataclass(frozen=True)
class FactorModelL14OnlineFiltering(FactorModelL13OnlineFiltering):
    priors: Any = field(default_factory=Level14ModelPriors)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_level12_runtime_batch(batch)
        context = _build_context(runtime_batch, cast(Any, self.priors))
        asset_sites = _sample_shared_structural_sites(batch, context)
        s_u = _sample_regime_scale(context)
        h = _sample_regime_path(context, s_u)
        u = _sample_total_scale(context, h, s_u)
        _log_regime_and_scale(batch, s_u, h, u)
        obs_dist = _build_observation_distribution(context, asset_sites, u)
        _log_observation_distribution(batch, obs_dist)
        _sample_observations(context, obs_dist)


@register_model("factor_model_l14_online_filtering")
def build_factor_model_l14_online_filtering(
    params: Mapping[str, Any],
) -> PyroModel:
    return _build_registered_model(
        params=params,
        model_type=FactorModelL14OnlineFiltering,
        prior_builder=_build_model_priors,
    )


def _build_model_priors(params: Mapping[str, Any]) -> Level14ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return Level14ModelPriors()
    extra = set(values) - {"mean", "shrinkage", "factors", "regime"}
    if extra:
        raise ConfigError(
            "Unknown factor_model_l14_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return Level14ModelPriors(
        mean=_build_mean_priors(values.get("mean")),
        shrinkage=_build_shrinkage_priors(values.get("shrinkage")),
        factors=_build_factor_priors(values.get("factors")),
        regime=_build_regime_priors(values.get("regime")),
    )


def _sample_shared_structural_sites(
    batch: ModelBatch,
    context: _ModelContext,
    plan: _StructuralSitePlan | None = None,
) -> _AssetSites:
    resolved = plan or _default_structural_site_plan()
    resolved.log_inputs(batch, context)
    shrinkage = resolved.sample_shrinkage(context)
    resolved.log_shrinkage(batch, shrinkage)
    loadings = resolved.sample_global_loadings(context)
    resolved.log_global_loadings(batch, loadings)
    asset_sites = resolved.sample_asset_sites(context, shrinkage, loadings)
    resolved.log_asset_sites(batch, asset_sites)
    return asset_sites


def _build_structural_site_plan(
    *,
    log_inputs: Any,
    shrinkage: _StageHooks,
    loadings: _StageHooks,
    assets: _StageHooks,
) -> _StructuralSitePlan:
    return _StructuralSitePlan(
        log_inputs=log_inputs,
        sample_shrinkage=shrinkage.sample,
        log_shrinkage=shrinkage.log,
        sample_global_loadings=loadings.sample,
        log_global_loadings=loadings.log,
        sample_asset_sites=assets.sample,
        log_asset_sites=assets.log,
    )


def _default_structural_site_plan() -> _StructuralSitePlan:
    return _build_structural_site_plan(
        log_inputs=_log_inputs,
        shrinkage=_StageHooks(
            sample=_sample_shrinkage,
            log=_log_shrinkage,
        ),
        loadings=_StageHooks(
            sample=_sample_global_loadings,
            log=_log_global_loadings,
        ),
        assets=_StageHooks(
            sample=_sample_asset_sites,
            log=_log_asset_sites,
        ),
    )


def _build_mean_priors(raw: object) -> MeanPriors:
    values = _coerce_mapping(raw, label="model.params.mean")
    base = MeanPriors()
    extra = set(values) - {
        "alpha_scale",
        "sigma_idio_scale",
        "sigma_idio_scale_fx",
        "sigma_idio_scale_index",
        "sigma_idio_scale_commodity",
        "beta0_scale",
        "tau_beta_scale",
    }
    if extra:
        raise ConfigError(
            "Unknown Level 14 mean priors",
            context={"params": ", ".join(sorted(extra))},
        )
    try:
        sigma_base = float(values.get("sigma_idio_scale", base.sigma_idio_scale))
        sigma_fx = float(values.get("sigma_idio_scale_fx", 0.6 * sigma_base))
        sigma_index = float(
            values.get("sigma_idio_scale_index", 1.0 * sigma_base)
        )
        sigma_commodity = float(
            values.get("sigma_idio_scale_commodity", 1.2 * sigma_base)
        )
        updated = MeanPriors(
            alpha_scale=float(values.get("alpha_scale", base.alpha_scale)),
            sigma_idio_scale=sigma_base,
            sigma_idio_scale_fx=sigma_fx,
            sigma_idio_scale_index=sigma_index,
            sigma_idio_scale_commodity=sigma_commodity,
            beta0_scale=float(values.get("beta0_scale", base.beta0_scale)),
            tau_beta_scale=float(
                values.get("tau_beta_scale", base.tau_beta_scale)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid Level 14 mean priors",
            context={"params": str(dict(values))},
        ) from exc
    _validate_positive_mean_priors(updated)
    return updated


def _validate_positive_mean_priors(priors: MeanPriors) -> None:
    positive_values = {
        "alpha_scale": priors.alpha_scale,
        "sigma_idio_scale": priors.sigma_idio_scale,
        "sigma_idio_scale_fx": priors.sigma_idio_scale_fx,
        "sigma_idio_scale_index": priors.sigma_idio_scale_index,
        "sigma_idio_scale_commodity": priors.sigma_idio_scale_commodity,
        "beta0_scale": priors.beta0_scale,
        "tau_beta_scale": priors.tau_beta_scale,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0.0]
    if invalid:
        raise ConfigError(
            "Level 14 mean priors must be positive",
            context={"params": ", ".join(sorted(invalid))},
        )


def _sample_asset_sites(
    context: _ModelContext,
    shrinkage: _ShrinkageSites,
    loadings: _GlobalLoadingSites,
) -> _AssetSites:
    mean_priors = cast(MeanPriors, context.priors.mean)
    factor_priors = context.priors.factors
    shrink_priors = context.priors.shrinkage
    sigma_scales = _sigma_idio_scales(context, mean_priors)
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
        sigma_idio = pyro.sample("sigma_idio", dist.HalfNormal(sigma_scales))
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
    return _AssetSites(
        alpha=alpha,
        sigma_idio=sigma_idio,
        w=w,
        beta=beta,
        B=B,
    )


def _sigma_idio_scales(
    context: _ModelContext, priors: MeanPriors
) -> torch.Tensor:
    class_scales = torch.tensor(
        [
            priors.sigma_idio_scale_fx,
            priors.sigma_idio_scale_index,
            priors.sigma_idio_scale_commodity,
        ],
        device=context.device,
        dtype=context.dtype,
    )
    return class_scales.index_select(
        dim=0,
        index=context.batch.asset_class_ids,
    ).unsqueeze(-1)
