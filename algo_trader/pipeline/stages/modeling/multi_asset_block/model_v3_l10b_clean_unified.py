from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
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
from algo_trader.pipeline.stages.modeling.runtime_support import (
    sample_time_observations,
)

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l10b_clean_unified import (
    MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering,
)
from .model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _cov_factor_path as _base_cov_factor_path,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from .predict_v3_l10b_clean_unified import (
    predict_multi_asset_block_v3_l10b_clean_unified,
)
from .shared_v3_l10b_clean_unified import (
    IndexTCopulaMixSamples,
    IndexTCopulaOverlayConfig,
    apply_index_t_copula_overlay,
)
from .v3_l10b_clean_defaults import (
    merge_nested_params,
    model_default_params_v3_l10b_clean,
)


@dataclass(frozen=True)
class V3L10BCleanUnifiedModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexTCopulaOverlayConfig = field(
        default_factory=IndexTCopulaOverlayConfig
    )


@dataclass
class MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering(PyroModel):
    priors: V3L10BCleanUnifiedModelPriors = field(
        default_factory=V3L10BCleanUnifiedModelPriors
    )

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mixes = _sample_index_t_copula_mix(
            context, self.priors.index_t_copula
        )
        obs_dist = _build_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            mixes=mixes,
            overlay=self.priors.index_t_copula,
        )
        sample_time_observations(
            time_count=context.shape.T,
            obs_dist=obs_dist,
            y_obs=runtime_batch.observations.y_obs,
            time_mask=runtime_batch.observations.time_mask,
            obs_scale=runtime_batch.observations.obs_scale,
        )

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        summaries = getattr(guide, "structural_predictive_summaries", None)
        if not callable(summaries):
            summaries = getattr(guide, "structural_posterior_means", None)
        if not callable(summaries):
            return None
        return predict_multi_asset_block_v3_l10b_clean_unified(
            model=self,
            guide=cast(
                MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering, guide
            ),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("multi_asset_block_model_v3_l10b_clean_unified_online_filtering")
def build_multi_asset_block_model_v3_l10b_clean_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v3_l10b_clean(), params)
    return MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V3L10BCleanUnifiedModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V3L10BCleanUnifiedModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "index_t_copula"}
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_model_v3_l10b_clean_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key]
        for key in ("mean", "factors", "regime")
        if key in values
    }
    return V3L10BCleanUnifiedModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
    )


def _build_index_t_copula_config(raw: object) -> IndexTCopulaOverlayConfig:
    values = _coerce_mapping(raw, label="model.params.index_t_copula")
    if not values:
        return IndexTCopulaOverlayConfig()
    base = IndexTCopulaOverlayConfig()
    return IndexTCopulaOverlayConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        broad_df=float(values.get("broad_df", base.broad_df)),
        us_diff_df=float(values.get("us_diff_df", base.us_diff_df)),
        us_diff_strength=float(
            values.get("us_diff_strength", base.us_diff_strength)
        ),
        eps=float(values.get("eps", base.eps)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_t_copula_mix(
    context: Any,
    overlay: IndexTCopulaOverlayConfig,
) -> IndexTCopulaMixSamples:
    if not overlay.enabled:
        ones = torch.ones(
            (context.shape.T,),
            device=context.device,
            dtype=context.dtype,
        )
        return IndexTCopulaMixSamples(broad=ones, us_diff=ones)
    broad_concentration = torch.full(
        (context.shape.T,),
        overlay.broad_df / 2.0,
        device=context.device,
        dtype=context.dtype,
    )
    us_diff_concentration = torch.full(
        (context.shape.T,),
        overlay.us_diff_df / 2.0,
        device=context.device,
        dtype=context.dtype,
    )
    broad_mix = pyro.sample(
        "index_t_copula_broad_mix",
        dist.Gamma(broad_concentration, broad_concentration).to_event(1),
    )
    us_diff_mix = pyro.sample(
        "index_t_copula_us_diff_mix",
        dist.Gamma(us_diff_concentration, us_diff_concentration).to_event(1),
    )
    return IndexTCopulaMixSamples(broad=broad_mix, us_diff=us_diff_mix)


def _build_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    mixes: IndexTCopulaMixSamples,
    overlay: IndexTCopulaOverlayConfig,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(context, structural)
    cov_factor = _base_cov_factor_path(context, structural, regime_path)
    base_cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    if not overlay.enabled:
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=cov_factor,
        cov_diag=base_cov_diag,
        assets=context.batch.assets,
        mixes=mixes,
        overlay=overlay,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    )


__all__ = [
    "MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering",
    "V3L10BCleanUnifiedModelPriors",
    "build_multi_asset_block_model_v3_l10b_clean_unified_online_filtering",
]
