from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import sample_time_observations

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l10_unified import MultiAssetBlockGuideV3L10UnifiedOnlineFiltering
from .model_v3_l6_unified import (
    V3L6UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _cov_factor_path as _base_cov_factor_path,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from .shared_v3_l10_unified import (
    INDEX_FLOW_MODULE_NAME,
    IndexBlockAffineCouplingFlow,
    IndexFlowConfig,
    build_index_block_affine_flow,
)
from .v3_l10_defaults import merge_nested_params, model_default_params_v3_l10
from .predict_v3_l10_unified import predict_multi_asset_block_v3_l10_unified


@dataclass(frozen=True)
class V3L10UnifiedModelPriors:
    base: V3L6UnifiedModelPriors = field(default_factory=V3L6UnifiedModelPriors)
    index_flow: IndexFlowConfig = field(default_factory=IndexFlowConfig)


@dataclass
class MultiAssetBlockModelV3L10UnifiedOnlineFiltering(PyroModel):
    priors: V3L10UnifiedModelPriors = field(default_factory=V3L10UnifiedModelPriors)
    _index_flow: IndexBlockAffineCouplingFlow | None = field(
        default=None, init=False, repr=False
    )
    _flow_signature: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        obs_dist = self._build_observation_distribution(context, structural, regime_path)
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
        return predict_multi_asset_block_v3_l10_unified(
            model=self,
            guide=cast(MultiAssetBlockGuideV3L10UnifiedOnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )

    def sync_index_flow(
        self, batch: ModelBatch
    ) -> IndexBlockAffineCouplingFlow | None:
        runtime_batch = (
            batch
            if hasattr(batch, "assets") and hasattr(batch, "observations")
            else build_v3_l1_unified_runtime_batch(batch)
        )
        flow = self._ensure_index_flow(runtime_batch)
        if flow is None:
            return None
        pyro.module(INDEX_FLOW_MODULE_NAME, flow)
        return flow

    def _ensure_index_flow(
        self, batch: Any
    ) -> IndexBlockAffineCouplingFlow | None:
        signature = tuple(batch.assets.asset_names)
        flow = self._index_flow
        if flow is not None and self._flow_signature == signature:
            return flow
        built = build_index_block_affine_flow(
            assets=batch.assets,
            config=self.priors.index_flow,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        )
        self._index_flow = built
        self._flow_signature = signature
        return built

    def _build_observation_distribution(
        self,
        context: Any,
        structural: Any,
        regime_path: Any,
    ) -> dist.TorchDistribution:
        loc = _base_mean_path(context, structural)
        cov_factor = _base_cov_factor_path(context, structural, regime_path)
        cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
        base_dist = dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
        )
        flow = self._ensure_index_flow(context.batch)
        if flow is None:
            return base_dist
        pyro.module(INDEX_FLOW_MODULE_NAME, flow)
        return dist.TransformedDistribution(base_dist, [flow])


@register_model("multi_asset_block_model_v3_l10_unified_online_filtering")
def build_multi_asset_block_model_v3_l10_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v3_l10(), params)
    return MultiAssetBlockModelV3L10UnifiedOnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V3L10UnifiedModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V3L10UnifiedModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "index_flow"}
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_model_v3_l10_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key]
        for key in ("mean", "factors", "regime")
        if key in values
    }
    return V3L10UnifiedModelPriors(
        base=_build_base_model_priors(base_payload),
        index_flow=_build_index_flow_config(values.get("index_flow")),
    )


def _build_index_flow_config(raw: object) -> IndexFlowConfig:
    values = _coerce_mapping(raw, label="model.params.index_flow")
    if not values:
        return IndexFlowConfig()
    base = IndexFlowConfig()
    return IndexFlowConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        hidden_dim=int(values.get("hidden_dim", base.hidden_dim)),
        log_scale_min_clip=float(
            values.get("log_scale_min_clip", base.log_scale_min_clip)
        ),
        log_scale_max_clip=float(
            values.get("log_scale_max_clip", base.log_scale_max_clip)
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)

__all__ = [
    "MultiAssetBlockModelV3L10UnifiedOnlineFiltering",
    "V3L10UnifiedModelPriors",
    "build_multi_asset_block_model_v3_l10_unified_online_filtering",
]
