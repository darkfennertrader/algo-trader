from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import sample_time_observations

from .defaults import merge_nested_params, model_default_params_v9_l2
from .guide import HybridMeasurementGuideV9L2OnlineFiltering
from .predict import predict_hybrid_measurement_v9_l2
from .shared import (
    HybridMeasurementConfig,
    HybridMeasurementFactorState,
    HybridMeasurementPosteriorMeans,
    apply_hybrid_measurement_residual_scale,
    build_hybrid_measurement_config,
    build_hybrid_measurement_factor_block,
    build_hybrid_measurement_structure,
    build_nonindex_cov_factor_path,
)

_STATE_COUNT = 3


@dataclass(frozen=True)
class V9L2ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    hybrid_measurement: HybridMeasurementConfig = field(
        default_factory=HybridMeasurementConfig
    )


@dataclass
class HybridMeasurementModelV9L2OnlineFiltering(PyroModel):
    priors: V9L2ModelPriors = field(default_factory=V9L2ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        measurement_params = _sample_hybrid_measurement_sites(
            assets=context.batch.assets,
            overlay=self.priors.hybrid_measurement,
            device=context.device,
            dtype=context.dtype,
        )
        obs_dist = _build_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            measurement_params=measurement_params,
            overlay=self.priors.hybrid_measurement,
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
        return predict_hybrid_measurement_v9_l2(
            model=self,
            guide=cast(HybridMeasurementGuideV9L2OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("hybrid_measurement_model_v9_l2_online_filtering")
def build_hybrid_measurement_model_v9_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v9_l2(), params)
    return HybridMeasurementModelV9L2OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V9L2ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V9L2ModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "hybrid_measurement"}
    if extra:
        raise ConfigError(
            "Unknown hybrid_measurement_model_v9_l2_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V9L2ModelPriors(
        base=_build_base_model_priors(base_payload),
        hybrid_measurement=build_hybrid_measurement_config(
            values.get("hybrid_measurement")
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_hybrid_measurement_sites(
    *,
    assets: Any,
    overlay: HybridMeasurementConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> HybridMeasurementPosteriorMeans:
    structure = build_hybrid_measurement_structure(
        assets=assets,
        config=overlay,
        device=device,
        dtype=dtype,
    )
    if not overlay.enabled:
        return HybridMeasurementPosteriorMeans(
            state_scale=torch.zeros((_STATE_COUNT,), device=device, dtype=dtype),
            state_corr_cholesky=torch.eye(_STATE_COUNT, device=device, dtype=dtype),
            loading_delta=torch.zeros_like(structure.anchor_loadings),
            residual_scale=torch.ones_like(structure.residual_anchor),
        )
    state_scale = pyro.sample(
        "hybrid_measurement_state_scale",
        dist.HalfNormal(
            torch.full(
                (_STATE_COUNT,),
                overlay.prior_scales.state_scale,
                device=device,
                dtype=dtype,
            )
        ).to_event(1),
    )
    state_corr_cholesky = pyro.sample(
        "hybrid_measurement_state_corr_cholesky",
        dist.LKJCholesky(
            dim=_STATE_COUNT,
            concentration=torch.tensor(
                overlay.prior_scales.correlation_concentration,
                device=device,
                dtype=dtype,
            ),
        ),
    )
    loading_delta = pyro.sample(
        "hybrid_measurement_loading_delta",
        dist.Normal(
            torch.zeros_like(structure.anchor_loadings),
            structure.loading_deviation_scale,
        ).to_event(2),
    )
    residual_scale = pyro.sample(
        "hybrid_measurement_residual_scale",
        dist.LogNormal(
            torch.log(structure.residual_anchor.clamp_min(float(overlay.eps))),
            structure.residual_prior_scale,
        ).to_event(1),
    )
    return HybridMeasurementPosteriorMeans(
        state_scale=state_scale,
        state_corr_cholesky=state_corr_cholesky,
        loading_delta=loading_delta,
        residual_scale=residual_scale,
    )


def _sample_hybrid_measurement_mix(
    *,
    df_value: float,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    concentration = torch.full(
        (time_count,),
        df_value / 2.0,
        device=device,
        dtype=dtype,
    )
    return pyro.sample(
        "hybrid_measurement_mix",
        dist.Gamma(concentration, concentration).to_event(1),
    )


def _build_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    measurement_params: HybridMeasurementPosteriorMeans,
    overlay: HybridMeasurementConfig,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(context, structural)
    base_cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    if not overlay.enabled:
        cov_factor = build_nonindex_cov_factor_path(
            loadings=structural.loadings,
            class_ids=context.batch.assets.class_ids,
            regime_path=regime_path,
            dtype=context.dtype,
        )
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
    structure = build_hybrid_measurement_structure(
        assets=context.batch.assets,
        config=overlay,
        device=context.device,
        dtype=context.dtype,
    )
    nonindex_block = build_nonindex_cov_factor_path(
        loadings=structural.loadings,
        class_ids=context.batch.assets.class_ids,
        regime_path=regime_path,
        dtype=context.dtype,
    )
    mix = _sample_hybrid_measurement_mix(
        df_value=overlay.state_df,
        time_count=context.shape.T,
        device=context.device,
        dtype=context.dtype,
    )
    index_block = build_hybrid_measurement_factor_block(
        structure=structure,
        state=HybridMeasurementFactorState(
            state_scale=measurement_params.state_scale,
            state_corr_cholesky=measurement_params.state_corr_cholesky,
            loading_delta=measurement_params.loading_delta,
            regime_scale=torch.exp(0.5 * regime_path[:, 2]),
            mix=mix,
            eps=overlay.eps,
        ),
    )
    cov_diag = apply_hybrid_measurement_residual_scale(
        cov_diag=base_cov_diag,
        residual_scale=measurement_params.residual_scale,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([nonindex_block, index_block], dim=-1),
        cov_diag=cov_diag,
    )


__all__ = [
    "HybridMeasurementModelV9L2OnlineFiltering",
    "V9L2ModelPriors",
    "build_hybrid_measurement_model_v9_l2_online_filtering",
]
