from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.shared import (
    HybridMeasurementFactorState,
)
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

from .defaults import merge_nested_params, model_default_params_v10_l1
from .guide import StateConditionedMeasurementGuideV10L1OnlineFiltering
from .predict import predict_state_conditioned_measurement_v10_l1
from .shared import (
    HybridMeasurementPosteriorMeans,
    StateConditionedMeasurementCoefficients,
    StateConditionedMeasurementConfig,
    apply_hybrid_measurement_residual_scale,
    build_base_hybrid_measurement_config,
    build_hybrid_measurement_factor_block,
    build_hybrid_measurement_structure,
    build_nonindex_cov_factor_path,
    build_state_conditioned_contrast_scale,
    build_state_conditioned_measurement_config,
    build_state_conditioned_measurement_gate_series,
    build_state_conditioned_residual_scale,
)

_STATE_COUNT = 3


@dataclass(frozen=True)
class V10L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    state_conditioned_measurement: StateConditionedMeasurementConfig = field(
        default_factory=StateConditionedMeasurementConfig
    )


@dataclass(frozen=True)
class _ObservationInputs:
    context: Any
    structural: Any
    regime_path: torch.Tensor
    measurement_params: HybridMeasurementPosteriorMeans
    coefficients: StateConditionedMeasurementCoefficients
    structure: Any


@dataclass
class StateConditionedMeasurementModelV10L1OnlineFiltering(PyroModel):
    priors: V10L1ModelPriors = field(default_factory=V10L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        measurement_params = _sample_measurement_sites(
            assets=context.batch.assets,
            overlay=self.priors.state_conditioned_measurement,
            device=context.device,
            dtype=context.dtype,
        )
        structure = build_hybrid_measurement_structure(
            assets=context.batch.assets,
            config=build_base_hybrid_measurement_config(
                self.priors.state_conditioned_measurement
            ),
            device=context.device,
            dtype=context.dtype,
        )
        coefficients = _sample_state_conditioned_coefficients(
            context=context,
            overlay=self.priors.state_conditioned_measurement,
        )
        obs_dist = _build_observation_distribution(
            inputs=_ObservationInputs(
                context=context,
                structural=structural,
                regime_path=regime_path,
                measurement_params=measurement_params,
                coefficients=coefficients,
                structure=structure,
            ),
            overlay=self.priors.state_conditioned_measurement,
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
        return predict_state_conditioned_measurement_v10_l1(
            model=self,
            guide=cast(StateConditionedMeasurementGuideV10L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("state_conditioned_measurement_model_v10_l1_online_filtering")
def build_state_conditioned_measurement_model_v10_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v10_l1(), params)
    return StateConditionedMeasurementModelV10L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V10L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V10L1ModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "state_conditioned_measurement",
    }
    if extra:
        raise ConfigError(
            "Unknown state_conditioned_measurement_model_v10_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V10L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        state_conditioned_measurement=build_state_conditioned_measurement_config(
            values.get("state_conditioned_measurement")
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_measurement_sites(
    *,
    assets: Any,
    overlay: StateConditionedMeasurementConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> HybridMeasurementPosteriorMeans:
    structure = build_hybrid_measurement_structure(
        assets=assets,
        config=build_base_hybrid_measurement_config(overlay),
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
        "state_conditioned_measurement_state_scale",
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
        "state_conditioned_measurement_state_corr_cholesky",
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
        "state_conditioned_measurement_loading_delta",
        dist.Normal(
            torch.zeros_like(structure.anchor_loadings),
            structure.loading_deviation_scale,
        ).to_event(2),
    )
    residual_scale = pyro.sample(
        "state_conditioned_measurement_residual_scale",
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


def _sample_state_conditioned_coefficients(
    *,
    context: Any,
    overlay: StateConditionedMeasurementConfig,
) -> StateConditionedMeasurementCoefficients:
    if not overlay.enabled:
        zeros = torch.zeros((), device=context.device, dtype=context.dtype)
        return StateConditionedMeasurementCoefficients(
            bias=zeros,
            global_weight=zeros,
            index_weight=zeros,
            contrast_strength=zeros,
            composite_residual_strength=zeros,
        )
    return StateConditionedMeasurementCoefficients(
        bias=pyro.sample(
            "state_conditioned_measurement_bias",
            dist.Normal(
                torch.zeros((), device=context.device, dtype=context.dtype),
                torch.tensor(
                    overlay.condition_prior_scales.bias,
                    device=context.device,
                    dtype=context.dtype,
                ),
            ),
        ),
        global_weight=_sample_positive_site(
            name="state_conditioned_measurement_global_weight",
            scale=overlay.condition_prior_scales.global_weight,
            device=context.device,
            dtype=context.dtype,
        ),
        index_weight=_sample_positive_site(
            name="state_conditioned_measurement_index_weight",
            scale=overlay.condition_prior_scales.index_weight,
            device=context.device,
            dtype=context.dtype,
        ),
        contrast_strength=_sample_positive_site(
            name="state_conditioned_measurement_contrast_strength",
            scale=overlay.condition_prior_scales.contrast_strength,
            device=context.device,
            dtype=context.dtype,
        ),
        composite_residual_strength=_sample_positive_site(
            name="state_conditioned_measurement_composite_residual_strength",
            scale=overlay.condition_prior_scales.composite_residual_strength,
            device=context.device,
            dtype=context.dtype,
        ),
    )


def _sample_positive_site(
    *,
    name: str,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return pyro.sample(
        name,
        dist.HalfNormal(torch.tensor(scale, device=device, dtype=dtype)),
    )


def _sample_state_conditioned_mix(
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
        "state_conditioned_measurement_mix",
        dist.Gamma(concentration, concentration).to_event(1),
    )


def _build_observation_distribution(
    *,
    inputs: _ObservationInputs,
    overlay: StateConditionedMeasurementConfig,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(inputs.context, inputs.structural)
    base_cov_diag = (
        inputs.structural.mean.sigma_idio.pow(2) + inputs.context.priors.regime.eps
    )
    if not overlay.enabled:
        cov_factor = build_nonindex_cov_factor_path(
            loadings=inputs.structural.loadings,
            class_ids=inputs.context.batch.assets.class_ids,
            regime_path=inputs.regime_path,
            dtype=inputs.context.dtype,
        )
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
    gate = build_state_conditioned_measurement_gate_series(
        X_asset=inputs.context.batch.X_asset,
        X_global=inputs.context.batch.X_global,
        assets=inputs.context.batch.assets,
        coefficients=inputs.coefficients,
        config=overlay,
    )
    nonindex_block = _build_nonindex_block(inputs)
    index_block = _build_index_block(inputs, overlay, gate)
    cov_diag = _build_cov_diag(inputs, overlay, base_cov_diag, gate)
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([nonindex_block, index_block], dim=-1),
        cov_diag=cov_diag,
    )


def _build_nonindex_block(inputs: _ObservationInputs) -> torch.Tensor:
    return build_nonindex_cov_factor_path(
        loadings=inputs.structural.loadings,
        class_ids=inputs.context.batch.assets.class_ids,
        regime_path=inputs.regime_path,
        dtype=inputs.context.dtype,
    )


def _build_index_block(
    inputs: _ObservationInputs,
    overlay: StateConditionedMeasurementConfig,
    gate: torch.Tensor,
) -> torch.Tensor:
    mix = _sample_state_conditioned_mix(
        df_value=overlay.state_df,
        time_count=inputs.context.shape.T,
        device=inputs.context.device,
        dtype=inputs.context.dtype,
    )
    contrast_scale = build_state_conditioned_contrast_scale(
        gate=gate,
        coefficients=inputs.coefficients,
        eps=overlay.eps,
    )
    return build_hybrid_measurement_factor_block(
        structure=inputs.structure,
        state=HybridMeasurementFactorState(
            state_scale=inputs.measurement_params.state_scale,
            state_corr_cholesky=inputs.measurement_params.state_corr_cholesky,
            loading_delta=inputs.measurement_params.loading_delta,
            regime_scale=torch.exp(0.5 * inputs.regime_path[:, 2]) * contrast_scale,
            mix=mix,
            eps=overlay.eps,
        ),
    )


def _build_cov_diag(
    inputs: _ObservationInputs,
    overlay: StateConditionedMeasurementConfig,
    base_cov_diag: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    residual_scale = build_state_conditioned_residual_scale(
        residual_scale=inputs.measurement_params.residual_scale,
        assets=inputs.context.batch.assets,
        gate=gate,
        coefficients=inputs.coefficients,
        eps=overlay.eps,
    )
    return apply_hybrid_measurement_residual_scale(
        cov_diag=base_cov_diag,
        residual_scale=residual_scale,
    )


__all__ = [
    "StateConditionedMeasurementModelV10L1OnlineFiltering",
    "V10L1ModelPriors",
    "build_state_conditioned_measurement_model_v10_l1_online_filtering",
]
