from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.hybrid_measurement.versions.v9_l2.shared import (
    HybridMeasurementFactorState,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l1_unified import (
    _initial_regime_samples,
    _mean_step,
    _move_filtering_state,
    _move_structural,
    _resolve_structural_means,
)
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide import StateConditionedMeasurementGuideV10L1OnlineFiltering
from .shared import (
    HybridMeasurementPosteriorMeans,
    StateConditionedMeasurementCoefficients,
    apply_hybrid_measurement_residual_scale,
    build_base_hybrid_measurement_config,
    build_hybrid_measurement_factor_block,
    build_hybrid_measurement_structure,
    build_nonindex_cov_factor_step,
    build_state_conditioned_contrast_scale,
    build_state_conditioned_measurement_gate_series,
    build_state_conditioned_residual_scale,
)

if TYPE_CHECKING:
    from .model import StateConditionedMeasurementModelV10L1OnlineFiltering


class _V10L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_state_conditioned_measurement_v10_l1(
            model=cast(Any, request.model),
            guide=cast(StateConditionedMeasurementGuideV10L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("state_conditioned_measurement_predict_v10_l1_online_filtering")
def build_state_conditioned_measurement_predict_v10_l1_online_filtering(
    params: Mapping[str, Any],
) -> _V10L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown state_conditioned_measurement_predict_v10_l1_online_filtering "
            f"params: {unknown}"
        )
    return _V10L1Predictor()


def predict_state_conditioned_measurement_v10_l1(
    *,
    model: StateConditionedMeasurementModelV10L1OnlineFiltering,
    guide: StateConditionedMeasurementGuideV10L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    measurement_params = guide.state_conditioned_measurement_posterior_means(
        batch=runtime_batch
    ).to(
        device=runtime_batch.X_asset.device,
        dtype=runtime_batch.X_asset.dtype,
    )
    coefficients = guide.state_conditioned_measurement_coefficients_posterior_means().to(
        device=runtime_batch.X_asset.device,
        dtype=runtime_batch.X_asset.dtype,
    )
    gate_series = build_state_conditioned_measurement_gate_series(
        X_asset=runtime_batch.X_asset,
        X_global=runtime_batch.X_global,
        assets=runtime_batch.assets,
        coefficients=coefficients,
        config=model.priors.state_conditioned_measurement,
    )
    draws = _rollout_samples(
        inputs=_PredictInputs(
            model=model,
            structural=structural,
            batch=runtime_batch,
            measurement_params=measurement_params,
            coefficients=coefficients,
            gate_series=gate_series,
        ),
        num_samples=int(num_samples),
    )
    return {
        "samples": draws,
        "mean": draws.mean(dim=0),
        "covariance": predictive_covariance(draws),
    }


@dataclass(frozen=True)
class _ObservationContext:
    batch: Any
    structural: Any
    measurement_params: HybridMeasurementPosteriorMeans
    coefficients: StateConditionedMeasurementCoefficients
    structure: Any
    overlay: Any


@dataclass(frozen=True)
class _PredictInputs:
    model: Any
    structural: Any
    batch: Any
    measurement_params: HybridMeasurementPosteriorMeans
    coefficients: StateConditionedMeasurementCoefficients
    gate_series: torch.Tensor


def _rollout_samples(
    *,
    inputs: _PredictInputs,
    num_samples: int,
) -> torch.Tensor:
    filtering_state = _move_filtering_state(
        batch=inputs.batch,
        filtering_state=inputs.batch.filtering_state,
    )
    structural = _move_structural(
        structural=inputs.structural,
        device=inputs.batch.X_asset.device,
        dtype=inputs.batch.X_asset.dtype,
    )
    regime = _initial_regime_samples(
        filtering_state=filtering_state,
        num_samples=num_samples,
    )
    overlay = inputs.model.priors.state_conditioned_measurement
    structure = build_hybrid_measurement_structure(
        assets=inputs.batch.assets,
        config=build_base_hybrid_measurement_config(overlay),
        device=inputs.batch.X_asset.device,
        dtype=inputs.batch.X_asset.dtype,
    )
    context = _ObservationContext(
        batch=inputs.batch,
        structural=structural,
        measurement_params=inputs.measurement_params,
        coefficients=inputs.coefficients,
        structure=structure,
        overlay=overlay,
    )
    phi = _phi_vector(
        inputs.model,
        device=inputs.batch.X_asset.device,
        dtype=inputs.batch.X_asset.dtype,
    )
    scales = structural.regime.as_tensor().to(
        device=inputs.batch.X_asset.device,
        dtype=inputs.batch.X_asset.dtype,
    )
    draws = []
    for time_index in range(int(inputs.batch.X_asset.shape[0])):
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(
            regime
        )
        draws.append(
            _sample_observation_step(
                context=context,
                regime=regime,
                gate_value=inputs.gate_series[time_index],
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: StateConditionedMeasurementModelV10L1OnlineFiltering,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [
            model.priors.base.regime.fx_broad.phi,
            model.priors.base.regime.fx_cross.phi,
            model.priors.base.regime.index.phi,
            model.priors.base.regime.commodity.phi,
        ],
        device=device,
        dtype=dtype,
    )


def _sample_observation_step(
    *,
    context: _ObservationContext,
    regime: torch.Tensor,
    gate_value: torch.Tensor,
    time_index: int,
) -> torch.Tensor:
    loc = _mean_step(
        structural=context.structural,
        batch=context.batch,
        time_index=time_index,
    )
    base_cov_diag = context.structural.sigma_idio.pow(2)
    nonindex_block = build_nonindex_cov_factor_step(
        structural=context.structural,
        assets=context.batch.assets,
        regime=regime,
        dtype=context.batch.X_asset.dtype,
    )
    gate_path = gate_value.expand(regime.shape[0])
    contrast_scale = build_state_conditioned_contrast_scale(
        gate=gate_path,
        coefficients=context.coefficients,
        eps=context.overlay.eps,
    )
    index_block = build_hybrid_measurement_factor_block(
        structure=context.structure,
        state=HybridMeasurementFactorState(
            state_scale=context.measurement_params.state_scale,
            state_corr_cholesky=context.measurement_params.state_corr_cholesky,
            loading_delta=context.measurement_params.loading_delta,
            regime_scale=torch.exp(0.5 * regime[:, 2]) * contrast_scale,
            mix=_sample_mix(
                num_samples=regime.shape[0],
                df_value=context.overlay.state_df,
                device=context.batch.X_asset.device,
                dtype=context.batch.X_asset.dtype,
            ),
            eps=context.overlay.eps,
        ),
    )
    residual_scale = build_state_conditioned_residual_scale(
        residual_scale=context.measurement_params.residual_scale,
        assets=context.batch.assets,
        gate=gate_path,
        coefficients=context.coefficients,
        eps=context.overlay.eps,
    )
    cov_diag = apply_hybrid_measurement_residual_scale(
        cov_diag=base_cov_diag,
        residual_scale=residual_scale,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([nonindex_block, index_block], dim=-1),
        cov_diag=cov_diag,
    ).rsample()


def _sample_mix(
    *,
    num_samples: int,
    df_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    concentration = torch.full(
        (num_samples,),
        df_value / 2.0,
        device=device,
        dtype=dtype,
    )
    return dist.Gamma(concentration, concentration).sample()


__all__ = [
    "build_state_conditioned_measurement_predict_v10_l1_online_filtering",
    "predict_state_conditioned_measurement_v10_l1",
]
