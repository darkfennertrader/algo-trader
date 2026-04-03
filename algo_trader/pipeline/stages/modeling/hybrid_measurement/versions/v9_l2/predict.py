from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
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

from .guide import HybridMeasurementGuideV9L2OnlineFiltering
from .shared import (
    HybridMeasurementFactorState,
    HybridMeasurementPosteriorMeans,
    apply_hybrid_measurement_residual_scale,
    build_hybrid_measurement_factor_block,
    build_hybrid_measurement_structure,
    build_nonindex_cov_factor_step,
)

if TYPE_CHECKING:
    from .model import HybridMeasurementModelV9L2OnlineFiltering


class _V9L2Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_hybrid_measurement_v9_l2(
            model=cast(Any, request.model),
            guide=cast(HybridMeasurementGuideV9L2OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("hybrid_measurement_predict_v9_l2_online_filtering")
def build_hybrid_measurement_predict_v9_l2_online_filtering(
    params: Mapping[str, Any],
) -> _V9L2Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown hybrid_measurement_predict_v9_l2_online_filtering params: "
            f"{unknown}"
        )
    return _V9L2Predictor()


def predict_hybrid_measurement_v9_l2(
    *,
    model: HybridMeasurementModelV9L2OnlineFiltering,
    guide: HybridMeasurementGuideV9L2OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    measurement_params = guide.hybrid_measurement_posterior_means(
        batch=runtime_batch
    ).to(
        device=runtime_batch.X_asset.device,
        dtype=runtime_batch.X_asset.dtype,
    )
    draws = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=int(num_samples),
        measurement_params=measurement_params,
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


def _rollout_samples(
    *,
    model: HybridMeasurementModelV9L2OnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
    measurement_params: HybridMeasurementPosteriorMeans,
) -> torch.Tensor:
    filtering_state = _move_filtering_state(
        batch=batch,
        filtering_state=batch.filtering_state,
    )
    structural = _move_structural(
        structural=structural,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    regime = _initial_regime_samples(
        filtering_state=filtering_state,
        num_samples=num_samples,
    )
    context = _ObservationContext(
        batch=batch,
        structural=structural,
        measurement_params=measurement_params,
    )
    structure = build_hybrid_measurement_structure(
        assets=batch.assets,
        config=model.priors.hybrid_measurement,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    phi = _phi_vector(model, device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    scales = structural.regime.as_tensor().to(
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    draws = []
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(
            regime
        )
        draws.append(
            _sample_observation_step(
                model=model,
                context=context,
                structure=structure,
                regime=regime,
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: HybridMeasurementModelV9L2OnlineFiltering,
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
    model: HybridMeasurementModelV9L2OnlineFiltering,
    context: _ObservationContext,
    structure: Any,
    regime: torch.Tensor,
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
    index_block = build_hybrid_measurement_factor_block(
        structure=structure,
        state=HybridMeasurementFactorState(
            state_scale=context.measurement_params.state_scale,
            state_corr_cholesky=context.measurement_params.state_corr_cholesky,
            loading_delta=context.measurement_params.loading_delta,
            regime_scale=torch.exp(0.5 * regime[:, 2]),
            mix=_sample_mix(
                num_samples=regime.shape[0],
                df_value=model.priors.hybrid_measurement.state_df,
                device=context.batch.X_asset.device,
                dtype=context.batch.X_asset.dtype,
            ),
            eps=model.priors.hybrid_measurement.eps,
        ),
    )
    cov_diag = apply_hybrid_measurement_residual_scale(
        cov_diag=base_cov_diag,
        residual_scale=context.measurement_params.residual_scale,
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
    "build_hybrid_measurement_predict_v9_l2_online_filtering",
    "predict_hybrid_measurement_v9_l2",
]
