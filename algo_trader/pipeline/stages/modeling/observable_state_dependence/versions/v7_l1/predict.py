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
    _cov_factor_step,
    _initial_regime_samples,
    _mean_step,
    _move_filtering_state,
    _move_structural,
    _resolve_structural_means,
)
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide import ObservableStateDependenceGuideV7L1OnlineFiltering
from .shared import (
    ObservableStateCoefficients,
    ObservableStateOverlayInputs,
    apply_observable_state_dependence_overlay,
    build_observable_state_gate_series,
)

if TYPE_CHECKING:
    from .model import ObservableStateDependenceModelV7L1OnlineFiltering


class _V7L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_observable_state_dependence_v7_l1(
            model=cast(Any, request.model),
            guide=cast(ObservableStateDependenceGuideV7L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("observable_state_dependence_predict_v7_l1_online_filtering")
def build_observable_state_dependence_predict_v7_l1_online_filtering(
    params: Mapping[str, Any],
) -> _V7L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown observable_state_dependence_predict_v7_l1_online_filtering "
            f"params: {unknown}"
        )
    return _V7L1Predictor()


def predict_observable_state_dependence_v7_l1(
    *,
    model: ObservableStateDependenceModelV7L1OnlineFiltering,
    guide: ObservableStateDependenceGuideV7L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    coefficients = guide.observable_state_posterior_means().to(
        device=runtime_batch.X_asset.device,
        dtype=runtime_batch.X_asset.dtype,
    )
    gate_series = build_observable_state_gate_series(
        X_asset=runtime_batch.X_asset,
        X_global=runtime_batch.X_global,
        assets=runtime_batch.assets,
        coefficients=coefficients,
        overlay=model.priors.observable_state_dependence,
    )
    overlay = _PredictiveOverlayState(
        coefficients=coefficients,
        gate_series=gate_series,
    )
    draws = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=int(num_samples),
        overlay=overlay,
    )
    return {
        "samples": draws,
        "mean": draws.mean(dim=0),
        "covariance": predictive_covariance(draws),
    }


@dataclass(frozen=True)
class _PredictiveOverlayState:
    coefficients: ObservableStateCoefficients
    gate_series: torch.Tensor


@dataclass(frozen=True)
class _ObservationContext:
    batch: Any
    structural: Any
    coefficients: ObservableStateCoefficients


def _rollout_samples(
    *,
    model: ObservableStateDependenceModelV7L1OnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
    overlay: _PredictiveOverlayState,
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
    draws = []
    phi = _phi_vector(model, device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    scales = structural.regime.as_tensor().to(
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    context = _ObservationContext(
        batch=batch,
        structural=structural,
        coefficients=overlay.coefficients,
    )
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(
            regime
        )
        draws.append(
            _sample_observation_step(
                model=model,
                context=context,
                regime=regime,
                gate_value=overlay.gate_series[time_index],
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: ObservableStateDependenceModelV7L1OnlineFiltering,
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
    model: ObservableStateDependenceModelV7L1OnlineFiltering,
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
    cov_factor = _cov_factor_step(
        structural=context.structural,
        batch=context.batch,
        regime=regime,
    )
    base_cov_diag = context.structural.sigma_idio.pow(2)
    scaled_factor, scaled_diag = apply_observable_state_dependence_overlay(
        inputs=ObservableStateOverlayInputs(
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
            gate=gate_value.expand(regime.shape[0]),
        ),
        assets=context.batch.assets,
        coefficients=context.coefficients,
        eps=model.priors.observable_state_dependence.eps,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    ).rsample()


__all__ = [
    "build_observable_state_dependence_predict_v7_l1_online_filtering",
    "predict_observable_state_dependence_v7_l1",
]
