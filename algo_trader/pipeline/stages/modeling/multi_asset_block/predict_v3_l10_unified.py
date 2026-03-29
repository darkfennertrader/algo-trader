from __future__ import annotations
# pylint: disable=duplicate-code

from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l10_unified import MultiAssetBlockGuideV3L10UnifiedOnlineFiltering
from .predict_v3_l6_unified import (
    _cov_factor_step,
    _initial_regime_samples,
    _mean_step,
    _move_filtering_state,
    _move_structural,
    _resolve_structural_means,
    _roll_regime_step,
)

if TYPE_CHECKING:
    from .model_v3_l10_unified import MultiAssetBlockModelV3L10UnifiedOnlineFiltering


class _V3L10UnifiedPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_multi_asset_block_v3_l10_unified(
            model=cast(Any, request.model),
            guide=cast(MultiAssetBlockGuideV3L10UnifiedOnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("multi_asset_block_predict_v3_l10_unified_online_filtering")
def build_multi_asset_block_predict_v3_l10_unified_online_filtering(
    params: Mapping[str, Any],
) -> _V3L10UnifiedPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l10_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L10UnifiedPredictor()


def predict_multi_asset_block_v3_l10_unified(
    *,
    model: MultiAssetBlockModelV3L10UnifiedOnlineFiltering,
    guide: MultiAssetBlockGuideV3L10UnifiedOnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=guide)
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    draws = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=int(num_samples),
    )
    return {
        "samples": draws,
        "mean": draws.mean(dim=0),
        "covariance": predictive_covariance(draws),
    }


def _rollout_samples(
    *,
    model: MultiAssetBlockModelV3L10UnifiedOnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
) -> torch.Tensor:
    filtering_state = _move_filtering_state(batch=batch, filtering_state=batch.filtering_state)
    structural = _move_structural(
        structural=structural,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    regime = _initial_regime_samples(
        filtering_state=filtering_state,
        num_samples=num_samples,
    )
    flow = model.sync_index_flow(batch)
    draws = []
    phi = _phi_vector(model, device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    scales = structural.regime.as_tensor().to(
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    spread_index = 3
    spread_df = torch.tensor(
        model.priors.base.regime.index_spread.df,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = _roll_regime_step(
            previous=regime,
            phi=phi,
            scales=scales,
            spread_index=spread_index,
            spread_df=spread_df,
        )
        draws.append(
            _sample_observation_step(
                structural=structural,
                batch=batch,
                regime=regime,
                time_index=time_index,
                flow=flow,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: MultiAssetBlockModelV3L10UnifiedOnlineFiltering,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [
            model.priors.base.regime.fx_broad.phi,
            model.priors.base.regime.fx_cross.phi,
            model.priors.base.regime.index.phi,
            model.priors.base.regime.index_spread.phi,
            model.priors.base.regime.commodity.phi,
        ],
        device=device,
        dtype=dtype,
    )


def _sample_observation_step(
    *,
    structural: Any,
    batch: Any,
    regime: torch.Tensor,
    time_index: int,
    flow: Any,
) -> torch.Tensor:
    loc = _mean_step(structural=structural, batch=batch, time_index=time_index)
    cov_factor = _cov_factor_step(structural=structural, batch=batch, regime=regime)
    cov_diag = structural.sigma_idio.pow(2)
    base_dist = dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    )
    if flow is None:
        return base_dist.rsample()
    pyro.module("multi_asset_block_v3_l10_index_flow", flow)
    obs_dist = dist.TransformedDistribution(base_dist, [flow])
    return obs_dist.rsample()


__all__ = [
    "build_multi_asset_block_predict_v3_l10_unified_online_filtering",
    "predict_multi_asset_block_v3_l10_unified",
]
