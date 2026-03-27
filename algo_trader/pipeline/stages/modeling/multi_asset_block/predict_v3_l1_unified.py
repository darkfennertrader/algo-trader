from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor
from algo_trader.pipeline.stages.modeling.runtime_support import (
    move_filtering_state,
)

from .guide_v3_l1_unified import (
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    build_v3_l1_unified_runtime_batch,
)
from .shared_v3_l1_unified import (
    CovarianceLoadings,
    FX_CLASS_ID,
    INDEX_CLASS_ID,
    COMMODITY_CLASS_ID,
    FilteringState,
    MeanTensorMeans,
    RegimePosteriorMeans,
    StructuralPosteriorMeans,
    StructuralTensorMeans,
    V3L1UnifiedRuntimeBatch,
    asset_class_mask,
    coerce_four_state_tensor,
)

if TYPE_CHECKING:
    from .model_v3_l1_unified import MultiAssetBlockModelV3L1UnifiedOnlineFiltering


class _V3L1UnifiedPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_multi_asset_block_v3_l1_unified(
            model=cast(Any, request.model),
            guide=cast(MultiAssetBlockGuideV3L1UnifiedOnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_multi_asset_block_v3_l1_unified(
    *,
    model: MultiAssetBlockModelV3L1UnifiedOnlineFiltering,
    guide: MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=guide)
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    sample_count = int(num_samples)
    draws = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=sample_count,
    )
    return {
        "samples": draws,
        "mean": draws.mean(dim=0),
        "covariance": predictive_covariance(draws),
    }


@register_predictor("multi_asset_block_predict_v3_l1_unified_online_filtering")
def build_multi_asset_block_predict_v3_l1_unified_online_filtering(
    params: Mapping[str, Any],
) -> _V3L1UnifiedPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l1_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L1UnifiedPredictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
) -> StructuralPosteriorMeans:
    payload = None if state is None else state.get("structural_posterior_means")
    if isinstance(payload, Mapping):
        return StructuralPosteriorMeans.from_mapping(payload)
    summaries = getattr(guide, "structural_predictive_summaries", None)
    if callable(summaries):
        return cast(StructuralPosteriorMeans, summaries())
    return guide.structural_posterior_means()


def _rollout_samples(
    *,
    model: MultiAssetBlockModelV3L1UnifiedOnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: V3L1UnifiedRuntimeBatch,
    num_samples: int,
) -> torch.Tensor:
    filtering_state = _move_filtering_state(
        batch=batch, filtering_state=batch.filtering_state
    )
    structural = _move_structural(
        structural=structural,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    regime = _initial_regime_samples(
        filtering_state=filtering_state, num_samples=num_samples
    )
    draws = []
    phi = _phi_vector(
        model,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    scales = structural.regime.as_tensor().to(device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(regime)
        draws.append(
            _sample_observation_step(
                structural=structural,
                batch=batch,
                regime=regime,
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: MultiAssetBlockModelV3L1UnifiedOnlineFiltering,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [
            model.priors.regime.fx_broad.phi,
            model.priors.regime.fx_cross.phi,
            model.priors.regime.index.phi,
            model.priors.regime.commodity.phi,
        ],
        device=device,
        dtype=dtype,
    )


def _initial_regime_samples(
    *, filtering_state: FilteringState, num_samples: int
) -> torch.Tensor:
    loc = filtering_state.h_loc.unsqueeze(0).expand(num_samples, -1)
    scale = filtering_state.h_scale.unsqueeze(0).expand(num_samples, -1)
    return loc + scale * torch.randn_like(loc)


def _sample_observation_step(
    *,
    structural: StructuralPosteriorMeans,
    batch: V3L1UnifiedRuntimeBatch,
    regime: torch.Tensor,
    time_index: int,
) -> torch.Tensor:
    loc = _mean_step(structural=structural, batch=batch, time_index=time_index)
    cov_factor = _cov_factor_step(structural=structural, batch=batch, regime=regime)
    cov_diag = structural.sigma_idio.pow(2)
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    ).rsample()


def _mean_step(
    *,
    structural: StructuralPosteriorMeans,
    batch: V3L1UnifiedRuntimeBatch,
    time_index: int,
) -> torch.Tensor:
    asset_term = (batch.X_asset[time_index] * structural.w).sum(dim=-1)
    global_term = batch.X_global[time_index] @ structural.beta.T
    return structural.alpha + asset_term + global_term


def _cov_factor_step(
    *,
    structural: StructuralPosteriorMeans,
    batch: V3L1UnifiedRuntimeBatch,
    regime: torch.Tensor,
) -> torch.Tensor:
    dtype = batch.X_asset.dtype
    class_ids = batch.assets.class_ids
    fx_mask = asset_class_mask(
        class_ids, class_id=FX_CLASS_ID, dtype=dtype
    ).unsqueeze(0).unsqueeze(-1)
    index_mask = asset_class_mask(
        class_ids, class_id=INDEX_CLASS_ID, dtype=dtype
    ).unsqueeze(0).unsqueeze(-1)
    commodity_mask = asset_class_mask(
        class_ids, class_id=COMMODITY_CLASS_ID, dtype=dtype
    ).unsqueeze(0).unsqueeze(-1)
    return torch.cat(
        [
            structural.B_global.unsqueeze(0).expand(regime.shape[0], -1, -1),
            structural.B_fx_broad.unsqueeze(0) * fx_mask * torch.exp(0.5 * regime[:, 0]).view(-1, 1, 1),
            structural.B_fx_cross.unsqueeze(0) * fx_mask * torch.exp(0.5 * regime[:, 1]).view(-1, 1, 1),
            structural.B_index.unsqueeze(0) * index_mask * torch.exp(0.5 * regime[:, 2]).view(-1, 1, 1),
            structural.B_commodity.unsqueeze(0) * commodity_mask * torch.exp(0.5 * regime[:, 3]).view(-1, 1, 1),
        ],
        dim=-1,
    )


def _move_filtering_state(
    *, batch: V3L1UnifiedRuntimeBatch, filtering_state: FilteringState | None
) -> FilteringState:
    if filtering_state is None:
        raise ConfigError("v3_l1_unified rollout requires filtering_state")
    return move_filtering_state(
        filtering_state=filtering_state,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        coerce_tensor=lambda value: coerce_four_state_tensor(
            value,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
    )


def _move_structural(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> StructuralPosteriorMeans:
    return StructuralPosteriorMeans(
        tensors=StructuralTensorMeans(
            mean=_move_mean_tensors(
                structural=structural,
                device=device,
                dtype=dtype,
            ),
            loadings=_move_covariance_loadings(
                structural=structural,
                device=device,
                dtype=dtype,
            ),
        ),
        regime=_move_regime_means(
            structural=structural,
            device=device,
            dtype=dtype,
        ),
    )


def _move_mean_tensors(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> MeanTensorMeans:
    return MeanTensorMeans(
        alpha=structural.alpha.to(device=device, dtype=dtype),
        sigma_idio=structural.sigma_idio.to(device=device, dtype=dtype),
        w=structural.w.to(device=device, dtype=dtype),
        beta=structural.beta.to(device=device, dtype=dtype),
    )


def _move_covariance_loadings(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> CovarianceLoadings:
    return CovarianceLoadings(
        B_global=structural.B_global.to(device=device, dtype=dtype),
        B_fx_broad=structural.B_fx_broad.to(device=device, dtype=dtype),
        B_fx_cross=structural.B_fx_cross.to(device=device, dtype=dtype),
        B_index=structural.B_index.to(device=device, dtype=dtype),
        B_commodity=structural.B_commodity.to(device=device, dtype=dtype),
    )


def _move_regime_means(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> RegimePosteriorMeans:
    return RegimePosteriorMeans(
        s_u_fx_broad_mean=structural.s_u_fx_broad_mean.to(
            device=device, dtype=dtype
        ),
        s_u_fx_cross_mean=structural.s_u_fx_cross_mean.to(
            device=device, dtype=dtype
        ),
        s_u_index_mean=structural.s_u_index_mean.to(
            device=device, dtype=dtype
        ),
        s_u_commodity_mean=structural.s_u_commodity_mean.to(
            device=device, dtype=dtype
        ),
    )
