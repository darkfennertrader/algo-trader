from __future__ import annotations
# pylint: disable=duplicate-code

from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor
from algo_trader.pipeline.stages.modeling.runtime_support import move_filtering_state

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l9_unified import MultiAssetBlockGuideV3L9UnifiedOnlineFiltering
from .shared_v3_l1_unified import (
    COMMODITY_CLASS_ID,
    FX_CLASS_ID,
    INDEX_CLASS_ID,
    CovarianceLoadings,
    FilteringState,
    MeanTensorMeans,
    StructuralTensorMeans,
    V3L1UnifiedRuntimeBatch,
    asset_class_mask,
)
from .shared_v3_l9_unified import (
    RegimePosteriorMeansV3L9,
    StructuralPosteriorMeansV3L9,
    build_dynamic_europe_core_region_block,
    build_dynamic_us_europe_region_block,
    coerce_v3_l9_state_tensor,
    v3_l9_commodity_state_index,
    v3_l9_eu_core_state_index,
    v3_l9_us_eu_region_state_index,
)

if TYPE_CHECKING:
    from .model_v3_l9_unified import MultiAssetBlockModelV3L9UnifiedOnlineFiltering


class _V3L9UnifiedPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_multi_asset_block_v3_l9_unified(
            model=cast(Any, request.model),
            guide=cast(MultiAssetBlockGuideV3L9UnifiedOnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_multi_asset_block_v3_l9_unified(
    *,
    model: MultiAssetBlockModelV3L9UnifiedOnlineFiltering,
    guide: MultiAssetBlockGuideV3L9UnifiedOnlineFiltering,
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


class _RegionalRolloutConfig:
    def __init__(
        self,
        *,
        us_eu_index: int,
        eu_core_index: int,
        us_eu_df: torch.Tensor,
        eu_core_df: torch.Tensor,
    ) -> None:
        self.us_eu_index = us_eu_index
        self.eu_core_index = eu_core_index
        self.us_eu_df = us_eu_df
        self.eu_core_df = eu_core_df


@register_predictor("multi_asset_block_predict_v3_l9_unified_online_filtering")
def build_multi_asset_block_predict_v3_l9_unified_online_filtering(
    params: Mapping[str, Any],
) -> _V3L9UnifiedPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l9_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L9UnifiedPredictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: MultiAssetBlockGuideV3L9UnifiedOnlineFiltering,
) -> StructuralPosteriorMeansV3L9:
    payload = None if state is None else state.get("structural_posterior_means")
    if isinstance(payload, Mapping):
        return StructuralPosteriorMeansV3L9.from_mapping(payload)
    summaries = getattr(guide, "structural_predictive_summaries", None)
    if callable(summaries):
        return cast(StructuralPosteriorMeansV3L9, summaries())
    return guide.structural_posterior_means()


def _rollout_samples(
    *,
    model: MultiAssetBlockModelV3L9UnifiedOnlineFiltering,
    structural: StructuralPosteriorMeansV3L9,
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
    phi = _phi_vector(model, device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    scales = structural.regime.as_tensor().to(device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    regional = _RegionalRolloutConfig(
        us_eu_index=v3_l9_us_eu_region_state_index(),
        eu_core_index=v3_l9_eu_core_state_index(),
        us_eu_df=torch.tensor(
            model.priors.regime.index_region_us_eu.df,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
        eu_core_df=torch.tensor(
            model.priors.regime.index_region_eu_core_vs_uk_ch.df,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
    )
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = _roll_regime_step(
            previous=regime,
            phi=phi,
            scales=scales,
            regional=regional,
        )
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
    model: MultiAssetBlockModelV3L9UnifiedOnlineFiltering,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [
            model.priors.regime.fx_broad.phi,
            model.priors.regime.fx_cross.phi,
            model.priors.regime.index.phi,
            model.priors.regime.index_region_us_eu.phi,
            model.priors.regime.index_region_eu_core_vs_uk_ch.phi,
            model.priors.regime.commodity.phi,
        ],
        device=device,
        dtype=dtype,
    )


def _initial_regime_samples(
    *, filtering_state: FilteringState, num_samples: int
) -> torch.Tensor:
    loc = coerce_v3_l9_state_tensor(
        filtering_state.h_loc,
        device=filtering_state.h_loc.device,
        dtype=filtering_state.h_loc.dtype,
    ).unsqueeze(0).expand(num_samples, -1)
    scale = coerce_v3_l9_state_tensor(
        filtering_state.h_scale,
        device=filtering_state.h_scale.device,
        dtype=filtering_state.h_scale.dtype,
    ).unsqueeze(0).expand(num_samples, -1)
    return loc + scale * torch.randn_like(loc)


def _roll_regime_step(
    *,
    previous: torch.Tensor,
    phi: torch.Tensor,
    scales: torch.Tensor,
    regional: _RegionalRolloutConfig,
) -> torch.Tensor:
    regime = phi.unsqueeze(0) * previous + scales.unsqueeze(0) * torch.randn_like(previous)
    us_eu_dist = dist.StudentT(
        regional.us_eu_df,
        loc=phi[regional.us_eu_index] * previous[:, regional.us_eu_index],
        scale=scales[regional.us_eu_index],
    )
    eu_core_dist = dist.StudentT(
        regional.eu_core_df,
        loc=phi[regional.eu_core_index] * previous[:, regional.eu_core_index],
        scale=scales[regional.eu_core_index],
    )
    regime[:, regional.us_eu_index] = us_eu_dist.sample()
    regime[:, regional.eu_core_index] = eu_core_dist.sample()
    return regime


def _sample_observation_step(
    *,
    structural: StructuralPosteriorMeansV3L9,
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
    structural: StructuralPosteriorMeansV3L9,
    batch: V3L1UnifiedRuntimeBatch,
    time_index: int,
) -> torch.Tensor:
    asset_term = (batch.X_asset[time_index] * structural.w).sum(dim=-1)
    global_term = batch.X_global[time_index] @ structural.beta.T
    return structural.alpha + asset_term + global_term


def _cov_factor_step(
    *,
    structural: StructuralPosteriorMeansV3L9,
    batch: V3L1UnifiedRuntimeBatch,
    regime: torch.Tensor,
) -> torch.Tensor:
    dtype = batch.X_asset.dtype
    fx_mask = asset_class_mask(batch.assets.class_ids, class_id=FX_CLASS_ID, dtype=dtype).unsqueeze(-1)
    index_mask = asset_class_mask(batch.assets.class_ids, class_id=INDEX_CLASS_ID, dtype=dtype).unsqueeze(-1)
    commodity_mask = asset_class_mask(batch.assets.class_ids, class_id=COMMODITY_CLASS_ID, dtype=dtype).unsqueeze(-1)
    loadings = structural.tensors.loadings
    fx_broad = _scaled_block(loadings=loadings.B_fx_broad, mask=fx_mask, regime=regime[:, 0])
    fx_cross = _scaled_block(loadings=loadings.B_fx_cross, mask=fx_mask, regime=regime[:, 1])
    index_static = loadings.B_index_static.unsqueeze(0) * index_mask.unsqueeze(0)
    index_block = _scaled_block(loadings=loadings.B_index, mask=index_mask, regime=regime[:, 2])
    us_eu_region = build_dynamic_us_europe_region_block(
        assets=batch.assets,
        region_state=regime[:, 3],
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    eu_core_region = build_dynamic_europe_core_region_block(
        assets=batch.assets,
        region_state=regime[:, 4],
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    commodity_block = _scaled_block(
        loadings=loadings.B_commodity,
        mask=commodity_mask,
        regime=regime[:, v3_l9_commodity_state_index()],
    )
    return torch.cat(
        [
            loadings.B_global.unsqueeze(0).expand(regime.shape[0], -1, -1),
            fx_broad,
            fx_cross,
            index_static.expand(regime.shape[0], -1, -1),
            index_block,
            us_eu_region,
            eu_core_region,
            commodity_block,
        ],
        dim=-1,
    )


def _scaled_block(
    *,
    loadings: torch.Tensor,
    mask: torch.Tensor,
    regime: torch.Tensor,
) -> torch.Tensor:
    return (
        loadings.unsqueeze(0)
        * mask.unsqueeze(0)
        * torch.exp(0.5 * regime).view(-1, 1, 1)
    )


def _move_filtering_state(
    *,
    batch: V3L1UnifiedRuntimeBatch,
    filtering_state: FilteringState | None,
) -> FilteringState:
    if filtering_state is None:
        raise ConfigError("v3_l9_unified rollout requires filtering_state")
    return move_filtering_state(
        filtering_state=filtering_state,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        coerce_tensor=lambda value: coerce_v3_l9_state_tensor(
            value,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
    )


def _move_structural(
    *,
    structural: StructuralPosteriorMeansV3L9,
    device: torch.device,
    dtype: torch.dtype,
) -> StructuralPosteriorMeansV3L9:
    return StructuralPosteriorMeansV3L9.from_mapping(
        {
            key: value.to(device=device, dtype=dtype)
            for key, value in structural.to_mapping().items()
        }
    )


__all__ = [
    "build_multi_asset_block_predict_v3_l9_unified_online_filtering",
    "predict_multi_asset_block_v3_l9_unified",
]
