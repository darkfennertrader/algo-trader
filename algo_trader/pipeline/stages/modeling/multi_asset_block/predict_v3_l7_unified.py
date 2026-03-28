from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor
from algo_trader.pipeline.stages.modeling.runtime_support import move_filtering_state

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .guide_v3_l7_unified import MultiAssetBlockGuideV3L7UnifiedOnlineFiltering
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
from .shared_v3_l7_unified import (
    RegimePosteriorMeansV3L7,
    StructuralPosteriorMeansV3L7,
    build_dynamic_europe_uk_ch_spread_block,
    build_dynamic_us_europe_spread_block,
    coerce_v3_l7_state_tensor,
    v3_l7_commodity_state_index,
    v3_l7_eu_vs_uk_ch_spread_state_index,
    v3_l7_us_eu_spread_state_index,
)

if TYPE_CHECKING:
    from .model_v3_l7_unified import MultiAssetBlockModelV3L7UnifiedOnlineFiltering


@dataclass(frozen=True)
class _SpreadRolloutConfig:
    index: int
    degrees_of_freedom: torch.Tensor


class _V3L7UnifiedPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_multi_asset_block_v3_l7_unified(
            model=cast(Any, request.model),
            guide=cast(MultiAssetBlockGuideV3L7UnifiedOnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_multi_asset_block_v3_l7_unified(
    *,
    model: MultiAssetBlockModelV3L7UnifiedOnlineFiltering,
    guide: MultiAssetBlockGuideV3L7UnifiedOnlineFiltering,
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


@register_predictor("multi_asset_block_predict_v3_l7_unified_online_filtering")
def build_multi_asset_block_predict_v3_l7_unified_online_filtering(
    params: Mapping[str, Any],
) -> _V3L7UnifiedPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l7_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L7UnifiedPredictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: MultiAssetBlockGuideV3L7UnifiedOnlineFiltering,
) -> StructuralPosteriorMeansV3L7:
    payload = None if state is None else state.get("structural_posterior_means")
    if isinstance(payload, Mapping):
        return StructuralPosteriorMeansV3L7.from_mapping(payload)
    summaries = getattr(guide, "structural_predictive_summaries", None)
    if callable(summaries):
        return cast(StructuralPosteriorMeansV3L7, summaries())
    return guide.structural_posterior_means()


def _rollout_samples(
    *,
    model: MultiAssetBlockModelV3L7UnifiedOnlineFiltering,
    structural: StructuralPosteriorMeansV3L7,
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
    scales = structural.regime.as_tensor().to(
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    spread_configs = _spread_rollout_configs(model=model, batch=batch)
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = _roll_regime_step(
            previous=regime,
            phi=phi,
            scales=scales,
            spread_configs=spread_configs,
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
    model: MultiAssetBlockModelV3L7UnifiedOnlineFiltering,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [
            model.priors.regime.fx_broad.phi,
            model.priors.regime.fx_cross.phi,
            model.priors.regime.index.phi,
            model.priors.regime.index_spread_us_eu.phi,
            model.priors.regime.index_spread_eu_vs_uk_ch.phi,
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


def _spread_rollout_configs(
    *,
    model: MultiAssetBlockModelV3L7UnifiedOnlineFiltering,
    batch: V3L1UnifiedRuntimeBatch,
) -> tuple[_SpreadRolloutConfig, _SpreadRolloutConfig]:
    return (
        _SpreadRolloutConfig(
            index=v3_l7_us_eu_spread_state_index(),
            degrees_of_freedom=torch.tensor(
                model.priors.regime.index_spread_us_eu.df,
                device=batch.X_asset.device,
                dtype=batch.X_asset.dtype,
            ),
        ),
        _SpreadRolloutConfig(
            index=v3_l7_eu_vs_uk_ch_spread_state_index(),
            degrees_of_freedom=torch.tensor(
                model.priors.regime.index_spread_eu_vs_uk_ch.df,
                device=batch.X_asset.device,
                dtype=batch.X_asset.dtype,
            ),
        ),
    )


def _roll_regime_step(
    *,
    previous: torch.Tensor,
    phi: torch.Tensor,
    scales: torch.Tensor,
    spread_configs: tuple[_SpreadRolloutConfig, _SpreadRolloutConfig],
) -> torch.Tensor:
    regime = phi.unsqueeze(0) * previous + scales.unsqueeze(0) * torch.randn_like(previous)
    for spread_config in spread_configs:
        regime[:, spread_config.index] = _sample_spread_state(
            previous=previous[:, spread_config.index],
            phi=phi[spread_config.index],
            scale=scales[spread_config.index],
            degrees_of_freedom=spread_config.degrees_of_freedom,
        )
    return regime


def _sample_spread_state(
    *,
    previous: torch.Tensor,
    phi: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
) -> torch.Tensor:
    return dist.StudentT(
        degrees_of_freedom,
        loc=phi * previous,
        scale=scale,
    ).sample()


def _sample_observation_step(
    *,
    structural: StructuralPosteriorMeansV3L7,
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
    structural: StructuralPosteriorMeansV3L7,
    batch: V3L1UnifiedRuntimeBatch,
    time_index: int,
) -> torch.Tensor:
    asset_term = (batch.X_asset[time_index] * structural.w).sum(dim=-1)
    global_term = batch.X_global[time_index] @ structural.beta.T
    return structural.alpha + asset_term + global_term


def _cov_factor_step(
    *,
    structural: StructuralPosteriorMeansV3L7,
    batch: V3L1UnifiedRuntimeBatch,
    regime: torch.Tensor,
) -> torch.Tensor:
    dtype = batch.X_asset.dtype
    class_ids = batch.assets.class_ids
    fx_mask = asset_class_mask(class_ids, class_id=FX_CLASS_ID, dtype=dtype).unsqueeze(0).unsqueeze(-1)
    index_mask = asset_class_mask(class_ids, class_id=INDEX_CLASS_ID, dtype=dtype).unsqueeze(0).unsqueeze(-1)
    commodity_mask = asset_class_mask(class_ids, class_id=COMMODITY_CLASS_ID, dtype=dtype).unsqueeze(0).unsqueeze(-1)
    us_eu_index = v3_l7_us_eu_spread_state_index()
    eu_uk_ch_index = v3_l7_eu_vs_uk_ch_spread_state_index()
    commodity_index = v3_l7_commodity_state_index()
    return torch.cat(
        [
            structural.B_global.unsqueeze(0).expand(regime.shape[0], -1, -1),
            structural.B_fx_broad.unsqueeze(0) * fx_mask * torch.exp(0.5 * regime[:, 0]).view(-1, 1, 1),
            structural.B_fx_cross.unsqueeze(0) * fx_mask * torch.exp(0.5 * regime[:, 1]).view(-1, 1, 1),
            structural.B_index_static.unsqueeze(0).expand(regime.shape[0], -1, -1) * index_mask,
            structural.B_index.unsqueeze(0) * index_mask * torch.exp(0.5 * regime[:, 2]).view(-1, 1, 1),
            build_dynamic_us_europe_spread_block(
                assets=batch.assets,
                spread_state=regime[:, us_eu_index],
                device=batch.X_asset.device,
                dtype=batch.X_asset.dtype,
            ),
            build_dynamic_europe_uk_ch_spread_block(
                assets=batch.assets,
                spread_state=regime[:, eu_uk_ch_index],
                device=batch.X_asset.device,
                dtype=batch.X_asset.dtype,
            ),
            structural.B_commodity.unsqueeze(0) * commodity_mask * torch.exp(0.5 * regime[:, commodity_index]).view(-1, 1, 1),
        ],
        dim=-1,
    )


def _move_filtering_state(
    *, batch: V3L1UnifiedRuntimeBatch, filtering_state: FilteringState | None
) -> FilteringState:
    if filtering_state is None:
        raise ConfigError("v3_l7_unified rollout requires filtering_state")
    return move_filtering_state(
        filtering_state=filtering_state,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        coerce_tensor=lambda value: coerce_v3_l7_state_tensor(
            value,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
    )


def _move_structural(
    *,
    structural: StructuralPosteriorMeansV3L7,
    device: torch.device,
    dtype: torch.dtype,
) -> StructuralPosteriorMeansV3L7:
    return StructuralPosteriorMeansV3L7(
        tensors=StructuralTensorMeans(
            mean=_move_mean_tensors(structural=structural, device=device, dtype=dtype),
            loadings=_move_covariance_loadings(
                structural=structural,
                device=device,
                dtype=dtype,
            ),
        ),
        regime=_move_regime_means(structural=structural, device=device, dtype=dtype),
    )


def _move_mean_tensors(
    *,
    structural: StructuralPosteriorMeansV3L7,
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
    structural: StructuralPosteriorMeansV3L7,
    device: torch.device,
    dtype: torch.dtype,
) -> CovarianceLoadings:
    return CovarianceLoadings(
        B_global=structural.B_global.to(device=device, dtype=dtype),
        B_fx_broad=structural.B_fx_broad.to(device=device, dtype=dtype),
        B_fx_cross=structural.B_fx_cross.to(device=device, dtype=dtype),
        B_index=structural.B_index.to(device=device, dtype=dtype),
        B_index_static=structural.B_index_static.to(device=device, dtype=dtype),
        index_group_scale=structural.index_group_scale.to(device=device, dtype=dtype),
        B_commodity=structural.B_commodity.to(device=device, dtype=dtype),
    )


def _move_regime_means(
    *,
    structural: StructuralPosteriorMeansV3L7,
    device: torch.device,
    dtype: torch.dtype,
) -> RegimePosteriorMeansV3L7:
    return RegimePosteriorMeansV3L7(
        s_u_fx_broad_mean=structural.s_u_fx_broad_mean.to(device=device, dtype=dtype),
        s_u_fx_cross_mean=structural.s_u_fx_cross_mean.to(device=device, dtype=dtype),
        s_u_index_mean=structural.s_u_index_mean.to(device=device, dtype=dtype),
        s_u_index_spread_us_eu_mean=structural.s_u_index_spread_us_eu_mean.to(
            device=device,
            dtype=dtype,
        ),
        s_u_index_spread_eu_vs_uk_ch_mean=structural.s_u_index_spread_eu_vs_uk_ch_mean.to(
            device=device,
            dtype=dtype,
        ),
        s_u_commodity_mean=structural.s_u_commodity_mean.to(device=device, dtype=dtype),
    )


__all__ = [
    "build_multi_asset_block_predict_v3_l7_unified_online_filtering",
    "predict_multi_asset_block_v3_l7_unified",
]
