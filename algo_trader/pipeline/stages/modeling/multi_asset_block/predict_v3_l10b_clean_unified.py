from __future__ import annotations
# pylint: disable=duplicate-code

from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.predictive_stats import (
    predictive_covariance,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PredictiveRequest,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide_v3_l10b_clean_unified import (
    MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering,
)
from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .predict_v3_l1_unified import (
    _cov_factor_step,
    _initial_regime_samples,
    _mean_step,
    _move_filtering_state,
    _move_structural,
    _resolve_structural_means,
)
from .shared_v3_l10b_clean_unified import (
    IndexTCopulaMixSamples,
    apply_index_t_copula_overlay,
)

if TYPE_CHECKING:
    from .model_v3_l10b_clean_unified import (
        MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
    )


class _V3L10BCleanUnifiedPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_multi_asset_block_v3_l10b_clean_unified(
            model=cast(Any, request.model),
            guide=cast(
                MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering,
                request.guide,
            ),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor(
    "multi_asset_block_predict_v3_l10b_clean_unified_online_filtering"
)
def build_multi_asset_block_predict_v3_l10b_clean_unified_online_filtering(
    params: Mapping[str, Any],
) -> _V3L10BCleanUnifiedPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l10b_clean_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L10BCleanUnifiedPredictor()


def predict_multi_asset_block_v3_l10b_clean_unified(
    *,
    model: MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
    guide: MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(
        state=state,
        guide=cast(Any, guide),
    )
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
    model: MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
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
    for time_index in range(int(batch.X_asset.shape[0])):
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(regime)
        draws.append(
            _sample_observation_step(
                model=model,
                structural=structural,
                batch=batch,
                regime=regime,
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
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
    model: MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
    structural: Any,
    batch: Any,
    regime: torch.Tensor,
    time_index: int,
) -> torch.Tensor:
    loc = _mean_step(structural=structural, batch=batch, time_index=time_index)
    cov_factor = _cov_factor_step(
        structural=structural,
        batch=batch,
        regime=regime,
    )
    base_cov_diag = structural.sigma_idio.pow(2)
    mixes = _sample_index_t_copula_mix(
        num_samples=regime.shape[0],
        model=model,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=cov_factor,
        cov_diag=base_cov_diag,
        assets=batch.assets,
        mixes=mixes,
        overlay=model.priors.index_t_copula,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    ).rsample()


def _sample_index_t_copula_mix(
    *,
    num_samples: int,
    model: MultiAssetBlockModelV3L10BCleanUnifiedOnlineFiltering,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexTCopulaMixSamples:
    overlay = model.priors.index_t_copula
    if not overlay.enabled:
        ones = torch.ones((num_samples,), device=device, dtype=dtype)
        return IndexTCopulaMixSamples(broad=ones, us_diff=ones)
    broad_concentration = torch.full(
        (num_samples,),
        overlay.broad_df / 2.0,
        device=device,
        dtype=dtype,
    )
    us_diff_concentration = torch.full(
        (num_samples,),
        overlay.us_diff_df / 2.0,
        device=device,
        dtype=dtype,
    )
    return IndexTCopulaMixSamples(
        broad=dist.Gamma(broad_concentration, broad_concentration).sample(),
        us_diff=dist.Gamma(us_diff_concentration, us_diff_concentration).sample(),
    )


__all__ = [
    "build_multi_asset_block_predict_v3_l10b_clean_unified_online_filtering",
    "predict_multi_asset_block_v3_l10b_clean_unified",
]
