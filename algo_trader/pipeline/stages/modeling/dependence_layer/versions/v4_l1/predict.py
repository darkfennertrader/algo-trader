from __future__ import annotations
# pylint: disable=duplicate-code

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
from algo_trader.pipeline.stages.modeling.predictive_stats import (
    predictive_covariance,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PredictiveRequest,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide import DependenceLayerGuideV4L1OnlineFiltering
from .shared import apply_index_t_copula_overlay

if TYPE_CHECKING:
    from .model import DependenceLayerModelV4L1OnlineFiltering


class _V4L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_dependence_layer_v4_l1(
            model=cast(Any, request.model),
            guide=cast(DependenceLayerGuideV4L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("dependence_layer_predict_v4_l1_online_filtering")
def build_dependence_layer_predict_v4_l1_online_filtering(
    params: Mapping[str, Any],
) -> _V4L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown dependence_layer_predict_v4_l1_online_filtering "
            f"params: {unknown}"
        )
    return _V4L1Predictor()


def predict_dependence_layer_v4_l1(
    *,
    model: DependenceLayerModelV4L1OnlineFiltering,
    guide: DependenceLayerGuideV4L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
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
    model: DependenceLayerModelV4L1OnlineFiltering,
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
        regime = phi.unsqueeze(0) * regime + scales.unsqueeze(0) * torch.randn_like(
            regime
        )
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
    model: DependenceLayerModelV4L1OnlineFiltering,
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
    model: DependenceLayerModelV4L1OnlineFiltering,
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
    mix = _sample_index_t_copula_mix(
        num_samples=regime.shape[0],
        model=model,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=cov_factor,
        cov_diag=base_cov_diag,
        assets=batch.assets,
        mix=mix,
        eps=model.priors.index_t_copula.eps,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    ).rsample()


def _sample_index_t_copula_mix(
    *,
    num_samples: int,
    model: DependenceLayerModelV4L1OnlineFiltering,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    overlay = model.priors.index_t_copula
    if not overlay.enabled:
        return torch.ones((num_samples,), device=device, dtype=dtype)
    concentration = torch.full(
        (num_samples,),
        overlay.df / 2.0,
        device=device,
        dtype=dtype,
    )
    return dist.Gamma(concentration, concentration).sample()


__all__ = [
    "build_dependence_layer_predict_v4_l1_online_filtering",
    "predict_dependence_layer_v4_l1",
]
