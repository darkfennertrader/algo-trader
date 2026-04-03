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

from .guide import IndexBasisGuideV8L1OnlineFiltering
from .shared import (
    IndexBasisFactorState,
    IndexBasisPosteriorMeans,
    build_index_basis_coordinates,
    build_index_basis_factor_block,
    build_nonindex_cov_factor_step,
)

if TYPE_CHECKING:
    from .model import IndexBasisModelV8L1OnlineFiltering


class _V8L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_index_basis_v8_l1(
            model=cast(Any, request.model),
            guide=cast(IndexBasisGuideV8L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("index_basis_predict_v8_l1_online_filtering")
def build_index_basis_predict_v8_l1_online_filtering(
    params: Mapping[str, Any],
) -> _V8L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown index_basis_predict_v8_l1_online_filtering params: "
            f"{unknown}"
        )
    return _V8L1Predictor()


def predict_index_basis_v8_l1(
    *,
    model: IndexBasisModelV8L1OnlineFiltering,
    guide: IndexBasisGuideV8L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    basis_params = guide.index_basis_posterior_means().to(
        device=runtime_batch.X_asset.device,
        dtype=runtime_batch.X_asset.dtype,
    )
    draws = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=int(num_samples),
        basis_params=basis_params,
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
    basis_params: IndexBasisPosteriorMeans


def _rollout_samples(
    *,
    model: IndexBasisModelV8L1OnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
    basis_params: IndexBasisPosteriorMeans,
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
        basis_params=basis_params,
    )
    phi = _phi_vector(model, device=batch.X_asset.device, dtype=batch.X_asset.dtype)
    scales = structural.regime.as_tensor().to(
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    draws = []
    coordinates = build_index_basis_coordinates(
        assets=batch.assets,
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
                context=context,
                coordinates=coordinates,
                regime=regime,
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _phi_vector(
    model: IndexBasisModelV8L1OnlineFiltering,
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
    model: IndexBasisModelV8L1OnlineFiltering,
    context: _ObservationContext,
    coordinates: Any,
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
    index_block = build_index_basis_factor_block(
        coordinates=coordinates,
        state=IndexBasisFactorState(
            global_scale=context.basis_params.global_scale,
            spread_scale=context.basis_params.spread_scale,
            spread_corr_cholesky=context.basis_params.spread_corr_cholesky,
            regime_scale=torch.exp(0.5 * regime[:, 2]),
            global_mix=_sample_mix(
                num_samples=regime.shape[0],
                df_value=model.priors.index_basis.global_df,
                device=context.batch.X_asset.device,
                dtype=context.batch.X_asset.dtype,
            ),
            spread_mix=_sample_mix(
                num_samples=regime.shape[0],
                df_value=model.priors.index_basis.spread_df,
                device=context.batch.X_asset.device,
                dtype=context.batch.X_asset.dtype,
            ),
            eps=model.priors.index_basis.eps,
        ),
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([nonindex_block, index_block], dim=-1),
        cov_diag=base_cov_diag,
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
    "build_index_basis_predict_v8_l1_online_filtering",
    "predict_index_basis_v8_l1",
]
