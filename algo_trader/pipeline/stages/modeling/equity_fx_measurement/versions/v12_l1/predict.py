from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, cast

import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.shared import (
    apply_index_t_copula_overlay,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.predict_v3_l1_unified import (
    _cov_factor_step as _base_cov_factor_step,
    _initial_regime_samples,
    _mean_step,
    _move_filtering_state,
    _move_structural,
    _resolve_structural_means,
)
from algo_trader.pipeline.stages.modeling.predictive_stats import predictive_covariance
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .guide import EquityFXMeasurementGuideV12L1OnlineFiltering
from .shared import (
    EquityFXMeasurementFactorState,
    EquityFXMeasurementPosteriorMeans,
    apply_equity_fx_measurement_residual_scale,
    build_equity_fx_measurement_factor_block,
    build_equity_fx_measurement_structure,
)

if TYPE_CHECKING:
    from .model import EquityFXMeasurementModelV12L1OnlineFiltering


class _V12L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_equity_fx_measurement_v12_l1(
            model=cast(Any, request.model),
            guide=cast(EquityFXMeasurementGuideV12L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("equity_fx_measurement_predict_v12_l1_online_filtering")
def build_equity_fx_measurement_predict_v12_l1_online_filtering(
    params: Mapping[str, Any],
) -> _V12L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown equity_fx_measurement_predict_v12_l1_online_filtering params: "
            f"{unknown}"
        )
    return _V12L1Predictor()


@dataclass(frozen=True)
class _ObservationContext:
    batch: Any
    structural: Any
    measurement_params: EquityFXMeasurementPosteriorMeans


def predict_equity_fx_measurement_v12_l1(
    *,
    model: EquityFXMeasurementModelV12L1OnlineFiltering,
    guide: EquityFXMeasurementGuideV12L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=cast(Any, guide))
    runtime_batch = build_v3_l1_unified_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    measurement_params = guide.equity_fx_measurement_posterior_means(
        batch=runtime_batch
    ).to(device=runtime_batch.X_asset.device, dtype=runtime_batch.X_asset.dtype)
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


def _rollout_samples(
    *,
    model: EquityFXMeasurementModelV12L1OnlineFiltering,
    structural: Any,
    batch: Any,
    num_samples: int,
    measurement_params: EquityFXMeasurementPosteriorMeans,
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
    structure = build_equity_fx_measurement_structure(
        assets=batch.assets,
        config=model.priors.equity_fx_measurement,
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
    model: EquityFXMeasurementModelV12L1OnlineFiltering,
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
    model: EquityFXMeasurementModelV12L1OnlineFiltering,
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
    base_cov_factor = _base_cov_factor_step(
        structural=context.structural,
        batch=context.batch,
        regime=regime,
    )
    index_mix = _sample_mix(
        num_samples=regime.shape[0],
        df_value=model.priors.index_t_copula.df,
        device=context.batch.X_asset.device,
        dtype=context.batch.X_asset.dtype,
    )
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=base_cov_factor,
        cov_diag=base_cov_diag,
        assets=context.batch.assets,
        mix=index_mix,
        eps=model.priors.index_t_copula.eps,
    )
    measurement_block = build_equity_fx_measurement_factor_block(
        structure=structure,
        state=EquityFXMeasurementFactorState(
            state_scale=context.measurement_params.state_scale,
            state_corr_cholesky=context.measurement_params.state_corr_cholesky,
            loading_delta=context.measurement_params.loading_delta,
            local_regime_scale=torch.exp(0.5 * regime[:, 2]),
            translation_regime_scale=torch.exp(0.5 * regime[:, 1]),
            mix=_sample_mix(
                num_samples=regime.shape[0],
                df_value=model.priors.equity_fx_measurement.state_df,
                device=context.batch.X_asset.device,
                dtype=context.batch.X_asset.dtype,
            ),
            eps=model.priors.equity_fx_measurement.eps,
        ),
    )
    cov_diag = apply_equity_fx_measurement_residual_scale(
        cov_diag=scaled_diag,
        residual_scale=context.measurement_params.residual_scale,
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([scaled_factor, measurement_block], dim=-1),
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
    "build_equity_fx_measurement_predict_v12_l1_online_filtering",
    "predict_equity_fx_measurement_v12_l1",
]
