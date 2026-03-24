from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
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

from .guide_v2_l2 import (
    FilteringState,
    FXCurrencyFactorGuideV2L2OnlineFiltering,
    StructuralPosteriorMeans,
    V2L2RuntimeBatch,
    build_v2_l2_runtime_batch,
)

if TYPE_CHECKING:
    from .model_v2_l2 import FXCurrencyFactorModelV2L2OnlineFiltering


@dataclass(frozen=True)
class V2L2PredictiveOutputs:
    samples: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class _PredictiveRegimeState:
    regime: torch.Tensor
    regime_var: torch.Tensor


class _V2L2Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_fx_currency_factor_v2_l2(
            model=cast(Any, request.model),
            guide=cast(FXCurrencyFactorGuideV2L2OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_fx_currency_factor_v2_l2(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    guide: FXCurrencyFactorGuideV2L2OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=guide)
    runtime_batch = build_v2_l2_runtime_batch(
        batch,
        currency_names=structural.currency_names,
        anchor_currency=structural.anchor_currency,
    )
    if runtime_batch.filtering_state is None:
        return None
    samples = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=num_samples,
    )
    outputs = V2L2PredictiveOutputs(
        samples=samples,
        mean=samples.mean(dim=0),
        covariance=predictive_covariance(samples),
    )
    return {
        "samples": outputs.samples,
        "mean": outputs.mean,
        "covariance": outputs.covariance,
    }


@register_predictor("fx_currency_factor_predict_v2_l2_online_filtering")
def build_fx_currency_factor_predict_v2_l2_online_filtering(
    params: Mapping[str, Any],
) -> _V2L2Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown fx_currency_factor_predict_v2_l2_online_filtering params: "
            f"{unknown}"
        )
    return _V2L2Predictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: FXCurrencyFactorGuideV2L2OnlineFiltering,
) -> StructuralPosteriorMeans:
    if state is not None:
        payload = state.get("structural_posterior_means")
        if isinstance(payload, Mapping):
            return StructuralPosteriorMeans.from_mapping(payload)
    predictive_summaries = getattr(guide, "structural_predictive_summaries", None)
    if callable(predictive_summaries):
        return cast(StructuralPosteriorMeans, predictive_summaries())
    return guide.structural_posterior_means()


def _rollout_samples(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: V2L2RuntimeBatch,
    num_samples: int,
) -> torch.Tensor:
    if batch.filtering_state is None:
        raise ValueError("V2 L2 rollout requires filtering_state")
    filtering_state = _move_filtering_state(
        batch=batch, filtering_state=batch.filtering_state
    )
    structural = _move_structural_means(
        structural=structural,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    regime_state = _PredictiveRegimeState(
        regime=_initial_regime_samples(
            filtering_state=filtering_state, num_samples=num_samples
        ),
        regime_var=_initial_regime_variance(filtering_state=filtering_state),
    )
    draws = []
    for time_index in range(int(batch.X_asset.shape[0])):
        next_regime_var = _forecast_regime_variance(
            model=model,
            regime_var=regime_state.regime_var,
            structural=structural,
        )
        next_regime = _sample_next_regime(
            model=model,
            regime=regime_state.regime,
            structural=structural,
        )
        regime_state = _PredictiveRegimeState(
            regime=next_regime,
            regime_var=next_regime_var,
        )
        draws.append(
            _sample_observation_step(
                model=model,
                structural=structural,
                batch=batch,
                regime_state=regime_state,
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _move_filtering_state(
    *, batch: V2L2RuntimeBatch, filtering_state: FilteringState
) -> FilteringState:
    return FilteringState(
        h_loc=filtering_state.h_loc.to(
            device=batch.X_asset.device, dtype=batch.X_asset.dtype
        ),
        h_scale=filtering_state.h_scale.to(
            device=batch.X_asset.device, dtype=batch.X_asset.dtype
        ),
        steps_seen=int(filtering_state.steps_seen),
    )


def _move_structural_means(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> StructuralPosteriorMeans:
    return StructuralPosteriorMeans(
        alpha=structural.alpha.to(device=device, dtype=dtype),
        sigma_idio=structural.sigma_idio.to(device=device, dtype=dtype),
        w=structural.w.to(device=device, dtype=dtype),
        gamma_currency=structural.gamma_currency.to(device=device, dtype=dtype),
        B_currency=structural.B_currency.to(device=device, dtype=dtype),
        s_u_mean=structural.s_u_mean.to(device=device, dtype=dtype),
        currency_names=structural.currency_names,
        anchor_currency=structural.anchor_currency,
    )


def _initial_regime_samples(
    *, filtering_state: FilteringState, num_samples: int
) -> torch.Tensor:
    return dist.Normal(
        filtering_state.h_loc, filtering_state.h_scale
    ).rsample((num_samples,))


def _initial_regime_variance(*, filtering_state: FilteringState) -> torch.Tensor:
    return filtering_state.h_scale.pow(2)


def _forecast_regime_variance(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    regime_var: torch.Tensor,
    structural: StructuralPosteriorMeans,
) -> torch.Tensor:
    phi = torch.tensor(
        model.priors.regime.phi,
        device=regime_var.device,
        dtype=regime_var.dtype,
    )
    return phi.pow(2) * regime_var + structural.s_u_mean.pow(2)


def _sample_next_regime(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    regime: torch.Tensor,
    structural: StructuralPosteriorMeans,
) -> torch.Tensor:
    return dist.Normal(
        model.priors.regime.phi * regime,
        structural.s_u_mean,
    ).rsample()


def _sample_observation_step(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: V2L2RuntimeBatch,
    regime_state: _PredictiveRegimeState,
    time_index: int,
) -> torch.Tensor:
    mu_t = _mean_step(
        structural=structural,
        batch=batch,
        time_index=time_index,
    )
    u_t = _sample_total_scale(
        model=model,
        batch=batch,
        regime=regime_state.regime,
        regime_var=regime_state.regime_var.to(
            device=mu_t.device, dtype=mu_t.dtype
        ),
    )
    pair_factor = batch.exposure_matrix @ structural.B_currency
    predictive_dist = dist.LowRankMultivariateNormal(
        loc=mu_t.unsqueeze(0).expand(int(regime_state.regime.shape[0]), -1),
        cov_factor=pair_factor.unsqueeze(0) * torch.rsqrt(u_t).unsqueeze(-1),
        cov_diag=structural.sigma_idio.pow(2).unsqueeze(0).expand_as(u_t),
    )
    return predictive_dist.rsample()


def _mean_step(
    *,
    structural: StructuralPosteriorMeans,
    batch: V2L2RuntimeBatch,
    time_index: int,
) -> torch.Tensor:
    X_asset_t = batch.X_asset[time_index]
    X_global_t = batch.X_global[time_index]
    mu_asset = (X_asset_t * structural.w).sum(dim=-1)
    currency_macro = X_global_t @ structural.gamma_currency.transpose(0, 1)
    mu_global = batch.exposure_matrix @ currency_macro
    return structural.alpha + mu_asset + mu_global


def _sample_total_scale(
    *,
    model: FXCurrencyFactorModelV2L2OnlineFiltering,
    batch: V2L2RuntimeBatch,
    regime: torch.Tensor,
    regime_var: torch.Tensor,
) -> torch.Tensor:
    nu = torch.tensor(
        model.priors.regime.nu,
        device=regime.device,
        dtype=regime.dtype,
    )
    v_t = dist.Gamma(nu / 2.0, nu / 2.0).rsample((int(regime.shape[0]),))
    u_scalar = torch.exp(regime - 0.5 * regime_var) * v_t
    return u_scalar.unsqueeze(-1).expand(-1, int(batch.X_asset.shape[1]))
