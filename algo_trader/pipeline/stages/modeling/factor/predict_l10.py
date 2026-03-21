from __future__ import annotations

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

from .guide_l10 import (
    FactorGuideL10OnlineFiltering,
    FilteringState,
    Level10RuntimeBatch,
    StructuralPosteriorMeans,
    build_level10_runtime_batch,
)

if TYPE_CHECKING:
    from .model_l10 import FactorModelL10OnlineFiltering


@dataclass(frozen=True)
class Level10PredictiveOutputs:
    samples: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class _PredictiveRegimeState:
    regime: torch.Tensor
    regime_var: torch.Tensor


class _Level10Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_factor_l10(
            model=cast(Any, request.model),
            guide=cast(FactorGuideL10OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_factor_l10(
    *,
    model: FactorModelL10OnlineFiltering,
    guide: FactorGuideL10OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    runtime_batch = build_level10_runtime_batch(batch)
    if runtime_batch.filtering_state is None:
        return None
    structural = _resolve_structural_means(state=state, guide=guide)
    samples = _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=num_samples,
    )
    outputs = Level10PredictiveOutputs(
        samples=samples,
        mean=samples.mean(dim=0),
        covariance=predictive_covariance(samples),
    )
    return {
        "samples": outputs.samples,
        "mean": outputs.mean,
        "covariance": outputs.covariance,
    }


@register_predictor("factor_predict_l10_online_filtering")
def build_factor_predict_l10_online_filtering(
    params: Mapping[str, Any],
) -> _Level10Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown factor_predict_l10_online_filtering params: "
            f"{unknown}"
        )
    return _Level10Predictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: FactorGuideL10OnlineFiltering,
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
    model: FactorModelL10OnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: Level10RuntimeBatch,
    num_samples: int,
) -> torch.Tensor:
    if batch.filtering_state is None:
        raise ValueError("Level 10 rollout requires filtering_state")
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
    for t in range(int(batch.X_asset.shape[0])):
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
                X_asset_t=batch.X_asset[t],
                X_global_t=batch.X_global[t],
                regime_state=regime_state,
            )
        )
    return torch.stack(draws, dim=1)


def _move_filtering_state(
    *, batch: Level10RuntimeBatch, filtering_state: FilteringState
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
        beta=structural.beta.to(device=device, dtype=dtype),
        B=structural.B.to(device=device, dtype=dtype),
        s_u_mean=structural.s_u_mean.to(device=device, dtype=dtype),
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
    model: FactorModelL10OnlineFiltering,
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
    model: FactorModelL10OnlineFiltering,
    regime: torch.Tensor,
    structural: StructuralPosteriorMeans,
) -> torch.Tensor:
    return dist.Normal(
        model.priors.regime.phi * regime,
        structural.s_u_mean,
    ).rsample()


def _sample_observation_step(
    *,
    model: FactorModelL10OnlineFiltering,
    structural: StructuralPosteriorMeans,
    X_asset_t: torch.Tensor,
    X_global_t: torch.Tensor,
    regime_state: _PredictiveRegimeState,
) -> torch.Tensor:
    mu_t = _mean_step(
        structural=structural,
        X_asset_t=X_asset_t,
        X_global_t=X_global_t,
    )
    u_t = _sample_total_scale(
        model=model,
        regime=regime_state.regime,
        regime_var=regime_state.regime_var.to(
            device=mu_t.device, dtype=mu_t.dtype
        ),
        dtype=mu_t.dtype,
        device=mu_t.device,
    )
    predictive_dist = dist.LowRankMultivariateNormal(
        loc=mu_t.unsqueeze(0).expand(int(regime_state.regime.shape[0]), -1),
        cov_factor=structural.B.unsqueeze(0) / torch.sqrt(u_t).view(-1, 1, 1),
        cov_diag=structural.sigma_idio.pow(2).unsqueeze(0) / u_t.view(-1, 1),
    )
    return predictive_dist.rsample()


def _mean_step(
    *,
    structural: StructuralPosteriorMeans,
    X_asset_t: torch.Tensor,
    X_global_t: torch.Tensor,
) -> torch.Tensor:
    mu_asset = (X_asset_t * structural.w).sum(dim=-1)
    mu_global = X_global_t @ structural.beta.transpose(0, 1)
    return structural.alpha + mu_asset + mu_global


def _sample_total_scale(
    *,
    model: FactorModelL10OnlineFiltering,
    regime: torch.Tensor,
    regime_var: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nu = torch.tensor(model.priors.regime.nu, device=device, dtype=dtype)
    v_t = dist.Gamma(nu / 2.0, nu / 2.0).rsample((int(regime.shape[0]),))
    s_regime = torch.exp(regime - 0.5 * regime_var)
    return s_regime * v_t
