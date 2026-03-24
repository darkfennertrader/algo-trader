from __future__ import annotations

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

from .guide_v2_l6 import (
    FXCurrencyFactorGuideV2L6OnlineFiltering,
    build_v2_l6_runtime_batch,
)
from .shared_v2_l6 import (
    CurrencyPosteriorMetadata,
    FilteringState,
    RegimePosteriorMeans,
    StructuralPosteriorMeans,
    StructuralTensorMeans,
    V2L6RuntimeBatch,
    coerce_two_state_tensor,
)

if TYPE_CHECKING:
    from .model_v2_l6 import FXCurrencyFactorModelV2L6OnlineFiltering

class _V2L6Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_fx_currency_factor_v2_l6(
            model=cast(Any, request.model),
            guide=cast(FXCurrencyFactorGuideV2L6OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def predict_fx_currency_factor_v2_l6(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    guide: FXCurrencyFactorGuideV2L6OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor] | None:
    structural = _resolve_structural_means(state=state, guide=guide)
    runtime_batch = build_v2_l6_runtime_batch(
        batch,
        currency_names=structural.currency_names,
        anchor_currency=structural.anchor_currency,
    )
    if runtime_batch.filtering_state is None:
        return None
    samples = _predictive_samples(
        model=model,
        structural=structural,
        runtime_batch=runtime_batch,
        num_samples=num_samples,
    )
    return _prediction_outputs(samples)


@register_predictor("fx_currency_factor_predict_v2_l6_online_filtering")
def build_fx_currency_factor_predict_v2_l6_online_filtering(
    params: Mapping[str, Any],
) -> _V2L6Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown fx_currency_factor_predict_v2_l6_online_filtering params: "
            f"{unknown}"
        )
    return _V2L6Predictor()


def _resolve_structural_means(
    *,
    state: Mapping[str, Any] | None,
    guide: FXCurrencyFactorGuideV2L6OnlineFiltering,
) -> StructuralPosteriorMeans:
    cached = _state_structural_means(state)
    if cached is not None:
        return cached
    return _guide_structural_means(guide)


def _guide_structural_means(
    guide: FXCurrencyFactorGuideV2L6OnlineFiltering,
) -> StructuralPosteriorMeans:
    predictive_summaries = getattr(guide, "structural_predictive_summaries", None)
    if not callable(predictive_summaries):
        return guide.structural_posterior_means()
    return cast(StructuralPosteriorMeans, predictive_summaries())


def _predictive_samples(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    structural: StructuralPosteriorMeans,
    runtime_batch: V2L6RuntimeBatch,
    num_samples: int,
) -> torch.Tensor:
    return _rollout_samples(
        model=model,
        structural=structural,
        batch=runtime_batch,
        num_samples=num_samples,
    )


def _rollout_samples(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: V2L6RuntimeBatch,
    num_samples: int,
) -> torch.Tensor:
    if batch.filtering_state is None:
        raise ValueError("V2 L6 rollout requires filtering_state")
    filtering_state, structural = _prepare_rollout_inputs(
        batch=batch,
        filtering_state=batch.filtering_state,
        structural=structural,
    )
    regime, regime_var = _initial_rollout_state(
        filtering_state=filtering_state,
        num_samples=num_samples,
    )
    draws = []
    for time_index in range(int(batch.X_asset.shape[0])):
        regime_var = _forecast_regime_variance(
            model=model,
            regime_var=regime_var,
            structural=structural,
        )
        regime = _sample_next_regime(
            model=model,
            regime=regime,
            structural=structural,
        )
        draws.append(
            _sample_observation_step(
                model=model,
                structural=structural,
                batch=batch,
                regime_state=(regime, regime_var),
                time_index=time_index,
            )
        )
    return torch.stack(draws, dim=1)


def _prediction_outputs(samples: torch.Tensor) -> Mapping[str, torch.Tensor]:
    return {
        "samples": samples,
        "mean": samples.mean(dim=0),
        "covariance": predictive_covariance(samples),
    }


def _state_structural_means(
    state: Mapping[str, Any] | None,
) -> StructuralPosteriorMeans | None:
    if state is None:
        return None
    payload = state.get("structural_posterior_means")
    if not isinstance(payload, Mapping):
        return None
    return StructuralPosteriorMeans.from_mapping(payload)


def _prepare_rollout_inputs(
    *,
    batch: V2L6RuntimeBatch,
    filtering_state: FilteringState,
    structural: StructuralPosteriorMeans,
) -> tuple[FilteringState, StructuralPosteriorMeans]:
    moved_filtering_state = _move_filtering_state(
        batch=batch, filtering_state=filtering_state
    )
    moved_structural = _move_structural_means(
        structural=structural,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
    )
    return moved_filtering_state, moved_structural


def _initial_rollout_state(
    *, filtering_state: FilteringState, num_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _initial_regime_samples(
            filtering_state=filtering_state, num_samples=num_samples
        ),
        _initial_regime_variance(filtering_state=filtering_state),
    )


def _move_filtering_state(
    *, batch: V2L6RuntimeBatch, filtering_state: FilteringState
) -> FilteringState:
    return FilteringState(
        h_loc=coerce_two_state_tensor(
            filtering_state.h_loc,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
        ),
        h_scale=coerce_two_state_tensor(
            filtering_state.h_scale,
            device=batch.X_asset.device,
            dtype=batch.X_asset.dtype,
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
        tensors=StructuralTensorMeans(
            alpha=structural.alpha.to(device=device, dtype=dtype),
            sigma_idio=structural.sigma_idio.to(device=device, dtype=dtype),
            w=structural.w.to(device=device, dtype=dtype),
            gamma_currency=structural.gamma_currency.to(
                device=device, dtype=dtype
            ),
            B_currency_broad=structural.B_currency_broad.to(
                device=device, dtype=dtype
            ),
            B_currency_cross=structural.B_currency_cross.to(
                device=device, dtype=dtype
            ),
        ),
        regime=RegimePosteriorMeans(
            s_u_broad_mean=structural.s_u_broad_mean.to(
                device=device, dtype=dtype
            ),
            s_u_cross_mean=structural.s_u_cross_mean.to(
                device=device, dtype=dtype
            ),
        ),
        metadata=CurrencyPosteriorMetadata(
            currency_names=structural.currency_names,
            anchor_currency=structural.anchor_currency,
        ),
    )


def _initial_regime_samples(
    *, filtering_state: FilteringState, num_samples: int
) -> torch.Tensor:
    return dist.Normal(
        filtering_state.h_loc,
        filtering_state.h_scale,
    ).rsample((num_samples,))


def _initial_regime_variance(*, filtering_state: FilteringState) -> torch.Tensor:
    return filtering_state.h_scale.pow(2)


def _forecast_regime_variance(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    regime_var: torch.Tensor,
    structural: StructuralPosteriorMeans,
) -> torch.Tensor:
    phi = _phi_vector(model=model, device=regime_var.device, dtype=regime_var.dtype)
    s_u_mean = _s_u_mean_vector(
        structural=structural, device=regime_var.device, dtype=regime_var.dtype
    )
    return phi.pow(2).unsqueeze(0) * regime_var + s_u_mean.pow(2).unsqueeze(0)


def _sample_next_regime(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    regime: torch.Tensor,
    structural: StructuralPosteriorMeans,
) -> torch.Tensor:
    phi = _phi_vector(model=model, device=regime.device, dtype=regime.dtype)
    s_u_mean = _s_u_mean_vector(
        structural=structural, device=regime.device, dtype=regime.dtype
    )
    return dist.Normal(phi.unsqueeze(0) * regime, s_u_mean.unsqueeze(0)).rsample()


def _sample_observation_step(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    structural: StructuralPosteriorMeans,
    batch: V2L6RuntimeBatch,
    regime_state: tuple[torch.Tensor, torch.Tensor],
    time_index: int,
) -> torch.Tensor:
    regime, regime_var = regime_state
    mu_t = _mean_step(
        structural=structural,
        batch=batch,
        time_index=time_index,
    )
    total_scale = _sample_total_scale(
        model=model,
        regime=regime,
        regime_var=regime_var.to(
            device=mu_t.device, dtype=mu_t.dtype
        ),
    )
    cov_factor = _covariance_factor(
        structural=structural,
        batch=batch,
        total_scale=total_scale,
    )
    expanded_mu = _expanded_mean(mu_t=mu_t, sample_count=int(regime.shape[0]))
    predictive_dist = dist.LowRankMultivariateNormal(
        loc=expanded_mu,
        cov_factor=cov_factor,
        cov_diag=structural.sigma_idio.pow(2).unsqueeze(0).expand_as(expanded_mu),
    )
    return predictive_dist.rsample()


def _covariance_factor(
    *,
    structural: StructuralPosteriorMeans,
    batch: V2L6RuntimeBatch,
    total_scale: torch.Tensor,
) -> torch.Tensor:
    pair_factor_broad = batch.exposure_matrix @ structural.B_currency_broad
    pair_factor_cross = batch.exposure_matrix @ structural.B_currency_cross
    broad_factor = pair_factor_broad.unsqueeze(0) * torch.rsqrt(
        total_scale[:, 0]
    ).unsqueeze(-1).unsqueeze(-1)
    cross_factor = pair_factor_cross.unsqueeze(0) * torch.rsqrt(
        total_scale[:, 1]
    ).unsqueeze(-1).unsqueeze(-1)
    return torch.cat([broad_factor, cross_factor], dim=-1)


def _expanded_mean(*, mu_t: torch.Tensor, sample_count: int) -> torch.Tensor:
    return mu_t.unsqueeze(0).expand(sample_count, -1)


def _mean_step(
    *,
    structural: StructuralPosteriorMeans,
    batch: V2L6RuntimeBatch,
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
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    regime: torch.Tensor,
    regime_var: torch.Tensor,
) -> torch.Tensor:
    nu = torch.tensor(
        [model.priors.regime.broad.nu, model.priors.regime.cross.nu],
        device=regime.device,
        dtype=regime.dtype,
    )
    v_t = dist.Gamma(nu / 2.0, nu / 2.0).rsample((int(regime.shape[0]),))
    u_scalar = torch.exp(regime - 0.5 * regime_var) * v_t
    return u_scalar


def _phi_vector(
    *,
    model: FXCurrencyFactorModelV2L6OnlineFiltering,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [model.priors.regime.broad.phi, model.priors.regime.cross.phi],
        device=device,
        dtype=dtype,
    )


def _s_u_mean_vector(
    *,
    structural: StructuralPosteriorMeans,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [structural.s_u_broad_mean, structural.s_u_cross_mean]
    ).to(device=device, dtype=dtype)
