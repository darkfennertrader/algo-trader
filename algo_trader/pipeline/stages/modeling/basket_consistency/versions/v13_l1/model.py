from __future__ import annotations
# pylint: disable=duplicate-code

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model import (
    _build_index_t_copula_config,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.predict import (
    predict_dependence_layer_v4_l1,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.shared import (
    IndexTCopulaOverlayConfig,
    apply_index_t_copula_overlay,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _cov_factor_path as _base_cov_factor_path,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroGuide,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import (
    sample_time_observations,
)

from .defaults import merge_nested_params, model_default_params_v13_l1
from .guide import BasketConsistencyGuideV13L1OnlineFiltering
from .shared import (
    BasketConsistencyConfig,
    BasketConsistencyCoordinates,
    BasketObservationGroup,
    BasketConsistencyPosteriorMeans,
    basket_scale_from_covariance,
    build_basket_observation_groups,
    build_basket_consistency_config,
    build_basket_consistency_coordinates,
    build_basket_consistency_transform,
    project_basket_covariance,
    project_basket_mean,
    whiten_basket_covariance,
    whiten_basket_observations,
)


@dataclass(frozen=True)
class V13L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexTCopulaOverlayConfig = field(
        default_factory=IndexTCopulaOverlayConfig
    )
    basket_consistency: BasketConsistencyConfig = field(
        default_factory=BasketConsistencyConfig
    )


@dataclass(frozen=True)
class _BasketObservationInputs:
    batch: Any
    loc: torch.Tensor
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    overlay: BasketConsistencyConfig
    coordinates: BasketConsistencyCoordinates
    basket_params: BasketConsistencyPosteriorMeans


@dataclass(frozen=True)
class _NamedTimeSite:
    name: str
    obs_dist: dist.TorchDistribution
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None


@dataclass(frozen=True)
class _GroupedBasketInputs:
    basket_df: float
    basket_obs: torch.Tensor
    basket_mean: torch.Tensor
    basket_scale: torch.Tensor


@dataclass
class BasketConsistencyModelV13L1OnlineFiltering(PyroModel):
    priors: V13L1ModelPriors = field(default_factory=V13L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mix = _sample_index_t_copula_mix(context, self.priors.index_t_copula)
        coordinates = build_basket_consistency_coordinates(
            assets=context.batch.assets,
            device=context.device,
            dtype=context.dtype,
        )
        basket_params = _sample_basket_consistency_sites(
            count=coordinates.basket_count,
            overlay=self.priors.basket_consistency,
            device=context.device,
            dtype=context.dtype,
        )
        loc, cov_factor, cov_diag, obs_dist = _build_raw_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            mix=mix,
            overlay=self.priors.index_t_copula,
        )
        sample_time_observations(
            time_count=context.shape.T,
            obs_dist=obs_dist,
            y_obs=runtime_batch.observations.y_obs,
            time_mask=runtime_batch.observations.time_mask,
            obs_scale=runtime_batch.observations.obs_scale,
        )
        _sample_basket_observations(
            inputs=_BasketObservationInputs(
                batch=context.batch,
                loc=loc,
                cov_factor=cov_factor,
                cov_diag=cov_diag,
                overlay=self.priors.basket_consistency,
                coordinates=coordinates,
                basket_params=basket_params,
            ),
        )

    def posterior_predict(
        self,
        *,
        guide: PyroGuide,
        batch: ModelBatch,
        num_samples: int,
        state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        summaries = getattr(guide, "structural_predictive_summaries", None)
        if not callable(summaries):
            summaries = getattr(guide, "structural_posterior_means", None)
        if not callable(summaries):
            return None
        return predict_basket_consistency_v13_l1(
            model=self,
            guide=cast(BasketConsistencyGuideV13L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("basket_consistency_model_v13_l1_online_filtering")
def build_basket_consistency_model_v13_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v13_l1(), params)
    return BasketConsistencyModelV13L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V13L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V13L1ModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "index_t_copula",
        "basket_consistency",
    }
    if extra:
        raise ConfigError(
            "Unknown basket_consistency_model_v13_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V13L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
        basket_consistency=build_basket_consistency_config(
            values.get("basket_consistency")
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_t_copula_mix(
    context: Any,
    overlay: IndexTCopulaOverlayConfig,
) -> torch.Tensor:
    if not overlay.enabled:
        return torch.ones((context.shape.T,), device=context.device, dtype=context.dtype)
    concentration = torch.full(
        (context.shape.T,),
        overlay.df / 2.0,
        device=context.device,
        dtype=context.dtype,
    )
    return pyro.sample(
        "index_t_copula_mix",
        dist.Gamma(concentration, concentration).to_event(1),
    )


def _sample_basket_consistency_sites(
    *,
    count: int,
    overlay: BasketConsistencyConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> BasketConsistencyPosteriorMeans:
    if not overlay.enabled or count == 0:
        return BasketConsistencyPosteriorMeans(
            basket_scale=torch.ones((count,), device=device, dtype=dtype),
        )
    basket_scale = pyro.sample(
        "basket_consistency_scale",
        dist.LogNormal(
            torch.full(
                (count,),
                float(overlay.prior_scales.scale_center),
                device=device,
                dtype=dtype,
            ).log(),
            torch.full(
                (count,),
                float(overlay.prior_scales.scale_log_scale),
                device=device,
                dtype=dtype,
            ),
        ).to_event(1),
    )
    return BasketConsistencyPosteriorMeans(basket_scale=basket_scale)


def _build_raw_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    mix: torch.Tensor,
    overlay: IndexTCopulaOverlayConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dist.LowRankMultivariateNormal]:
    loc = _base_mean_path(context, structural)
    cov_factor = _base_cov_factor_path(context, structural, regime_path)
    base_cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    if not overlay.enabled:
        obs_dist = dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
        return loc, cov_factor, base_cov_diag.unsqueeze(0).expand_as(loc), obs_dist
    scaled_factor, scaled_diag = apply_index_t_copula_overlay(
        cov_factor=cov_factor,
        cov_diag=base_cov_diag,
        assets=context.batch.assets,
        mix=mix,
        eps=overlay.eps,
    )
    obs_dist = dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=scaled_factor,
        cov_diag=scaled_diag,
    )
    return loc, scaled_factor, scaled_diag, obs_dist


def _sample_basket_observations(
    *,
    inputs: _BasketObservationInputs,
) -> None:
    if not inputs.overlay.enabled or inputs.coordinates.basket_count == 0:
        return
    y_obs = inputs.batch.observations.y_obs
    if y_obs is None:
        return
    basket_obs = project_basket_mean(loc=y_obs, basis=inputs.coordinates.basis)
    transform = build_basket_consistency_transform(
        observations=basket_obs,
        config=inputs.overlay,
    )
    basket_mean = project_basket_mean(
        loc=inputs.loc,
        basis=inputs.coordinates.basis,
    )
    basket_cov = project_basket_covariance(
        cov_factor=inputs.cov_factor,
        cov_diag=inputs.cov_diag,
        basis=inputs.coordinates.basis,
        eps=inputs.overlay.eps,
    )
    whitened_mean = whiten_basket_observations(
        values=basket_mean,
        transform=transform,
    )
    whitened_obs = whiten_basket_observations(
        values=basket_obs,
        transform=transform,
    )
    whitened_cov = whiten_basket_covariance(
        covariance=basket_cov,
        transform=transform,
    )
    basket_scale = basket_scale_from_covariance(
        covariance=whitened_cov,
        basket_scale=inputs.basket_params.basket_scale,
        eps=inputs.overlay.eps,
    )
    for group in build_basket_observation_groups(
        config=inputs.overlay,
        basket_names=inputs.coordinates.basket_names,
        device=whitened_mean.device,
    ):
        _sample_named_time_site(
            time_count=int(whitened_mean.shape[0]),
            site=_build_group_time_site(
                grouped_inputs=_GroupedBasketInputs(
                    basket_df=inputs.overlay.df,
                    basket_obs=whitened_obs,
                    basket_mean=whitened_mean,
                    basket_scale=basket_scale,
                ),
                group=group,
                time_mask=inputs.batch.observations.time_mask,
                base_obs_scale=inputs.batch.observations.obs_scale,
            ),
        )


def _basket_obs_scale(
    *,
    base_obs_scale: float | None,
    obs_weight: float,
) -> float | None:
    if base_obs_scale is None:
        return obs_weight
    return float(base_obs_scale) * obs_weight


def _build_group_time_site(
    *,
    grouped_inputs: _GroupedBasketInputs,
    group: BasketObservationGroup,
    time_mask: torch.BoolTensor | None,
    base_obs_scale: float | None,
) -> _NamedTimeSite:
    return _NamedTimeSite(
        name=group.name,
        obs_dist=_build_group_distribution(
            basket_df=grouped_inputs.basket_df,
            basket_mean=grouped_inputs.basket_mean,
            basket_scale=grouped_inputs.basket_scale,
            mask=group.mask,
        ),
        y_obs=grouped_inputs.basket_obs[:, group.mask],
        time_mask=time_mask,
        obs_scale=_basket_obs_scale(
            base_obs_scale=base_obs_scale,
            obs_weight=group.obs_weight,
        ),
    )


def _build_group_distribution(
    *,
    basket_df: float,
    basket_mean: torch.Tensor,
    basket_scale: torch.Tensor,
    mask: torch.BoolTensor,
) -> dist.TorchDistribution:
    return dist.StudentT(
        df=basket_df,
        loc=basket_mean[:, mask],
        scale=basket_scale[:, mask],
    ).to_event(1)


def _sample_named_time_site(
    *,
    time_count: int,
    site: _NamedTimeSite,
) -> None:
    with pyro.plate(f"{site.name}_time", time_count, dim=-1):
        with _optional_mask(site.time_mask):
            with _optional_scale(site.obs_scale):
                pyro.sample(site.name, site.obs_dist, obs=site.y_obs)


@contextmanager
def _optional_mask(mask: torch.BoolTensor | None) -> Iterator[None]:
    if mask is None:
        yield
        return
    with _managed_context(poutine.mask(mask=mask)):
        yield


@contextmanager
def _optional_scale(scale: float | None) -> Iterator[None]:
    if scale is None:
        yield
        return
    with _managed_context(poutine.scale(scale=float(scale))):
        yield


@contextmanager
def _managed_context(handler_obj: object) -> Iterator[None]:
    enter = getattr(handler_obj, "__enter__", None)
    exit_handler = getattr(handler_obj, "__exit__", None)
    if not callable(enter) or not callable(exit_handler):
        raise ConfigError("Invalid Pyro context manager")
    enter()
    try:
        yield
    finally:
        exit_handler(None, None, None)


def predict_basket_consistency_v13_l1(
    *,
    model: BasketConsistencyModelV13L1OnlineFiltering,
    guide: BasketConsistencyGuideV13L1OnlineFiltering,
    batch: ModelBatch,
    num_samples: int,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    return predict_dependence_layer_v4_l1(
        model=cast(Any, model),
        guide=cast(Any, guide),
        batch=batch,
        num_samples=num_samples,
        state=state,
    )


__all__ = [
    "BasketConsistencyModelV13L1OnlineFiltering",
    "V13L1ModelPriors",
    "build_basket_consistency_model_v13_l1_online_filtering",
    "predict_basket_consistency_v13_l1",
]
