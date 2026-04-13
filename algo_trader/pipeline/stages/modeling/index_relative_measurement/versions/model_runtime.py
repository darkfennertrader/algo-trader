from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    _build_raw_observation_distribution,
)
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.shared import (
    BasketObservationGroup,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model import (
    _build_index_t_copula_config,
    _sample_index_t_copula_mix,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.predict import (
    predict_dependence_layer_v4_l1,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.shared import (
    IndexTCopulaOverlayConfig,
)
from algo_trader.pipeline.stages.modeling.config_support import coerce_mapping
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from algo_trader.pipeline.stages.modeling.protocols import (
    ModelBatch,
    PyroGuide,
    PyroModel,
)
from algo_trader.pipeline.stages.modeling.runtime_support import (
    sample_time_observations,
    supports_structural_predictive_summaries,
)

from .shared_common import (
    IndexRelativeMeasurementConfig,
    IndexRelativeMeasurementCoordinates,
    build_index_coordinate_transform,
    coordinate_scale_from_covariance,
    project_basket_covariance,
    project_basket_mean,
    standardize_index_coordinates,
    standardize_index_covariance,
)

CoordinateBuilder = Callable[..., IndexRelativeMeasurementCoordinates]
GroupBuilder = Callable[..., tuple[BasketObservationGroup, ...]]
ConfigBuilder = Callable[[object], IndexRelativeMeasurementConfig]


@dataclass(frozen=True)
class IndexRelativeMeasurementModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexTCopulaOverlayConfig = field(
        default_factory=IndexTCopulaOverlayConfig
    )
    index_relative_measurement: IndexRelativeMeasurementConfig = field(
        default_factory=IndexRelativeMeasurementConfig
    )


@dataclass(frozen=True)
class _IndexRelativeObservationInputs:
    batch: Any
    loc: torch.Tensor
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    overlay: IndexRelativeMeasurementConfig
    coordinates: IndexRelativeMeasurementCoordinates


@dataclass(frozen=True)
class _NamedTimeSite:
    name: str
    obs_dist: dist.TorchDistribution
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None


@dataclass(frozen=True)
class _GroupTimeSiteInputs:
    base_obs_scale: float | None
    time_mask: torch.BoolTensor | None
    coordinate_mean: torch.Tensor
    coordinate_obs: torch.Tensor
    coordinate_scale: torch.Tensor
    df: float


@dataclass(frozen=True)
class _RuntimeObservationInputs:
    runtime_batch: Any
    batch: Any
    time_count: int
    device: torch.device
    coordinates: IndexRelativeMeasurementCoordinates
    group_builder: GroupBuilder
    raw_state: "_RuntimeObservationState"


@dataclass(frozen=True)
class _RuntimeObservationState:
    loc: torch.Tensor
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    full_obs_dist: dist.TorchDistribution


@dataclass
class IndexRelativeMeasurementModelRuntime(PyroModel):
    priors: IndexRelativeMeasurementModelPriors = field(
        default_factory=IndexRelativeMeasurementModelPriors
    )
    coordinate_builder: CoordinateBuilder | None = None
    group_builder: GroupBuilder | None = None

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        if self.coordinate_builder is None or self.group_builder is None:
            raise ConfigError("Index relative measurement runtime is incomplete")
        coordinate_builder = self.coordinate_builder
        group_builder = self.group_builder
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mix = _sample_index_t_copula_mix(context, self.priors.index_t_copula)
        loc, cov_factor, cov_diag, full_obs_dist = _build_raw_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            mix=mix,
            overlay=self.priors.index_t_copula,
        )
        coordinates = coordinate_builder(
            assets=context.batch.assets,
            device=context.device,
            dtype=context.dtype,
        )
        self._sample_observations(
            inputs=_RuntimeObservationInputs(
                runtime_batch=runtime_batch,
                batch=context.batch,
                time_count=context.shape.T,
                device=context.device,
                coordinates=coordinates,
                group_builder=group_builder,
                raw_state=_RuntimeObservationState(
                    loc=loc,
                    cov_factor=cov_factor,
                    cov_diag=cov_diag,
                    full_obs_dist=full_obs_dist,
                ),
            ),
        )

    def _sample_observations(
        self,
        *,
        inputs: _RuntimeObservationInputs,
    ) -> None:
        if (
            not self.priors.index_relative_measurement.enabled
            or inputs.coordinates.coordinate_count == 0
        ):
            self._sample_full_raw_head(inputs)
            return
        sample_non_index_raw_observations(
            batch=inputs.batch,
            loc=inputs.raw_state.loc,
            cov_factor=inputs.raw_state.cov_factor,
            cov_diag=inputs.raw_state.cov_diag,
        )
        self._sample_auxiliary_observations(inputs)

    def _sample_full_raw_head(self, inputs: _RuntimeObservationInputs) -> None:
        sample_time_observations(
            time_count=inputs.time_count,
            obs_dist=inputs.raw_state.full_obs_dist,
            y_obs=inputs.runtime_batch.observations.y_obs,
            time_mask=inputs.runtime_batch.observations.time_mask,
            obs_scale=inputs.runtime_batch.observations.obs_scale,
        )

    def _sample_auxiliary_observations(
        self,
        inputs: _RuntimeObservationInputs,
    ) -> None:
        sample_index_relative_observations(
            inputs=_IndexRelativeObservationInputs(
                batch=inputs.batch,
                loc=inputs.raw_state.loc,
                cov_factor=inputs.raw_state.cov_factor,
                cov_diag=inputs.raw_state.cov_diag,
                overlay=self.priors.index_relative_measurement,
                coordinates=inputs.coordinates,
            ),
            groups=inputs.group_builder(
                config=self.priors.index_relative_measurement,
                coordinate_names=inputs.coordinates.coordinate_names,
                device=inputs.device,
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
        if not supports_structural_predictive_summaries(guide):
            return None
        return predict_dependence_layer_v4_l1(
            model=cast(Any, self),
            guide=cast(Any, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


def build_index_relative_measurement_model_priors(
    *,
    params: Mapping[str, Any],
    config_builder: ConfigBuilder,
    label: str,
    param_key: str = "index_relative_measurement",
) -> IndexRelativeMeasurementModelPriors:
    values = coerce_mapping(params, label="model.params")
    if not values:
        return IndexRelativeMeasurementModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "index_t_copula",
        param_key,
    }
    if extra:
        raise ConfigError(
            f"Unknown {label} params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return IndexRelativeMeasurementModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
        index_relative_measurement=config_builder(values.get(param_key)),
    )


def sample_non_index_raw_observations(
    *,
    batch: Any,
    loc: torch.Tensor,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
) -> None:
    non_index_mask = cast(torch.BoolTensor, ~batch.assets.index_mask)
    if not bool(non_index_mask.any()):
        return
    site = _NamedTimeSite(
        name="non_index_raw_obs",
        obs_dist=dist.LowRankMultivariateNormal(
            loc=loc[:, non_index_mask],
            cov_factor=cov_factor[:, non_index_mask, :],
            cov_diag=cov_diag[:, non_index_mask],
        ),
        y_obs=_masked_observations(batch.observations.y_obs, non_index_mask),
        time_mask=batch.observations.time_mask,
        obs_scale=batch.observations.obs_scale,
    )
    _sample_named_time_site(time_count=int(loc.shape[0]), site=site)


def sample_index_relative_observations(
    *,
    inputs: _IndexRelativeObservationInputs,
    groups: tuple[BasketObservationGroup, ...],
) -> None:
    observed = inputs.batch.observations.y_obs
    if observed is None:
        return
    standardized = _standardize_index_relative_inputs(
        observed=observed,
        inputs=inputs,
    )
    for group in groups:
        _sample_named_time_site(
            time_count=int(standardized.coordinate_mean.shape[0]),
            site=_build_group_time_site(
                group=group,
                inputs=standardized,
            ),
        )


def _standardize_index_relative_inputs(
    *,
    observed: torch.Tensor,
    inputs: _IndexRelativeObservationInputs,
) -> _GroupTimeSiteInputs:
    coordinate_obs = project_basket_mean(
        loc=observed[:, inputs.coordinates.index_mask],
        basis=inputs.coordinates.basis,
    )
    transform = build_index_coordinate_transform(
        observations=coordinate_obs,
        config=inputs.overlay,
    )
    coordinate_mean = standardize_index_coordinates(
        values=project_basket_mean(
            loc=inputs.loc[:, inputs.coordinates.index_mask],
            basis=inputs.coordinates.basis,
        ),
        transform=transform,
    )
    coordinate_covariance = standardize_index_covariance(
        covariance=project_basket_covariance(
            cov_factor=inputs.cov_factor[:, inputs.coordinates.index_mask, :],
            cov_diag=inputs.cov_diag[:, inputs.coordinates.index_mask],
            basis=inputs.coordinates.basis,
            eps=inputs.overlay.eps,
        ),
        transform=transform,
    )
    return _GroupTimeSiteInputs(
        base_obs_scale=inputs.batch.observations.obs_scale,
        time_mask=inputs.batch.observations.time_mask,
        coordinate_mean=coordinate_mean,
        coordinate_obs=standardize_index_coordinates(
            values=coordinate_obs,
            transform=transform,
        ),
        coordinate_scale=coordinate_scale_from_covariance(
            covariance=coordinate_covariance,
            eps=inputs.overlay.eps,
        ),
        df=inputs.overlay.df,
    )


def _build_group_time_site(
    *,
    group: BasketObservationGroup,
    inputs: _GroupTimeSiteInputs,
) -> _NamedTimeSite:
    mask = group.mask
    return _NamedTimeSite(
        name=group.name,
        obs_dist=dist.StudentT(
            df=inputs.df,
            loc=inputs.coordinate_mean[:, mask],
            scale=inputs.coordinate_scale[:, mask],
        ).to_event(1),
        y_obs=inputs.coordinate_obs[:, mask],
        time_mask=inputs.time_mask,
        obs_scale=_resolved_obs_scale(inputs.base_obs_scale, group.obs_weight),
    )


def _masked_observations(
    values: torch.Tensor | None,
    mask: torch.BoolTensor,
) -> torch.Tensor | None:
    if values is None:
        return None
    return values[:, mask]


def _resolved_obs_scale(
    base_obs_scale: float | None,
    obs_weight: float,
) -> float | None:
    if base_obs_scale is None:
        return float(obs_weight)
    return float(base_obs_scale) * float(obs_weight)


def _sample_named_time_site(*, time_count: int, site: _NamedTimeSite) -> None:
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


__all__ = [
    "IndexRelativeMeasurementModelPriors",
    "IndexRelativeMeasurementModelRuntime",
    "build_index_relative_measurement_model_priors",
    "sample_index_relative_observations",
    "sample_non_index_raw_observations",
]
