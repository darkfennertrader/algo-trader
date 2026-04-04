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

from .defaults import merge_nested_params, model_default_params_v11_l1
from .guide import IndexSubspaceConsistencyGuideV11L1OnlineFiltering
from .shared import (
    IndexSubspaceConfig,
    IndexSubspacePosteriorMeans,
    build_index_subspace_config,
    build_index_subspace_coordinates,
    global_scale_from_covariance,
    project_subspace_covariance,
    project_subspace_mean,
    spread_scale_tril_from_covariance,
    subspace_time_mask,
)

_SUBSPACE_DIM = 5


@dataclass(frozen=True)
class V11L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_t_copula: IndexTCopulaOverlayConfig = field(
        default_factory=IndexTCopulaOverlayConfig
    )
    index_subspace: IndexSubspaceConfig = field(default_factory=IndexSubspaceConfig)


@dataclass(frozen=True)
class _SubspaceObservationInputs:
    batch: Any
    loc: torch.Tensor
    cov_factor: torch.Tensor
    cov_diag: torch.Tensor
    overlay: IndexSubspaceConfig
    subspace_params: IndexSubspacePosteriorMeans


@dataclass(frozen=True)
class _NamedTimeSite:
    name: str
    obs_dist: dist.TorchDistribution
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None


@dataclass
class IndexSubspaceConsistencyModelV11L1OnlineFiltering(PyroModel):
    priors: V11L1ModelPriors = field(default_factory=V11L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        mix = _sample_index_t_copula_mix(context, self.priors.index_t_copula)
        subspace_params = _sample_index_subspace_sites(
            overlay=self.priors.index_subspace,
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
        _sample_subspace_observations(
            inputs=_SubspaceObservationInputs(
                batch=context.batch,
                loc=loc,
                cov_factor=cov_factor,
                cov_diag=cov_diag,
                overlay=self.priors.index_subspace,
                subspace_params=subspace_params,
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
        return predict_index_subspace_consistency_v11_l1(
            model=self,
            guide=cast(IndexSubspaceConsistencyGuideV11L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("index_subspace_consistency_model_v11_l1_online_filtering")
def build_index_subspace_consistency_model_v11_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v11_l1(), params)
    return IndexSubspaceConsistencyModelV11L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V11L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V11L1ModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "index_t_copula",
        "index_subspace_consistency",
    }
    if extra:
        raise ConfigError(
            "Unknown index_subspace_consistency_model_v11_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V11L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        index_t_copula=_build_index_t_copula_config(values.get("index_t_copula")),
        index_subspace=build_index_subspace_config(
            values.get("index_subspace_consistency")
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


def _sample_index_subspace_sites(
    *,
    overlay: IndexSubspaceConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexSubspacePosteriorMeans:
    if not overlay.enabled:
        return IndexSubspacePosteriorMeans(
            global_scale=torch.ones((), device=device, dtype=dtype),
            spread_scale=torch.ones((4,), device=device, dtype=dtype),
            spread_corr_cholesky=torch.eye(4, device=device, dtype=dtype),
        )
    global_scale = pyro.sample(
        "index_subspace_global_scale",
        dist.LogNormal(
            torch.tensor(
                overlay.prior_scales.global_scale_center,
                device=device,
                dtype=dtype,
            ).log(),
            torch.tensor(
                overlay.prior_scales.global_scale_log_scale,
                device=device,
                dtype=dtype,
            ),
        ),
    )
    spread_scale = pyro.sample(
        "index_subspace_spread_scale",
        dist.LogNormal(
            torch.full(
                (4,),
                float(overlay.prior_scales.spread_scale_center),
                device=device,
                dtype=dtype,
            ).log(),
            torch.full(
                (4,),
                float(overlay.prior_scales.spread_scale_log_scale),
                device=device,
                dtype=dtype,
            ),
        ).to_event(1),
    )
    spread_corr_cholesky = pyro.sample(
        "index_subspace_spread_corr_cholesky",
        dist.LKJCholesky(
            dim=4,
            concentration=torch.tensor(
                overlay.prior_scales.correlation_concentration,
                device=device,
                dtype=dtype,
            ),
        ),
    )
    return IndexSubspacePosteriorMeans(
        global_scale=global_scale,
        spread_scale=spread_scale,
        spread_corr_cholesky=spread_corr_cholesky,
    )


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


def _sample_subspace_observations(
    *,
    inputs: _SubspaceObservationInputs,
) -> None:
    if not inputs.overlay.enabled:
        return
    coordinates = build_index_subspace_coordinates(
        assets=inputs.batch.assets,
        device=inputs.loc.device,
        dtype=inputs.loc.dtype,
    )
    z_mean = project_subspace_mean(loc=inputs.loc, basis=coordinates.basis)
    z_cov = project_subspace_covariance(
        cov_factor=inputs.cov_factor,
        cov_diag=inputs.cov_diag,
        basis=coordinates.basis,
        eps=inputs.overlay.eps,
    )
    z_obs = None
    if inputs.batch.observations.y_obs is not None:
        z_obs = project_subspace_mean(
            loc=inputs.batch.observations.y_obs,
            basis=coordinates.basis,
        )
    global_dist = dist.StudentT(
        df=inputs.overlay.global_df,
        loc=z_mean[:, 0],
        scale=global_scale_from_covariance(
            subspace_covariance=z_cov,
            global_scale=inputs.subspace_params.global_scale,
            eps=inputs.overlay.eps,
        ),
    )
    spread_dist = dist.MultivariateStudentT(
        df=inputs.overlay.spread_df,
        loc=z_mean[:, 1:],
        scale_tril=spread_scale_tril_from_covariance(
            subspace_covariance=z_cov,
            spread_scale=inputs.subspace_params.spread_scale,
            spread_corr_cholesky=inputs.subspace_params.spread_corr_cholesky,
            eps=inputs.overlay.eps,
        ),
    )
    time_mask = subspace_time_mask(
        time_mask=inputs.batch.observations.time_mask,
        index_mask=coordinates.index_mask,
    )
    obs_scale = _subspace_obs_scale(
        base_obs_scale=inputs.batch.observations.obs_scale,
        overlay=inputs.overlay,
    )
    _sample_named_time_site(
        time_count=int(z_mean.shape[0]),
        site=_NamedTimeSite(
            name="index_subspace_global_obs",
            obs_dist=global_dist,
            y_obs=None if z_obs is None else z_obs[:, 0],
            time_mask=time_mask,
            obs_scale=obs_scale,
        ),
    )
    _sample_named_time_site(
        time_count=int(z_mean.shape[0]),
        site=_NamedTimeSite(
            name="index_subspace_spread_obs",
            obs_dist=spread_dist,
            y_obs=None if z_obs is None else z_obs[:, 1:],
            time_mask=time_mask,
            obs_scale=obs_scale,
        ),
    )


def _subspace_obs_scale(
    *,
    base_obs_scale: float | None,
    overlay: IndexSubspaceConfig,
) -> float | None:
    if base_obs_scale is None:
        return overlay.obs_weight
    return float(base_obs_scale) * overlay.obs_weight


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


def predict_index_subspace_consistency_v11_l1(
    *,
    model: IndexSubspaceConsistencyModelV11L1OnlineFiltering,
    guide: IndexSubspaceConsistencyGuideV11L1OnlineFiltering,
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
    "IndexSubspaceConsistencyModelV11L1OnlineFiltering",
    "V11L1ModelPriors",
    "build_index_subspace_consistency_model_v11_l1_online_filtering",
    "predict_index_subspace_consistency_v11_l1",
]
