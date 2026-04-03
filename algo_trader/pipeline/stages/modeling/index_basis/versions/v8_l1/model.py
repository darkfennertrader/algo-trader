from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.model_v3_l1_unified import (
    V3L1UnifiedModelPriors,
    _build_context as _build_base_context,
    _build_model_priors as _build_base_model_priors,
    _mean_path as _base_mean_path,
    _sample_regime_path as _sample_base_regime_path,
    _sample_regime_scales as _sample_base_regime_scales,
    _sample_structural_sites as _sample_base_structural_sites,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide, PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.runtime_support import sample_time_observations

from .defaults import merge_nested_params, model_default_params_v8_l1
from .guide import IndexBasisGuideV8L1OnlineFiltering
from .predict import predict_index_basis_v8_l1
from .shared import (
    IndexBasisConfig,
    IndexBasisFactorState,
    IndexBasisPosteriorMeans,
    build_index_basis_config,
    build_index_basis_coordinates,
    build_index_basis_factor_block,
    build_nonindex_cov_factor_path,
)

_SPREAD_DIM = 4


@dataclass(frozen=True)
class V8L1ModelPriors:
    base: V3L1UnifiedModelPriors = field(default_factory=V3L1UnifiedModelPriors)
    index_basis: IndexBasisConfig = field(default_factory=IndexBasisConfig)


@dataclass
class IndexBasisModelV8L1OnlineFiltering(PyroModel):
    priors: V8L1ModelPriors = field(default_factory=V8L1ModelPriors)

    def supported_training_methods(self) -> tuple[str, ...]:
        return ("online_filtering",)

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_base_context(runtime_batch, self.priors.base)
        structural = _sample_base_structural_sites(context)
        regime_scales = _sample_base_regime_scales(context)
        regime_path = _sample_base_regime_path(context, regime_scales)
        basis_params = _sample_index_basis_sites(
            overlay=self.priors.index_basis,
            device=context.device,
            dtype=context.dtype,
        )
        obs_dist = _build_observation_distribution(
            context=context,
            structural=structural,
            regime_path=regime_path,
            basis_params=basis_params,
            overlay=self.priors.index_basis,
        )
        sample_time_observations(
            time_count=context.shape.T,
            obs_dist=obs_dist,
            y_obs=runtime_batch.observations.y_obs,
            time_mask=runtime_batch.observations.time_mask,
            obs_scale=runtime_batch.observations.obs_scale,
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
        return predict_index_basis_v8_l1(
            model=self,
            guide=cast(IndexBasisGuideV8L1OnlineFiltering, guide),
            batch=batch,
            num_samples=num_samples,
            state=state,
        )


@register_model("index_basis_model_v8_l1_online_filtering")
def build_index_basis_model_v8_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v8_l1(), params)
    return IndexBasisModelV8L1OnlineFiltering(priors=_build_model_priors(merged_params))


def _build_model_priors(params: Mapping[str, Any]) -> V8L1ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V8L1ModelPriors()
    extra = set(values) - {"mean", "factors", "regime", "index_basis"}
    if extra:
        raise ConfigError(
            "Unknown index_basis_model_v8_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V8L1ModelPriors(
        base=_build_base_model_priors(base_payload),
        index_basis=build_index_basis_config(values.get("index_basis")),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_basis_sites(
    *,
    overlay: IndexBasisConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> IndexBasisPosteriorMeans:
    if not overlay.enabled:
        zeros = torch.zeros((), device=device, dtype=dtype)
        return IndexBasisPosteriorMeans(
            global_scale=zeros,
            spread_scale=torch.zeros((_SPREAD_DIM,), device=device, dtype=dtype),
            spread_corr_cholesky=torch.eye(_SPREAD_DIM, device=device, dtype=dtype),
        )
    global_scale = pyro.sample(
        "index_basis_global_scale",
        dist.HalfNormal(
            torch.tensor(
                overlay.prior_scales.global_scale,
                device=device,
                dtype=dtype,
            )
        ),
    )
    spread_scale = pyro.sample(
        "index_basis_spread_scale",
        dist.HalfNormal(
            torch.full(
                (_SPREAD_DIM,),
                overlay.prior_scales.spread_scale,
                device=device,
                dtype=dtype,
            )
        ).to_event(1),
    )
    spread_corr_cholesky = pyro.sample(
        "index_basis_spread_corr_cholesky",
        dist.LKJCholesky(
            dim=_SPREAD_DIM,
            concentration=torch.tensor(
                overlay.prior_scales.correlation_concentration,
                device=device,
                dtype=dtype,
            ),
        ),
    )
    return IndexBasisPosteriorMeans(
        global_scale=global_scale,
        spread_scale=spread_scale,
        spread_corr_cholesky=spread_corr_cholesky,
    )


def _sample_index_basis_mix(
    *,
    df_value: float,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    concentration = torch.full(
        (time_count,),
        df_value / 2.0,
        device=device,
        dtype=dtype,
    )
    return pyro.sample(name, dist.Gamma(concentration, concentration).to_event(1))


def _build_observation_distribution(
    *,
    context: Any,
    structural: Any,
    regime_path: torch.Tensor,
    basis_params: IndexBasisPosteriorMeans,
    overlay: IndexBasisConfig,
) -> dist.LowRankMultivariateNormal:
    loc = _base_mean_path(context, structural)
    base_cov_diag = structural.mean.sigma_idio.pow(2) + context.priors.regime.eps
    if not overlay.enabled:
        cov_factor = build_nonindex_cov_factor_path(
            loadings=structural.loadings,
            class_ids=context.batch.assets.class_ids,
            regime_path=regime_path,
            dtype=context.dtype,
        )
        return dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=cov_factor,
            cov_diag=base_cov_diag,
        )
    coordinates = build_index_basis_coordinates(
        assets=context.batch.assets,
        device=context.device,
        dtype=context.dtype,
    )
    nonindex_block = build_nonindex_cov_factor_path(
        loadings=structural.loadings,
        class_ids=context.batch.assets.class_ids,
        regime_path=regime_path,
        dtype=context.dtype,
    )
    regime_scale = torch.exp(0.5 * regime_path[:, 2])
    global_mix = _sample_index_basis_mix(
        df_value=overlay.global_df,
        time_count=context.shape.T,
        device=context.device,
        dtype=context.dtype,
        name="index_basis_global_mix",
    )
    spread_mix = _sample_index_basis_mix(
        df_value=overlay.spread_df,
        time_count=context.shape.T,
        device=context.device,
        dtype=context.dtype,
        name="index_basis_spread_mix",
    )
    index_block = build_index_basis_factor_block(
        coordinates=coordinates,
        state=IndexBasisFactorState(
            global_scale=basis_params.global_scale,
            spread_scale=basis_params.spread_scale,
            spread_corr_cholesky=basis_params.spread_corr_cholesky,
            regime_scale=regime_scale,
            global_mix=global_mix,
            spread_mix=spread_mix,
            eps=overlay.eps,
        ),
    )
    return dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=torch.cat([nonindex_block, index_block], dim=-1),
        cov_diag=base_cov_diag,
    )


__all__ = [
    "IndexBasisModelV8L1OnlineFiltering",
    "V8L1ModelPriors",
    "build_index_basis_model_v8_l1_online_filtering",
]
