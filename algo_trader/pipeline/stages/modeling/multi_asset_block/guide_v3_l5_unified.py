from __future__ import annotations
# pylint: disable=duplicate-code

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import (
    FilteringState,
    _coerce_mapping,
    _lognormal_mean,
    _lognormal_median,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_v3_l1_unified import build_v3_l1_unified_runtime_batch
from .shared_v3_l1_unified import (
    CovarianceLoadings,
    FactorCountConfig,
    MeanTensorMeans,
    PanelDimensions,
    StructuralTensorMeans,
    V3L1UnifiedRuntimeBatch,
    build_factor_count_config,
    build_panel_dimensions,
)
from .shared_v3_l5_unified import (
    RegimePosteriorMeansV3L5,
    StructuralPosteriorMeansV3L5,
    coerce_v3_l5_state_tensor,
    v3_l5_state_count,
)
from .v3_l5_defaults import guide_default_params_v3_l5, merge_nested_params

_INIT_LOGSCALE = 0.30
_INIT_POSITIVE_LOC = math.log(0.08)


@dataclass(frozen=True)
class RegimePhiConfigV3L5:
    phi_fx_broad: float = 0.95
    phi_fx_cross: float = 0.985
    phi_index: float = 0.97
    phi_index_group: float = 0.985
    phi_commodity: float = 0.97


@dataclass(frozen=True)
class V3L5UnifiedGuideConfig:
    counts: FactorCountConfig = FactorCountConfig()
    phi: RegimePhiConfigV3L5 = RegimePhiConfigV3L5()


@dataclass(frozen=True)
class _GuideContext:
    batch: V3L1UnifiedRuntimeBatch
    device: torch.device
    dtype: torch.dtype
    shape: PanelDimensions


@dataclass
class MultiAssetBlockGuideV3L5UnifiedOnlineFiltering(PyroGuide):
    config: V3L5UnifiedGuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_structural_sites(context=context, config=self.config)
        _sample_regime_scale_sites(context)
        _sample_local_state_sites(context)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        h_loc = pyro.get_param_store().get_param("h_loc")
        h_scale = pyro.get_param_store().get_param("h_scale")
        return FilteringState(
            h_loc=h_loc[-1].detach(),
            h_scale=h_scale[-1].detach(),
            steps_seen=_next_steps_seen(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeansV3L5:
        store = pyro.get_param_store()
        return StructuralPosteriorMeansV3L5(
            tensors=StructuralTensorMeans(
                mean=MeanTensorMeans(
                    alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
                    sigma_idio=_sigma_summary(use_median=False),
                    w=store.get_param("w_loc").detach(),
                    beta=store.get_param("beta_loc").detach(),
                ),
                loadings=CovarianceLoadings(
                    B_global=store.get_param("B_global_loc").detach(),
                    B_fx_broad=store.get_param("B_fx_broad_loc").detach(),
                    B_fx_cross=store.get_param("B_fx_cross_loc").detach(),
                    B_index=store.get_param("B_index_loc").detach(),
                    B_index_static=store.get_param("B_index_static_loc").detach(),
                    index_group_scale=_optional_positive_summary(
                        store=store,
                        name="index_group_scale",
                        use_median=False,
                    ),
                    B_commodity=store.get_param("B_commodity_loc").detach(),
                ),
            ),
            regime=_regime_means(store=store, use_median=False),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeansV3L5:
        store = pyro.get_param_store()
        return StructuralPosteriorMeansV3L5(
            tensors=StructuralTensorMeans(
                mean=MeanTensorMeans(
                    alpha=store.get_param("alpha_loc").squeeze(-1).detach(),
                    sigma_idio=_sigma_summary(use_median=True),
                    w=store.get_param("w_loc").detach(),
                    beta=store.get_param("beta_loc").detach(),
                ),
                loadings=CovarianceLoadings(
                    B_global=store.get_param("B_global_loc").detach(),
                    B_fx_broad=store.get_param("B_fx_broad_loc").detach(),
                    B_fx_cross=store.get_param("B_fx_cross_loc").detach(),
                    B_index=store.get_param("B_index_loc").detach(),
                    B_index_static=store.get_param("B_index_static_loc").detach(),
                    index_group_scale=_optional_positive_summary(
                        store=store,
                        name="index_group_scale",
                        use_median=True,
                    ),
                    B_commodity=store.get_param("B_commodity_loc").detach(),
                ),
            ),
            regime=_regime_means(store=store, use_median=True),
        )


@register_guide("multi_asset_block_guide_v3_l5_unified_online_filtering")
def build_multi_asset_block_guide_v3_l5_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v3_l5(), params)
    return MultiAssetBlockGuideV3L5UnifiedOnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V3L5UnifiedGuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V3L5UnifiedGuideConfig()
    extra = set(values) - {
        "global_factor_count",
        "fx_broad_factor_count",
        "fx_cross_factor_count",
        "index_factor_count",
        "index_static_factor_count",
        "commodity_factor_count",
        "phi_fx_broad",
        "phi_fx_cross",
        "phi_index",
        "phi_index_group",
        "phi_commodity",
    }
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_guide_v3_l5_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = V3L5UnifiedGuideConfig()
    return V3L5UnifiedGuideConfig(
        counts=build_factor_count_config(values, base=base.counts),
        phi=RegimePhiConfigV3L5(
            phi_fx_broad=float(
                values.get("phi_fx_broad", base.phi.phi_fx_broad)
            ),
            phi_fx_cross=float(
                values.get("phi_fx_cross", base.phi.phi_fx_cross)
            ),
            phi_index=float(values.get("phi_index", base.phi.phi_index)),
            phi_index_group=float(
                values.get("phi_index_group", base.phi.phi_index_group)
            ),
            phi_commodity=float(
                values.get("phi_commodity", base.phi.phi_commodity)
            ),
        ),
    )


def _build_context(batch: V3L1UnifiedRuntimeBatch) -> _GuideContext:
    return _GuideContext(
        batch=batch,
        device=batch.X_asset.device,
        dtype=batch.X_asset.dtype,
        shape=build_panel_dimensions(batch),
    )


def _sample_structural_sites(
    *, context: _GuideContext, config: V3L5UnifiedGuideConfig
) -> None:
    _sample_alpha(context)
    _sample_sigma_idio(context)
    _sample_tau0(context)
    _sample_w(context)
    _sample_beta(context)
    _sample_loading_matrix(context, "B_global", config.counts.global_factor_count)
    _sample_loading_matrix(
        context, "B_fx_broad", config.counts.fx_broad_factor_count
    )
    _sample_loading_matrix(
        context, "B_fx_cross", config.counts.fx_cross_factor_count
    )
    _sample_loading_matrix(context, "B_index", config.counts.index_factor_count)
    _sample_loading_matrix(
        context, "B_index_static", config.counts.index_static_factor_count
    )
    _sample_index_group_scale(context)
    _sample_loading_matrix(
        context, "B_commodity", config.counts.commodity_factor_count
    )


def _sample_alpha(context: _GuideContext) -> None:
    with pyro.plate("asset", context.shape.A, dim=-2):
        loc = pyro.param(
            "alpha_loc",
            torch.zeros((context.shape.A, 1), device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            "alpha_scale",
            torch.full((context.shape.A, 1), 0.05, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("alpha", dist.Normal(loc, scale))


def _sample_sigma_idio(context: _GuideContext) -> None:
    base_scale = _class_sigma_base(context)
    with pyro.plate("asset_sigma", context.shape.A, dim=-1):
        loc = pyro.param("sigma_idio_loc", base_scale.log())
        scale = pyro.param(
            "sigma_idio_scale",
            torch.full((context.shape.A,), _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample("sigma_idio", dist.LogNormal(loc, scale))


def _class_sigma_base(context: _GuideContext) -> torch.Tensor:
    class_ids = context.batch.assets.class_ids
    base = torch.full((context.shape.A,), 0.05, device=context.device, dtype=context.dtype)
    base = torch.where(class_ids == 0, torch.full_like(base, 0.03), base)
    base = torch.where(class_ids == 1, torch.full_like(base, 0.06), base)
    return torch.where(class_ids == 2, torch.full_like(base, 0.04), base)


def _sample_w(context: _GuideContext) -> None:
    with pyro.plate("asset_w", context.shape.A, dim=-2):
        with pyro.plate("feature_w", context.shape.F, dim=-1):
            loc = pyro.param(
                "w_loc",
                torch.zeros((context.shape.A, context.shape.F), device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "w_scale",
                torch.full((context.shape.A, context.shape.F), 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample("w", dist.Normal(loc, scale))


def _sample_tau0(context: _GuideContext) -> None:
    loc = pyro.param(
        "tau0_loc",
        torch.tensor(math.log(0.05), device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "tau0_scale",
        torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    pyro.sample("tau0", dist.LogNormal(loc, scale))


def _sample_beta(context: _GuideContext) -> None:
    with pyro.plate("asset_beta", context.shape.A, dim=-2):
        with pyro.plate("global_beta", context.shape.G, dim=-1):
            loc = pyro.param(
                "beta_loc",
                torch.zeros((context.shape.A, context.shape.G), device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                "beta_scale",
                torch.full((context.shape.A, context.shape.G), 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample("beta", dist.Normal(loc, scale))


def _sample_loading_matrix(
    context: _GuideContext, name: str, factor_count: int
) -> None:
    if factor_count < 1:
        empty = torch.zeros((context.shape.A, 0), device=context.device, dtype=context.dtype)
        pyro.param(f"{name}_loc", empty)
        pyro.param(f"{name}_scale", empty, constraint=constraints.positive)
        return
    with pyro.plate(f"{name}_asset", context.shape.A, dim=-2):
        with pyro.plate(f"{name}_k", factor_count, dim=-1):
            loc = pyro.param(
                f"{name}_loc",
                torch.zeros((context.shape.A, factor_count), device=context.device, dtype=context.dtype),
            )
            scale = pyro.param(
                f"{name}_scale",
                torch.full((context.shape.A, factor_count), 0.05, device=context.device, dtype=context.dtype),
                constraint=constraints.positive,
            )
            pyro.sample(name, dist.Normal(loc, scale))


def _sample_index_group_scale(context: _GuideContext) -> None:
    group_count = int(context.batch.assets.index_group_count)
    if group_count < 1:
        empty = torch.zeros(0, device=context.device, dtype=context.dtype)
        pyro.param("index_group_scale_loc", empty)
        pyro.param(
            "index_group_scale_scale",
            empty,
            constraint=constraints.positive,
        )
        return
    loc = pyro.param(
        "index_group_scale_loc",
        torch.full((group_count,), _INIT_POSITIVE_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "index_group_scale_scale",
        torch.full((group_count,), _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    with pyro.plate("index_group", group_count, dim=-1):
        pyro.sample("index_group_scale", dist.LogNormal(loc, scale))


def _sample_regime_scale_sites(context: _GuideContext) -> None:
    for name in ("s_u_fx_broad", "s_u_fx_cross", "s_u_index", "s_u_commodity"):
        loc = pyro.param(
            f"{name}_loc",
            torch.tensor(_INIT_POSITIVE_LOC, device=context.device, dtype=context.dtype),
        )
        scale = pyro.param(
            f"{name}_scale",
            torch.tensor(_INIT_LOGSCALE, device=context.device, dtype=context.dtype),
            constraint=constraints.positive,
        )
        pyro.sample(name, dist.LogNormal(loc, scale))
    _sample_index_group_regime_scale(context)


def _sample_index_group_regime_scale(context: _GuideContext) -> None:
    group_count = int(context.batch.assets.index_group_count)
    if group_count < 1:
        empty = torch.zeros(0, device=context.device, dtype=context.dtype)
        pyro.param("s_u_index_group_loc", empty)
        pyro.param(
            "s_u_index_group_scale",
            empty,
            constraint=constraints.positive,
        )
        return
    loc = pyro.param(
        "s_u_index_group_loc",
        torch.full((group_count,), _INIT_POSITIVE_LOC, device=context.device, dtype=context.dtype),
    )
    scale = pyro.param(
        "s_u_index_group_scale",
        torch.full((group_count,), _INIT_LOGSCALE, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    with pyro.plate("index_group_regime", group_count, dim=-1):
        pyro.sample("s_u_index_group", dist.LogNormal(loc, scale))


def _sample_local_state_sites(context: _GuideContext) -> None:
    state_count = v3_l5_state_count(
        group_count=int(context.batch.assets.index_group_count)
    )
    prior_loc = _initial_h_loc(context, state_count=state_count)
    loc = pyro.param("h_loc", prior_loc.clone())
    scale = pyro.param(
        "h_scale",
        torch.full((context.shape.T, state_count), 0.15, device=context.device, dtype=context.dtype),
        constraint=constraints.positive,
    )
    for index in range(context.shape.T):
        pyro.sample(f"h_{index + 1}", dist.Normal(loc[index], scale[index]).to_event(1))


def _initial_h_loc(
    context: _GuideContext, *, state_count: int
) -> torch.Tensor:
    base = torch.zeros((context.shape.T, state_count), device=context.device, dtype=context.dtype)
    if context.batch.filtering_state is None:
        return base
    base[0] = coerce_v3_l5_state_tensor(
        context.batch.filtering_state.h_loc,
        device=context.device,
        dtype=context.dtype,
        group_count=int(context.batch.assets.index_group_count),
    )
    return base


def _regime_means(*, store: Any, use_median: bool) -> RegimePosteriorMeansV3L5:
    return RegimePosteriorMeansV3L5(
        s_u_fx_broad_mean=_positive_summary(
            store=store, name="s_u_fx_broad", use_median=use_median
        ),
        s_u_fx_cross_mean=_positive_summary(
            store=store, name="s_u_fx_cross", use_median=use_median
        ),
        s_u_index_mean=_positive_summary(
            store=store, name="s_u_index", use_median=use_median
        ),
        s_u_index_group_mean=_optional_positive_summary(
            store=store, name="s_u_index_group", use_median=use_median
        ),
        s_u_commodity_mean=_positive_summary(
            store=store, name="s_u_commodity", use_median=use_median
        ),
    )


def _sigma_summary(*, use_median: bool) -> torch.Tensor:
    store = pyro.get_param_store()
    loc = store.get_param("sigma_idio_loc")
    if use_median:
        return torch.exp(loc).detach()
    scale = store.get_param("sigma_idio_scale")
    return torch.exp(loc + 0.5 * scale.pow(2)).detach()


def _positive_summary(*, store: Any, name: str, use_median: bool) -> torch.Tensor:
    loc = store.get_param(f"{name}_loc")
    if use_median:
        return _lognormal_median(loc).detach()
    scale = store.get_param(f"{name}_scale")
    return _lognormal_mean(loc, scale).detach()


def _optional_positive_summary(
    *, store: Any, name: str, use_median: bool
) -> torch.Tensor:
    loc_name = f"{name}_loc"
    if loc_name not in set(store.keys()):
        return torch.zeros(0)
    return _positive_summary(store=store, name=name, use_median=use_median)


def _next_steps_seen(batch: V3L1UnifiedRuntimeBatch) -> int:
    previous = 0 if batch.filtering_state is None else int(batch.filtering_state.steps_seen)
    return previous + int(batch.observations.y_input.shape[0])


__all__ = [
    "MultiAssetBlockGuideV3L5UnifiedOnlineFiltering",
    "RegimePhiConfigV3L5",
    "V3L5UnifiedGuideConfig",
    "build_multi_asset_block_guide_v3_l5_unified_online_filtering",
]
