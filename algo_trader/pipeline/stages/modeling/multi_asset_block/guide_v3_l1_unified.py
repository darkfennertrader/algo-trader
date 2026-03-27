from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.batch_utils import resolve_batch_shape
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import (
    FilteringState,
    _coerce_mapping,
    _lognormal_mean,
    _lognormal_median,
    _resolve_filtering_state,
    _resolve_time_mask,
    _resolve_y_input,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide
from algo_trader.pipeline.stages.modeling.runtime_support import (
    build_runtime_observations,
)

from .shared_v3_l1_unified import (
    CovarianceLoadings,
    FactorCountConfig,
    MeanTensorMeans,
    PanelDimensions,
    RegimePosteriorMeans,
    RuntimeAssetMetadata,
    StructuralPosteriorMeans,
    StructuralTensorMeans,
    V3L1UnifiedRuntimeBatch,
    build_asset_class_ids,
    build_factor_count_config,
    build_panel_dimensions,
    coerce_four_state_tensor,
)

_INIT_LOGSCALE = 0.30
_INIT_POSITIVE_LOC = math.log(0.08)
_STATE_COUNT = 4


@dataclass(frozen=True)
class RegimePhiConfig:
    phi_fx_broad: float = 0.95
    phi_fx_cross: float = 0.985
    phi_index: float = 0.97
    phi_commodity: float = 0.97


@dataclass(frozen=True)
class V3L1UnifiedGuideConfig:
    counts: FactorCountConfig = FactorCountConfig()
    phi: RegimePhiConfig = RegimePhiConfig()


@dataclass(frozen=True)
class _GuideContext:
    batch: V3L1UnifiedRuntimeBatch
    device: torch.device
    dtype: torch.dtype
    shape: PanelDimensions


@dataclass
class MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(PyroGuide):
    config: V3L1UnifiedGuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_structural_sites(context=context, config=self.config)
        _sample_regime_scale_sites(context)
        _sample_local_state_sites(context)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch = build_v3_l1_unified_runtime_batch(batch)
        context = _build_context(runtime_batch)
        h_loc = pyro.get_param_store().get_param("h_loc")
        h_scale = pyro.get_param_store().get_param("h_scale")
        return FilteringState(
            h_loc=h_loc[-1].detach(),
            h_scale=h_scale[-1].detach(),
            steps_seen=_next_steps_seen(runtime_batch),
        )

    def structural_posterior_means(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
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
                    B_commodity=store.get_param("B_commodity_loc").detach(),
                ),
            ),
            regime=_regime_means(store=store, use_median=False),
        )

    def structural_predictive_summaries(self) -> StructuralPosteriorMeans:
        store = pyro.get_param_store()
        return StructuralPosteriorMeans(
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
                    B_commodity=store.get_param("B_commodity_loc").detach(),
                ),
            ),
            regime=_regime_means(store=store, use_median=True),
        )


@register_guide("multi_asset_block_guide_v3_l1_unified_online_filtering")
def build_multi_asset_block_guide_v3_l1_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(
        config=_build_guide_config(params)
    )


def build_v3_l1_unified_runtime_batch(
    batch: ModelBatch,
) -> V3L1UnifiedRuntimeBatch:
    shape = resolve_batch_shape(batch)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is None or X_asset.ndim != 3:
        raise ConfigError("v3_l1_unified runtime requires batch.X_asset with shape [T, A, F]")
    if batch.X_global is None or batch.X_global.ndim != 2:
        raise ConfigError("v3_l1_unified runtime requires batch.X_global with shape [T, G]")
    asset_names = _validated_asset_names(batch.asset_names, expected=shape.A)
    observations = build_runtime_observations(
        y_input=_resolve_y_input(shape),
        y_obs=shape.y_obs,
        time_mask=_resolve_time_mask(batch, shape.T, shape.A),
        obs_scale=batch.obs_scale,
    )
    return V3L1UnifiedRuntimeBatch(
        X_asset=X_asset.to(device=shape.device, dtype=shape.dtype),
        X_global=batch.X_global.to(device=shape.device, dtype=shape.dtype),
        observations=observations,
        assets=RuntimeAssetMetadata(
            asset_names=asset_names,
            class_ids=build_asset_class_ids(asset_names, device=shape.device),
        ),
        filtering_state=_resolve_filtering_state(
            batch.filtering_state,
            device=shape.device,
            dtype=shape.dtype,
        ),
    )


def _validated_asset_names(
    asset_names: Sequence[str] | None, *, expected: int
) -> tuple[str, ...]:
    if asset_names is None:
        raise ConfigError("v3_l1_unified requires batch.asset_names")
    names = tuple(str(name) for name in asset_names)
    if len(names) != expected:
        raise ConfigError("batch.asset_names must align with the asset dimension")
    return names


def _build_guide_config(params: Mapping[str, Any]) -> V3L1UnifiedGuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V3L1UnifiedGuideConfig()
    extra = set(values) - {
        "global_factor_count",
        "fx_broad_factor_count",
        "fx_cross_factor_count",
        "index_factor_count",
        "commodity_factor_count",
        "phi_fx_broad",
        "phi_fx_cross",
        "phi_index",
        "phi_commodity",
    }
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_guide_v3_l1_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = V3L1UnifiedGuideConfig()
    return V3L1UnifiedGuideConfig(
        counts=build_factor_count_config(values, base=base.counts),
        phi=RegimePhiConfig(
            phi_fx_broad=float(
                values.get("phi_fx_broad", base.phi.phi_fx_broad)
            ),
            phi_fx_cross=float(
                values.get("phi_fx_cross", base.phi.phi_fx_cross)
            ),
            phi_index=float(values.get("phi_index", base.phi.phi_index)),
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
    *, context: _GuideContext, config: V3L1UnifiedGuideConfig
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
        context, "B_commodity", config.counts.commodity_factor_count
    )


def _sample_alpha(context: _GuideContext) -> None:
    with pyro.plate("asset", context.shape.A, dim=-2):
        loc = pyro.param(
            "alpha_loc",
            torch.zeros(
                (context.shape.A, 1), device=context.device, dtype=context.dtype
            ),
        )
        scale = pyro.param(
            "alpha_scale",
            torch.full(
                (context.shape.A, 1),
                0.05,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("alpha", dist.Normal(loc, scale))


def _sample_sigma_idio(context: _GuideContext) -> None:
    base_scale = _class_sigma_base(context)
    with pyro.plate("asset_sigma", context.shape.A, dim=-1):
        loc = pyro.param("sigma_idio_loc", base_scale.log())
        scale = pyro.param(
            "sigma_idio_scale",
            torch.full(
                (context.shape.A,),
                _INIT_LOGSCALE,
                device=context.device,
                dtype=context.dtype,
            ),
            constraint=constraints.positive,
        )
        pyro.sample("sigma_idio", dist.LogNormal(loc, scale))


def _class_sigma_base(context: _GuideContext) -> torch.Tensor:
    class_ids = context.batch.assets.class_ids
    base = torch.full(
        (context.shape.A,), 0.05, device=context.device, dtype=context.dtype
    )
    base = torch.where(class_ids == 0, torch.full_like(base, 0.03), base)
    base = torch.where(class_ids == 1, torch.full_like(base, 0.06), base)
    return torch.where(class_ids == 2, torch.full_like(base, 0.04), base)


def _sample_w(context: _GuideContext) -> None:
    with pyro.plate("asset_w", context.shape.A, dim=-2):
        with pyro.plate("feature_w", context.shape.F, dim=-1):
            loc = pyro.param(
                "w_loc",
                torch.zeros(
                    (context.shape.A, context.shape.F),
                    device=context.device,
                    dtype=context.dtype,
                ),
            )
            scale = pyro.param(
                "w_scale",
                torch.full(
                    (context.shape.A, context.shape.F),
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample("w", dist.Normal(loc, scale))


def _sample_tau0(context: _GuideContext) -> None:
    loc = pyro.param(
        "tau0_loc",
        torch.tensor(
            math.log(0.05), device=context.device, dtype=context.dtype
        ),
    )
    scale = pyro.param(
        "tau0_scale",
        torch.tensor(
            _INIT_LOGSCALE, device=context.device, dtype=context.dtype
        ),
        constraint=constraints.positive,
    )
    pyro.sample("tau0", dist.LogNormal(loc, scale))


def _sample_beta(context: _GuideContext) -> None:
    with pyro.plate("asset_beta", context.shape.A, dim=-2):
        with pyro.plate("global_beta", context.shape.G, dim=-1):
            loc = pyro.param(
                "beta_loc",
                torch.zeros(
                    (context.shape.A, context.shape.G),
                    device=context.device,
                    dtype=context.dtype,
                ),
            )
            scale = pyro.param(
                "beta_scale",
                torch.full(
                    (context.shape.A, context.shape.G),
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample("beta", dist.Normal(loc, scale))


def _sample_loading_matrix(
    context: _GuideContext, name: str, factor_count: int
) -> None:
    with pyro.plate(f"{name}_asset", context.shape.A, dim=-2):
        with pyro.plate(f"{name}_k", factor_count, dim=-1):
            loc = pyro.param(
                f"{name}_loc",
                torch.zeros(
                    (context.shape.A, factor_count),
                    device=context.device,
                    dtype=context.dtype,
                ),
            )
            scale = pyro.param(
                f"{name}_scale",
                torch.full(
                    (context.shape.A, factor_count),
                    0.05,
                    device=context.device,
                    dtype=context.dtype,
                ),
                constraint=constraints.positive,
            )
            pyro.sample(name, dist.Normal(loc, scale))


def _sample_regime_scale_sites(context: _GuideContext) -> None:
    for name in _regime_param_names():
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


def _sample_local_state_sites(context: _GuideContext) -> None:
    prior_loc = _initial_h_loc(context)
    loc = pyro.param("h_loc", prior_loc.clone())
    scale = pyro.param(
        "h_scale",
        torch.full(
            (context.shape.T, _STATE_COUNT),
            0.15,
            device=context.device,
            dtype=context.dtype,
        ),
        constraint=constraints.positive,
    )
    for index in range(context.shape.T):
        pyro.sample(f"h_{index + 1}", dist.Normal(loc[index], scale[index]).to_event(1))


def _initial_h_loc(context: _GuideContext) -> torch.Tensor:
    base = torch.zeros(
        (context.shape.T, _STATE_COUNT),
        device=context.device,
        dtype=context.dtype,
    )
    if context.batch.filtering_state is None:
        return base
    base[0] = coerce_four_state_tensor(
        context.batch.filtering_state.h_loc,
        device=context.device,
        dtype=context.dtype,
    )
    return base


def _regime_param_names() -> tuple[str, ...]:
    return ("s_u_fx_broad", "s_u_fx_cross", "s_u_index", "s_u_commodity")


def _regime_means(
    *, store: Any, use_median: bool
) -> RegimePosteriorMeans:
    return RegimePosteriorMeans(
        s_u_fx_broad_mean=_positive_summary(
            store=store,
            name="s_u_fx_broad",
            use_median=use_median,
        ),
        s_u_fx_cross_mean=_positive_summary(
            store=store,
            name="s_u_fx_cross",
            use_median=use_median,
        ),
        s_u_index_mean=_positive_summary(
            store=store,
            name="s_u_index",
            use_median=use_median,
        ),
        s_u_commodity_mean=_positive_summary(
            store=store,
            name="s_u_commodity",
            use_median=use_median,
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


def _next_steps_seen(batch: V3L1UnifiedRuntimeBatch) -> int:
    previous = (
        0 if batch.filtering_state is None else int(batch.filtering_state.steps_seen)
    )
    return previous + int(batch.observations.y_input.shape[0])
