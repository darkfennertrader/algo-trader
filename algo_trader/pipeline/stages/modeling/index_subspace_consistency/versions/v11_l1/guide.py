from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState
from algo_trader.pipeline.stages.modeling.multi_asset_block.guide_v3_l1_unified import (
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    V3L1UnifiedGuideConfig,
    _build_context as _build_base_context,
    _build_guide_config as _build_base_guide_config,
    _sample_local_state_sites as _sample_base_local_state_sites,
    _sample_regime_scale_sites as _sample_base_regime_scale_sites,
    _sample_structural_sites as _sample_base_structural_sites,
    build_v3_l1_unified_runtime_batch,
)
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v11_l1, merge_nested_params
from .shared import initial_index_subspace_posterior_means


@dataclass(frozen=True)
class V11L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_t_copula_enabled: bool = True
    index_subspace_enabled: bool = True


@dataclass
class IndexSubspaceConsistencyGuideV11L1OnlineFiltering(PyroGuide):
    config: V11L1GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        context = _build_base_context(build_v3_l1_unified_runtime_batch(batch))
        _sample_base_structural_sites(context=context, config=self.config.base)
        _sample_base_regime_scale_sites(context)
        _sample_base_local_state_sites(context)
        _sample_index_t_copula_mix_sites(
            time_count=int(context.shape.T),
            device=context.device,
            dtype=context.dtype,
            enabled=self.config.index_t_copula_enabled,
        )
        _sample_index_subspace_sites(
            device=context.device,
            dtype=context.dtype,
            enabled=self.config.index_subspace_enabled,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("index_subspace_consistency_guide_v11_l1_online_filtering")
def build_index_subspace_consistency_guide_v11_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v11_l1(), params)
    return IndexSubspaceConsistencyGuideV11L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V11L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V11L1GuideConfig()
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
        "phi_commodity",
        "index_group_enabled",
        "index_t_copula_enabled",
        "index_subspace_consistency_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown index_subspace_consistency_guide_v11_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value
        for key, value in values.items()
        if key not in {"index_t_copula_enabled", "index_subspace_consistency_enabled"}
    }
    return V11L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        index_t_copula_enabled=bool(values.get("index_t_copula_enabled", True)),
        index_subspace_enabled=bool(
            values.get("index_subspace_consistency_enabled", True)
        ),
    )


def _sample_index_subspace_sites(
    *,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    if not enabled:
        return
    initial = initial_index_subspace_posterior_means(device=device, dtype=dtype)
    global_scale = pyro.param(
        "index_subspace_global_scale_q",
        initial.global_scale,
        constraint=constraints.positive,
    )
    spread_scale = pyro.param(
        "index_subspace_spread_scale_q",
        initial.spread_scale,
        constraint=constraints.positive,
    )
    spread_corr_cholesky = pyro.param(
        "index_subspace_spread_corr_cholesky_q",
        initial.spread_corr_cholesky,
        constraint=constraints.corr_cholesky,
    )
    pyro.sample("index_subspace_global_scale", dist.Delta(global_scale))
    pyro.sample(
        "index_subspace_spread_scale",
        dist.Delta(spread_scale, event_dim=1),
    )
    pyro.sample(
        "index_subspace_spread_corr_cholesky",
        dist.Delta(spread_corr_cholesky, event_dim=2),
    )


def _sample_index_t_copula_mix_sites(
    *,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    if not enabled:
        return
    concentration = pyro.param(
        "index_t_copula_mix_concentration",
        torch.full((time_count,), 3.0, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    rate = pyro.param(
        "index_t_copula_mix_rate",
        torch.full((time_count,), 3.0, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample("index_t_copula_mix", dist.Gamma(concentration, rate).to_event(1))


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "IndexSubspaceConsistencyGuideV11L1OnlineFiltering",
    "V11L1GuideConfig",
    "build_index_subspace_consistency_guide_v11_l1_online_filtering",
]
