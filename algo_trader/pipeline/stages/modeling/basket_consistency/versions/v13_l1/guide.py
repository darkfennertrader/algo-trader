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

from .defaults import guide_default_params_v13_l1, merge_nested_params
from .shared import (
    build_basket_consistency_coordinates,
    initial_basket_consistency_posterior_means,
)


@dataclass(frozen=True)
class V13L1GuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_t_copula_enabled: bool = True
    basket_consistency_enabled: bool = True


@dataclass
class BasketConsistencyGuideV13L1OnlineFiltering(PyroGuide):
    config: V13L1GuideConfig

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
        _sample_basket_consistency_sites(
            batch=context.batch,
            device=context.device,
            dtype=context.dtype,
            enabled=self.config.basket_consistency_enabled,
        )

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(config=self.config.base)


@register_guide("basket_consistency_guide_v13_l1_online_filtering")
def build_basket_consistency_guide_v13_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v13_l1(), params)
    return BasketConsistencyGuideV13L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V13L1GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V13L1GuideConfig()
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
        "basket_consistency_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown basket_consistency_guide_v13_l1_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value
        for key, value in values.items()
        if key not in {"index_t_copula_enabled", "basket_consistency_enabled"}
    }
    return V13L1GuideConfig(
        base=_build_base_guide_config(base_payload),
        index_t_copula_enabled=bool(values.get("index_t_copula_enabled", True)),
        basket_consistency_enabled=bool(
            values.get("basket_consistency_enabled", True)
        ),
    )


def _sample_basket_consistency_sites(
    *,
    batch: Any,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    if not enabled:
        return
    coordinates = build_basket_consistency_coordinates(
        assets=batch.assets,
        device=device,
        dtype=dtype,
    )
    if coordinates.basket_count == 0:
        return
    initial = initial_basket_consistency_posterior_means(
        count=coordinates.basket_count,
        device=device,
        dtype=dtype,
    )
    scale = pyro.param(
        "basket_consistency_scale_q",
        initial.basket_scale,
        constraint=constraints.positive,
    )
    pyro.sample("basket_consistency_scale", dist.Delta(scale, event_dim=1))


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
    "BasketConsistencyGuideV13L1OnlineFiltering",
    "V13L1GuideConfig",
    "build_basket_consistency_guide_v13_l1_online_filtering",
]
