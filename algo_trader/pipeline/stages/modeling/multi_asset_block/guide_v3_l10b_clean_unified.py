from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_v3_l1_unified import (
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    V3L1UnifiedGuideConfig,
    _build_context as _build_base_context,
    _build_guide_config as _build_base_guide_config,
    _sample_local_state_sites as _sample_base_local_state_sites,
    _sample_regime_scale_sites as _sample_base_regime_scale_sites,
    _sample_structural_sites as _sample_base_structural_sites,
    build_v3_l1_unified_runtime_batch,
)
from .v3_l10b_clean_defaults import (
    guide_default_params_v3_l10b_clean,
    merge_nested_params,
)

_INDEX_T_COPULA_BROAD_DF_INIT = 6.0
_INDEX_T_COPULA_US_DIFF_DF_INIT = 10.0


@dataclass(frozen=True)
class V3L10BCleanUnifiedGuideConfig:
    base: V3L1UnifiedGuideConfig = field(default_factory=V3L1UnifiedGuideConfig)
    index_t_copula_enabled: bool = True


@dataclass
class MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering(PyroGuide):
    config: V3L10BCleanUnifiedGuideConfig

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

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        return self._base_guide().build_filtering_state(batch)

    def structural_posterior_means(self) -> Any:
        return self._base_guide().structural_posterior_means()

    def structural_predictive_summaries(self) -> Any:
        return self._base_guide().structural_predictive_summaries()

    def _base_guide(self) -> MultiAssetBlockGuideV3L1UnifiedOnlineFiltering:
        return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(
            config=self.config.base
        )


@register_guide("multi_asset_block_guide_v3_l10b_clean_unified_online_filtering")
def build_multi_asset_block_guide_v3_l10b_clean_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v3_l10b_clean(), params)
    return MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(
    params: Mapping[str, Any]
) -> V3L10BCleanUnifiedGuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V3L10BCleanUnifiedGuideConfig()
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
    }
    if extra:
        raise ConfigError(
            "Unknown multi_asset_block_guide_v3_l10b_clean_unified_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: value
        for key, value in values.items()
        if key != "index_t_copula_enabled"
    }
    return V3L10BCleanUnifiedGuideConfig(
        base=_build_base_guide_config(base_payload),
        index_t_copula_enabled=bool(values.get("index_t_copula_enabled", True)),
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _sample_index_t_copula_mix_sites(
    *,
    time_count: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    if not enabled:
        return
    broad_concentration = pyro.param(
        "index_t_copula_broad_mix_concentration",
        torch.full(
            (time_count,),
            _INDEX_T_COPULA_BROAD_DF_INIT / 2.0,
            device=device,
            dtype=dtype,
        ),
        constraint=constraints.positive,
    )
    broad_rate = pyro.param(
        "index_t_copula_broad_mix_rate",
        torch.full(
            (time_count,),
            _INDEX_T_COPULA_BROAD_DF_INIT / 2.0,
            device=device,
            dtype=dtype,
        ),
        constraint=constraints.positive,
    )
    pyro.sample(
        "index_t_copula_broad_mix",
        dist.Gamma(broad_concentration, broad_rate).to_event(1),
    )
    us_diff_concentration = pyro.param(
        "index_t_copula_us_diff_mix_concentration",
        torch.full(
            (time_count,),
            _INDEX_T_COPULA_US_DIFF_DF_INIT / 2.0,
            device=device,
            dtype=dtype,
        ),
        constraint=constraints.positive,
    )
    us_diff_rate = pyro.param(
        "index_t_copula_us_diff_mix_rate",
        torch.full(
            (time_count,),
            _INDEX_T_COPULA_US_DIFF_DF_INIT / 2.0,
            device=device,
            dtype=dtype,
        ),
        constraint=constraints.positive,
    )
    pyro.sample(
        "index_t_copula_us_diff_mix",
        dist.Gamma(us_diff_concentration, us_diff_rate).to_event(1),
    )


__all__ = [
    "MultiAssetBlockGuideV3L10BCleanUnifiedOnlineFiltering",
    "V3L10BCleanUnifiedGuideConfig",
    "build_multi_asset_block_guide_v3_l10b_clean_unified_online_filtering",
]
