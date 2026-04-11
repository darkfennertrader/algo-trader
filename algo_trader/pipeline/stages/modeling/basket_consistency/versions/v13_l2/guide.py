from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide import (
    BasketConsistencyGuideV13L1OnlineFiltering,
    V13L1GuideConfig,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v13_l2, merge_nested_params


@register_guide("basket_consistency_guide_v13_l2_online_filtering")
def build_basket_consistency_guide_v13_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v13_l2(), params)
    return BasketConsistencyGuideV13L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


__all__ = [
    "BasketConsistencyGuideV13L1OnlineFiltering",
    "V13L1GuideConfig",
    "build_basket_consistency_guide_v13_l2_online_filtering",
]
