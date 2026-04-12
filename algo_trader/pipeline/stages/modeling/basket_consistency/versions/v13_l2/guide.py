from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide import (
    BasketConsistencyGuideV13L1OnlineFiltering,
    V13L1GuideConfig,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v13_l2, merge_nested_params
from ..runtime_helpers import build_followup_guide


@register_guide("basket_consistency_guide_v13_l2_online_filtering")
def build_basket_consistency_guide_v13_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_followup_guide(
        params=params,
        guide_defaults=guide_default_params_v13_l2,
        guide_factory=BasketConsistencyGuideV13L1OnlineFiltering,
        merge_params=merge_nested_params,
    )


__all__ = [
    "BasketConsistencyGuideV13L1OnlineFiltering",
    "V13L1GuideConfig",
    "build_basket_consistency_guide_v13_l2_online_filtering",
]
