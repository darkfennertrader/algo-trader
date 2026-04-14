from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    build_dependence_followup_guide,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
    V4L1GuideConfig,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v16_l1


class PairwiseIndexRelativeGuideV16L1OnlineFiltering(
    DependenceLayerGuideV4L1OnlineFiltering
):
    pass


@register_guide("pairwise_index_relative_guide_v16_l1_online_filtering")
def build_pairwise_index_relative_guide_v16_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_dependence_followup_guide(
        params=params,
        defaults=guide_default_params_v16_l1(),
        guide_type=PairwiseIndexRelativeGuideV16L1OnlineFiltering,
    )


__all__ = [
    "PairwiseIndexRelativeGuideV16L1OnlineFiltering",
    "V4L1GuideConfig",
    "build_pairwise_index_relative_guide_v16_l1_online_filtering",
]
