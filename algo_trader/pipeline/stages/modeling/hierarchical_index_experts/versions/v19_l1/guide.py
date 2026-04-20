from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    build_dependence_followup_guide,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v19_l1


class HierarchicalIndexExpertsGuideV19L1OnlineFiltering(
    DependenceLayerGuideV4L1OnlineFiltering
):
    pass


@register_guide("hierarchical_index_experts_guide_v19_l1_online_filtering")
def build_hierarchical_index_experts_guide_v19_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_dependence_followup_guide(
        params=params,
        defaults=guide_default_params_v19_l1(),
        guide_type=HierarchicalIndexExpertsGuideV19L1OnlineFiltering,
    )


__all__ = [
    "HierarchicalIndexExpertsGuideV19L1OnlineFiltering",
    "build_hierarchical_index_experts_guide_v19_l1_online_filtering",
]
