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

from .defaults import guide_default_params_v18_l1


class PairStateConditionedCuratedPairGuideV18L1OnlineFiltering(
    DependenceLayerGuideV4L1OnlineFiltering
):
    pass


@register_guide(
    "pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering"
)
def build_pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_dependence_followup_guide(
        params=params,
        defaults=guide_default_params_v18_l1(),
        guide_type=PairStateConditionedCuratedPairGuideV18L1OnlineFiltering,
    )


__all__ = [
    "PairStateConditionedCuratedPairGuideV18L1OnlineFiltering",
    "build_pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering",
]
