from __future__ import annotations

from typing import Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    DependenceFollowupPredictor,
    build_dependence_followup_predictor,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor


@register_predictor("hierarchical_index_experts_predict_v19_l1_online_filtering")
def build_hierarchical_index_experts_predict_v19_l1_online_filtering(
    params: Mapping[str, object]
) -> DependenceFollowupPredictor:
    return build_dependence_followup_predictor(
        params=params,
        label="hierarchical_index_experts_predict_v19_l1_online_filtering",
    )


__all__ = [
    "build_hierarchical_index_experts_predict_v19_l1_online_filtering",
]
