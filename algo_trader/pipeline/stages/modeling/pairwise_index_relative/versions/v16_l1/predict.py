from __future__ import annotations

from typing import Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    DependenceFollowupPredictor,
    build_dependence_followup_predictor,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor


@register_predictor("pairwise_index_relative_predict_v16_l1_online_filtering")
def build_pairwise_index_relative_predict_v16_l1_online_filtering(
    params: Mapping[str, object]
) -> DependenceFollowupPredictor:
    return build_dependence_followup_predictor(
        params=params,
        label="pairwise_index_relative_predict_v16_l1_online_filtering",
    )


__all__ = [
    "build_pairwise_index_relative_predict_v16_l1_online_filtering",
]
