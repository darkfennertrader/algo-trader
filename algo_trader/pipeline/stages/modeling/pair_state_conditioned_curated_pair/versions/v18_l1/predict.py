from __future__ import annotations

from typing import Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    DependenceFollowupPredictor,
    build_dependence_followup_predictor,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor


@register_predictor(
    "pair_state_conditioned_curated_pair_predict_v18_l1_online_filtering"
)
def build_pair_state_conditioned_curated_pair_predict_v18_l1_online_filtering(
    params: Mapping[str, object]
) -> DependenceFollowupPredictor:
    return build_dependence_followup_predictor(
        params=params,
        label="pair_state_conditioned_curated_pair_predict_v18_l1_online_filtering",
    )


__all__ = [
    "build_pair_state_conditioned_curated_pair_predict_v18_l1_online_filtering",
]
