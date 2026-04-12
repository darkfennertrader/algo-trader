from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    predict_basket_consistency_v13_l1,
)
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from ..runtime_helpers import BasketConsistencyFollowupPredictor, build_followup_predictor


@register_predictor("basket_consistency_predict_v13_l3_online_filtering")
def build_basket_consistency_predict_v13_l3_online_filtering(
    params: Mapping[str, Any],
) -> BasketConsistencyFollowupPredictor:
    return build_followup_predictor(
        params=params,
        predictor_name="basket_consistency_predict_v13_l3_online_filtering",
    )


__all__ = [
    "build_basket_consistency_predict_v13_l3_online_filtering",
    "predict_basket_consistency_v13_l1",
]
