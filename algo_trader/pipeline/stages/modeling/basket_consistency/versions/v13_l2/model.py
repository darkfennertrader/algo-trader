from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    BasketConsistencyModelV13L1OnlineFiltering,
    V13L1ModelPriors,
    _build_model_priors,
    predict_basket_consistency_v13_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .defaults import merge_nested_params, model_default_params_v13_l2


@register_model("basket_consistency_model_v13_l2_online_filtering")
def build_basket_consistency_model_v13_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v13_l2(), params)
    return BasketConsistencyModelV13L1OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


__all__ = [
    "BasketConsistencyModelV13L1OnlineFiltering",
    "V13L1ModelPriors",
    "build_basket_consistency_model_v13_l2_online_filtering",
    "predict_basket_consistency_v13_l1",
]
