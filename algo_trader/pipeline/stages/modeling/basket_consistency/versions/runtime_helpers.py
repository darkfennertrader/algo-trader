from __future__ import annotations

from typing import Any, Callable, Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide import (
    BasketConsistencyGuideV13L1OnlineFiltering,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    predict_basket_consistency_v13_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest, PyroGuide


class BasketConsistencyFollowupPredictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_basket_consistency_v13_l1(
            model=cast(Any, request.model),
            guide=cast(BasketConsistencyGuideV13L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


def build_followup_guide(
    *,
    params: Mapping[str, Any],
    guide_defaults: Callable[[], dict[str, Any]],
    guide_factory: Callable[[Any], PyroGuide],
    merge_params: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
) -> PyroGuide:
    merged_params = merge_params(guide_defaults(), params)
    return guide_factory(_build_guide_config(merged_params))


def build_followup_predictor(
    *,
    params: Mapping[str, Any],
    predictor_name: str,
) -> BasketConsistencyFollowupPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(f"Unknown {predictor_name} params: {unknown}")
    return BasketConsistencyFollowupPredictor()


__all__ = [
    "BasketConsistencyFollowupPredictor",
    "build_followup_guide",
    "build_followup_predictor",
]
