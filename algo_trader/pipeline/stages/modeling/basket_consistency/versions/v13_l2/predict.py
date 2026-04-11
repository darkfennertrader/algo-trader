from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.guide import (
    BasketConsistencyGuideV13L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
    predict_basket_consistency_v13_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

if TYPE_CHECKING:
    from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.model import (
        BasketConsistencyModelV13L1OnlineFiltering,
    )


class _V13L2Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_basket_consistency_v13_l1(
            model=cast(Any, request.model),
            guide=cast(BasketConsistencyGuideV13L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("basket_consistency_predict_v13_l2_online_filtering")
def build_basket_consistency_predict_v13_l2_online_filtering(
    params: Mapping[str, Any],
) -> _V13L2Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown basket_consistency_predict_v13_l2_online_filtering "
            f"params: {unknown}"
        )
    return _V13L2Predictor()


__all__ = [
    "build_basket_consistency_predict_v13_l2_online_filtering",
    "predict_basket_consistency_v13_l1",
]
