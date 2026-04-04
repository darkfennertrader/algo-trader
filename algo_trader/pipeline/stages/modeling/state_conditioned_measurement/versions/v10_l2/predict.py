from __future__ import annotations
# pylint: disable=duplicate-code

from typing import Any, Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor
from algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.guide import (
    StateConditionedMeasurementGuideV10L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.predict import (
    predict_state_conditioned_measurement_v10_l1,
)


class _V10L2Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, Any] | None:
        return predict_state_conditioned_measurement_v10_l1(
            model=cast(Any, request.model),
            guide=cast(StateConditionedMeasurementGuideV10L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("state_conditioned_measurement_predict_v10_l2_online_filtering")
def build_state_conditioned_measurement_predict_v10_l2_online_filtering(
    params: Mapping[str, Any],
) -> _V10L2Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown state_conditioned_measurement_predict_v10_l2_online_filtering "
            f"params: {unknown}"
        )
    return _V10L2Predictor()


__all__ = ["build_state_conditioned_measurement_predict_v10_l2_online_filtering"]
