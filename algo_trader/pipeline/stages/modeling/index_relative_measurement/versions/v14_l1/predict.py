from __future__ import annotations

from typing import Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.model import (
    DependenceLayerModelV4L1OnlineFiltering,
)
from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.predict import (
    predict_dependence_layer_v4_l1,
)
from algo_trader.pipeline.stages.modeling.protocols import PredictiveRequest
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor


class _V14L1Predictor:
    def __call__(self, request: PredictiveRequest) -> Mapping[str, object] | None:
        return predict_dependence_layer_v4_l1(
            model=cast(DependenceLayerModelV4L1OnlineFiltering, request.model),
            guide=cast(DependenceLayerGuideV4L1OnlineFiltering, request.guide),
            batch=request.batch,
            num_samples=request.num_samples,
            state=request.state,
        )


@register_predictor("index_relative_measurement_predict_v14_l1_online_filtering")
def build_index_relative_measurement_predict_v14_l1_online_filtering(
    params: Mapping[str, object]
) -> _V14L1Predictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown index_relative_measurement_predict_v14_l1_online_filtering "
            f"params: {unknown}"
        )
    return _V14L1Predictor()


__all__ = [
    "build_index_relative_measurement_predict_v14_l1_online_filtering",
    "predict_dependence_layer_v4_l1",
]
