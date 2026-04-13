from __future__ import annotations

from typing import Mapping

from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from ..runtime_helpers import (
    IndexRelativeMeasurementPredictor,
    build_index_relative_measurement_predictor,
)


@register_predictor("index_relative_measurement_predict_v14_l2_online_filtering")
def build_index_relative_measurement_predict_v14_l2_online_filtering(
    params: Mapping[str, object]
) -> IndexRelativeMeasurementPredictor:
    return build_index_relative_measurement_predictor(
        params=params,
        label="index_relative_measurement_predict_v14_l2_online_filtering",
    )


__all__ = [
    "build_index_relative_measurement_predict_v14_l2_online_filtering",
]
