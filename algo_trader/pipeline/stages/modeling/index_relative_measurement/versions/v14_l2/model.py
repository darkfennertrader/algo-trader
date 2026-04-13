from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from ..model_runtime import (
    IndexRelativeMeasurementModelPriors,
    IndexRelativeMeasurementModelRuntime,
    build_index_relative_measurement_model_priors,
)
from ..runtime_helpers import (
    predict_index_relative_measurement_runtime as predict_index_relative_measurement_v14_l2,
)
from .defaults import merge_nested_params, model_default_params_v14_l2
from .shared import (
    build_index_relative_measurement_config,
    build_index_relative_measurement_coordinates,
    build_index_relative_observation_groups,
)

V14L2ModelPriors = IndexRelativeMeasurementModelPriors


class IndexRelativeMeasurementModelV14L2OnlineFiltering(
    IndexRelativeMeasurementModelRuntime
):
    def __init__(
        self,
        priors: V14L2ModelPriors | None = None,
    ) -> None:
        super().__init__(
            priors=priors or V14L2ModelPriors(),
            coordinate_builder=build_index_relative_measurement_coordinates,
            group_builder=build_index_relative_observation_groups,
        )


@register_model("index_relative_measurement_model_v14_l2_online_filtering")
def build_index_relative_measurement_model_v14_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v14_l2(), params)
    return IndexRelativeMeasurementModelV14L2OnlineFiltering(
        priors=build_index_relative_measurement_model_priors(
            params=merged_params,
            config_builder=build_index_relative_measurement_config,
            label="index_relative_measurement_model_v14_l2_online_filtering",
        )
    )


__all__ = [
    "IndexRelativeMeasurementModelV14L2OnlineFiltering",
    "V14L2ModelPriors",
    "build_index_relative_measurement_model_v14_l2_online_filtering",
    "predict_index_relative_measurement_v14_l2",
]
