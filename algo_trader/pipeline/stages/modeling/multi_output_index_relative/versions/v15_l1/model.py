from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.model_runtime import (
    _RuntimeObservationInputs,
    IndexRelativeMeasurementModelPriors,
    IndexRelativeMeasurementModelRuntime,
    build_index_relative_measurement_model_priors,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .defaults import merge_nested_params, model_default_params_v15_l1
from .shared import (
    build_index_relative_measurement_coordinates,
    build_index_relative_observation_groups,
    build_multi_output_index_relative_config,
)

V15L1ModelPriors = IndexRelativeMeasurementModelPriors


class MultiOutputIndexRelativeModelV15L1OnlineFiltering(
    IndexRelativeMeasurementModelRuntime
):
    def __init__(
        self,
        priors: V15L1ModelPriors | None = None,
    ) -> None:
        super().__init__(
            priors=priors or V15L1ModelPriors(),
            coordinate_builder=build_index_relative_measurement_coordinates,
            group_builder=build_index_relative_observation_groups,
        )

    def _sample_observations(
        self,
        *,
        inputs: _RuntimeObservationInputs,
    ) -> None:
        self._sample_full_raw_head(inputs)
        if (
            not self.priors.index_relative_measurement.enabled
            or inputs.coordinates.coordinate_count == 0
        ):
            return
        self._sample_auxiliary_observations(inputs)


@register_model("multi_output_index_relative_model_v15_l1_online_filtering")
def build_multi_output_index_relative_model_v15_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v15_l1(), params)
    return MultiOutputIndexRelativeModelV15L1OnlineFiltering(
        priors=build_index_relative_measurement_model_priors(
            params=merged_params,
            config_builder=build_multi_output_index_relative_config,
            label="multi_output_index_relative_model_v15_l1_online_filtering",
            param_key="multi_output_index_relative",
        )
    )


__all__ = [
    "MultiOutputIndexRelativeModelV15L1OnlineFiltering",
    "V15L1ModelPriors",
    "build_multi_output_index_relative_model_v15_l1_online_filtering",
]
