from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    IndexRelativeFollowupModelBuildSpec,
    RawPlusAuxiliaryIndexRelativeRuntime,
    build_index_relative_followup_model,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.model_runtime import (
    IndexRelativeMeasurementModelPriors,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .defaults import model_default_params_v15_l1
from .shared import (
    build_index_relative_measurement_coordinates,
    build_index_relative_observation_groups,
    build_multi_output_index_relative_config,
)

V15L1ModelPriors = IndexRelativeMeasurementModelPriors


class MultiOutputIndexRelativeModelV15L1OnlineFiltering(
    RawPlusAuxiliaryIndexRelativeRuntime
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

_MODEL_BUILD_SPEC = IndexRelativeFollowupModelBuildSpec(
    defaults=model_default_params_v15_l1,
    runtime_type=MultiOutputIndexRelativeModelV15L1OnlineFiltering,
    config_builder=build_multi_output_index_relative_config,
    label="multi_output_index_relative_model_v15_l1_online_filtering",
    param_key="multi_output_index_relative",
)


@register_model("multi_output_index_relative_model_v15_l1_online_filtering")
def build_multi_output_index_relative_model_v15_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    return build_index_relative_followup_model(params=params, spec=_MODEL_BUILD_SPEC)


__all__ = [
    "MultiOutputIndexRelativeModelV15L1OnlineFiltering",
    "V15L1ModelPriors",
    "build_multi_output_index_relative_model_v15_l1_online_filtering",
]
