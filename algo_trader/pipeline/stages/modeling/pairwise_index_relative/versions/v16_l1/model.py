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

from .defaults import model_default_params_v16_l1
from .shared import (
    build_pairwise_index_relative_config,
    build_pairwise_index_relative_coordinates,
    build_pairwise_index_relative_observation_groups,
)

V16L1ModelPriors = IndexRelativeMeasurementModelPriors


class PairwiseIndexRelativeModelV16L1OnlineFiltering(
    RawPlusAuxiliaryIndexRelativeRuntime
):
    def __init__(
        self,
        priors: V16L1ModelPriors | None = None,
    ) -> None:
        super().__init__(
            priors=priors or V16L1ModelPriors(),
            coordinate_builder=build_pairwise_index_relative_coordinates,
            group_builder=build_pairwise_index_relative_observation_groups,
        )

_MODEL_BUILD_SPEC = IndexRelativeFollowupModelBuildSpec(
    defaults=model_default_params_v16_l1,
    runtime_type=PairwiseIndexRelativeModelV16L1OnlineFiltering,
    config_builder=build_pairwise_index_relative_config,
    label="pairwise_index_relative_model_v16_l1_online_filtering",
    param_key="pairwise_index_relative",
)


@register_model("pairwise_index_relative_model_v16_l1_online_filtering")
def build_pairwise_index_relative_model_v16_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    return build_index_relative_followup_model(params=params, spec=_MODEL_BUILD_SPEC)


__all__ = [
    "PairwiseIndexRelativeModelV16L1OnlineFiltering",
    "V16L1ModelPriors",
    "build_pairwise_index_relative_model_v16_l1_online_filtering",
]
