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

from .defaults import model_default_params_v19_l1
from .shared import (
    build_hierarchical_index_experts_config,
    build_hierarchical_index_experts_coordinates,
    build_hierarchical_index_experts_observation_groups,
)

V19L1ModelPriors = IndexRelativeMeasurementModelPriors


class HierarchicalIndexExpertsModelV19L1OnlineFiltering(
    RawPlusAuxiliaryIndexRelativeRuntime
):
    def __init__(
        self,
        priors: V19L1ModelPriors | None = None,
    ) -> None:
        super().__init__(
            priors=priors or V19L1ModelPriors(),
            coordinate_builder=build_hierarchical_index_experts_coordinates,
            group_builder=build_hierarchical_index_experts_observation_groups,
        )


_MODEL_BUILD_SPEC = IndexRelativeFollowupModelBuildSpec(
    defaults=model_default_params_v19_l1,
    runtime_type=HierarchicalIndexExpertsModelV19L1OnlineFiltering,
    config_builder=build_hierarchical_index_experts_config,
    label="hierarchical_index_experts_model_v19_l1_online_filtering",
    param_key="hierarchical_index_experts",
)


@register_model("hierarchical_index_experts_model_v19_l1_online_filtering")
def build_hierarchical_index_experts_model_v19_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    return build_index_relative_followup_model(params=params, spec=_MODEL_BUILD_SPEC)


__all__ = [
    "HierarchicalIndexExpertsModelV19L1OnlineFiltering",
    "V19L1ModelPriors",
    "build_hierarchical_index_experts_model_v19_l1_online_filtering",
]
