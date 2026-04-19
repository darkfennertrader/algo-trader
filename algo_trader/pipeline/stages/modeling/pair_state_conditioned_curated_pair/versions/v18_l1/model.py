from __future__ import annotations

from typing import Any, Mapping, cast

from algo_trader.pipeline.stages.modeling.dependence_followup_support import (
    IndexRelativeFollowupModelBuildSpec,
    RawPlusAuxiliaryIndexRelativeRuntime,
    build_index_relative_followup_model,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.model_runtime import (
    IndexRelativeGroupSiteOptions,
    IndexRelativeMeasurementModelPriors,
    sample_index_relative_group_site,
    standardize_runtime_index_relative_group_inputs,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .defaults import model_default_params_v18_l1
from .shared import (
    PairStateConditionedCuratedPairConfig,
    build_pair_state_conditioned_curated_pair_config,
    build_pair_state_conditioned_curated_pair_coordinates,
    build_pair_state_conditioned_curated_pair_observation_groups,
    build_pair_state_conditioned_range_mask,
    state_window_from_config,
)

V18L1ModelPriors = IndexRelativeMeasurementModelPriors


class PairStateConditionedCuratedPairModelV18L1OnlineFiltering(
    RawPlusAuxiliaryIndexRelativeRuntime
):
    def __init__(
        self,
        priors: V18L1ModelPriors | None = None,
    ) -> None:
        super().__init__(
            priors=priors or V18L1ModelPriors(),
            coordinate_builder=build_pair_state_conditioned_curated_pair_coordinates,
            group_builder=build_pair_state_conditioned_curated_pair_observation_groups,
        )

    def _sample_auxiliary_observations(self, inputs: Any) -> None:
        observed = inputs.runtime_batch.observations.y_obs
        if observed is None or self.group_builder is None:
            return
        config = cast(
            PairStateConditionedCuratedPairConfig,
            self.priors.index_relative_measurement,
        )
        groups = self.group_builder(
            config=self.priors.index_relative_measurement,
            coordinate_names=inputs.coordinates.coordinate_names,
            device=inputs.device,
        )
        standardized = standardize_runtime_index_relative_group_inputs(
            observed=observed,
            runtime_inputs=inputs,
            overlay=config,
        )
        range_mask = build_pair_state_conditioned_range_mask(
            observed=observed,
            coordinates=inputs.coordinates,
            time_mask=inputs.runtime_batch.observations.time_mask,
            state_window=state_window_from_config(config),
        )
        for group in groups:
            if group.name == "pair_state_conditioned_curated_pair_obs":
                sample_index_relative_group_site(
                    group=group,
                    inputs=standardized,
                    options=IndexRelativeGroupSiteOptions(
                        name="pair_state_conditioned_curated_pair_range_obs",
                        obs_weight=config.weights.relative_obs_weight,
                        time_mask=range_mask,
                    ),
                )
                continue
            sample_index_relative_group_site(
                group=group,
                inputs=standardized,
                options=IndexRelativeGroupSiteOptions(),
            )


_MODEL_BUILD_SPEC = IndexRelativeFollowupModelBuildSpec(
    defaults=model_default_params_v18_l1,
    runtime_type=PairStateConditionedCuratedPairModelV18L1OnlineFiltering,
    config_builder=build_pair_state_conditioned_curated_pair_config,
    label="pair_state_conditioned_curated_pair_model_v18_l1_online_filtering",
    param_key="pair_state_conditioned_curated_pair",
)


@register_model("pair_state_conditioned_curated_pair_model_v18_l1_online_filtering")
def build_pair_state_conditioned_curated_pair_model_v18_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    return build_index_relative_followup_model(params=params, spec=_MODEL_BUILD_SPEC)


__all__ = [
    "PairStateConditionedCuratedPairModelV18L1OnlineFiltering",
    "V18L1ModelPriors",
    "build_pair_state_conditioned_curated_pair_model_v18_l1_online_filtering",
]
