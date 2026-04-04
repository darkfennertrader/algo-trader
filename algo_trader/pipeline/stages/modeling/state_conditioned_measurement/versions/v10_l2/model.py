from __future__ import annotations
# pylint: disable=duplicate-code

from typing import Any, Mapping

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model
from algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.model import (
    StateConditionedMeasurementModelV10L1OnlineFiltering,
    V10L1ModelPriors,
    _build_base_model_priors,
)

from .defaults import merge_nested_params, model_default_params_v10_l2
from .shared import build_state_conditioned_measurement_config

V10L2ModelPriors = V10L1ModelPriors
StateConditionedMeasurementModelV10L2OnlineFiltering = (
    StateConditionedMeasurementModelV10L1OnlineFiltering
)


@register_model("state_conditioned_measurement_model_v10_l2_online_filtering")
def build_state_conditioned_measurement_model_v10_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v10_l2(), params)
    return StateConditionedMeasurementModelV10L2OnlineFiltering(
        priors=_build_model_priors(merged_params)
    )


def _build_model_priors(params: Mapping[str, Any]) -> V10L2ModelPriors:
    values = _coerce_mapping(params, label="model.params")
    if not values:
        return V10L2ModelPriors()
    extra = set(values) - {
        "mean",
        "factors",
        "regime",
        "state_conditioned_measurement",
    }
    if extra:
        raise ConfigError(
            "Unknown state_conditioned_measurement_model_v10_l2_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    base_payload = {
        key: values[key] for key in ("mean", "factors", "regime") if key in values
    }
    return V10L2ModelPriors(
        base=_build_base_model_priors(base_payload),
        state_conditioned_measurement=build_state_conditioned_measurement_config(
            values.get("state_conditioned_measurement")
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "StateConditionedMeasurementModelV10L2OnlineFiltering",
    "V10L2ModelPriors",
    "build_state_conditioned_measurement_model_v10_l2_online_filtering",
]
