from __future__ import annotations
# pylint: disable=duplicate-code

from typing import Any, Mapping

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide
from algo_trader.pipeline.stages.modeling.state_conditioned_measurement.versions.v10_l1.guide import (
    StateConditionedMeasurementGuideV10L1OnlineFiltering,
    V10L1GuideConfig,
    _build_base_guide_config,
)

from .defaults import guide_default_params_v10_l2, merge_nested_params

V10L2GuideConfig = V10L1GuideConfig
StateConditionedMeasurementGuideV10L2OnlineFiltering = (
    StateConditionedMeasurementGuideV10L1OnlineFiltering
)


@register_guide("state_conditioned_measurement_guide_v10_l2_online_filtering")
def build_state_conditioned_measurement_guide_v10_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v10_l2(), params)
    return StateConditionedMeasurementGuideV10L2OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


def _build_guide_config(params: Mapping[str, Any]) -> V10L2GuideConfig:
    values = _coerce_mapping(params, label="model.guide_params")
    if not values:
        return V10L2GuideConfig()
    extra = set(values) - {
        "global_factor_count",
        "fx_broad_factor_count",
        "fx_cross_factor_count",
        "index_factor_count",
        "index_static_factor_count",
        "commodity_factor_count",
        "phi_fx_broad",
        "phi_fx_cross",
        "phi_index",
        "phi_commodity",
        "index_group_enabled",
        "state_conditioned_measurement_enabled",
    }
    if extra:
        raise ConfigError(
            "Unknown state_conditioned_measurement_guide_v10_l2_online_filtering params",
            context={"params": ", ".join(sorted(extra))},
        )
    return V10L2GuideConfig(
        base=_build_base_guide_config(
            {
                key: value
                for key, value in values.items()
                if key != "state_conditioned_measurement_enabled"
            }
        ),
        state_conditioned_measurement_enabled=bool(
            values.get("state_conditioned_measurement_enabled", True)
        ),
    )


def _coerce_mapping(raw: object, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = [
    "StateConditionedMeasurementGuideV10L2OnlineFiltering",
    "V10L2GuideConfig",
    "build_state_conditioned_measurement_guide_v10_l2_online_filtering",
]
