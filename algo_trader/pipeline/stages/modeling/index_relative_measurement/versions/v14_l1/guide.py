from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
    V4L1GuideConfig,
    _build_guide_config,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .defaults import guide_default_params_v14_l1, merge_nested_params


class IndexRelativeMeasurementGuideV14L1OnlineFiltering(
    DependenceLayerGuideV4L1OnlineFiltering
):
    pass


@register_guide("index_relative_measurement_guide_v14_l1_online_filtering")
def build_index_relative_measurement_guide_v14_l1_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v14_l1(), params)
    return IndexRelativeMeasurementGuideV14L1OnlineFiltering(
        config=_build_guide_config(merged_params)
    )


__all__ = [
    "IndexRelativeMeasurementGuideV14L1OnlineFiltering",
    "V4L1GuideConfig",
    "build_index_relative_measurement_guide_v14_l1_online_filtering",
]
