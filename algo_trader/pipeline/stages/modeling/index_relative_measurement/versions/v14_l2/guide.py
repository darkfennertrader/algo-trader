from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.guide import (
    DependenceLayerGuideV4L1OnlineFiltering,
    V4L1GuideConfig,
)
from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from ..runtime_helpers import build_index_relative_measurement_guide
from .defaults import guide_default_params_v14_l2, merge_nested_params


class IndexRelativeMeasurementGuideV14L2OnlineFiltering(
    DependenceLayerGuideV4L1OnlineFiltering
):
    pass


@register_guide("index_relative_measurement_guide_v14_l2_online_filtering")
def build_index_relative_measurement_guide_v14_l2_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return build_index_relative_measurement_guide(
        params=params,
        defaults=guide_default_params_v14_l2(),
        guide_type=IndexRelativeMeasurementGuideV14L2OnlineFiltering,
    )


__all__ = [
    "IndexRelativeMeasurementGuideV14L2OnlineFiltering",
    "V4L1GuideConfig",
    "build_index_relative_measurement_guide_v14_l2_online_filtering",
    "merge_nested_params",
]
