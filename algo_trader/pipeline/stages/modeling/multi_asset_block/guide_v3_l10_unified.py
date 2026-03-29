from __future__ import annotations
# pylint: disable=duplicate-code

from typing import Any, Mapping, cast

from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_v3_l6_unified import (
    MultiAssetBlockGuideV3L6UnifiedOnlineFiltering,
    build_multi_asset_block_guide_v3_l6_unified_online_filtering,
)
from .v3_l10_defaults import guide_default_params_v3_l10, merge_nested_params

MultiAssetBlockGuideV3L10UnifiedOnlineFiltering = (
    MultiAssetBlockGuideV3L6UnifiedOnlineFiltering
)


@register_guide("multi_asset_block_guide_v3_l10_unified_online_filtering")
def build_multi_asset_block_guide_v3_l10_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v3_l10(), params)
    return cast(
        PyroGuide,
        build_multi_asset_block_guide_v3_l6_unified_online_filtering(merged_params),
    )


__all__ = [
    "MultiAssetBlockGuideV3L10UnifiedOnlineFiltering",
    "build_multi_asset_block_guide_v3_l10_unified_online_filtering",
]
