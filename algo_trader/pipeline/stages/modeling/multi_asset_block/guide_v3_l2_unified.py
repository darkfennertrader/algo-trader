from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.protocols import PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_v3_l1_unified import (
    MultiAssetBlockGuideV3L1UnifiedOnlineFiltering,
    _build_guide_config,
)
from .v3_l2_defaults import guide_default_params_v3_l2, merge_nested_params


@register_guide("multi_asset_block_guide_v3_l2_unified_online_filtering")
def build_multi_asset_block_guide_v3_l2_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    merged_params = merge_nested_params(guide_default_params_v3_l2(), params)
    return MultiAssetBlockGuideV3L1UnifiedOnlineFiltering(
        config=_build_guide_config(merged_params)
    )
