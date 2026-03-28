from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.protocols import PyroModel
from algo_trader.pipeline.stages.modeling.registry_core import register_model

from .model_v3_l1_unified import (
    MultiAssetBlockModelV3L1UnifiedOnlineFiltering,
    _build_model_priors,
)
from .v3_l2_defaults import merge_nested_params, model_default_params_v3_l2


@register_model("multi_asset_block_model_v3_l2_unified_online_filtering")
def build_multi_asset_block_model_v3_l2_unified_online_filtering(
    params: Mapping[str, Any]
) -> PyroModel:
    merged_params = merge_nested_params(model_default_params_v3_l2(), params)
    return MultiAssetBlockModelV3L1UnifiedOnlineFiltering(
        priors=_build_model_priors(merged_params)
    )
