from __future__ import annotations

from typing import Any, Mapping

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.protocols import PyroPredictor
from algo_trader.pipeline.stages.modeling.registry_core import register_predictor

from .predict_v3_l1_unified import _V3L1UnifiedPredictor


@register_predictor("multi_asset_block_predict_v3_l4_unified_online_filtering")
def build_multi_asset_block_predict_v3_l4_unified_online_filtering(
    params: Mapping[str, Any],
) -> PyroPredictor:
    if params:
        unknown = ", ".join(sorted(str(key) for key in params))
        raise ConfigError(
            "Unknown multi_asset_block_predict_v3_l4_unified_online_filtering "
            f"params: {unknown}"
        )
    return _V3L1UnifiedPredictor()
