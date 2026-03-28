from __future__ import annotations

from copy import deepcopy
from typing import Any

from .v3_l2_defaults import build_guide_defaults, merge_nested_params
from .v3_l3_defaults import model_default_params_v3_l3


_FACTOR_COUNT_DEFAULTS_V3_L4 = {
    "global_factor_count": 1,
    "fx_broad_factor_count": 1,
    "fx_cross_factor_count": 1,
    "index_factor_count": 1,
    "index_static_factor_count": 0,
    "commodity_factor_count": 1,
}


def guide_default_params_v3_l4() -> dict[str, Any]:
    return {
        **build_guide_defaults(_FACTOR_COUNT_DEFAULTS_V3_L4),
        "index_group_enabled": True,
    }


def model_default_params_v3_l4() -> dict[str, Any]:
    defaults = deepcopy(model_default_params_v3_l3())
    factors = defaults["factors"]
    factors.update(_FACTOR_COUNT_DEFAULTS_V3_L4)
    factors["index_static_b_scale"] = 0.0
    factors["index_group_scale"] = 0.08
    return defaults


__all__ = [
    "guide_default_params_v3_l4",
    "merge_nested_params",
    "model_default_params_v3_l4",
]
