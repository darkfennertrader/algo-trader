from __future__ import annotations
# pylint: disable=duplicate-code

from copy import deepcopy
from typing import Any

from .v3_l2_defaults import build_guide_defaults, merge_nested_params
from .v3_l6_defaults import model_default_params_v3_l6


_FACTOR_COUNT_DEFAULTS_V3_L8 = {
    "global_factor_count": 1,
    "fx_broad_factor_count": 1,
    "fx_cross_factor_count": 1,
    "index_factor_count": 1,
    "index_static_factor_count": 1,
    "commodity_factor_count": 1,
}


def guide_default_params_v3_l8() -> dict[str, Any]:
    return {
        **build_guide_defaults(_FACTOR_COUNT_DEFAULTS_V3_L8),
        "phi_index_region": 0.985,
    }


def model_default_params_v3_l8() -> dict[str, Any]:
    defaults = deepcopy(model_default_params_v3_l6())
    defaults["factors"].update(_FACTOR_COUNT_DEFAULTS_V3_L8)
    defaults["factors"]["index_static_b_scale"] = 0.02
    defaults["regime"].pop("index_spread", None)
    defaults["regime"]["index_region"] = {
        "phi": 0.985,
        "s_u_scale": 0.01,
        "df": 6.0,
    }
    return defaults


__all__ = [
    "guide_default_params_v3_l8",
    "merge_nested_params",
    "model_default_params_v3_l8",
]
