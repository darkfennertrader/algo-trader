from __future__ import annotations
# pylint: disable=duplicate-code

from copy import deepcopy
from typing import Any

from .v3_l2_defaults import build_guide_defaults, merge_nested_params
from .v3_l4_defaults import model_default_params_v3_l4


_FACTOR_COUNT_DEFAULTS_V3_L5 = {
    "global_factor_count": 1,
    "fx_broad_factor_count": 1,
    "fx_cross_factor_count": 1,
    "index_factor_count": 1,
    "index_static_factor_count": 0,
    "commodity_factor_count": 1,
}


def guide_default_params_v3_l5() -> dict[str, Any]:
    return {
        **build_guide_defaults(_FACTOR_COUNT_DEFAULTS_V3_L5),
        "phi_index_group": 0.985,
    }


def model_default_params_v3_l5() -> dict[str, Any]:
    defaults = deepcopy(model_default_params_v3_l4())
    defaults["factors"].update(_FACTOR_COUNT_DEFAULTS_V3_L5)
    defaults["factors"]["index_static_b_scale"] = 0.0
    defaults["factors"]["index_group_scale"] = 0.08
    defaults["regime"]["index"] = {"phi": 0.97, "s_u_scale": 0.015}
    defaults["regime"]["index_group"] = {"phi": 0.985, "s_u_scale": 0.01}
    return defaults


__all__ = [
    "guide_default_params_v3_l5",
    "merge_nested_params",
    "model_default_params_v3_l5",
]
