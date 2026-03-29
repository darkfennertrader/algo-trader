from __future__ import annotations
# pylint: disable=duplicate-code

from copy import deepcopy
from typing import Any

from .v3_l6_defaults import (
    guide_default_params_v3_l6,
    merge_nested_params,
    model_default_params_v3_l6,
)


def guide_default_params_v3_l10() -> dict[str, Any]:
    return deepcopy(guide_default_params_v3_l6())


def model_default_params_v3_l10() -> dict[str, Any]:
    defaults = deepcopy(model_default_params_v3_l6())
    defaults["index_flow"] = {
        "enabled": True,
        "hidden_dim": 16,
        "log_scale_min_clip": -0.4,
        "log_scale_max_clip": 0.4,
    }
    return defaults


__all__ = [
    "guide_default_params_v3_l10",
    "merge_nested_params",
    "model_default_params_v3_l10",
]
