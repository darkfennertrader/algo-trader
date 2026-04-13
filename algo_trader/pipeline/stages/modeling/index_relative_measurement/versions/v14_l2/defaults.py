from __future__ import annotations

from typing import Any

from ..default_support import (
    guide_default_params_index_relative_measurement,
    merge_nested_params,
    model_default_params_index_relative_measurement,
)

_DEFAULTS = {
    "enabled": True,
    "df": 8.0,
    "obs_weight": 0.06,
    "level_obs_weight": 0.02,
    "relative_obs_weight": 0.14,
    "residual_obs_weight": 0.02,
    "mad_floor": 1e-4,
    "eps": 1e-6,
}


guide_default_params_v14_l2 = guide_default_params_index_relative_measurement


def model_default_params_v14_l2() -> dict[str, Any]:
    return model_default_params_index_relative_measurement(_DEFAULTS)


__all__ = [
    "guide_default_params_v14_l2",
    "merge_nested_params",
    "model_default_params_v14_l2",
]
