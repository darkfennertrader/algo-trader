from __future__ import annotations

from typing import Any

from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.defaults import (
    guide_default_params_v4_l1,
    merge_nested_params,
    model_default_params_v4_l1,
)

_AUXILIARY_DEFAULTS = {
    "enabled": True,
    "df": 8.0,
    "obs_weight": 0.04,
    "level_obs_weight": 0.015,
    "relative_obs_weight": 0.05,
    "residual_obs_weight": 0.015,
    "mad_floor": 1e-4,
    "eps": 1e-6,
}


def guide_default_params_v15_l1() -> dict[str, Any]:
    return guide_default_params_v4_l1()


def model_default_params_v15_l1() -> dict[str, Any]:
    return merge_nested_params(
        model_default_params_v4_l1(),
        {"multi_output_index_relative": dict(_AUXILIARY_DEFAULTS)},
    )


__all__ = [
    "guide_default_params_v15_l1",
    "merge_nested_params",
    "model_default_params_v15_l1",
]
