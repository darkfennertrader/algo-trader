from __future__ import annotations

from typing import Any

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.defaults import (
    guide_default_params_v13_l1,
    merge_nested_params,
    model_default_params_v13_l1,
)


def guide_default_params_v13_l3() -> dict[str, Any]:
    return guide_default_params_v13_l1()


def model_default_params_v13_l3() -> dict[str, Any]:
    return merge_nested_params(
        model_default_params_v13_l1(),
        {
            "basket_consistency": {
                "level_obs_weight": 0.08,
                "spread_obs_weight": 0.16,
            }
        },
    )


__all__ = [
    "guide_default_params_v13_l3",
    "merge_nested_params",
    "model_default_params_v13_l3",
]
