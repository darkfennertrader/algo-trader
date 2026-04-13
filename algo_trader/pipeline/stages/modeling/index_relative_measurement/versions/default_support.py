from __future__ import annotations

from typing import Any, Mapping

from algo_trader.pipeline.stages.modeling.dependence_layer.versions.v4_l1.defaults import (
    guide_default_params_v4_l1,
    merge_nested_params,
    model_default_params_v4_l1,
)


def guide_default_params_index_relative_measurement() -> dict[str, Any]:
    return guide_default_params_v4_l1()


def model_default_params_index_relative_measurement(
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    return merge_nested_params(
        model_default_params_v4_l1(),
        {"index_relative_measurement": dict(overrides)},
    )


__all__ = [
    "guide_default_params_index_relative_measurement",
    "merge_nested_params",
    "model_default_params_index_relative_measurement",
]
