from __future__ import annotations

from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    IndexRelativeMeasurementConfig,
    build_index_relative_config,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.v14_l1.shared import (
    build_index_relative_measurement_coordinates,
    build_index_relative_observation_groups,
)


def build_multi_output_index_relative_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    return build_index_relative_config(
        raw,
        label="model.params.multi_output_index_relative",
    )


__all__ = [
    "IndexRelativeMeasurementConfig",
    "build_index_relative_measurement_coordinates",
    "build_index_relative_observation_groups",
    "build_multi_output_index_relative_config",
]
