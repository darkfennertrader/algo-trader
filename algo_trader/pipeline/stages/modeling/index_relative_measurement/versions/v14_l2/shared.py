from __future__ import annotations

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)
from algo_trader.pipeline.stages.modeling.vector_support import (
    VectorBuildConfig,
    equal_weight_vector,
)

from ..shared_common import (
    BasketObservationGroup,
    IndexCoordinateTransform,
    IndexRelativeMeasurementConfig,
    IndexRelativeMeasurementCoordinates,
    IndexRelativeMeasurementWeights,
    SeedEntry,
    build_index_coordinate_transform,
    build_index_relative_measurement_config,
    coordinate_scale_from_covariance,
    make_coordinate_builder,
    make_group_builder,
    project_basket_covariance,
    project_basket_mean,
    standardize_index_coordinates,
    standardize_index_covariance,
)

_LEVEL_NAMES = frozenset(("index_level",))


def _build_seed_entries(
    index_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    vector_config = VectorBuildConfig(device=device, dtype=dtype, strict=False)
    level = equal_weight_vector(index_names, index_names, config=vector_config)
    entries: list[SeedEntry] = [("index_level", level)]
    for name in index_names:
        asset_level = equal_weight_vector(
            index_names,
            (name,),
            config=vector_config,
        )
        relative = asset_level - level
        if bool(relative.abs().sum() > 0.0):
            entries.append((f"index_relative_{name}", relative))
    return tuple(entries)


def _relative_names(coordinate_names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        name for name in coordinate_names if name.startswith("index_relative_")
    )


build_index_relative_measurement_coordinates = make_coordinate_builder(
    _build_seed_entries
)
build_index_relative_observation_groups = make_group_builder(
    level_names=_LEVEL_NAMES,
    relative_names=_relative_names,
)
