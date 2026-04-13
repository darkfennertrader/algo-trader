from __future__ import annotations

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)
from algo_trader.pipeline.stages.modeling.vector_support import (
    VectorBuildConfig,
    equal_weight_vector,
    spread_vector,
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

_US = ("IBUS30", "IBUS500", "IBUST100")
_EUROPE = ("IBDE40", "IBES35", "IBEU50", "IBFR40", "IBGB100", "IBNL25")
_LEVEL_NAMES = frozenset(("index_level",))
_RELATIVE_NAMES = frozenset(
    ("us_relative_index", "europe_relative_index", "us_minus_europe")
)


def _build_seed_entries(
    index_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    vector_config = VectorBuildConfig(device=device, dtype=dtype, strict=False)
    level = equal_weight_vector(index_names, index_names, config=vector_config)
    return (
        ("index_level", level),
        (
            "us_relative_index",
            equal_weight_vector(index_names, _US, config=vector_config) - level,
        ),
        (
            "europe_relative_index",
            equal_weight_vector(index_names, _EUROPE, config=vector_config) - level,
        ),
        (
            "us_minus_europe",
            spread_vector(
                index_names,
                _US,
                _EUROPE,
                config=vector_config,
            ),
        ),
    )


build_index_relative_measurement_coordinates = make_coordinate_builder(
    _build_seed_entries
)
build_index_relative_observation_groups = make_group_builder(
    level_names=_LEVEL_NAMES,
    relative_names=_RELATIVE_NAMES,
)
