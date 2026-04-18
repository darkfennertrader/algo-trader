from __future__ import annotations

from typing import cast

import torch

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.shared import (
    BasketObservationGroup,
)
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)

from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    IndexCoordinateTransform,
    IndexRelativeMeasurementConfig,
    IndexRelativeMeasurementCoordinates,
    IndexRelativeMeasurementWeights,
    PairSpec,
    SeedEntry,
    build_custom_relative_config,
    build_pair_seed_entries,
    build_index_coordinate_transform,
    coordinate_scale_from_covariance,
    make_coordinate_builder,
    make_custom_group_builder,
    project_basket_covariance,
    project_basket_mean,
    standardize_index_coordinates,
    standardize_index_covariance,
)

_ANCHOR_INDEX = "IBUST100"
_US_COMPLEMENT = "IBUS30"
_EUROPEAN_PRIORITY = (
    "IBDE40",
    "IBFR40",
    "IBGB100",
    "IBES35",
    "IBNL25",
    "IBCH20",
)


def build_pairwise_index_relative_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    return build_custom_relative_config(
        raw,
        label="model.params.pairwise_index_relative",
        relative_weight_key="pairwise_obs_weight",
    )


def _build_seed_entries(
    index_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    return build_pair_seed_entries(
        index_names=index_names,
        pair_specs=_pairwise_specs(index_names),
        device=device,
        dtype=dtype,
    )


def _pairwise_specs(
    index_names: tuple[str, ...],
) -> tuple[PairSpec, ...]:
    present = frozenset(index_names)
    specs = []
    if _US_COMPLEMENT in present and _ANCHOR_INDEX in present:
        specs.append(
            (
                "pairwise_ibus30_vs_ibust100",
                (_US_COMPLEMENT,),
                (_ANCHOR_INDEX,),
            )
        )
    if _ANCHOR_INDEX in present:
        specs.extend(
            (
                f"pairwise_ibust100_vs_{name.lower()}",
                (_ANCHOR_INDEX,),
                (name,),
            )
            for name in _EUROPEAN_PRIORITY
            if name in present
        )
    if specs:
        return tuple(specs)
    anchor = index_names[0]
    return tuple(
        (
            f"pairwise_{anchor.lower()}_vs_{name.lower()}",
            (anchor,),
            (name,),
        )
        for name in index_names[1:]
    )


build_pairwise_index_relative_coordinates = make_coordinate_builder(_build_seed_entries)
build_pairwise_index_relative_observation_groups = make_custom_group_builder(
    relative_group_name="pairwise_index_relative_obs",
    relative_names=lambda coordinate_names: frozenset(
        name for name in coordinate_names if name.startswith("pairwise_")
    ),
    residual_group_name="pairwise_index_residual_obs",
)


__all__ = [
    "BasketObservationGroup",
    "IndexCoordinateTransform",
    "IndexRelativeMeasurementConfig",
    "IndexRelativeMeasurementCoordinates",
    "IndexRelativeMeasurementWeights",
    "RuntimeAssetMetadata",
    "build_index_coordinate_transform",
    "build_pairwise_index_relative_config",
    "build_pairwise_index_relative_coordinates",
    "build_pairwise_index_relative_observation_groups",
    "coordinate_scale_from_covariance",
    "project_basket_covariance",
    "project_basket_mean",
    "standardize_index_coordinates",
    "standardize_index_covariance",
]
