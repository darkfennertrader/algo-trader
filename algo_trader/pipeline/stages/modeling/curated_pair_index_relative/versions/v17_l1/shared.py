from __future__ import annotations

from algo_trader.pipeline.stages.modeling.curated_pair_support import (
    make_curated_pair_seed_builder,
)
from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    IndexRelativeMeasurementConfig,
    PairSpec,
    build_custom_relative_config,
    make_coordinate_builder,
    make_custom_group_builder,
)

_CURATED_PAIR_SPECS: tuple[PairSpec, ...] = (
    (
        "curated_pair_ibch20_vs_ibde40",
        ("IBCH20",),
        ("IBDE40",),
    ),
    (
        "curated_pair_ibus30_vs_ibust100",
        ("IBUS30",),
        ("IBUST100",),
    ),
)


def build_curated_pair_index_relative_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    return build_custom_relative_config(
        raw,
        label="model.params.curated_pair_index_relative",
        relative_weight_key="curated_pair_obs_weight",
    )


build_curated_pair_index_relative_coordinates = make_coordinate_builder(
    make_curated_pair_seed_builder(
        pair_specs=_CURATED_PAIR_SPECS,
        fallback_prefix="curated_pair",
    )
)
build_curated_pair_index_relative_observation_groups = make_custom_group_builder(
    relative_group_name="curated_pair_index_relative_obs",
    relative_names=lambda coordinate_names: frozenset(
        name for name in coordinate_names if name.startswith("curated_pair_")
    ),
    residual_group_name="curated_pair_index_residual_obs",
)

__all__ = [
    "build_curated_pair_index_relative_config",
    "build_curated_pair_index_relative_coordinates",
    "build_curated_pair_index_relative_observation_groups",
]
