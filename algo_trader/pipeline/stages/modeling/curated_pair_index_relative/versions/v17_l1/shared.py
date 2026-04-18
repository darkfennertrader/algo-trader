from __future__ import annotations

import torch

from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    IndexRelativeMeasurementConfig,
    PairSpec,
    SeedEntry,
    build_custom_relative_config,
    build_pair_seed_entries,
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


def _build_seed_entries(
    index_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    pair_specs = tuple(
        spec
        for spec in _CURATED_PAIR_SPECS
        if _pair_is_present(index_names, spec)
    )
    if not pair_specs:
        pair_specs = _fallback_pair_specs(index_names)
    return build_pair_seed_entries(
        index_names=index_names,
        pair_specs=pair_specs,
        device=device,
        dtype=dtype,
    )


def _pair_is_present(
    index_names: tuple[str, ...],
    pair_spec: PairSpec,
) -> bool:
    _, long_assets, short_assets = pair_spec
    present = frozenset(index_names)
    required = frozenset(long_assets + short_assets)
    return required.issubset(present)


def _fallback_pair_specs(index_names: tuple[str, ...]) -> tuple[PairSpec, ...]:
    if len(index_names) < 2:
        return ()
    long_name, short_name = index_names[0], index_names[1]
    return (
        (
            f"curated_pair_{long_name.lower()}_vs_{short_name.lower()}",
            (long_name,),
            (short_name,),
        ),
    )


build_curated_pair_index_relative_coordinates = make_coordinate_builder(
    _build_seed_entries
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
