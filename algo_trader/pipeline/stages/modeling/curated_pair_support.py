from __future__ import annotations

from collections.abc import Callable

import torch

from algo_trader.pipeline.stages.modeling.index_relative_measurement.versions.shared_common import (
    PairSpec,
    SeedEntry,
    build_pair_seed_entries,
)


def build_curated_pair_seed_entries(
    *,
    index_names: tuple[str, ...],
    pair_specs: tuple[PairSpec, ...],
    fallback_prefix: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    present_specs = tuple(
        spec for spec in pair_specs if pair_spec_is_present(index_names, spec)
    )
    if not present_specs:
        present_specs = fallback_pair_specs(
            index_names=index_names,
            fallback_prefix=fallback_prefix,
        )
    return build_pair_seed_entries(
        index_names=index_names,
        pair_specs=present_specs,
        device=device,
        dtype=dtype,
    )


def pair_spec_is_present(
    index_names: tuple[str, ...],
    pair_spec: PairSpec,
) -> bool:
    _, long_assets, short_assets = pair_spec
    present = frozenset(index_names)
    required = frozenset(long_assets + short_assets)
    return required.issubset(present)


def fallback_pair_specs(
    *,
    index_names: tuple[str, ...],
    fallback_prefix: str,
) -> tuple[PairSpec, ...]:
    if len(index_names) < 2:
        return ()
    long_name, short_name = index_names[0], index_names[1]
    return (
        (
            f"{fallback_prefix}_{long_name.lower()}_vs_{short_name.lower()}",
            (long_name,),
            (short_name,),
        ),
    )


def make_curated_pair_seed_builder(
    *,
    pair_specs: tuple[PairSpec, ...],
    fallback_prefix: str,
) -> Callable[[tuple[str, ...], torch.device, torch.dtype], tuple[SeedEntry, ...]]:
    def builder(
        index_names: tuple[str, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[SeedEntry, ...]:
        return build_curated_pair_seed_entries(
            index_names=index_names,
            pair_specs=pair_specs,
            fallback_prefix=fallback_prefix,
            device=device,
            dtype=dtype,
        )

    return builder


__all__ = [
    "build_curated_pair_seed_entries",
    "fallback_pair_specs",
    "make_curated_pair_seed_builder",
    "pair_spec_is_present",
]
