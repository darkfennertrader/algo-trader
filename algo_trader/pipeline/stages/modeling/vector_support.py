from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VectorBuildConfig:
    device: torch.device
    dtype: torch.dtype
    strict: bool


def resolve_requested_indices(
    asset_names: tuple[str, ...],
    requested: tuple[str, ...],
    *,
    strict: bool,
) -> tuple[int, ...]:
    index_by_name = {name: idx for idx, name in enumerate(asset_names)}
    if strict and any(name not in index_by_name for name in requested):
        return ()
    resolved = [index_by_name[name] for name in requested if name in index_by_name]
    return tuple(resolved)


def equal_weight_vector(
    asset_names: tuple[str, ...],
    requested: tuple[str, ...],
    *,
    config: VectorBuildConfig,
) -> torch.Tensor:
    indices = resolve_requested_indices(
        asset_names,
        requested,
        strict=config.strict,
    )
    vector = torch.zeros(
        (len(asset_names),),
        device=config.device,
        dtype=config.dtype,
    )
    if not indices:
        return vector
    weight = 1.0 / float(len(indices))
    for index in indices:
        vector[index] = weight
    return vector


def spread_vector(
    asset_names: tuple[str, ...],
    long_assets: tuple[str, ...],
    short_assets: tuple[str, ...],
    *,
    config: VectorBuildConfig,
) -> torch.Tensor:
    long_indices = resolve_requested_indices(
        asset_names,
        long_assets,
        strict=config.strict,
    )
    short_indices = resolve_requested_indices(
        asset_names,
        short_assets,
        strict=config.strict,
    )
    vector = torch.zeros(
        (len(asset_names),),
        device=config.device,
        dtype=config.dtype,
    )
    if not long_indices or not short_indices:
        return vector
    long_weight = 0.5 / float(len(long_indices))
    short_weight = -0.5 / float(len(short_indices))
    for index in long_indices:
        vector[index] = long_weight
    for index in short_indices:
        vector[index] = short_weight
    return vector


__all__ = [
    "equal_weight_vector",
    "resolve_requested_indices",
    "spread_vector",
    "VectorBuildConfig",
]
