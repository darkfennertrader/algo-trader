from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, cast

import torch

from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.shared import (
    BasketObservationGroup,
    build_custom_basket_observation_groups,
    project_basket_covariance,
    project_basket_mean,
    resolve_basket_group_weight,
)
from algo_trader.pipeline.stages.modeling.config_support import coerce_mapping
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)
from algo_trader.pipeline.stages.modeling.vector_support import (
    VectorBuildConfig,
    spread_vector,
)

SeedEntry = tuple[str, torch.Tensor]
SeedBuilder = Callable[
    [tuple[str, ...], torch.device, torch.dtype],
    tuple[SeedEntry, ...],
]
PairSpec = tuple[str, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class IndexRelativeMeasurementWeights:
    obs_weight: float = 0.06
    level_obs_weight: float | None = None
    relative_obs_weight: float | None = None
    residual_obs_weight: float | None = None


@dataclass(frozen=True)
class IndexRelativeMeasurementConfig:
    enabled: bool = True
    df: float = 8.0
    weights: IndexRelativeMeasurementWeights = field(
        default_factory=IndexRelativeMeasurementWeights
    )
    mad_floor: float = 1e-4
    eps: float = 1e-6


@dataclass(frozen=True)
class IndexRelativeMeasurementCoordinates:
    index_mask: torch.BoolTensor
    index_names: tuple[str, ...]
    basis: torch.Tensor
    coordinate_names: tuple[str, ...]

    @property
    def coordinate_count(self) -> int:
        return int(self.basis.shape[-1])


@dataclass(frozen=True)
class IndexCoordinateTransform:
    center: torch.Tensor
    mad: torch.Tensor


CoordinateBuilder = Callable[
    [RuntimeAssetMetadata, torch.device, torch.dtype],
    IndexRelativeMeasurementCoordinates,
]
GroupBuilder = Callable[
    [IndexRelativeMeasurementConfig, tuple[str, ...], torch.device],
    tuple[BasketObservationGroup, ...],
]


def build_index_relative_config(
    raw: object,
    *,
    label: str,
) -> IndexRelativeMeasurementConfig:
    values = coerce_mapping(raw, label=label)
    if not values:
        return IndexRelativeMeasurementConfig()
    base = IndexRelativeMeasurementConfig()
    return IndexRelativeMeasurementConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        df=float(values.get("df", base.df)),
        weights=IndexRelativeMeasurementWeights(
            obs_weight=float(values.get("obs_weight", base.weights.obs_weight)),
            level_obs_weight=_optional_float(
                values.get("level_obs_weight", base.weights.level_obs_weight)
            ),
            relative_obs_weight=_optional_float(
                values.get("relative_obs_weight", base.weights.relative_obs_weight)
            ),
            residual_obs_weight=_optional_float(
                values.get(
                    "residual_obs_weight",
                    base.weights.residual_obs_weight,
                )
            ),
        ),
        mad_floor=float(values.get("mad_floor", base.mad_floor)),
        eps=float(values.get("eps", base.eps)),
    )


def build_index_relative_measurement_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    return build_index_relative_config(
        raw,
        label="model.params.index_relative_measurement",
    )


def build_custom_relative_config(
    raw: object,
    *,
    label: str,
    relative_weight_key: str,
) -> IndexRelativeMeasurementConfig:
    values = coerce_mapping(raw, label=label)
    if not values:
        return IndexRelativeMeasurementConfig()
    relative_weight = values.get(relative_weight_key)
    residual_weight = values.get("residual_obs_weight")
    return IndexRelativeMeasurementConfig(
        enabled=bool(values.get("enabled", True)),
        df=float(values.get("df", 8.0)),
        weights=IndexRelativeMeasurementWeights(
            obs_weight=float(values.get("obs_weight", 0.05)),
            level_obs_weight=None,
            relative_obs_weight=(
                None if relative_weight is None else float(relative_weight)
            ),
            residual_obs_weight=(
                None if residual_weight is None else float(residual_weight)
            ),
        ),
        mad_floor=float(values.get("mad_floor", 1e-4)),
        eps=float(values.get("eps", 1e-6)),
    )


def build_index_relative_measurement_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
    seed_builder: SeedBuilder,
) -> IndexRelativeMeasurementCoordinates:
    index_mask = assets.index_mask
    index_names = tuple(
        name
        for name, is_index in zip(
            assets.asset_names, index_mask.tolist(), strict=True
        )
        if bool(is_index)
    )
    if len(index_names) < 1:
        return IndexRelativeMeasurementCoordinates(
            index_mask=index_mask,
            index_names=(),
            basis=torch.empty((0, 0), device=device, dtype=dtype),
            coordinate_names=(),
        )
    seed_entries = seed_builder(index_names, device, dtype)
    basis_entries = _complete_basis(
        index_names=index_names,
        seed_entries=seed_entries,
        device=device,
        dtype=dtype,
    )
    coordinate_names = tuple(name for name, _ in basis_entries)
    basis = torch.stack([vector for _, vector in basis_entries], dim=-1)
    return IndexRelativeMeasurementCoordinates(
        index_mask=index_mask,
        index_names=index_names,
        basis=basis,
        coordinate_names=coordinate_names,
    )


def build_index_relative_observation_groups(
    *,
    config: IndexRelativeMeasurementConfig,
    coordinate_names: tuple[str, ...],
    device: torch.device,
    level_names: frozenset[str],
    relative_names: frozenset[str],
) -> tuple[BasketObservationGroup, ...]:
    residual_names = frozenset(
        name for name in coordinate_names if name.startswith("index_residual_")
    )
    specs = (
        (
            "index_relative_level_obs",
            level_names,
            resolve_basket_group_weight(
                configured=config.weights.level_obs_weight,
                fallback=config.weights.obs_weight,
            ),
        ),
        (
            "index_relative_relative_obs",
            relative_names,
            resolve_basket_group_weight(
                configured=config.weights.relative_obs_weight,
                fallback=config.weights.obs_weight,
            ),
        ),
        (
            "index_relative_residual_obs",
            residual_names,
            resolve_basket_group_weight(
                configured=config.weights.residual_obs_weight,
                fallback=config.weights.obs_weight,
            ),
        ),
    )
    return build_custom_basket_observation_groups(
        basket_names=coordinate_names,
        specs=specs,
        device=device,
        fallback_weight=config.weights.obs_weight,
    )


def make_coordinate_builder(seed_builder: SeedBuilder) -> CoordinateBuilder:
    def builder(
        assets: RuntimeAssetMetadata,
        device: torch.device,
        dtype: torch.dtype,
    ) -> IndexRelativeMeasurementCoordinates:
        return build_index_relative_measurement_coordinates(
            assets=assets,
            device=device,
            dtype=dtype,
            seed_builder=seed_builder,
        )

    return builder


def make_group_builder(
    *,
    level_names: frozenset[str],
    relative_names: frozenset[str] | Callable[[tuple[str, ...]], frozenset[str]],
) -> GroupBuilder:
    def builder(
        config: IndexRelativeMeasurementConfig,
        coordinate_names: tuple[str, ...],
        device: torch.device,
    ) -> tuple[BasketObservationGroup, ...]:
        resolved_relative = (
            relative_names(coordinate_names)
            if callable(relative_names)
            else relative_names
        )
        return build_index_relative_observation_groups(
            config=config,
            coordinate_names=coordinate_names,
            device=device,
            level_names=level_names,
            relative_names=resolved_relative,
        )

    return builder


def make_custom_group_builder(
    *,
    relative_group_name: str,
    relative_names: frozenset[str] | Callable[[tuple[str, ...]], frozenset[str]],
    residual_group_name: str,
) -> GroupBuilder:
    def builder(
        config: IndexRelativeMeasurementConfig,
        coordinate_names: tuple[str, ...],
        device: torch.device,
    ) -> tuple[BasketObservationGroup, ...]:
        resolved_relative = (
            relative_names(coordinate_names)
            if callable(relative_names)
            else relative_names
        )
        residual_names = frozenset(
            name for name in coordinate_names if name.startswith("index_residual_")
        )
        specs = (
            (
                relative_group_name,
                resolved_relative,
                resolve_basket_group_weight(
                    configured=config.weights.relative_obs_weight,
                    fallback=config.weights.obs_weight,
                ),
            ),
            (
                residual_group_name,
                residual_names,
                resolve_basket_group_weight(
                    configured=config.weights.residual_obs_weight,
                    fallback=config.weights.obs_weight,
                ),
            ),
        )
        return build_custom_basket_observation_groups(
            basket_names=coordinate_names,
            specs=specs,
            device=device,
            fallback_weight=config.weights.obs_weight,
        )

    return builder


def build_index_coordinate_transform(
    *,
    observations: torch.Tensor,
    config: IndexRelativeMeasurementConfig,
) -> IndexCoordinateTransform:
    center = observations.median(dim=0).values
    mad = (observations - center.unsqueeze(0)).abs().median(dim=0).values
    return IndexCoordinateTransform(
        center=center,
        mad=mad.clamp_min(float(config.mad_floor)),
    )


def standardize_index_coordinates(
    *,
    values: torch.Tensor,
    transform: IndexCoordinateTransform,
) -> torch.Tensor:
    return (values - transform.center.unsqueeze(0)) / transform.mad.unsqueeze(0)


def standardize_index_covariance(
    *,
    covariance: torch.Tensor,
    transform: IndexCoordinateTransform,
) -> torch.Tensor:
    scale = transform.mad.view(1, -1, 1) * transform.mad.view(1, 1, -1)
    return covariance / scale


def coordinate_scale_from_covariance(
    *,
    covariance: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(float(eps)).sqrt()


def complete_basis(
    *,
    index_names: tuple[str, ...],
    seed_entries: tuple[SeedEntry, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    accepted = _orthonormalize_entries(seed_entries, eps=1e-6)
    residual_entries = tuple(
        (
            f"index_residual_{name}",
            _canonical_vector(
                count=len(index_names),
                index=offset,
                device=device,
                dtype=dtype,
            ),
        )
        for offset, name in enumerate(index_names)
    )
    full = accepted + _orthonormalize_entries(
        residual_entries,
        eps=1e-6,
        initial_basis=tuple(vector for _, vector in accepted),
    )
    return full[: len(index_names)]


def build_pair_seed_entries(
    *,
    index_names: tuple[str, ...],
    pair_specs: tuple[PairSpec, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    vector_config = VectorBuildConfig(device=device, dtype=dtype, strict=False)
    return tuple(
        (
            name,
            spread_vector(
                index_names,
                long_assets=long_assets,
                short_assets=short_assets,
                config=vector_config,
            ),
        )
        for name, long_assets, short_assets in pair_specs
    )


def _complete_basis(
    *,
    index_names: tuple[str, ...],
    seed_entries: tuple[SeedEntry, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SeedEntry, ...]:
    return complete_basis(
        index_names=index_names,
        seed_entries=seed_entries,
        device=device,
        dtype=dtype,
    )


def _orthonormalize_entries(
    entries: tuple[SeedEntry, ...],
    *,
    eps: float,
    initial_basis: tuple[torch.Tensor, ...] = (),
) -> tuple[SeedEntry, ...]:
    accepted_basis = list(initial_basis)
    accepted_entries: list[SeedEntry] = []
    for name, vector in entries:
        candidate = vector.clone()
        for basis_vector in accepted_basis:
            projection = torch.sum(candidate * basis_vector)
            candidate = candidate - projection * basis_vector
        norm = float(candidate.norm())
        if norm <= float(eps):
            continue
        normalized = candidate / norm
        accepted_basis.append(normalized)
        accepted_entries.append((name, normalized))
    return tuple(accepted_entries)


def _canonical_vector(
    *,
    count: int,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vector = torch.zeros((count,), device=device, dtype=dtype)
    vector[index] = 1.0
    return vector


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    return float(cast(float, raw))
