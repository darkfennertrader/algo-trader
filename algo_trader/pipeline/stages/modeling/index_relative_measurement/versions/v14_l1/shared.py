from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

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
    equal_weight_vector,
    spread_vector,
)

_US = ("IBUS30", "IBUS500", "IBUST100")
_EUROPE = ("IBDE40", "IBES35", "IBEU50", "IBFR40", "IBGB100", "IBNL25")
_LEVEL_NAMES = frozenset(("index_level",))
_RELATIVE_NAMES = frozenset(
    ("us_relative_index", "europe_relative_index", "us_minus_europe")
)


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


def build_index_relative_measurement_config(
    raw: object,
) -> IndexRelativeMeasurementConfig:
    values = coerce_mapping(raw, label="model.params.index_relative_measurement")
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


def build_index_relative_measurement_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
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
    seed_entries = _build_seed_entries(index_names, device=device, dtype=dtype)
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
) -> tuple[BasketObservationGroup, ...]:
    specs = (
        (
            "index_relative_level_obs",
            frozenset(
                name for name in coordinate_names if name in _LEVEL_NAMES
            ),
            resolve_basket_group_weight(
                configured=config.weights.level_obs_weight,
                fallback=config.weights.obs_weight,
            ),
        ),
        (
            "index_relative_relative_obs",
            frozenset(
                name for name in coordinate_names if name in _RELATIVE_NAMES
            ),
            resolve_basket_group_weight(
                configured=config.weights.relative_obs_weight,
                fallback=config.weights.obs_weight,
            ),
        ),
        (
            "index_relative_residual_obs",
            frozenset(
                name
                for name in coordinate_names
                if name.startswith("index_residual_")
            ),
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


def _build_seed_entries(
    index_names: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[str, torch.Tensor], ...]:
    vector_config = VectorBuildConfig(device=device, dtype=dtype, strict=False)
    entries = (
        (
            "index_level",
            equal_weight_vector(
                index_names,
                index_names,
                config=vector_config,
            ),
        ),
        (
            "us_relative_index",
            equal_weight_vector(
                index_names,
                _US,
                config=vector_config,
            )
            - equal_weight_vector(
                index_names,
                index_names,
                config=vector_config,
            ),
        ),
        (
            "europe_relative_index",
            equal_weight_vector(
                index_names,
                _EUROPE,
                config=vector_config,
            )
            - equal_weight_vector(
                index_names,
                index_names,
                config=vector_config,
            ),
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
    return tuple(
        (name, vector) for name, vector in entries if bool(vector.abs().sum() > 0.0)
    )


def _complete_basis(
    *,
    index_names: tuple[str, ...],
    seed_entries: tuple[tuple[str, torch.Tensor], ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[str, torch.Tensor], ...]:
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


def _orthonormalize_entries(
    entries: tuple[tuple[str, torch.Tensor], ...],
    *,
    eps: float,
    initial_basis: tuple[torch.Tensor, ...] = (),
) -> tuple[tuple[str, torch.Tensor], ...]:
    accepted_basis = list(initial_basis)
    accepted_entries: list[tuple[str, torch.Tensor]] = []
    for name, vector in entries:
        candidate = vector.clone()
        for basis_vector in accepted_basis:
            candidate = candidate - candidate.dot(basis_vector) * basis_vector
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


__all__ = [
    "BasketObservationGroup",
    "IndexCoordinateTransform",
    "IndexRelativeMeasurementConfig",
    "IndexRelativeMeasurementCoordinates",
    "IndexRelativeMeasurementWeights",
    "build_index_coordinate_transform",
    "build_index_relative_measurement_config",
    "build_index_relative_measurement_coordinates",
    "build_index_relative_observation_groups",
    "coordinate_scale_from_covariance",
    "project_basket_covariance",
    "project_basket_mean",
    "standardize_index_coordinates",
    "standardize_index_covariance",
]
