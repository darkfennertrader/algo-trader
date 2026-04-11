from __future__ import annotations
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)

_US = ("IBUS30", "IBUS500", "IBUST100")
_EUROPE = ("IBDE40", "IBES35", "IBEU50", "IBFR40", "IBGB100", "IBNL25")
_BASKET_ORDER = (
    "us_index",
    "europe_index",
    "us_minus_europe",
    "index_equal_weight",
)
_LEVEL_BASKETS = frozenset(("us_index", "europe_index", "index_equal_weight"))
_SPREAD_BASKETS = frozenset(("us_minus_europe",))


@dataclass(frozen=True)
class BasketConsistencyPriorScaleConfig:
    scale_center: float = 0.85
    scale_log_scale: float = 0.12
    covariance_shrinkage: float = 0.35
    covariance_floor: float = 0.05
    mad_floor: float = 1e-4


@dataclass(frozen=True)
class BasketConsistencyConfig:
    enabled: bool = True
    df: float = 8.0
    obs_weight: float = 0.08
    level_obs_weight: float | None = None
    spread_obs_weight: float | None = None
    prior_scales: BasketConsistencyPriorScaleConfig = (
        BasketConsistencyPriorScaleConfig()
    )
    eps: float = 1e-6


@dataclass(frozen=True)
class BasketConsistencyPosteriorMeans:
    basket_scale: torch.Tensor


@dataclass(frozen=True)
class BasketConsistencyCoordinates:
    basis: torch.Tensor
    basket_names: tuple[str, ...]

    @property
    def basket_count(self) -> int:
        return int(self.basis.shape[-1])


@dataclass(frozen=True)
class BasketConsistencyTransform:
    center: torch.Tensor
    mad: torch.Tensor
    whitening: torch.Tensor


@dataclass(frozen=True)
class BasketObservationGroup:
    name: str
    mask: torch.BoolTensor
    obs_weight: float


@dataclass(frozen=True)
class _BasketObservationGroupSpec:
    name: str
    selected_names: frozenset[str]
    obs_weight: float


def build_basket_consistency_config(raw: object) -> BasketConsistencyConfig:
    values = _coerce_mapping(raw, label="model.params.basket_consistency")
    if not values:
        return BasketConsistencyConfig()
    base = BasketConsistencyConfig()
    scale_values = _coerce_mapping(
        values.get("prior_scales"),
        label="model.params.basket_consistency.prior_scales",
    )
    return BasketConsistencyConfig(
        enabled=bool(values.get("enabled", base.enabled)),
        df=float(values.get("df", base.df)),
        obs_weight=float(values.get("obs_weight", base.obs_weight)),
        level_obs_weight=_optional_float(values.get("level_obs_weight")),
        spread_obs_weight=_optional_float(values.get("spread_obs_weight")),
        prior_scales=BasketConsistencyPriorScaleConfig(
            scale_center=float(
                scale_values.get("scale_center", base.prior_scales.scale_center)
            ),
            scale_log_scale=float(
                scale_values.get("scale_log_scale", base.prior_scales.scale_log_scale)
            ),
            covariance_shrinkage=float(
                scale_values.get(
                    "covariance_shrinkage",
                    base.prior_scales.covariance_shrinkage,
                )
            ),
            covariance_floor=float(
                scale_values.get(
                    "covariance_floor",
                    base.prior_scales.covariance_floor,
                )
            ),
            mad_floor=float(
                scale_values.get("mad_floor", base.prior_scales.mad_floor)
            ),
        ),
        eps=float(values.get("eps", base.eps)),
    )


def build_basket_consistency_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> BasketConsistencyCoordinates:
    asset_names = assets.asset_names
    entries = (
        ("us_index", _equal_weight_vector(asset_names, _US, device=device, dtype=dtype)),
        (
            "europe_index",
            _equal_weight_vector(asset_names, _EUROPE, device=device, dtype=dtype),
        ),
        (
            "us_minus_europe",
            _spread_vector(asset_names, _US, _EUROPE, device=device, dtype=dtype),
        ),
        (
            "index_equal_weight",
            _equal_weight_vector(
                asset_names,
                tuple(name for name in asset_names if name.startswith("IB")),
                device=device,
                dtype=dtype,
            ),
        ),
    )
    active = [(name, vector) for name, vector in entries if bool(vector.abs().sum() > 0.0)]
    if not active:
        return BasketConsistencyCoordinates(
            basis=torch.empty(
                (len(asset_names), 0),
                device=device,
                dtype=dtype,
            ),
            basket_names=(),
        )
    names = tuple(name for name, _ in active)
    basis = torch.stack([vector for _, vector in active], dim=-1)
    return BasketConsistencyCoordinates(basis=basis, basket_names=names)


def build_basket_consistency_transform(
    *,
    observations: torch.Tensor,
    config: BasketConsistencyConfig,
) -> BasketConsistencyTransform:
    center = observations.median(dim=0).values
    centered = observations - center.unsqueeze(0)
    mad = centered.abs().median(dim=0).values.clamp_min(
        float(config.prior_scales.mad_floor)
    )
    standardized = centered / mad.unsqueeze(0)
    covariance = _centered_covariance(standardized)
    shrunk = _shrink_covariance(covariance, config)
    whitening = _whitening_matrix(shrunk)
    return BasketConsistencyTransform(
        center=center,
        mad=mad,
        whitening=whitening,
    )


def project_basket_mean(*, loc: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ta,aj->tj", loc, basis)


def project_basket_covariance(
    *,
    cov_factor: torch.Tensor,
    cov_diag: torch.Tensor,
    basis: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    transformed_factor = torch.einsum("tak,aj->tjk", cov_factor, basis)
    low_rank_cov = transformed_factor @ transformed_factor.transpose(-1, -2)
    diagonal_cov = torch.einsum("ai,ta,aj->tij", basis, cov_diag, basis)
    jitter = torch.eye(
        basis.shape[-1],
        device=cov_factor.device,
        dtype=cov_factor.dtype,
    ).unsqueeze(0) * float(eps)
    return low_rank_cov + diagonal_cov + jitter


def whiten_basket_observations(
    *,
    values: torch.Tensor,
    transform: BasketConsistencyTransform,
) -> torch.Tensor:
    standardized = (values - transform.center.unsqueeze(0)) / transform.mad.unsqueeze(0)
    return standardized @ transform.whitening.transpose(-1, -2)


def whiten_basket_covariance(
    *,
    covariance: torch.Tensor,
    transform: BasketConsistencyTransform,
) -> torch.Tensor:
    scale = transform.mad.view(1, -1, 1) * transform.mad.view(1, 1, -1)
    standardized = covariance / scale
    whitening = transform.whitening.unsqueeze(0)
    return whitening @ standardized @ whitening.transpose(-1, -2)


def basket_scale_from_covariance(
    *,
    covariance: torch.Tensor,
    basket_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    base_std = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(float(eps)).sqrt()
    return base_std * basket_scale.unsqueeze(0)


def initial_basket_consistency_posterior_means(
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BasketConsistencyPosteriorMeans:
    return BasketConsistencyPosteriorMeans(
        basket_scale=torch.full(
            (count,),
            0.85,
            device=device,
            dtype=dtype,
        )
    )


def build_basket_observation_groups(
    *,
    config: BasketConsistencyConfig,
    basket_names: tuple[str, ...],
    device: torch.device,
) -> tuple[BasketObservationGroup, ...]:
    if not _uses_split_weights(config):
        return (
            BasketObservationGroup(
                name="basket_consistency_obs",
                mask=cast(
                    torch.BoolTensor,
                    torch.ones(
                        len(basket_names),
                        device=device,
                        dtype=torch.bool,
                    ),
                ),
                obs_weight=float(config.obs_weight),
            ),
        )
    groups: list[BasketObservationGroup] = []
    _append_basket_group(
        groups=groups,
        basket_names=basket_names,
        spec=_BasketObservationGroupSpec(
            name="basket_consistency_level_obs",
            selected_names=_LEVEL_BASKETS,
            obs_weight=_resolve_group_weight(
                configured=config.level_obs_weight,
                fallback=config.obs_weight,
            ),
        ),
        device=device,
    )
    _append_basket_group(
        groups=groups,
        basket_names=basket_names,
        spec=_BasketObservationGroupSpec(
            name="basket_consistency_spread_obs",
            selected_names=_SPREAD_BASKETS,
            obs_weight=_resolve_group_weight(
                configured=config.spread_obs_weight,
                fallback=config.obs_weight,
            ),
        ),
        device=device,
    )
    if groups:
        return tuple(groups)
    return (
        BasketObservationGroup(
            name="basket_consistency_obs",
            mask=cast(
                torch.BoolTensor,
                torch.ones(
                    len(basket_names),
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            obs_weight=float(config.obs_weight),
        ),
    )


def _equal_weight_vector(
    asset_names: tuple[str, ...],
    requested: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    indices = _resolve_requested_indices(asset_names, requested)
    vector = torch.zeros((len(asset_names),), device=device, dtype=dtype)
    if not indices:
        return vector
    weight = 1.0 / float(len(indices))
    for index in indices:
        vector[index] = weight
    return vector


def _spread_vector(
    asset_names: tuple[str, ...],
    long_assets: tuple[str, ...],
    short_assets: tuple[str, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    long_indices = _resolve_requested_indices(asset_names, long_assets)
    short_indices = _resolve_requested_indices(asset_names, short_assets)
    vector = torch.zeros((len(asset_names),), device=device, dtype=dtype)
    if not long_indices or not short_indices:
        return vector
    long_weight = 0.5 / float(len(long_indices))
    short_weight = -0.5 / float(len(short_indices))
    for index in long_indices:
        vector[index] = long_weight
    for index in short_indices:
        vector[index] = short_weight
    return vector


def _resolve_requested_indices(
    asset_names: tuple[str, ...],
    requested: tuple[str, ...],
) -> tuple[int, ...]:
    index_by_name = {name: idx for idx, name in enumerate(asset_names)}
    if any(name not in index_by_name for name in requested):
        return ()
    return tuple(index_by_name[name] for name in requested)


def _centered_covariance(values: torch.Tensor) -> torch.Tensor:
    if int(values.shape[0]) <= 1:
        return torch.eye(
            values.shape[-1],
            device=values.device,
            dtype=values.dtype,
        )
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered / float(values.shape[0] - 1)


def _shrink_covariance(
    covariance: torch.Tensor,
    config: BasketConsistencyConfig,
) -> torch.Tensor:
    diagonal = torch.diag(torch.diagonal(covariance))
    shrinkage = float(config.prior_scales.covariance_shrinkage)
    floor = torch.eye(
        covariance.shape[-1],
        device=covariance.device,
        dtype=covariance.dtype,
    ) * float(config.prior_scales.covariance_floor)
    return ((1.0 - shrinkage) * covariance) + (shrinkage * diagonal) + floor


def _whitening_matrix(covariance: torch.Tensor) -> torch.Tensor:
    covariance_numpy = covariance.detach().cpu().numpy()
    cholesky = np.linalg.cholesky(covariance_numpy)
    identity = np.eye(covariance_numpy.shape[-1], dtype=covariance_numpy.dtype)
    whitening = np.linalg.solve(cholesky, identity)
    return torch.as_tensor(
        whitening,
        device=covariance.device,
        dtype=covariance.dtype,
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


def _append_basket_group(
    *,
    groups: list[BasketObservationGroup],
    basket_names: tuple[str, ...],
    spec: _BasketObservationGroupSpec,
    device: torch.device,
) -> None:
    mask = cast(
        torch.BoolTensor,
        torch.as_tensor(
            [basket_name in spec.selected_names for basket_name in basket_names],
            device=device,
            dtype=torch.bool,
        ),
    )
    if bool(mask.any()):
        groups.append(
            BasketObservationGroup(
                name=spec.name,
                mask=mask,
                obs_weight=spec.obs_weight,
            )
        )


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    return float(raw)


def _resolve_group_weight(*, configured: float | None, fallback: float) -> float:
    if configured is None:
        return float(fallback)
    return float(configured)


def _uses_split_weights(config: BasketConsistencyConfig) -> bool:
    return (
        config.level_obs_weight is not None
        or config.spread_obs_weight is not None
    )


__all__ = [
    "BasketConsistencyConfig",
    "BasketConsistencyCoordinates",
    "BasketObservationGroup",
    "BasketConsistencyPosteriorMeans",
    "BasketConsistencyPriorScaleConfig",
    "BasketConsistencyTransform",
    "basket_scale_from_covariance",
    "build_basket_observation_groups",
    "build_basket_consistency_config",
    "build_basket_consistency_coordinates",
    "build_basket_consistency_transform",
    "initial_basket_consistency_posterior_means",
    "project_basket_covariance",
    "project_basket_mean",
    "whiten_basket_covariance",
    "whiten_basket_observations",
]
