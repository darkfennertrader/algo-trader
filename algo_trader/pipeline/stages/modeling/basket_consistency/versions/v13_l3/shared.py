from __future__ import annotations

import torch

from algo_trader.pipeline.stages.modeling.multi_asset_block.shared_v3_l1_unified import (
    RuntimeAssetMetadata,
)
from algo_trader.pipeline.stages.modeling.basket_consistency.versions.v13_l1.shared import (
    BasketConsistencyConfig,
    BasketConsistencyCoordinates,
    BasketConsistencyPosteriorMeans,
    BasketConsistencyPriorScaleConfig,
    BasketConsistencyTransform,
    BasketObservationGroup,
    basket_scale_from_covariance,
    build_custom_basket_observation_groups,
    build_basket_consistency_config,
    build_basket_consistency_coordinates,
    build_basket_consistency_transform,
    project_basket_covariance,
    project_basket_mean,
    resolve_basket_group_weight,
    initial_basket_consistency_posterior_means,
    whiten_basket_covariance,
    whiten_basket_observations,
)

_RELATIVE_BASKETS = frozenset(("us_relative_index", "europe_relative_index"))
_SPREAD_BASKETS = frozenset(("us_minus_europe",))

def build_relative_basket_consistency_coordinates(
    *,
    assets: RuntimeAssetMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> BasketConsistencyCoordinates:
    base = build_basket_consistency_coordinates(
        assets=assets,
        device=device,
        dtype=dtype,
    )
    if base.basket_count == 0:
        return base
    named_basis = {name: base.basis[:, idx] for idx, name in enumerate(base.basket_names)}
    entries = _build_relative_entries(named_basis)
    if not entries:
        return BasketConsistencyCoordinates(
            basis=torch.empty((base.basis.shape[0], 0), device=device, dtype=dtype),
            basket_names=(),
        )
    names = tuple(name for name, _ in entries)
    basis = torch.stack([vector for _, vector in entries], dim=-1)
    return BasketConsistencyCoordinates(basis=basis, basket_names=names)


def build_relative_basket_observation_groups(
    *,
    config: BasketConsistencyConfig,
    basket_names: tuple[str, ...],
    device: torch.device,
) -> tuple[BasketObservationGroup, ...]:
    if config.level_obs_weight is None and config.spread_obs_weight is None:
        return build_custom_basket_observation_groups(
            basket_names=basket_names,
            specs=(),
            device=device,
            fallback_weight=config.obs_weight,
        )
    return build_custom_basket_observation_groups(
        basket_names=basket_names,
        specs=_build_relative_weight_specs(config),
        device=device,
        fallback_weight=config.obs_weight,
    )


def _build_relative_weight_specs(
    config: BasketConsistencyConfig,
) -> tuple[tuple[str, frozenset[str], float], ...]:
    configured_specs = (
        ("basket_consistency_relative_obs", _RELATIVE_BASKETS, config.level_obs_weight),
        ("basket_consistency_spread_obs", _SPREAD_BASKETS, config.spread_obs_weight),
    )
    return tuple(
        (
            name,
            selected_names,
            resolve_basket_group_weight(
                configured=configured_weight,
                fallback=config.obs_weight,
            ),
        )
        for name, selected_names, configured_weight in configured_specs
    )


def _build_relative_entries(
    named_basis: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    entries: list[tuple[str, torch.Tensor]] = []
    equal_weight = named_basis.get("index_equal_weight")
    if equal_weight is not None:
        us_basis = named_basis.get("us_index")
        europe_basis = named_basis.get("europe_index")
        if us_basis is not None:
            entries.append(("us_relative_index", us_basis - equal_weight))
        if europe_basis is not None:
            entries.append(("europe_relative_index", europe_basis - equal_weight))
    spread_basis = named_basis.get("us_minus_europe")
    if spread_basis is not None:
        entries.append(("us_minus_europe", spread_basis))
    return entries
