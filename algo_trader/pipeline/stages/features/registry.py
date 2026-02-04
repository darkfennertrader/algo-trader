from __future__ import annotations

from dataclasses import dataclass, field

from algo_trader.domain import ConfigError
from .protocols import FeatureGroup
from .horizons import HorizonSpec
from .momentum import DEFAULT_HORIZON_DAYS, MomentumConfig, MomentumFeatureGroup
from .mean_reversion import (
    DEFAULT_HORIZON_DAYS as DEFAULT_MEAN_REV_DAYS,
    MeanReversionConfig,
    MeanReversionFeatureGroup,
)


@dataclass
class FeatureRegistry:
    _items: dict[str, FeatureGroup] = field(default_factory=dict)

    def register(self, name: str, group: FeatureGroup) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Feature group '{name}' is already registered",
                context={"feature_group": normalized},
            )
        self._items[normalized] = group

    def get(self, name: str) -> FeatureGroup:
        normalized = _normalize_name(name)
        group = self._items.get(normalized)
        if group is None:
            raise ConfigError(
                f"Unknown feature group '{name}'",
                context={"feature_group": normalized},
            )
        return group

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


def default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    # Register feature groups here so the CLI can discover them.
    default_horizons = [
        HorizonSpec(days=days, weeks=days // 5)
        for days in DEFAULT_HORIZON_DAYS
    ]
    registry.register(
        "momentum",
        MomentumFeatureGroup(
            MomentumConfig(
                horizons=default_horizons,
                return_type="simple",
            )
        ),
    )
    mean_rev_horizons = [
        HorizonSpec(days=days, weeks=days // 5) for days in DEFAULT_MEAN_REV_DAYS
    ]
    registry.register(
        "mean_reversion",
        MeanReversionFeatureGroup(
            MeanReversionConfig(horizons=mean_rev_horizons)
        ),
    )
    return registry


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("feature group name must not be empty")
    return normalized
