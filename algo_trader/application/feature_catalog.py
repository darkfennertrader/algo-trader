from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from algo_trader.domain import ConfigError
from algo_trader.application.yaml_support import load_yaml_mapping

DEFAULT_FEATURE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "features.yml"
)


@dataclass(frozen=True)
class TechnicalFeatureCatalog:
    groups: Mapping[str, tuple[str, ...]]

    def ordered_group_names(self) -> tuple[str, ...]:
        return tuple(self.groups.keys())

    def features_for_group(self, group_name: str) -> tuple[str, ...]:
        features = self.groups.get(group_name)
        if features is None:
            raise ConfigError(
                "Technical feature group is not configured",
                context={"group": group_name},
            )
        return features


@dataclass(frozen=True)
class ExogenousFeatureCatalog:
    asset_features: tuple[str, ...]
    global_features: tuple[str, ...]


@dataclass(frozen=True)
class FeatureCatalogConfig:
    technical: TechnicalFeatureCatalog
    exogenous: ExogenousFeatureCatalog
    config_path: Path

    @classmethod
    def load(
        cls, config_path: Path | None = None
    ) -> "FeatureCatalogConfig":
        path = config_path or DEFAULT_FEATURE_CONFIG_PATH
        mapping = _load_yaml_mapping(path)
        technical = _load_technical_catalog(mapping, path)
        exogenous = _load_exogenous_catalog(mapping, path)
        return cls(
            technical=technical,
            exogenous=exogenous,
            config_path=path,
        )


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    return load_yaml_mapping(
        config_path,
        missing_message=f"Feature config not found at {config_path}",
        invalid_mapping_message=(
            f"Feature config must be a mapping in {config_path}"
        ),
    )


def _load_technical_catalog(
    mapping: Mapping[str, Any], config_path: Path
) -> TechnicalFeatureCatalog:
    technical = _coerce_mapping(mapping.get("technical"), label="technical")
    groups = _coerce_mapping(
        technical.get("groups"), label="technical.groups"
    )
    resolved: dict[str, tuple[str, ...]] = {}
    for group_name, raw_features in groups.items():
        normalized_group = _normalize_name(
            group_name, field="technical.groups key", config_path=config_path
        )
        resolved[normalized_group] = _coerce_name_list(
            raw_features,
            field=f"technical.groups.{normalized_group}",
            config_path=config_path,
        )
    if not resolved:
        raise ConfigError(
            f"technical.groups must not be empty in {config_path}"
        )
    return TechnicalFeatureCatalog(groups=resolved)


def _load_exogenous_catalog(
    mapping: Mapping[str, Any], config_path: Path
) -> ExogenousFeatureCatalog:
    exogenous = _coerce_mapping(mapping.get("exogenous"), label="exogenous")
    asset_features = _coerce_name_list(
        exogenous.get("asset_features"),
        field="exogenous.asset_features",
        config_path=config_path,
    )
    global_features = _coerce_name_list(
        exogenous.get("global_features"),
        field="exogenous.global_features",
        config_path=config_path,
    )
    return ExogenousFeatureCatalog(
        asset_features=asset_features,
        global_features=global_features,
    )


def _coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return raw


def _coerce_name_list(
    raw: object, *, field: str, config_path: Path
) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigError(f"{field} must be a list in {config_path}")
    values = tuple(
        _normalize_name(item, field=field, config_path=config_path)
        for item in raw
    )
    if not values:
        raise ConfigError(f"{field} must not be empty in {config_path}")
    if len(set(values)) != len(values):
        raise ConfigError(
            f"{field} must not contain duplicates in {config_path}"
        )
    return values


def _normalize_name(
    raw: object, *, field: str, config_path: Path
) -> str:
    if not isinstance(raw, str):
        raise ConfigError(f"{field} entries must be strings in {config_path}")
    normalized = raw.strip()
    if not normalized:
        raise ConfigError(f"{field} entries must not be empty in {config_path}")
    return normalized
