from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from algo_trader.domain import ConfigError


def require_string(
    value: object, *, field: str, config_path: Path
) -> str:
    if value is None:
        raise ConfigError(f"{field} is required in {config_path}")
    raw = str(value).strip()
    if not raw:
        raise ConfigError(f"{field} must be non-empty in {config_path}")
    return raw


def coerce_mapping(
    value: object, *, field: str, config_path: Path
) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping in {config_path}")
    return dict(value)


def require_bool(
    value: object, *, field: str, config_path: Path | None
) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    location = field if config_path is None else f"{field} in {config_path}"
    raise ConfigError(f"{location} must be a boolean")
