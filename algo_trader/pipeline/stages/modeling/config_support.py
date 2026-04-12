from __future__ import annotations

from typing import Any, Mapping

from algo_trader.domain import ConfigError


def coerce_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(raw)


__all__ = ["coerce_mapping"]
