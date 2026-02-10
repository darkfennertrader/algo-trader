from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from algo_trader.domain import ConfigError

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Generic string -> builder function registry."""

    _builders: dict[str, Callable[..., T]] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        normalized = _normalize_name(name)

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            if normalized in self._builders:
                raise ConfigError(
                    f"Duplicate registration for name '{name}'.",
                    context={"name": normalized},
                )
            self._builders[normalized] = fn
            return fn

        return decorator

    def build(self, name: str, **kwargs: Any) -> T:
        normalized = _normalize_name(name)
        builder = self._builders.get(normalized)
        if builder is None:
            raise ConfigError(
                f"Unknown registry key '{name}'.",
                context={"name": normalized},
            )
        return builder(**kwargs)

    def list_names(self) -> list[str]:
        return sorted(self._builders.keys())


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("name must not be empty")
    return normalized
