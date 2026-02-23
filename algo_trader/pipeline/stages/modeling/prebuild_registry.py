from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

from algo_trader.domain import ConfigError
from .prebuild import PrebuildHook

TPrebuild = TypeVar("TPrebuild", bound=PrebuildHook)


@dataclass
class PrebuildRegistry:
    _items: dict[str, PrebuildHook] = field(default_factory=dict)

    def register(self, name: str, hook: PrebuildHook) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Prebuild '{name}' is already registered",
                context={"prebuild": normalized},
            )
        self._items[normalized] = hook

    def get(self, name: str) -> PrebuildHook:
        normalized = _normalize_name(name)
        hook = self._items.get(normalized)
        if hook is None:
            raise ConfigError(
                f"Unknown prebuild '{name}'",
                context={"prebuild": normalized},
            )
        return hook

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


_PREBUILD_REGISTRY = PrebuildRegistry()


def register_prebuild(name: str) -> Callable[[TPrebuild], TPrebuild]:
    def decorator(hook: TPrebuild) -> TPrebuild:
        _PREBUILD_REGISTRY.register(name, hook)
        return hook

    return decorator


def default_prebuild_registry() -> PrebuildRegistry:
    # Add new prebuild modules here so decorators execute on registry creation.
    return _PREBUILD_REGISTRY


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("name must not be empty")
    return normalized
