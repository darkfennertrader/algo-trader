from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypeVar

from algo_trader.domain import ConfigError
from .protocols import PyroGuide, PyroModel

ModelBuilder = Callable[[Mapping[str, Any]], PyroModel]
GuideBuilder = Callable[[Mapping[str, Any]], PyroGuide]
TModelBuilder = TypeVar("TModelBuilder", bound=ModelBuilder)
TGuideBuilder = TypeVar("TGuideBuilder", bound=GuideBuilder)


@dataclass
class ModelRegistry:
    _items: dict[str, ModelBuilder] = field(default_factory=dict)

    def register(self, name: str, builder: ModelBuilder) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Model '{name}' is already registered",
                context={"model": normalized},
            )
        self._items[normalized] = builder

    def get(
        self, name: str, params: Mapping[str, Any] | None = None
    ) -> PyroModel:
        normalized = _normalize_name(name)
        builder = self._items.get(normalized)
        if builder is None:
            raise ConfigError(
                f"Unknown model '{name}'",
                context={"model": normalized},
            )
        return builder(dict(params or {}))

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


@dataclass
class GuideRegistry:
    _items: dict[str, GuideBuilder] = field(default_factory=dict)

    def register(self, name: str, builder: GuideBuilder) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Guide '{name}' is already registered",
                context={"guide": normalized},
            )
        self._items[normalized] = builder

    def get(
        self, name: str, params: Mapping[str, Any] | None = None
    ) -> PyroGuide:
        normalized = _normalize_name(name)
        builder = self._items.get(normalized)
        if builder is None:
            raise ConfigError(
                f"Unknown guide '{name}'",
                context={"guide": normalized},
            )
        return builder(dict(params or {}))

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


_MODEL_REGISTRY = ModelRegistry()
_GUIDE_REGISTRY = GuideRegistry()


def register_model(name: str) -> Callable[[TModelBuilder], TModelBuilder]:
    def decorator(builder: TModelBuilder) -> TModelBuilder:
        _MODEL_REGISTRY.register(name, builder)
        return builder

    return decorator


def register_guide(name: str) -> Callable[[TGuideBuilder], TGuideBuilder]:
    def decorator(builder: TGuideBuilder) -> TGuideBuilder:
        _GUIDE_REGISTRY.register(name, builder)
        return builder

    return decorator


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("name must not be empty")
    return normalized
