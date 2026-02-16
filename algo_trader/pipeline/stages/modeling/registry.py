from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

from algo_trader.domain import ConfigError
from .protocols import PyroGuide, PyroModel

ModelBuilder = Callable[[], PyroModel]
GuideBuilder = Callable[[], PyroGuide]
TModelBuilder = TypeVar("TModelBuilder", bound=ModelBuilder)
TGuideBuilder = TypeVar("TGuideBuilder", bound=GuideBuilder)


@dataclass
class ModelRegistry:
    _items: dict[str, PyroModel] = field(default_factory=dict)

    def register(self, name: str, model: PyroModel) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Model '{name}' is already registered",
                context={"model": normalized},
            )
        self._items[normalized] = model

    def get(self, name: str) -> PyroModel:
        normalized = _normalize_name(name)
        model = self._items.get(normalized)
        if model is None:
            raise ConfigError(
                f"Unknown model '{name}'",
                context={"model": normalized},
            )
        return model

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


@dataclass
class GuideRegistry:
    _items: dict[str, PyroGuide] = field(default_factory=dict)

    def register(self, name: str, guide: PyroGuide) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Guide '{name}' is already registered",
                context={"guide": normalized},
            )
        self._items[normalized] = guide

    def get(self, name: str) -> PyroGuide:
        normalized = _normalize_name(name)
        guide = self._items.get(normalized)
        if guide is None:
            raise ConfigError(
                f"Unknown guide '{name}'",
                context={"guide": normalized},
            )
        return guide

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


_MODEL_REGISTRY = ModelRegistry()
_GUIDE_REGISTRY = GuideRegistry()


def register_model(name: str) -> Callable[[TModelBuilder], TModelBuilder]:
    def decorator(builder: TModelBuilder) -> TModelBuilder:
        _MODEL_REGISTRY.register(name, builder())
        return builder

    return decorator


def register_guide(name: str) -> Callable[[TGuideBuilder], TGuideBuilder]:
    def decorator(builder: TGuideBuilder) -> TGuideBuilder:
        _GUIDE_REGISTRY.register(name, builder())
        return builder

    return decorator


def default_model_registry() -> ModelRegistry:
    return _MODEL_REGISTRY


def default_guide_registry() -> GuideRegistry:
    return _GUIDE_REGISTRY


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("name must not be empty")
    return normalized
