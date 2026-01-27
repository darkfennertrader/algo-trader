from __future__ import annotations

from dataclasses import dataclass, field

from algo_trader.domain import ConfigError
from .identity import IdentityPreprocessor
from .pca import PCAPreprocessor
from .zscore import ZScorePreprocessor
from .protocols import Preprocessor


@dataclass
class PreprocessorRegistry:
    _items: dict[str, Preprocessor] = field(default_factory=dict)

    def register(self, name: str, preprocessor: Preprocessor) -> None:
        normalized = _normalize_name(name)
        if normalized in self._items:
            raise ConfigError(
                f"Preprocessor '{name}' is already registered",
                context={"preprocessor": normalized},
            )
        self._items[normalized] = preprocessor

    def get(self, name: str) -> Preprocessor:
        normalized = _normalize_name(name)
        preprocessor = self._items.get(normalized)
        if preprocessor is None:
            raise ConfigError(
                f"Unknown preprocessor '{name}'",
                context={"preprocessor": normalized},
            )
        return preprocessor

    def list_names(self) -> list[str]:
        return sorted(self._items.keys())


def default_registry() -> PreprocessorRegistry:
    registry = PreprocessorRegistry()
    # Register new preprocessors here so the CLI can discover them.
    registry.register("identity", IdentityPreprocessor())
    registry.register("pca", PCAPreprocessor())
    registry.register("zscore", ZScorePreprocessor())
    return registry


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ConfigError("preprocessor name must not be empty")
    return normalized
