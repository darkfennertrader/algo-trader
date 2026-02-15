from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._builders:
                raise ValueError(f"Duplicate registration for name '{name}'.")
            self._builders[name] = fn
            return fn

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._builders:
            raise KeyError(
                f"Unknown registry key '{name}'. Available: {list(self._builders.keys())}"
            )
        return self._builders[name](**kwargs)
