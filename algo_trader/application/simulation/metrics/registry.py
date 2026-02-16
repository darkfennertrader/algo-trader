from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import Registry

MetricFn = Callable[[torch.Tensor, Mapping[str, Any], Mapping[str, Any]], float]

_METRIC_REGISTRY = Registry()
_METRIC_CACHE: dict[tuple[str, str, tuple[tuple[str, Any], ...]], MetricFn] = {}


def register_metric(name: str, *, scope: str) -> Callable[[Callable[..., MetricFn]], Callable[..., MetricFn]]:
    key = _scoped_key(scope, name)
    return _METRIC_REGISTRY.register(key)


def build_metric_scorer(
    score_spec: Mapping[str, Any], *, scope: str
) -> MetricFn:
    metric_name = _resolve_metric_name(score_spec)
    return build_metric(scope=scope, name=metric_name, spec=score_spec)


def build_metric(
    *, scope: str, name: str, spec: Mapping[str, Any]
) -> MetricFn:
    key = _scoped_key(scope, name)
    cache_key = (scope, name, _freeze_spec(spec))
    cached = _METRIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scorer = _METRIC_REGISTRY.build(key, spec=spec)
    _METRIC_CACHE[cache_key] = scorer
    return scorer


def _resolve_metric_name(score_spec: Mapping[str, Any]) -> str:
    raw = score_spec.get("metric_name")
    if raw is None:
        raise ConfigError("scoring.spec.metric_name is required")
    name = str(raw).strip().lower()
    if not name:
        raise ConfigError("scoring.spec.metric_name must not be empty")
    return name


def _scoped_key(scope: str, name: str) -> str:
    scope_norm = scope.strip().lower()
    if not scope_norm:
        raise ConfigError("metric scope must not be empty")
    name_norm = name.strip().lower()
    if not name_norm:
        raise ConfigError("metric name must not be empty")
    return f"{scope_norm}:{name_norm}"


def _freeze_spec(spec: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = []
    for key in sorted(spec.keys()):
        items.append((key, _freeze_value(spec[key])))
    return tuple(items)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((k, _freeze_value(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value
