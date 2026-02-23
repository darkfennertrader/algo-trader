from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    TuningAggregateConfig,
    TuningConfig,
    TuningParamSpec,
    TuningParamType,
    TuningRayConfig,
    TuningResourcesConfig,
    TuningTransform,
)
from .config_utils import require_string

TuningEngine = Literal["local", "ray"]
AggregateMethod = Literal["mean", "median", "mean_minus_std"]


def build_tuning_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningConfig:
    section = raw.get("tuning")
    if not isinstance(section, Mapping):
        raise ConfigError(f"tuning must be a mapping in {config_path}")
    try:
        space = _build_tuning_space(
            section.get("space", None), config_path
        )
        aggregate = _build_tuning_aggregate_config(
            section.get("aggregate", {}), config_path
        )
        resources = _build_tuning_resources_config(
            section.get("resources", {}), config_path
        )
        ray = _build_tuning_ray_config(
            section.get("ray", {}), resources, config_path
        )
        tuning = TuningConfig(
            space=space,
            num_samples=int(section.get("num_samples", 1)),
            seed=int(section.get("seed", 0)),
            kwargs=section.get("kwargs", {}),
            engine=section.get("engine", "local"),
            aggregate=aggregate,
            ray=ray,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid tuning configuration in {config_path}",
            context={"section": "tuning"},
        ) from exc
    return _normalize_tuning_config(tuning, config_path)


def _normalize_tuning_config(
    tuning: TuningConfig, config_path: Path
) -> TuningConfig:
    if tuning.num_samples <= 0:
        raise ConfigError(
            f"tuning.num_samples must be positive in {config_path}"
        )
    if tuning.seed < 0:
        raise ConfigError(
            f"tuning.seed must be non-negative in {config_path}"
        )
    return replace(
        tuning,
        engine=_normalize_tuning_engine(tuning.engine),
        aggregate=_normalize_aggregate_config(
            tuning.aggregate, config_path
        ),
        ray=_normalize_tuning_ray_config(tuning.ray, config_path),
    )


def _build_tuning_space(
    raw: object, config_path: Path
) -> tuple[TuningParamSpec, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list):
        raise ConfigError(f"tuning.space must be a list in {config_path}")
    specs: list[TuningParamSpec] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"tuning.space[{idx}] must be a mapping in {config_path}"
            )
        spec = _build_tuning_param_spec(
            item, config_path=config_path, index=idx
        )
        if spec.path in seen:
            raise ConfigError(
                f"Duplicate tuning.space path '{spec.path}' in {config_path}"
            )
        seen.add(spec.path)
        specs.append(spec)
    return tuple(specs)


def _build_tuning_param_spec(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
    index: int,
) -> TuningParamSpec:
    path = require_string(
        raw.get("path"),
        field=f"tuning.space[{index}].path",
        config_path=config_path,
    )
    param_type = _normalize_param_type(
        raw.get("type"),
        field=f"tuning.space[{index}].type",
    )
    transform = _normalize_transform(
        raw.get("transform"),
        field=f"tuning.space[{index}].transform",
        config_path=config_path,
    )
    when = _parse_when(
        raw.get("when"),
        field=f"tuning.space[{index}].when",
        config_path=config_path,
    )
    bounds = raw.get("bounds")
    values = raw.get("values")
    if param_type in {"float", "int"}:
        if values is not None:
            raise ConfigError(
                f"tuning.space[{index}].values is not allowed for {param_type} params in {config_path}"
            )
        bounds_tuple = _parse_bounds(
            bounds,
            field=f"tuning.space[{index}].bounds",
            config_path=config_path,
        )
        _validate_continuous_transform(
            transform=transform,
            bounds=bounds_tuple,
            field=f"tuning.space[{index}].transform",
            config_path=config_path,
        )
        return TuningParamSpec(
            path=path,
            param_type=param_type,
            bounds=bounds_tuple,
            values=None,
            transform=transform,
            when=when,
        )
    values_tuple = _parse_values(
        values,
        field=f"tuning.space[{index}].values",
        config_path=config_path,
    )
    if bounds is not None:
        raise ConfigError(
            f"tuning.space[{index}].bounds is not allowed for {param_type} params in {config_path}"
        )
    if transform != "none":
        raise ConfigError(
            f"tuning.space[{index}].transform must be 'none' for {param_type} params in {config_path}"
        )
    if param_type == "bool" and not all(
        isinstance(item, bool) for item in values_tuple
    ):
        raise ConfigError(
            f"tuning.space[{index}].values must be booleans in {config_path}"
        )
    return TuningParamSpec(
        path=path,
        param_type=param_type,
        bounds=None,
        values=values_tuple,
        transform=transform,
        when=when,
    )


def _parse_bounds(
    raw: object, *, field: str, config_path: Path
) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(
            f"{field} must be a 2-item list in {config_path}"
        )
    try:
        min_value = float(raw[0])
        max_value = float(raw[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be numeric in {config_path}") from exc
    if min_value >= max_value:
        raise ConfigError(
            f"{field} min must be less than max in {config_path}"
        )
    return min_value, max_value


def _parse_values(
    raw: object, *, field: str, config_path: Path
) -> tuple[Any, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            f"{field} must be a non-empty list in {config_path}"
        )
    return tuple(raw)


def _parse_when(
    raw: object, *, field: str, config_path: Path
) -> Mapping[str, tuple[Any, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{field} must be a mapping in {config_path}")
    when: dict[str, tuple[Any, ...]] = {}
    for key, value in raw.items():
        name = require_string(
            key, field=f"{field} key", config_path=config_path
        )
        allowed = _normalize_when_values(
            value, field=f"{field}.{name}", config_path=config_path
        )
        when[name] = allowed
    return when


def _normalize_when_values(
    raw: object, *, field: str, config_path: Path
) -> tuple[Any, ...]:
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ConfigError(
                f"{field} must not be empty in {config_path}"
            )
        return tuple(raw)
    if raw is None:
        raise ConfigError(f"{field} must not be null in {config_path}")
    return (raw,)


def _normalize_param_type(
    value: object, *, field: str
) -> TuningParamType:
    raw = str(value).strip().lower()
    if raw in {"float", "int", "categorical", "bool"}:
        return cast(TuningParamType, raw)
    raise ConfigError(f"{field} must be float, int, categorical, or bool")


def _normalize_transform(
    value: object, *, field: str, config_path: Path
) -> TuningTransform:
    if value is None:
        raise ConfigError(f"{field} is required in {config_path}")
    raw = str(value).strip().lower()
    if raw in {"linear", "log", "log10", "none"}:
        return cast(TuningTransform, raw)
    raise ConfigError(
        f"{field} must be linear, log, log10, or none in {config_path}"
    )


def _validate_continuous_transform(
    *,
    transform: str,
    bounds: tuple[float, float],
    field: str,
    config_path: Path,
) -> None:
    if transform == "none":
        raise ConfigError(
            f"{field} must not be 'none' for continuous params in {config_path}"
        )
    if transform in {"log", "log10"} and bounds[0] <= 0:
        raise ConfigError(
            f"{field} requires positive bounds in {config_path}"
        )


def _build_tuning_aggregate_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningAggregateConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"tuning.aggregate must be a mapping in {config_path}"
        )
    return TuningAggregateConfig(
        method=raw.get("method", "mean"),
        penalty=float(raw.get("penalty", 0.5)),
    )


def _normalize_aggregate_config(
    aggregate: TuningAggregateConfig, config_path: Path
) -> TuningAggregateConfig:
    if aggregate.penalty < 0:
        raise ConfigError(
            f"tuning.aggregate.penalty must be >= 0 in {config_path}"
        )
    return replace(
        aggregate,
        method=_normalize_aggregate_method(aggregate.method),
    )


def _build_tuning_resources_config(
    raw: Mapping[str, Any], config_path: Path
) -> TuningResourcesConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"tuning.resources must be a mapping in {config_path}"
        )
    cpu = raw.get("cpu")
    gpu = raw.get("gpu")
    return TuningResourcesConfig(
        cpu=float(cpu) if cpu is not None else None,
        gpu=float(gpu) if gpu is not None else None,
    )


def _normalize_tuning_resources_config(
    resources: TuningResourcesConfig, config_path: Path
) -> TuningResourcesConfig:
    if resources.cpu is not None and resources.cpu <= 0:
        raise ConfigError(
            f"tuning.resources.cpu must be > 0 in {config_path}"
        )
    if resources.gpu is not None and resources.gpu < 0:
        raise ConfigError(
            f"tuning.resources.gpu must be >= 0 in {config_path}"
        )
    return resources


def _build_tuning_ray_config(
    raw: Mapping[str, Any],
    resources: TuningResourcesConfig,
    config_path: Path,
) -> TuningRayConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"tuning.ray must be a mapping in {config_path}")
    raw_resources = raw.get("resources")
    if raw_resources is not None:
        if not isinstance(raw_resources, Mapping):
            raise ConfigError(
                f"tuning.ray.resources must be a mapping in {config_path}"
            )
        resources = _build_tuning_resources_config(
            raw_resources, config_path
        )
    address = raw.get("address")
    if address is not None:
        address = str(address).strip()
    return TuningRayConfig(address=address or None, resources=resources)


def _normalize_tuning_ray_config(
    ray: TuningRayConfig,
    config_path: Path,
) -> TuningRayConfig:
    resources = _normalize_tuning_resources_config(
        ray.resources, config_path
    )
    if ray.address is None:
        return replace(ray, resources=resources)
    normalized = ray.address.strip()
    if not normalized:
        return TuningRayConfig(address=None, resources=resources)
    return TuningRayConfig(address=normalized, resources=resources)


def _normalize_tuning_engine(value: object) -> TuningEngine:
    raw = str(value).strip().lower()
    if raw == "local":
        return "local"
    if raw == "ray":
        return "ray"
    raise ConfigError("tuning.engine must be local or ray")


def _normalize_aggregate_method(value: object) -> AggregateMethod:
    raw = str(value).strip().lower()
    if raw == "mean":
        return "mean"
    if raw == "median":
        return "median"
    if raw in {"mean_minus_std", "mean-std", "mean_std"}:
        return "mean_minus_std"
    raise ConfigError(
        "tuning.aggregate must be mean, median, or mean_minus_std"
    )
