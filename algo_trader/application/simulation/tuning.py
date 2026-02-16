from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
import copy
import json

import numpy as np

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import TuningConfig


@dataclass(frozen=True)
class _ContinuousParam:
    name: str
    min_value: float
    max_value: float
    dtype: str


@dataclass(frozen=True)
class _ParsedParamSpace:
    fixed: dict[str, Any]
    continuous: list[_ContinuousParam]
    discrete: dict[str, list[Any]]


def resolve_candidates(*, tuning: TuningConfig) -> list[dict[str, Any]]:
    if tuning.sampling.pre_sampled_path:
        return _load_candidates(Path(tuning.sampling.pre_sampled_path))
    return build_candidates(
        param_space=tuning.param_space,
        num_samples=tuning.num_samples,
        seed=tuning.sampling.seed,
        sampling_method=tuning.sampling.method,
    )


def build_candidates(
    *,
    param_space: Mapping[str, Any],
    num_samples: int,
    seed: int,
    sampling_method: str,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ConfigError("tuning.num_samples must be positive")
    if not param_space:
        return [{}]
    if sampling_method == "grid":
        return _grid_candidates(param_space, num_samples, seed)
    parsed = _parse_param_space(param_space)
    continuous_values = _sample_continuous(
        continuous=parsed.continuous,
        num_samples=num_samples,
        seed=seed,
        sampling_method=sampling_method,
    )
    discrete_values = _sample_discrete(
        discrete=parsed.discrete,
        num_samples=num_samples,
        seed=seed,
    )
    return _assemble_candidates(
        fixed=parsed.fixed,
        continuous=parsed.continuous,
        continuous_values=continuous_values,
        discrete_values=discrete_values,
        num_samples=num_samples,
    )


def apply_param_updates(
    base_config: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base_config))
    for key, value in updates.items():
        _set_config_path(merged, key, value)
    return merged


def _set_config_path(
    target: MutableMapping[str, Any],
    path: str,
    value: Any,
) -> None:
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        raise ConfigError("param_space keys must not be empty")
    cursor: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _grid_candidates(
    param_space: Mapping[str, Any],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if _has_continuous(param_space):
        raise ConfigError(
            "Continuous ranges require sampling_method of random, sobol, or lhs"
        )
    configs = expand_param_space(param_space)
    return select_candidates(configs, num_samples, seed)


def expand_param_space(param_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    values: list[Sequence[Any]] = []
    for key in keys:
        raw = param_space[key]
        if isinstance(raw, Mapping):
            raise ConfigError(
                f"param_space '{key}' must be list/tuple for grid sampling"
            )
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ConfigError(f"param_space '{key}' must not be empty")
            values.append(list(raw))
        else:
            values.append([raw])
    configs = [dict(zip(keys, combo, strict=False)) for combo in product(*values)]
    return configs or [{}]


def select_candidates(
    configs: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ConfigError("tuning.num_samples must be positive")
    if len(configs) <= num_samples:
        return configs
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(configs), size=num_samples, replace=False)
    return [configs[int(i)] for i in indices]


def _has_continuous(param_space: Mapping[str, Any]) -> bool:
    for raw in param_space.values():
        if isinstance(raw, Mapping) and {"min", "max"} <= set(raw.keys()):
            return True
    return False


def _parse_param_space(
    param_space: Mapping[str, Any],
) -> _ParsedParamSpace:
    fixed: dict[str, Any] = {}
    continuous: list[_ContinuousParam] = []
    discrete: dict[str, list[Any]] = {}
    for name, raw in param_space.items():
        if isinstance(raw, Mapping):
            continuous.append(_parse_continuous(name, raw))
            continue
        if isinstance(raw, (list, tuple)):
            values = list(raw)
            if not values:
                raise ConfigError(f"param_space '{name}' must not be empty")
            discrete[name] = values
            continue
        fixed[name] = raw
    return _ParsedParamSpace(
        fixed=fixed, continuous=continuous, discrete=discrete
    )


def _parse_continuous(
    name: str,
    raw: Mapping[str, Any],
) -> _ContinuousParam:
    if not {"min", "max"} <= set(raw.keys()):
        raise ConfigError(
            f"param_space '{name}' must include min and max"
        )
    min_value = float(raw["min"])
    max_value = float(raw["max"])
    if min_value >= max_value:
        raise ConfigError(
            f"param_space '{name}' min must be less than max"
        )
    dtype = _normalize_dtype(raw.get("dtype", "float"))
    return _ContinuousParam(
        name=name,
        min_value=min_value,
        max_value=max_value,
        dtype=dtype,
    )


def _normalize_dtype(value: object) -> str:
    raw = str(value).strip().lower()
    if raw in {"float", "float32", "float64"}:
        return "float"
    if raw in {"int", "int32", "int64", "integer"}:
        return "int"
    raise ConfigError("param_space dtype must be float or int")


def _sample_continuous(
    *,
    continuous: Sequence[_ContinuousParam],
    num_samples: int,
    seed: int,
    sampling_method: str,
) -> np.ndarray:
    if not continuous:
        return np.empty((num_samples, 0))
    bounds_min = np.array([p.min_value for p in continuous], dtype=float)
    bounds_max = np.array([p.max_value for p in continuous], dtype=float)
    if sampling_method in {"sobol", "lhs"}:
        qmc = _require_qmc()
        if sampling_method == "sobol":
            sobol_cls: Any = qmc.Sobol
            sampler = sobol_cls(
                d=len(continuous), scramble=True, seed=seed
            )
            raw = sampler.random(n=num_samples)
        else:
            lhs_cls: Any = qmc.LatinHypercube
            sampler = lhs_cls(d=len(continuous), seed=seed)
            raw = sampler.random(n=num_samples)
        scaled = qmc.scale(raw, bounds_min, bounds_max)
    else:
        rng = np.random.default_rng(seed)
        raw = rng.random((num_samples, len(continuous)))
        scaled = bounds_min + raw * (bounds_max - bounds_min)
    return _cast_continuous(scaled, continuous)


def _cast_continuous(
    values: np.ndarray, continuous: Sequence[_ContinuousParam]
) -> np.ndarray:
    casted = values.copy()
    for idx, param in enumerate(continuous):
        if param.dtype == "int":
            rounded = np.rint(casted[:, idx])
            clipped = np.clip(rounded, param.min_value, param.max_value)
            casted[:, idx] = clipped.astype(int)
    return casted


def _sample_discrete(
    *,
    discrete: Mapping[str, list[Any]],
    num_samples: int,
    seed: int,
) -> dict[str, list[Any]]:
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[Any]] = {}
    for name, values in discrete.items():
        sampled[name] = _balanced_choices(values, num_samples, rng)
    return sampled


def _assemble_candidates(
    *,
    fixed: Mapping[str, Any],
    continuous: Sequence[_ContinuousParam],
    continuous_values: np.ndarray,
    discrete_values: Mapping[str, Sequence[Any]],
    num_samples: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx in range(num_samples):
        candidate = dict(fixed)
        _assign_continuous(
            candidate, continuous, continuous_values, idx
        )
        _assign_discrete(candidate, discrete_values, idx)
        candidates.append(candidate)
    return candidates


def _assign_continuous(
    target: MutableMapping[str, Any],
    continuous: Sequence[_ContinuousParam],
    values: np.ndarray,
    idx: int,
) -> None:
    for j, param in enumerate(continuous):
        target[param.name] = _to_python_scalar(values[idx, j])


def _assign_discrete(
    target: MutableMapping[str, Any],
    discrete_values: Mapping[str, Sequence[Any]],
    idx: int,
) -> None:
    for name, values in discrete_values.items():
        target[name] = values[idx]


def _balanced_choices(
    values: Sequence[Any],
    num_samples: int,
    rng: np.random.Generator,
) -> list[Any]:
    if not values:
        raise ConfigError("param_space values must not be empty")
    if len(values) == 1:
        return [values[0]] * num_samples
    repeats, remainder = divmod(num_samples, len(values))
    pool: list[Any] = list(values) * repeats
    if remainder:
        pool.extend(
            rng.choice(list(values), size=remainder, replace=False).tolist()
        )
    rng.shuffle(pool)
    return pool


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_qmc():
    try:
        from scipy.stats import qmc  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ConfigError(
            "scipy is required for sobol/lhs sampling; install with `uv add scipy`"
        ) from exc
    return qmc


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"pre_sampled_path not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"pre_sampled_path is not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ConfigError(
            "pre_sampled_path must contain a JSON list of objects"
        )
    return [dict(item) for item in payload]
