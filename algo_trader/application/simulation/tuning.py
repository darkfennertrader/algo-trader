from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import TuningParamSpec


@dataclass(frozen=True)
class _SpaceGroups:
    categorical: tuple[TuningParamSpec, ...]
    continuous: tuple[TuningParamSpec, ...]


def build_candidates(
    *,
    space: Sequence[TuningParamSpec],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ConfigError("tuning.num_samples must be positive")
    if not space:
        return [{}]
    groups = _split_space(space)
    _validate_when_refs(space=space, categorical=groups.categorical)
    combos = _build_categorical_combos(groups.categorical)
    if not combos:
        return [{}]
    n_combos = len(combos)
    per_combo = _resolve_per_combo_samples(num_samples, n_combos)
    rng = np.random.default_rng(seed)
    counts = [per_combo] * n_combos
    seeds = _spawn_seeds(rng, len(combos))
    candidates: list[dict[str, Any]] = []
    for combo, count, sub_seed in zip(combos, counts, seeds, strict=False):
        if count <= 0:
            continue
        active_continuous = _active_continuous(groups.continuous, combo)
        candidates.extend(
            _sample_for_combo(
                combo=combo,
                continuous=active_continuous,
                num_samples=count,
                seed=sub_seed,
            )
        )
    return candidates


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
        raise ConfigError("tuning space paths must not be empty")
    cursor: MutableMapping[str, Any] = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _split_space(space: Sequence[TuningParamSpec]) -> _SpaceGroups:
    categorical: list[TuningParamSpec] = []
    continuous: list[TuningParamSpec] = []
    for spec in space:
        if spec.param_type in {"categorical", "bool"}:
            categorical.append(spec)
        else:
            continuous.append(spec)
    return _SpaceGroups(
        categorical=tuple(categorical),
        continuous=tuple(continuous),
    )


def _validate_when_refs(
    *,
    space: Sequence[TuningParamSpec],
    categorical: Sequence[TuningParamSpec],
) -> None:
    categorical_paths = {spec.path for spec in categorical}
    for spec in space:
        for key in spec.when:
            if key not in categorical_paths:
                raise ConfigError(
                    "when conditions must reference categorical or bool params"
                )


def _build_categorical_combos(
    categorical: Sequence[TuningParamSpec],
) -> list[dict[str, Any]]:
    if not categorical:
        return [{}]
    ordered = _order_categorical(categorical)
    combos: list[dict[str, Any]] = [{}]
    for spec in ordered:
        next_combos: list[dict[str, Any]] = []
        values = spec.values or ()
        for combo in combos:
            if _when_satisfied(spec.when, combo):
                for value in values:
                    updated = dict(combo)
                    updated[spec.path] = value
                    next_combos.append(updated)
            else:
                next_combos.append(combo)
        combos = next_combos
    return combos


def _order_categorical(
    categorical: Sequence[TuningParamSpec],
) -> list[TuningParamSpec]:
    param_map = {spec.path: spec for spec in categorical}
    order_index = {spec.path: idx for idx, spec in enumerate(categorical)}
    deps = {
        spec.path: set(spec.when.keys()) for spec in categorical
    }
    remaining = set(param_map)
    ready = [
        path for path, dep in deps.items() if not dep
    ]
    ready.sort(key=lambda item: order_index[item])
    ordered: list[TuningParamSpec] = []
    while ready:
        path = ready.pop(0)
        ordered.append(param_map[path])
        remaining.discard(path)
        for other in list(remaining):
            other_deps = deps[other]
            if path in other_deps:
                other_deps.remove(path)
                if not other_deps:
                    ready.append(other)
                    ready.sort(key=lambda item: order_index[item])
    if remaining:
        raise ConfigError(
            "Detected circular when dependencies in tuning.space"
        )
    return ordered


def _when_satisfied(
    when: Mapping[str, Sequence[Any]],
    combo: Mapping[str, Any],
) -> bool:
    if not when:
        return True
    for key, allowed in when.items():
        if key not in combo:
            return False
        if combo[key] not in set(allowed):
            return False
    return True


def _resolve_per_combo_samples(
    num_samples: int, n_combos: int
) -> int:
    if n_combos <= 0:
        raise ConfigError("No categorical combos available for sampling")
    if num_samples % n_combos != 0:
        raise ConfigError(
            "tuning.num_samples must be divisible by the number of categorical combos"
        )
    per_combo = num_samples // n_combos
    if not _is_power_of_two(per_combo):
        raise ConfigError(
            "tuning.num_samples per combo must be a power of two"
        )
    return per_combo


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _spawn_seeds(
    rng: np.random.Generator, count: int
) -> list[int]:
    if count <= 0:
        return []
    return rng.integers(0, 2**32 - 1, size=count).tolist()


def _active_continuous(
    continuous: Sequence[TuningParamSpec],
    combo: Mapping[str, Any],
) -> tuple[TuningParamSpec, ...]:
    active: list[TuningParamSpec] = []
    for spec in continuous:
        if _when_satisfied(spec.when, combo):
            active.append(spec)
    return tuple(active)


def _sample_for_combo(
    *,
    combo: Mapping[str, Any],
    continuous: Sequence[TuningParamSpec],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not continuous:
        return [dict(combo) for _ in range(num_samples)]
    raw = _sobol_samples(
        num_samples=num_samples, dim=len(continuous), seed=seed
    )
    mapped = _map_samples(raw, continuous)
    candidates: list[dict[str, Any]] = []
    for idx in range(num_samples):
        candidate = dict(combo)
        for j, spec in enumerate(continuous):
            candidate[spec.path] = _to_python_scalar(mapped[idx, j])
        candidates.append(candidate)
    return candidates


def _sobol_samples(*, num_samples: int, dim: int, seed: int) -> np.ndarray:
    qmc = _require_qmc()
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return sampler.random(n=num_samples)


def _map_samples(
    raw: np.ndarray, params: Sequence[TuningParamSpec]
) -> np.ndarray:
    mapped = np.empty_like(raw, dtype=float)
    for idx, spec in enumerate(params):
        if spec.bounds is None:
            raise ConfigError("Continuous param missing bounds")
        lower, upper = spec.bounds
        values = _apply_transform(
            raw[:, idx], lower=lower, upper=upper, transform=spec.transform
        )
        if spec.param_type == "int":
            values = np.rint(values)
            values = np.clip(values, lower, upper)
        mapped[:, idx] = values
    return mapped


def _apply_transform(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    transform: str,
) -> np.ndarray:
    if transform == "linear":
        return lower + values * (upper - lower)
    if transform == "log":
        lower_log = np.log(lower)
        upper_log = np.log(upper)
        return np.exp(lower_log + values * (upper_log - lower_log))
    if transform == "log10":
        lower_log = np.log10(lower)
        upper_log = np.log10(upper)
        return np.power(10.0, lower_log + values * (upper_log - lower_log))
    raise ConfigError(f"Unsupported transform '{transform}'")


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_qmc():
    try:
        from scipy.stats import qmc  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ConfigError(
            "scipy is required for sobol sampling; install with `uv add scipy`"
        ) from exc
    return qmc
