from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import ModelPrebuildConfig, OuterFold
from algo_trader.pipeline import modeling

from .artifacts import SimulationArtifacts

@dataclass(frozen=True)
class PrebuildRunResult:
    result: modeling.PrebuildResult
    train_idx: np.ndarray
    train_group_ids: list[int]


@dataclass(frozen=True)
class PrebuildInputs:
    X: torch.Tensor
    y: torch.Tensor
    M: torch.Tensor
    outer_folds: Sequence[OuterFold]
    group_by_index: np.ndarray
    feature_names: Sequence[str]
    assets: Sequence[str]


def run_prebuild(
    *,
    prebuild: ModelPrebuildConfig,
    inputs: PrebuildInputs,
) -> PrebuildRunResult:
    if not prebuild.enabled:
        raise ConfigError(
            "Prebuild is disabled; call should be skipped",
            context={"prebuild": prebuild.name},
        )
    train_idx = _resolve_train_idx(inputs.outer_folds)
    if train_idx.size == 0:
        raise SimulationError(
            "No training samples available for prebuild",
            context={"prebuild": prebuild.name},
        )
    context = _build_prebuild_context(
        prebuild=prebuild,
        inputs=inputs,
        train_idx=train_idx,
    )
    registry = modeling.default_prebuild_registry()
    hook = registry.get(prebuild.name)
    result = hook(context)
    if not isinstance(result, modeling.PrebuildResult):
        raise SimulationError(
            "Prebuild hook must return PrebuildResult",
            context={"prebuild": prebuild.name},
        )
    train_group_ids = _train_group_ids(
        train_idx=train_idx, group_by_index=inputs.group_by_index
    )
    return PrebuildRunResult(
        result=result,
        train_idx=train_idx,
        train_group_ids=train_group_ids,
    )


def maybe_run_prebuild(
    *,
    prebuild: ModelPrebuildConfig | None,
    inputs: PrebuildInputs,
    artifacts: SimulationArtifacts,
) -> PrebuildRunResult | None:
    if prebuild is None or not prebuild.enabled:
        return None
    result = run_prebuild(prebuild=prebuild, inputs=inputs)
    artifacts.write_prebuild(payload=_build_prebuild_payload(prebuild, result))
    return result


def apply_prebuild(
    base_config: Mapping[str, Any],
    prebuild: PrebuildRunResult,
) -> Mapping[str, Any]:
    merged = dict(base_config)
    model_section = dict(merged.get("model", {}))
    model_params = _deep_merge(
        model_section.get("params", {}),
        prebuild.result.model_params,
    )
    guide_params = _deep_merge(
        model_section.get("guide_params", {}),
        prebuild.result.guide_params,
    )
    model_section["params"] = model_params
    model_section["guide_params"] = guide_params
    merged["model"] = model_section
    return merged


def _build_prebuild_payload(
    prebuild: ModelPrebuildConfig, result: PrebuildRunResult
) -> Mapping[str, Any]:
    return {
        "name": prebuild.name,
        "enabled": prebuild.enabled,
        "params": dict(prebuild.params),
        "n_train_samples": int(result.train_idx.size),
        "train_group_ids": list(result.train_group_ids),
        "model_params": dict(result.result.model_params),
        "guide_params": dict(result.result.guide_params),
        "metadata": dict(result.result.metadata),
    }


def _deep_merge(
    base: Mapping[str, Any], updates: Mapping[str, Any]
) -> Mapping[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_prebuild_context(
    *,
    prebuild: ModelPrebuildConfig,
    inputs: PrebuildInputs,
    train_idx: np.ndarray,
) -> modeling.PrebuildContext:
    return modeling.PrebuildContext(
        X_train=inputs.X[train_idx],
        y_train=inputs.y[train_idx],
        M_train=inputs.M[train_idx],
        feature_names=inputs.feature_names,
        assets=inputs.assets,
        params=prebuild.params,
    )


def _resolve_train_idx(outer_folds: Sequence[OuterFold]) -> np.ndarray:
    if not outer_folds:
        raise SimulationError("No outer folds available for prebuild")
    intersection: set[int] | None = None
    for fold in outer_folds:
        fold_idx = set(int(item) for item in fold.train_idx.tolist())
        if intersection is None:
            intersection = fold_idx
        else:
            intersection &= fold_idx
    if intersection is None:
        return np.array([], dtype=int)
    return np.array(sorted(intersection), dtype=int)


def _train_group_ids(
    *, train_idx: np.ndarray, group_by_index: np.ndarray
) -> list[int]:
    if train_idx.size == 0:
        return []
    groups = group_by_index[train_idx]
    group_ids = {
        int(value) for value in groups.tolist() if int(value) >= 0
    }
    return sorted(group_ids)
