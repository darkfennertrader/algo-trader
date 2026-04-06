from __future__ import annotations

from dataclasses import replace
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    CVParams,
    ModelConfig,
    OuterFold,
    SimulationFlags,
    TrainingConfig,
)
from .index_ranges import indices_to_ranges


def with_fold_seed(cv: CVParams, fold_id: int) -> CVParams:
    return replace(
        cv,
        cpcv=replace(cv.cpcv, seed=_fold_seed(cv.cpcv.seed, fold_id)),
    )


def _fold_seed(base_seed: int, fold_id: int) -> int:
    return base_seed + 10_000 * fold_id


def outer_fold_seed(cv: CVParams, fold_id: int) -> int:
    return _fold_seed(cv.cpcv.seed, fold_id)


def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_group_by_index(
    *, groups: Sequence[np.ndarray], total_len: int
) -> np.ndarray:
    group_by_index = np.full(total_len, -1, dtype=int)
    for group_id, indices in enumerate(groups):
        np.put(group_by_index, np.asarray(indices, dtype=int), group_id)
    return group_by_index


def should_stop_after(phase: str, flags: SimulationFlags) -> bool:
    if flags.execution_mode == "model_research":
        return phase == "inner"
    if flags.simulation_mode == "dry_run":
        return phase == "cv"
    return False


def with_run_meta(
    results: Mapping[str, Any], flags: SimulationFlags
) -> Mapping[str, Any]:
    enriched = dict(results)
    enriched["run_mode"] = flags.simulation_mode
    enriched["execution_mode"] = flags.execution_mode
    return enriched


def resolve_outer_test_group_ids(
    outer_ids: list[int] | None,
    outer_last_n: int | None,
    n_groups: int,
) -> list[int]:
    if outer_ids is not None:
        return outer_ids
    if outer_last_n is not None:
        if outer_last_n <= 0:
            raise ConfigError("outer.last_n must be positive")
        start = max(0, n_groups - outer_last_n)
        return list(range(start, n_groups))
    n_outer = min(5, n_groups)
    return list(range(n_groups - n_outer, n_groups))


def build_base_config(
    model: ModelConfig,
    training: TrainingConfig,
    flags: SimulationFlags,
    debug_output_dir: str | None,
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": model.model_name,
            "guide_name": model.guide_name,
            "predict_name": model.predict_name,
            "params": dict(model.params),
            "guide_params": dict(model.guide_params),
            "predict_params": dict(model.predict_params),
        },
        "debug": {
            "enabled": flags.smoke_test_debug,
            "output_dir": debug_output_dir,
        },
        "training": {
            "method": training.method,
            "target_normalization": training.target_normalization,
            "log_prob_scaling": training.log_prob_scaling,
            "svi_shared": {
                "learning_rate": training.svi_shared.learning_rate,
                "grad_accum_steps": training.svi_shared.grad_accum_steps,
                "num_elbo_particles": (
                    training.svi_shared.num_elbo_particles
                ),
                "log_every": training.svi_shared.log_every,
            },
            "online_filtering": {
                "steps_per_observation": (
                    training.online_filtering.steps_per_observation
                ),
            },
            "tbptt": {
                "num_steps": training.tbptt.num_steps,
                "window_len": training.tbptt.window_len,
                "burn_in_len": training.tbptt.burn_in_len,
            },
        },
    }


def outer_fold_payload(outer_fold: OuterFold) -> Mapping[str, Any]:
    return {
        "k_test": int(outer_fold.k_test),
        "train_ranges": indices_to_ranges(outer_fold.train_idx),
        "test_ranges": indices_to_ranges(outer_fold.test_idx),
        "inner_group_ids": list(outer_fold.inner_group_ids),
    }
