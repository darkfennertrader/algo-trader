from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    CVParams,
    ModelConfig,
    OuterFold,
    SimulationFlags,
    TrainingConfig,
)


def with_fold_seed(cv: CVParams, fold_id: int) -> CVParams:
    return replace(
        cv,
        cpcv=replace(cv.cpcv, seed=_fold_seed(cv.cpcv.seed, fold_id)),
    )


def _fold_seed(base_seed: int, fold_id: int) -> int:
    return base_seed + 10_000 * fold_id


def build_group_by_index(
    *, groups: Sequence[np.ndarray], total_len: int
) -> np.ndarray:
    group_by_index = np.full(total_len, -1, dtype=int)
    for group_id, indices in enumerate(groups):
        np.put(group_by_index, np.asarray(indices, dtype=int), group_id)
    return group_by_index


def should_stop_after(phase: str, flags: SimulationFlags) -> bool:
    if flags.stop_after is not None:
        return phase == flags.stop_after
    if flags.simulation_mode == "dry_run":
        return phase == "cv"
    return False


def with_run_meta(
    results: Mapping[str, Any], flags: SimulationFlags
) -> Mapping[str, Any]:
    enriched = dict(results)
    enriched["run_mode"] = flags.simulation_mode
    enriched["stop_after"] = flags.stop_after
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
            "params": dict(model.params),
            "guide_params": dict(model.guide_params),
        },
        "debug": {
            "enabled": flags.smoke_test_debug,
            "output_dir": debug_output_dir,
        },
        "training": {
            "target_normalization": training.target_normalization,
            "log_prob_scaling": training.log_prob_scaling,
            "svi": {
                "num_steps": training.svi.num_steps,
                "learning_rate": training.svi.learning_rate,
                "tbptt_window_len": training.svi.tbptt_window_len,
                "tbptt_burn_in_len": training.svi.tbptt_burn_in_len,
                "grad_accum_steps": training.svi.grad_accum_steps,
                "num_elbo_particles": training.svi.num_elbo_particles,
                "log_every": training.svi.log_every,
            }
        },
    }


def outer_fold_payload(outer_fold: OuterFold) -> Mapping[str, Any]:
    return {
        "k_test": int(outer_fold.k_test),
        "train_idx": outer_fold.train_idx.tolist(),
        "test_idx": outer_fold.test_idx.tolist(),
        "inner_group_ids": list(outer_fold.inner_group_ids),
    }
