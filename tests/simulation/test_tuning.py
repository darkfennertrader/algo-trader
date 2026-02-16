from __future__ import annotations

import pytest

from algo_trader.application.simulation.tuning import (
    apply_param_updates,
    build_candidates,
)
from algo_trader.domain import ConfigError


def test_build_candidates_random_mixed() -> None:
    param_space = {
        "training.learning_rate": {"min": 1e-4, "max": 1e-2, "dtype": "float"},
        "training.num_steps": [500, 1000],
        "model_params.activation": ["relu", "tanh"],
    }
    candidates = build_candidates(
        param_space=param_space,
        num_samples=5,
        seed=7,
        sampling_method="random",
    )
    assert len(candidates) == 5
    for candidate in candidates:
        assert 1e-4 <= candidate["training.learning_rate"] <= 1e-2
        assert candidate["training.num_steps"] in {500, 1000}
        assert candidate["model_params.activation"] in {"relu", "tanh"}


def test_apply_param_updates_dot_paths() -> None:
    base = {
        "training": {"learning_rate": 1e-3, "num_steps": 1000},
        "model_params": {},
    }
    merged = apply_param_updates(
        base,
        {
            "training.learning_rate": 5e-4,
            "model_params.activation": "relu",
        },
    )
    assert merged["training"]["learning_rate"] == 5e-4
    assert merged["training"]["num_steps"] == 1000
    assert merged["model_params"]["activation"] == "relu"


def test_grid_rejects_continuous_ranges() -> None:
    param_space = {"training.learning_rate": {"min": 1e-4, "max": 1e-2}}
    with pytest.raises(ConfigError):
        build_candidates(
            param_space=param_space,
            num_samples=2,
            seed=1,
            sampling_method="grid",
        )
