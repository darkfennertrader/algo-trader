from __future__ import annotations

import pytest

from algo_trader.application.simulation.tuning import (
    apply_param_updates,
    build_candidates,
)
from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import TuningParamSpec


def test_build_candidates_sobol_stratified() -> None:
    space = (
        TuningParamSpec(
            path="model.params.activation",
            param_type="categorical",
            values=("relu", "tanh"),
            transform="none",
        ),
        TuningParamSpec(
            path="training.svi.learning_rate",
            param_type="float",
            bounds=(1e-4, 1e-2),
            transform="log10",
            when={"model.params.activation": ("relu",)},
        ),
        TuningParamSpec(
            path="training.svi.num_steps",
            param_type="int",
            bounds=(500, 1500),
            transform="linear",
            when={},
        ),
    )
    candidates = build_candidates(space=space, num_samples=8, seed=7)
    assert len(candidates) == 8
    for candidate in candidates:
        assert candidate["model.params.activation"] in {"relu", "tanh"}
        steps = candidate["training.svi.num_steps"]
        assert 500 <= steps <= 1500
        assert float(steps).is_integer()
        if candidate["model.params.activation"] == "relu":
            lr = candidate["training.svi.learning_rate"]
            assert 1e-4 <= lr <= 1e-2
        else:
            assert "training.svi.learning_rate" not in candidate


def test_build_candidates_rejects_invalid_when() -> None:
    space = (
        TuningParamSpec(
            path="training.svi.num_steps",
            param_type="categorical",
            values=(500, 1000),
            transform="none",
            when={"training.svi.learning_rate": (1e-3,)},
        ),
    )
    with pytest.raises(ConfigError):
        build_candidates(space=space, num_samples=2, seed=1)


def test_apply_param_updates_dot_paths() -> None:
    base = {
        "training": {"svi": {"learning_rate": 1e-3, "num_steps": 1000}},
        "model": {"params": {}},
    }
    merged = apply_param_updates(
        base,
        {
            "training.svi.learning_rate": 5e-4,
            "model.params.activation": "relu",
        },
    )
    assert merged["training"]["svi"]["learning_rate"] == 5e-4
    assert merged["training"]["svi"]["num_steps"] == 1000
    assert merged["model"]["params"]["activation"] == "relu"
