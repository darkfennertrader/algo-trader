from __future__ import annotations

from typing import Any

import pytest
import torch

from algo_trader.application.simulation import hooks
from algo_trader.domain import ConfigError

_TEST_ASSET_NAMES_2 = ("AUD.CAD", "IBUS500")


def test_level10_prediction_uses_custom_online_filtering_rollout() -> None:
    config = _level10_online_filtering_config()
    X_train = torch.randn((3, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 2), dtype=torch.float32)
    X_pred = torch.randn((2, 2, 1), dtype=torch.float32)
    X_pred_global = torch.randn((2, 1), dtype=torch.float32)

    state = hooks._fit_pyro(  # pylint: disable=protected-access
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
        config=config,
    )
    pred = hooks._predict_pyro(  # pylint: disable=protected-access
        X_pred=X_pred,
        X_pred_global=X_pred_global,
        state=state,
        config=config,
        num_samples=8,
    )

    assert pred["samples"].shape == (8, 2, 2)
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert "structural_posterior_means" in state
    assert torch.is_tensor(state["structural_posterior_means"]["alpha"])


def test_level10_prediction_supports_explicit_predictor_config() -> None:
    config = _level10_online_filtering_config(
        predict_name="factor_predict_l10_online_filtering"
    )
    X_train = torch.randn((3, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 2), dtype=torch.float32)
    X_pred = torch.randn((2, 2, 1), dtype=torch.float32)
    X_pred_global = torch.randn((2, 1), dtype=torch.float32)

    state = hooks._fit_pyro(  # pylint: disable=protected-access
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
        config=config,
    )
    pred = hooks._predict_pyro(  # pylint: disable=protected-access
        X_pred=X_pred,
        X_pred_global=X_pred_global,
        state=state,
        config=config,
        num_samples=4,
    )

    assert pred["samples"].shape == (4, 2, 2)


def test_level10_rejects_tbptt_training_method() -> None:
    config = _level10_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l10_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_level11_prediction_supports_explicit_predictor_config() -> None:
    config = _level11_online_filtering_config(
        predict_name="factor_predict_l11_online_filtering"
    )
    X_train = torch.randn((3, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 2), dtype=torch.float32)
    X_pred = torch.randn((2, 2, 1), dtype=torch.float32)
    X_pred_global = torch.randn((2, 1), dtype=torch.float32)

    state = hooks._fit_pyro(  # pylint: disable=protected-access
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
        config=config,
    )
    pred = hooks._predict_pyro(  # pylint: disable=protected-access
        X_pred=X_pred,
        X_pred_global=X_pred_global,
        state=state,
        config=config,
        num_samples=4,
    )

    assert pred["samples"].shape == (4, 2, 2)


def test_level11_rejects_tbptt_training_method() -> None:
    config = _level11_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l11_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_level12_prediction_supports_explicit_predictor_config() -> None:
    config = _level12_online_filtering_config(
        predict_name="factor_predict_l12_online_filtering"
    )
    X_train = torch.randn((3, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 2), dtype=torch.float32)
    X_pred = torch.randn((2, 2, 1), dtype=torch.float32)
    X_pred_global = torch.randn((2, 1), dtype=torch.float32)

    state = hooks._fit_pyro(  # pylint: disable=protected-access
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
        config=config,
    )
    pred = hooks._predict_pyro(  # pylint: disable=protected-access
        X_pred=X_pred,
        X_pred_global=X_pred_global,
        state=state,
        config=config,
        num_samples=4,
    )

    assert pred["samples"].shape == (4, 2, 2)


def test_level12_rejects_tbptt_training_method() -> None:
    config = _level12_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l12_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_level13_prediction_supports_explicit_predictor_config() -> None:
    config = _level13_online_filtering_config(
        predict_name="factor_predict_l13_online_filtering"
    )
    X_train = torch.randn((3, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 2), dtype=torch.float32)
    X_pred = torch.randn((2, 2, 1), dtype=torch.float32)
    X_pred_global = torch.randn((2, 1), dtype=torch.float32)

    state = hooks._fit_pyro(  # pylint: disable=protected-access
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=y_train,
        config=config,
    )
    pred = hooks._predict_pyro(  # pylint: disable=protected-access
        X_pred=X_pred,
        X_pred_global=X_pred_global,
        state=state,
        config=config,
        num_samples=4,
    )

    assert pred["samples"].shape == (4, 2, 2)


def test_level13_rejects_tbptt_training_method() -> None:
    config = _level13_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l13_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def _level10_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l10_online_filtering",
            "guide_name": "factor_guide_l10_online_filtering",
            "predict_name": predict_name,
        },
        "training": {
            "method": "online_filtering",
            "svi_shared": {
                "learning_rate": 1e-3,
                "num_elbo_particles": 1,
                "grad_accum_steps": 1,
                "log_every": None,
            },
            "tbptt": {
                "num_steps": 1,
                "window_len": None,
                "burn_in_len": 0,
            },
            "online_filtering": {
                "steps_per_observation": 1,
            },
            "log_prob_scaling": False,
            "target_normalization": False,
        },
    }


def _level11_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l11_online_filtering",
            "guide_name": "factor_guide_l11_online_filtering",
            "predict_name": predict_name,
        },
        "training": {
            "method": "online_filtering",
            "svi_shared": {
                "learning_rate": 1e-3,
                "num_elbo_particles": 1,
                "grad_accum_steps": 1,
                "log_every": None,
            },
            "tbptt": {
                "num_steps": 1,
                "window_len": None,
                "burn_in_len": 0,
            },
            "online_filtering": {
                "steps_per_observation": 1,
            },
            "log_prob_scaling": False,
            "target_normalization": False,
        },
    }


def _level12_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l12_online_filtering",
            "guide_name": "factor_guide_l12_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_ASSET_NAMES_2),
        },
        "training": {
            "method": "online_filtering",
            "svi_shared": {
                "learning_rate": 1e-3,
                "num_elbo_particles": 1,
                "grad_accum_steps": 1,
                "log_every": None,
            },
            "tbptt": {
                "num_steps": 1,
                "window_len": None,
                "burn_in_len": 0,
            },
            "online_filtering": {
                "steps_per_observation": 1,
            },
            "log_prob_scaling": False,
            "target_normalization": False,
        },
    }


def _level13_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l13_online_filtering",
            "guide_name": "factor_guide_l13_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_ASSET_NAMES_2),
        },
        "training": {
            "method": "online_filtering",
            "svi_shared": {
                "learning_rate": 1e-3,
                "num_elbo_particles": 1,
                "grad_accum_steps": 1,
                "log_every": None,
            },
            "tbptt": {
                "num_steps": 1,
                "window_len": None,
                "burn_in_len": 0,
            },
            "online_filtering": {
                "steps_per_observation": 1,
            },
            "log_prob_scaling": False,
            "target_normalization": False,
        },
    }
