from __future__ import annotations

from typing import Any

import pytest
import torch

from algo_trader.application.simulation import hooks
from algo_trader.domain import ConfigError

_TEST_UNIFIED_ASSET_NAMES_3 = ("EUR.USD", "IBUS500", "XAU.USD")


def test_v3_l10a_clean_unified_prediction_supports_explicit_predictor_config() -> None:
    config = _v3_l10a_clean_unified_online_filtering_config(
        predict_name=(
            "multi_asset_block_predict_v3_l10a_clean_unified_online_filtering"
        )
    )
    X_train = torch.randn((3, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((3, 1), dtype=torch.float32)
    y_train = torch.randn((3, 3), dtype=torch.float32)
    X_pred = torch.randn((2, 3, 1), dtype=torch.float32)
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

    assert pred["samples"].shape == (4, 2, 3)
    assert pred["mean"].shape == (2, 3)
    assert pred["covariance"].shape == (2, 3, 3)
    assert state["filtering_state"]["steps_seen"] == 3
    assert state["filtering_state"]["h_loc"].shape == (4,)


def test_v3_l10a_clean_unified_rejects_tbptt_training_method() -> None:
    config = _v3_l10a_clean_unified_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 3), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "multi_asset_block_model_v3_l10a_clean_unified_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def _v3_l10a_clean_unified_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "multi_asset_block_model_v3_l10a_clean_unified_online_filtering",
            "guide_name": "multi_asset_block_guide_v3_l10a_clean_unified_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_UNIFIED_ASSET_NAMES_3),
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
