from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.application.simulation import hooks
from algo_trader.domain import ConfigError
from algo_trader.infrastructure.data import symbol_directory

_TICKERS_CONFIG = Path(__file__).resolve().parents[2] / "config" / "tickers.yml"


def _configured_asset_names(count: int) -> tuple[str, ...]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG)
    names = tuple(symbol_directory(ticker) for ticker in config.tickers)
    if len(names) < count:
        raise AssertionError(
            "Not enough configured assets for prediction tests"
        )
    return names[:count]


def _configured_fx_asset_names(count: int) -> tuple[str, ...]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG)
    names = tuple(
        symbol_directory(ticker)
        for ticker in config.tickers
        if "." in symbol_directory(ticker)
    )
    if len(names) < count:
        raise AssertionError("Not enough configured FX assets for prediction tests")
    return names[:count]


_TEST_ASSET_NAMES_2 = _configured_asset_names(2)
_TEST_FX_ASSET_NAMES_2 = _configured_fx_asset_names(2)
_TEST_UNIFIED_ASSET_NAMES_3 = ("EUR.USD", "IBUS500", "XAUUSD")


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


def test_level14_prediction_supports_explicit_predictor_config() -> None:
    config = _level14_online_filtering_config(
        predict_name="factor_predict_l14_online_filtering"
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


def test_level14_rejects_tbptt_training_method() -> None:
    config = _level14_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l14_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_level15_prediction_supports_explicit_predictor_config() -> None:
    config = _level15_online_filtering_config(
        predict_name="factor_predict_l15_online_filtering"
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


def test_level15_rejects_tbptt_training_method() -> None:
    config = _level15_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "factor_model_l15_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l1_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l1_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l1_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert torch.is_tensor(state["structural_posterior_means"]["gamma_currency"])


def test_v2_l1_rejects_tbptt_training_method() -> None:
    config = _v2_l1_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l1_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l2_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l2_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l2_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert torch.is_tensor(state["structural_posterior_means"]["gamma_currency"])


def test_v2_l2_rejects_tbptt_training_method() -> None:
    config = _v2_l2_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l2_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l3_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l3_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l3_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert torch.is_tensor(state["structural_posterior_means"]["omega_currency"])


def test_v2_l3_rejects_tbptt_training_method() -> None:
    config = _v2_l3_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l3_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l4_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l4_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l4_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert torch.is_tensor(state["structural_posterior_means"]["alpha_currency"])
    assert torch.is_tensor(state["structural_posterior_means"]["theta_currency"])


def test_v2_l4_rejects_tbptt_training_method() -> None:
    config = _v2_l4_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l4_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l5_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l5_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l5_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert torch.is_tensor(state["structural_posterior_means"]["alpha"])
    assert torch.is_tensor(state["structural_posterior_means"]["alpha_currency"])
    assert torch.is_tensor(state["structural_posterior_means"]["theta_currency"])


def test_v2_l5_rejects_tbptt_training_method() -> None:
    config = _v2_l5_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l5_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v2_l6_prediction_supports_explicit_predictor_config() -> None:
    config = _v2_l6_online_filtering_config(
        predict_name="fx_currency_factor_predict_v2_l6_online_filtering"
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
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)
    assert state["filtering_state"]["steps_seen"] == 3
    assert state["filtering_state"]["h_loc"].shape == (2,)
    assert torch.is_tensor(
        state["structural_posterior_means"]["B_currency_broad"]
    )
    assert torch.is_tensor(
        state["structural_posterior_means"]["B_currency_cross"]
    )


def test_v2_l6_rejects_tbptt_training_method() -> None:
    config = _v2_l6_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 2, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 2), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "fx_currency_factor_model_v2_l6_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v3_l1_unified_prediction_supports_explicit_predictor_config() -> None:
    config = _v3_l1_unified_online_filtering_config(
        predict_name="multi_asset_block_predict_v3_l1_unified_online_filtering"
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
    assert torch.is_tensor(state["structural_posterior_means"]["B_global"])
    assert torch.is_tensor(state["structural_posterior_means"]["B_index"])


def test_v3_l1_unified_rejects_tbptt_training_method() -> None:
    config = _v3_l1_unified_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 3), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "multi_asset_block_model_v3_l1_unified_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v3_l2_unified_prediction_supports_explicit_predictor_config() -> None:
    config = _v3_l2_unified_online_filtering_config(
        predict_name="multi_asset_block_predict_v3_l2_unified_online_filtering"
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
    assert torch.is_tensor(state["structural_posterior_means"]["B_global"])
    assert torch.is_tensor(state["structural_posterior_means"]["B_index"])


def test_v3_l2_unified_rejects_tbptt_training_method() -> None:
    config = _v3_l2_unified_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 3), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "multi_asset_block_model_v3_l2_unified_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v3_l3_unified_prediction_supports_explicit_predictor_config() -> None:
    config = _v3_l3_unified_online_filtering_config(
        predict_name="multi_asset_block_predict_v3_l3_unified_online_filtering"
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
    assert torch.is_tensor(state["structural_posterior_means"]["B_global"])
    assert torch.is_tensor(state["structural_posterior_means"]["B_index_static"])


def test_v3_l3_unified_rejects_tbptt_training_method() -> None:
    config = _v3_l3_unified_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 3), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "multi_asset_block_model_v3_l3_unified_online_filtering "
            "does not support training.method=tbptt"
        ),
    ):
        hooks._fit_pyro(  # pylint: disable=protected-access
            X_train=X_train,
            X_train_global=X_train_global,
            y_train=y_train,
            config=config,
        )


def test_v3_l4_unified_prediction_supports_explicit_predictor_config() -> None:
    config = _v3_l4_unified_online_filtering_config(
        predict_name="multi_asset_block_predict_v3_l4_unified_online_filtering"
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
    assert torch.is_tensor(state["structural_posterior_means"]["B_global"])
    assert torch.is_tensor(
        state["structural_posterior_means"]["index_group_scale"]
    )


def test_v3_l4_unified_rejects_tbptt_training_method() -> None:
    config = _v3_l4_unified_online_filtering_config()
    config["training"]["method"] = "tbptt"
    config["training"]["target_normalization"] = True
    X_train = torch.randn((2, 3, 1), dtype=torch.float32)
    X_train_global = torch.randn((2, 1), dtype=torch.float32)
    y_train = torch.randn((2, 3), dtype=torch.float32)

    with pytest.raises(
        ConfigError,
        match=(
            "multi_asset_block_model_v3_l4_unified_online_filtering "
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


def _level14_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l14_online_filtering",
            "guide_name": "factor_guide_l14_online_filtering",
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


def _level15_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "factor_model_l15_online_filtering",
            "guide_name": "factor_guide_l15_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l1_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l1_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l1_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l2_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l2_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l2_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l3_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l3_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l3_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l4_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l4_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l4_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l5_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l5_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l5_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v2_l6_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "fx_currency_factor_model_v2_l6_online_filtering",
            "guide_name": "fx_currency_factor_guide_v2_l6_online_filtering",
            "predict_name": predict_name,
        },
        "run_context": {
            "asset_names": list(_TEST_FX_ASSET_NAMES_2),
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


def _v3_l1_unified_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "multi_asset_block_model_v3_l1_unified_online_filtering",
            "guide_name": "multi_asset_block_guide_v3_l1_unified_online_filtering",
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


def _v3_l2_unified_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "multi_asset_block_model_v3_l2_unified_online_filtering",
            "guide_name": "multi_asset_block_guide_v3_l2_unified_online_filtering",
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


def _v3_l3_unified_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "multi_asset_block_model_v3_l3_unified_online_filtering",
            "guide_name": "multi_asset_block_guide_v3_l3_unified_online_filtering",
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


def _v3_l4_unified_online_filtering_config(
    *, predict_name: str | None = None
) -> dict[str, Any]:
    return {
        "model": {
            "model_name": "multi_asset_block_model_v3_l4_unified_online_filtering",
            "guide_name": "multi_asset_block_guide_v3_l4_unified_online_filtering",
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
