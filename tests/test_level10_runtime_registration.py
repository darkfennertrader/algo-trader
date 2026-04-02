from __future__ import annotations

from pathlib import Path

import pyro
from pyro import poutine
import torch

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.infrastructure.data import symbol_directory
from algo_trader.pipeline.stages import modeling


_TICKERS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "tickers.yml"


def _configured_asset_names(count: int) -> tuple[str, ...]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG)
    names = tuple(symbol_directory(ticker) for ticker in config.tickers)
    if len(names) < count:
        raise AssertionError("Not enough configured assets for runtime tests")
    return names[:count]


def _configured_fx_asset_names(count: int) -> tuple[str, ...]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG)
    names = tuple(
        symbol_directory(ticker)
        for ticker in config.tickers
        if "." in symbol_directory(ticker)
    )
    if len(names) < count:
        raise AssertionError("Not enough configured FX assets for runtime tests")
    return names[:count]


def _runtime_batch(*, with_targets: bool) -> modeling.ModelBatch:
    X_asset = torch.zeros((2, 3, 4), dtype=torch.float32)
    X_global = torch.zeros((2, 2), dtype=torch.float32)
    y = None
    if with_targets:
        y = torch.zeros((2, 3), dtype=torch.float32)
    return modeling.ModelBatch(
        X_asset=X_asset,
        X_global=X_global,
        y=y,
        asset_names=_configured_asset_names(3),
    )


def _runtime_batch_fx(*, with_targets: bool) -> modeling.ModelBatch:
    X_asset = torch.zeros((2, 2, 4), dtype=torch.float32)
    X_global = torch.zeros((2, 2), dtype=torch.float32)
    y = None
    if with_targets:
        y = torch.zeros((2, 2), dtype=torch.float32)
    return modeling.ModelBatch(
        X_asset=X_asset,
        X_global=X_global,
        y=y,
        asset_names=_configured_fx_asset_names(2),
    )


def _runtime_batch_unified(*, with_targets: bool) -> modeling.ModelBatch:
    X_asset = torch.zeros((2, 3, 4), dtype=torch.float32)
    X_global = torch.zeros((2, 2), dtype=torch.float32)
    y = None
    if with_targets:
        y = torch.zeros((2, 3), dtype=torch.float32)
    return modeling.ModelBatch(
        X_asset=X_asset,
        X_global=X_global,
        y=y,
        asset_names=("EUR.USD", "IBUS500", "XAUUSD"),
    )


def test_level10_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l10_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l10_online_filtering"
    )
    train_batch = _runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level10_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l10_online_filtering"
    )
    predict_batch = _runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_level11_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l11_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l11_online_filtering"
    )
    train_batch = _runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level11_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l11_online_filtering"
    )
    predict_batch = _runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_level12_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l12_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l12_online_filtering"
    )
    train_batch = _runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level12_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l12_online_filtering"
    )
    predict_batch = _runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_level13_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l13_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l13_online_filtering"
    )
    train_batch = _runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level13_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l13_online_filtering"
    )
    predict_batch = _runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_level14_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l14_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l14_online_filtering"
    )
    train_batch = _runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level14_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l14_online_filtering"
    )
    predict_batch = _runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_level15_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "factor_model_l15_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l15_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_level15_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "factor_guide_l15_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l1_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l1_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l2_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l2_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l2_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l2_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l2_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l1_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l1_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l1_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l1_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l1_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l2_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l2_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l2_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l2_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l2_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l3_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l3_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l3_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l3_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l3_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l4_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l4_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l4_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l4_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l4_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l6_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l6_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l6_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l6_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l6_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l7_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l7_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l7_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l7_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l7_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l8_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l8_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l8_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l8_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l8_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l9_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l9_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l9_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l9_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l9_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l10_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l10_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l10_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l10a_clean_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l10a_clean_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10a_clean_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l10a_clean_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10a_clean_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l10b_clean_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l10b_clean_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10b_clean_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l10b_clean_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10b_clean_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v3_l10c_clean_unified_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "multi_asset_block_model_v3_l10c_clean_unified_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10c_clean_unified_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v3_l10c_clean_unified_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "multi_asset_block_guide_v3_l10c_clean_unified_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v4_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "dependence_layer_model_v4_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l1_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v4_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l1_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v4_l2_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "dependence_layer_model_v4_l2_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l2_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v4_l2_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l2_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v4_l3_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "dependence_layer_model_v4_l3_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l3_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v4_l3_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "dependence_layer_guide_v4_l3_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v5_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "residual_copula_model_v5_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "residual_copula_guide_v5_l1_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v5_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "residual_copula_guide_v5_l1_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v5_l2_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "residual_copula_model_v5_l2_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "residual_copula_guide_v5_l2_online_filtering"
    )
    train_batch = _runtime_batch_unified(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v5_l2_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "residual_copula_guide_v5_l2_online_filtering"
    )
    predict_batch = _runtime_batch_unified(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l3_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l3_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l3_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l3_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l3_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l4_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l4_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l4_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l4_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l4_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l5_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l5_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l5_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l5_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l5_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v2_l6_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "fx_currency_factor_model_v2_l6_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l6_online_filtering"
    )
    train_batch = _runtime_batch_fx(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v2_l6_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "fx_currency_factor_guide_v2_l6_online_filtering"
    )
    predict_batch = _runtime_batch_fx(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)
