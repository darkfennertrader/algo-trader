from __future__ import annotations

import pyro
from pyro import poutine
import torch

from algo_trader.pipeline.stages import modeling


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
        asset_names=("AUD.CAD", "IBUS500", "XAG.USD"),
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
