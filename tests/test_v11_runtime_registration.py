from __future__ import annotations
# pylint: disable=duplicate-code

import pyro
from pyro import poutine
import torch

from algo_trader.pipeline.stages import modeling


def _runtime_batch_index_basis(*, with_targets: bool) -> modeling.ModelBatch:
    asset_names = (
        "EUR.USD",
        "IBUS30",
        "IBUS500",
        "IBUST100",
        "IBDE40",
        "IBES35",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBNL25",
        "IBCH20",
        "XAU.USD",
    )
    X_asset = torch.zeros((2, len(asset_names), 4), dtype=torch.float32)
    X_global = torch.zeros((2, 2), dtype=torch.float32)
    y = None
    if with_targets:
        y = torch.zeros((2, len(asset_names)), dtype=torch.float32)
    return modeling.ModelBatch(
        X_asset=X_asset,
        X_global=X_global,
        y=y,
        asset_names=asset_names,
    )


def test_v11_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "index_subspace_consistency_model_v11_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "index_subspace_consistency_guide_v11_l1_online_filtering"
    )
    train_batch = _runtime_batch_index_basis(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v11_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "index_subspace_consistency_guide_v11_l1_online_filtering"
    )
    predict_batch = _runtime_batch_index_basis(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)


def test_v11_l2_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "index_subspace_consistency_model_v11_l2_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "index_subspace_consistency_guide_v11_l2_online_filtering"
    )
    train_batch = _runtime_batch_index_basis(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    poutine.trace(model).get_trace(train_batch)


def test_v11_l2_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "index_subspace_consistency_guide_v11_l2_online_filtering"
    )
    predict_batch = _runtime_batch_index_basis(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)
