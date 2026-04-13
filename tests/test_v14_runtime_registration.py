from __future__ import annotations

import pyro
from pyro import poutine

from algo_trader.pipeline.stages import modeling
from tests.support.v14_fixtures import build_v14_runtime_batch


def test_v14_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "index_relative_measurement_model_v14_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "index_relative_measurement_guide_v14_l1_online_filtering"
    )
    train_batch = build_v14_runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    trace = poutine.trace(model).get_trace(train_batch)

    assert "non_index_raw_obs" in trace.nodes
    assert "index_relative_level_obs" in trace.nodes
    assert "index_relative_relative_obs" in trace.nodes
    assert "index_relative_residual_obs" in trace.nodes


def test_v14_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "index_relative_measurement_guide_v14_l1_online_filtering"
    )
    predict_batch = build_v14_runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)
