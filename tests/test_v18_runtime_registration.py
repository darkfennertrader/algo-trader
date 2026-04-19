from __future__ import annotations

import pyro
from pyro import poutine

from algo_trader.pipeline.stages import modeling
from tests.support.v14_fixtures import build_v14_runtime_batch


def test_v18_l1_runtime_registry_builds_model_and_guide() -> None:
    model = modeling.default_model_registry().get(
        "pair_state_conditioned_curated_pair_model_v18_l1_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering"
    )
    train_batch = build_v14_runtime_batch(with_targets=True)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(train_batch)
    trace = poutine.trace(model).get_trace(train_batch)

    assert "obs" in trace.nodes
    assert "pair_state_conditioned_curated_pair_range_obs" in trace.nodes
    assert "pair_state_conditioned_curated_pair_residual_obs" in trace.nodes


def test_v18_l1_runtime_guide_accepts_prediction_batch() -> None:
    guide = modeling.default_guide_registry().get(
        "pair_state_conditioned_curated_pair_guide_v18_l1_online_filtering"
    )
    predict_batch = build_v14_runtime_batch(with_targets=False)

    pyro.clear_param_store()
    poutine.trace(guide).get_trace(predict_batch)
