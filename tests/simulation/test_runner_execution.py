from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from algo_trader.application.simulation import runner
from algo_trader.application.simulation.runner_helpers import (
    should_stop_after,
    with_run_meta,
)
from algo_trader.domain.simulation import SimulationFlags
from algo_trader.domain.simulation import OuterFold


def test_model_research_execution_stops_after_inner() -> None:
    flags = SimulationFlags(
        simulation_mode="full",
        execution_mode="model_research",
    )

    assert not should_stop_after("cv", flags)
    assert should_stop_after("inner", flags)
    assert not should_stop_after("outer", flags)


def test_dry_run_execution_stops_after_cv() -> None:
    flags = SimulationFlags(
        simulation_mode="dry_run",
        execution_mode="full",
    )

    assert should_stop_after("cv", flags)
    assert not should_stop_after("inner", flags)


def test_with_run_meta_includes_execution_mode() -> None:
    flags = SimulationFlags(
        simulation_mode="full",
        execution_mode="walkforward",
    )

    result = with_run_meta({"status": "ok"}, flags)

    assert result["run_mode"] == "full"
    assert result["execution_mode"] == "walkforward"


def test_run_outer_evaluation_sets_fold_seed(
    monkeypatch,
) -> None:
    captured: list[int] = []

    def _capture_seed(seed: int) -> None:
        captured.append(seed)

    def _evaluate_outer_walk_forward(**_: Any) -> tuple[dict[str, str], None]:
        return {"status": "ok"}, None

    def _write_outer_result(**_: Any) -> None:
        return None

    monkeypatch.setattr(
        runner,
        "set_runtime_seed",
        _capture_seed,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_outer_walk_forward",
        _evaluate_outer_walk_forward,
    )
    monkeypatch.setattr(
        runner,
        "_build_portfolio_specs_for_outer",
        lambda *_: (),
    )

    outer_context = runner.OuterFoldContext(
        config=cast(
            Any,
            SimpleNamespace(
                cv=SimpleNamespace(cpcv=SimpleNamespace(seed=7)),
                evaluation=SimpleNamespace(
                    predictive=SimpleNamespace(num_samples_outer=8),
                    allocation=SimpleNamespace(primary=None, baselines=()),
                    cost=SimpleNamespace(spec={}),
                ),
            ),
        ),
        context=cast(
            Any,
            SimpleNamespace(
                X=torch.zeros((1, 1, 1), dtype=torch.float32),
                M=torch.zeros((1, 1, 1), dtype=torch.bool),
                X_global=None,
                M_global=None,
                global_feature_names=(),
                y=torch.zeros((1, 1), dtype=torch.float32),
                timestamps=(0,),
                preprocess_spec=SimpleNamespace(),
                assets=("A",),
            ),
        ),
        base_config={},
        hooks=cast(Any, SimpleNamespace()),
        artifacts=cast(
            Any,
            SimpleNamespace(write_outer_result=_write_outer_result),
        ),
        flags=cast(Any, SimpleNamespace(execution_mode="walkforward")),
        ray_storage_path=None,
    )
    outer_fold = OuterFold(
        k_test=41,
        train_idx=np.array([0], dtype=int),
        test_idx=np.array([0], dtype=int),
        inner_group_ids=[0],
    )

    runner._run_outer_evaluation(  # pylint: disable=protected-access
        outer_context=outer_context,
        outer_fold=outer_fold,
        best_config={},
        week_progress=None,
    )

    assert captured == [410007]
