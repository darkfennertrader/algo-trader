from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from algo_trader.application.simulation import artifacts as simulation_artifacts
from algo_trader.application.simulation import runner
from algo_trader.application.simulation.artifacts import SimulationArtifacts
from algo_trader.domain.simulation import CandidateSpec, CPCVSplit, OuterFold


def test_inner_only_writes_splits_before_post_tune_selection(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts = SimulationArtifacts(tmp_path)
    inner_splits = [
        CPCVSplit(
            train_idx=np.array([0, 1], dtype=int),
            test_idx=np.array([2], dtype=int),
            test_group_ids=(3,),
        )
    ]
    outer_fold = OuterFold(
        k_test=17,
        train_idx=np.array([0, 1], dtype=int),
        test_idx=np.array([2], dtype=int),
        inner_group_ids=[3],
    )
    outer_context = runner.OuterFoldContext(
        config=SimpleNamespace(
            modeling=SimpleNamespace(tuning=SimpleNamespace(engine="ray")),
            evaluation=SimpleNamespace(
                model_selection=SimpleNamespace(enable=True),
            ),
        ),
        context=SimpleNamespace(
            cv=SimpleNamespace(warmup_idx=np.array([0, 1], dtype=int)),
        ),
        base_config={},
        hooks=SimpleNamespace(),
        artifacts=artifacts,
        flags=SimpleNamespace(use_gpu=False),
        ray_storage_path=tmp_path / "ray_results",
    )

    monkeypatch.setattr(
        simulation_artifacts,
        "write_splits_timeline_plot",
        lambda **_: None,
    )
    monkeypatch.setattr(
        runner,
        "_build_inner_splits",
        lambda **_: inner_splits,
    )
    monkeypatch.setattr(
        runner,
        "_build_inner_objective",
        lambda **_: runner.InnerObjectiveBundle(
            objective=object(),
            context=SimpleNamespace(params=SimpleNamespace(score_spec={})),
            hooks=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_postprocess_metadata",
        lambda **_: None,
    )
    monkeypatch.setattr(
        runner,
        "resolve_ray_selection_plan",
        lambda **_: SimpleNamespace(
            resume_requested=False,
            experiment_name="exp",
            resume_experiment_dir=None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_inner_outputs",
        lambda **_: None,
    )
    monkeypatch.setattr(
        runner,
        "_log_outer_complete",
        lambda **_: None,
    )

    def _fake_select_best_config(*args, **kwargs):
        splits_path = tmp_path / "inner" / "outer_17" / "splits.json"
        assert splits_path.exists()
        return {"model": {"selected": True}}

    monkeypatch.setattr(
        runner,
        "select_best_config",
        _fake_select_best_config,
    )

    result = runner._run_outer_fold_inner_only(  # pylint: disable=protected-access
        outer_context=outer_context,
        outer_fold=outer_fold,
        candidates=(CandidateSpec(candidate_id=0, params={}),),
        resume_state=SimpleNamespace(enabled=False),
        resume_tracker=None,
    )

    assert result == {"model": {"selected": True}}
