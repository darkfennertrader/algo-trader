from __future__ import annotations

from pathlib import Path

import pytest

from algo_trader.application.simulation.resume_flow import (
    RaySelectionContext,
    ResumeState,
    resolve_ray_selection_plan,
)
from algo_trader.application.simulation.resume_manifest import (
    SimulationResumeTracker,
)
from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import TuningConfig


def test_resume_tracker_persists_progress(tmp_path: Path) -> None:
    tracker = SimulationResumeTracker(
        base_dir=tmp_path,
        outer_ids=[1, 2, 3],
        model_selection_enabled=False,
        resume_requested=False,
    )
    run_id = tracker.run_id

    tracker.mark_inner_started(1)
    tracker.mark_inner_completed(1)
    tracker.mark_outer_started(1)
    tracker.mark_outer_completed(1)

    resumed = SimulationResumeTracker(
        base_dir=tmp_path,
        outer_ids=[1, 2, 3],
        model_selection_enabled=False,
        resume_requested=True,
    )

    assert resumed.run_id == run_id
    assert resumed.is_inner_completed(1)
    assert resumed.is_outer_completed(1)
    assert not resumed.is_outer_completed(2)


def test_resume_tracker_rejects_completed_run(tmp_path: Path) -> None:
    tracker = SimulationResumeTracker(
        base_dir=tmp_path,
        outer_ids=[1, 2],
        model_selection_enabled=False,
        resume_requested=False,
    )
    tracker.mark_outer_completed(1)
    tracker.mark_outer_completed(2)
    tracker.mark_run_completed()

    with pytest.raises(
        ConfigError,
        match="No interrupted Ray Tune experiment to resume",
    ):
        SimulationResumeTracker(
            base_dir=tmp_path,
            outer_ids=[1, 2],
            model_selection_enabled=False,
            resume_requested=True,
        )


def test_resolve_ray_selection_plan_uses_resume_dir(tmp_path: Path) -> None:
    tracker = SimulationResumeTracker(
        base_dir=tmp_path,
        outer_ids=[5],
        model_selection_enabled=False,
        resume_requested=False,
    )
    storage_path = tmp_path / "ray_results"
    resume_state = ResumeState(enabled=True)

    plan = resolve_ray_selection_plan(
        context=RaySelectionContext(
            tuning=TuningConfig(engine="ray"),
            tuning_engine="ray",
            ray_storage_path=storage_path,
            outer_k=5,
            resume_tracker=tracker,
        ),
        resume_state=resume_state,
    )

    assert plan.resume_requested
    assert (
        plan.resume_experiment_dir
        == storage_path / f"algotrader_{tracker.run_id}_outer_5"
    )
    assert not resume_state.enabled
