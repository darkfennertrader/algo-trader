from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from algo_trader.application.simulation.hooks_types import SimulationHooks
from algo_trader.application.simulation.inner_objective import (
    InnerObjectiveContext,
)
from algo_trader.application.simulation.tune_runner import (
    RayTuneContext,
    RayTuneInputs,
    RayTuneRuntimeSpec,
    RayTuneSpec,
    _build_scheduler,
    _build_trainable,
    _train_candidate,
)
from algo_trader.domain.simulation import (
    CandidateSpec,
    TuningRayConfig,
    TuningRayEarlyStoppingConfig,
    TuningResourcesConfig,
)


class _FakeTune:
    def __init__(self) -> None:
        self.trainable: Any = None
        self.kwargs: dict[str, Any] = {}
        self.schedulers = _FakeSchedulers()

    def with_parameters(self, trainable: Any, **kwargs: Any) -> str:
        self.trainable = trainable
        self.kwargs = kwargs
        return "wrapped-trainable"


class _FakeSchedulers:
    def MedianStoppingRule(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        return kwargs


def test_build_trainable_uses_with_parameters_for_large_context() -> None:
    tune = _FakeTune()
    inner_context = cast(InnerObjectiveContext, object())
    hooks = cast(SimulationHooks, object())
    context = RayTuneContext(
        inputs=RayTuneInputs(
            objective=object(),
            inner_context=inner_context,
            hooks=hooks,
        ),
        spec=RayTuneSpec(
            base_config={"training": {}},
            candidates=(CandidateSpec(candidate_id=0, params={}),),
            resources=TuningResourcesConfig(),
            ray_config=TuningRayConfig(),
            use_gpu=False,
            runtime=RayTuneRuntimeSpec(
                storage_path=Path("/tmp"),
                experiment_name="exp",
                address=None,
                resume_experiment_dir=None,
                logs_enabled=False,
            ),
        ),
    )

    wrapped = _build_trainable(context, tune)

    assert wrapped == "wrapped-trainable"
    assert tune.trainable is _train_candidate
    assert tune.kwargs["base_config"] == {"training": {}}
    assert tune.kwargs["inner_context"] is inner_context
    assert tune.kwargs["hooks"] is hooks


def test_build_scheduler_returns_none_when_disabled() -> None:
    scheduler = _build_scheduler(
        tune=_FakeTune(),
        ray_config=TuningRayConfig(
            early_stopping=TuningRayEarlyStoppingConfig(enabled=False)
        ),
    )

    assert scheduler is None


def test_build_scheduler_builds_median_rule_on_completed_splits() -> None:
    scheduler = _build_scheduler(
        tune=_FakeTune(),
        ray_config=TuningRayConfig(
            early_stopping=TuningRayEarlyStoppingConfig(
                enabled=True,
                method="median",
                grace_period=16,
                min_samples_required=4,
            )
        ),
    )

    assert scheduler == {
        "time_attr": "completed_splits",
        "grace_period": 16,
        "min_samples_required": 4,
    }
