from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import SimulationConfig, TuningConfig

from .resume_manifest import SimulationResumeTracker
from .tune_runner import resolve_ray_tune_storage_path


@dataclass
class ResumeState:
    enabled: bool

    def consume(self) -> bool:
        if not self.enabled:
            return False
        self.enabled = False
        return True


@dataclass(frozen=True)
class RaySelectionPlan:
    resume_requested: bool
    experiment_name: str | None
    resume_experiment_dir: Path | None


@dataclass(frozen=True)
class CompletedOuterFold:
    best_config: Mapping[str, Any]
    outer_result: Mapping[str, Any] | None


@dataclass(frozen=True)
class RaySelectionContext:
    tuning: TuningConfig
    tuning_engine: str
    ray_storage_path: Path | None
    outer_k: int
    resume_tracker: SimulationResumeTracker | None


def validate_resume_request(
    *, config: SimulationConfig, resume: bool
) -> None:
    if resume and config.modeling.tuning.engine != "ray":
        raise ConfigError("resume requires tuning.engine=ray")


def resolve_ray_storage_path(tuning_engine: str) -> Path | None:
    if tuning_engine != "ray":
        return None
    return resolve_ray_tune_storage_path()


def build_resume_tracker(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    model_selection_enabled: bool,
    tuning_engine: str,
    resume: bool,
) -> SimulationResumeTracker | None:
    if tuning_engine != "ray":
        return None
    return SimulationResumeTracker(
        base_dir=base_dir,
        outer_ids=outer_ids,
        model_selection_enabled=model_selection_enabled,
        resume_requested=resume,
    )


def resolve_resume_requested(
    *, tuning: TuningConfig, resume_state: ResumeState
) -> bool:
    if tuning.engine == "ray":
        return resume_state.consume()
    if resume_state.enabled:
        raise ConfigError("resume requires tuning.engine=ray")
    return False


def resolve_ray_selection_plan(
    *,
    context: RaySelectionContext,
    resume_state: ResumeState,
) -> RaySelectionPlan:
    resume_requested = resolve_resume_requested(
        tuning=context.tuning,
        resume_state=resume_state,
    )
    if context.tuning_engine != "ray":
        return RaySelectionPlan(
            resume_requested=resume_requested,
            experiment_name=None,
            resume_experiment_dir=None,
        )
    if context.resume_tracker is None:
        raise ConfigError("Ray Tune resume tracker is not initialized")
    if context.ray_storage_path is None:
        raise ConfigError("Ray Tune storage path is not initialized")
    experiment_name = context.resume_tracker.experiment_name_for_outer(
        context.outer_k
    )
    resume_dir: Path | None = None
    if resume_requested:
        resume_dir = context.ray_storage_path / experiment_name
    return RaySelectionPlan(
        resume_requested=resume_requested,
        experiment_name=experiment_name,
        resume_experiment_dir=resume_dir,
    )


def mark_outer_complete_for_all(
    *,
    resume_tracker: SimulationResumeTracker,
    outer_ids: Sequence[int],
) -> None:
    for outer_k in outer_ids:
        resume_tracker.mark_outer_completed(int(outer_k))


def load_completed_outer_fold(
    *, base_dir: Path, outer_k: int, inner_only: bool
) -> CompletedOuterFold:
    best_config = load_best_config(base_dir=base_dir, outer_k=outer_k)
    if inner_only:
        return CompletedOuterFold(best_config=best_config, outer_result=None)
    outer_result = load_outer_result(base_dir=base_dir, outer_k=outer_k)
    return CompletedOuterFold(
        best_config=best_config,
        outer_result=outer_result,
    )


def load_best_config(*, base_dir: Path, outer_k: int) -> Mapping[str, Any]:
    path = base_dir / "inner" / f"outer_{outer_k}" / "best_config.json"
    return read_json_mapping(
        path=path,
        message="Missing best_config for completed outer fold",
    )


def load_outer_result(*, base_dir: Path, outer_k: int) -> Mapping[str, Any]:
    path = base_dir / "outer" / f"outer_{outer_k}" / "result.json"
    return read_json_mapping(
        path=path,
        message="Missing outer result for completed outer fold",
    )


def read_json_mapping(*, path: Path, message: str) -> Mapping[str, Any]:
    if not path.exists():
        raise SimulationError(message, context={"path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimulationError(
            "Failed to parse JSON artifact",
            context={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise SimulationError(
            "JSON artifact payload must be a mapping",
            context={"path": str(path)},
        )
    return payload
