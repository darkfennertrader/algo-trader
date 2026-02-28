from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionConfig,
    TuningConfig,
)

from .diagnostics import FanChartDiagnosticsContext, run_fan_chart_diagnostics
from .inner_objective import InnerObjectiveContext
from .model_selection import (
    GlobalSelectionContext,
    PostTuneSelectionContext,
    select_best_candidate_global,
    select_best_candidate_post_tune,
)
from .tune_runner import (
    RayTuneContext,
    RayTuneInputs,
    RayTuneRuntimeSpec,
    RayTuneSpec,
    select_best_with_ray,
)
from .tuning import apply_param_updates, with_candidate_context
from .hooks import SimulationHooks
from .artifacts import SimulationArtifacts


@dataclass(frozen=True)
class BestConfigResources:
    tuning: TuningConfig
    use_gpu: bool
    model_selection: ModelSelectionConfig
    artifacts: SimulationArtifacts
    outer_k: int


@dataclass(frozen=True)
class BestConfigContext:
    objective: Any
    inner_context: InnerObjectiveContext
    hooks: SimulationHooks
    base_config: Mapping[str, Any]
    candidates: Sequence[CandidateSpec]
    resources: BestConfigResources


@dataclass(frozen=True)
class GlobalBestConfig:
    best_config: Mapping[str, Any]
    candidate_id: int


def select_best_config(
    context: BestConfigContext,
    *,
    resume_requested: bool,
    ray_experiment_name: str | None = None,
    ray_storage_path: Path | None = None,
    ray_resume_experiment_dir: Path | None = None,
) -> Mapping[str, Any]:
    resources = context.resources
    if resume_requested and resources.tuning.engine != "ray":
        raise ConfigError("resume requires tuning.engine=ray")
    if resources.tuning.engine == "ray":
        if ray_experiment_name is None:
            raise ConfigError("Ray Tune experiment name is required")
        if ray_storage_path is None:
            raise ConfigError("Ray Tune storage path is required")
        best_config = select_best_with_ray(
            RayTuneContext(
                inputs=RayTuneInputs(
                    objective=context.objective,
                    inner_context=context.inner_context,
                    hooks=context.hooks,
                ),
                spec=RayTuneSpec(
                    base_config=context.base_config,
                    candidates=context.candidates,
                    resources=resources.tuning.ray.resources,
                    use_gpu=resources.use_gpu,
                    runtime=RayTuneRuntimeSpec(
                        storage_path=ray_storage_path,
                        experiment_name=ray_experiment_name,
                        address=resources.tuning.ray.address,
                        resume_experiment_dir=(
                            ray_resume_experiment_dir
                            if resume_requested
                            else None
                        ),
                        logs_enabled=resources.tuning.ray.logs_enabled,
                    ),
                ),
            )
        )
    else:
        best_config = _select_best_config_local(
            objective=context.objective,
            base_config=context.base_config,
            candidates=context.candidates,
        )
    if not resources.model_selection.enable:
        return best_config
    selection = select_best_candidate_post_tune(
        PostTuneSelectionContext(
            artifacts=resources.artifacts,
            outer_k=resources.outer_k,
            candidates=context.candidates,
            model_selection=resources.model_selection,
            use_gpu=resources.use_gpu,
        )
    )
    params = _candidate_params_by_id(
        candidates=context.candidates,
        candidate_id=selection.best_candidate_id,
    )
    return apply_param_updates(context.base_config, params)


def select_global_best_config(
    *,
    artifacts: SimulationArtifacts,
    outer_ids: Sequence[int],
    candidates: Sequence[CandidateSpec],
    model_selection: ModelSelectionConfig,
    base_config: Mapping[str, Any],
) -> GlobalBestConfig:
    selection = select_best_candidate_global(
        GlobalSelectionContext(
            artifacts=artifacts,
            outer_ids=outer_ids,
            candidates=candidates,
            model_selection=model_selection,
        )
    )
    params = _candidate_params_by_id(
        candidates=candidates, candidate_id=selection.best_candidate_id
    )
    best_config = apply_param_updates(base_config, params)
    artifacts.write_global_best_config(payload=best_config)
    return GlobalBestConfig(
        best_config=best_config,
        candidate_id=int(selection.best_candidate_id),
    )


def run_global_diagnostics(
    *,
    base_dir,
    outer_ids: Sequence[int],
    candidate_id: int,
    diagnostics,
    model_selection_enabled: bool,
) -> None:
    if not diagnostics.fan_charts.enable:
        return
    if not model_selection_enabled:
        raise SimulationError(
            "Diagnostics require model_selection.enable"
        )
    run_fan_chart_diagnostics(
        FanChartDiagnosticsContext(
            base_dir=base_dir,
            outer_ids=outer_ids,
            candidate_id=candidate_id,
            config=diagnostics.fan_charts,
        )
    )


def _select_best_config_local(
    *,
    objective,
    base_config: Mapping[str, Any],
    candidates: Sequence[CandidateSpec],
) -> Mapping[str, Any]:
    best_score = float("-inf")
    best_config: Mapping[str, Any] = dict(base_config)

    for candidate in candidates:
        merged = apply_param_updates(base_config, candidate.params)
        merged = with_candidate_context(merged, candidate.candidate_id)
        score = float(objective(merged))
        if score > best_score:
            best_score = score
            best_config = merged

    return best_config


def _candidate_params_by_id(
    *, candidates: Sequence[CandidateSpec], candidate_id: int
) -> Mapping[str, Any]:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate.params
    raise SimulationError(
        "Post-tune selection returned unknown candidate id",
        context={"candidate_id": str(candidate_id)},
    )
