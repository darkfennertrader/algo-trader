# pylint: disable=too-many-lines
from __future__ import annotations
# pylint: disable=duplicate-code

import logging
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import torch

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.application.research.posterior_signal import (
    run_posterior_signal_analysis,
    run_posterior_signal_seed_stability,
)
from algo_trader.domain.simulation import (
    AllocationConfig,
    CandidateSpec,
    CPCVSplit,
    DataPaths,
    FeatureCleaningState,
    OuterFold,
    PanelDataset,
    SimulationConfig,
    SimulationFlags,
)
from algo_trader.infrastructure import log_boundary
from algo_trader.infrastructure.data import load_panel_tensor_dataset

from .config import DEFAULT_CONFIG_PATH, load_config
from .cv_groups import make_cpcv_splits
from .hooks import SimulationHooks, default_hooks, stub_hooks
from .inner_objective import (
    InnerObjectiveContext,
    InnerObjectiveData,
    InnerObjectiveParams,
    make_inner_objective,
)
from .artifacts import (
    CVStructureInputs,
    SimulationArtifacts,
    SimulationInputs,
    resolve_simulation_output_dir,
)
from .data_source_metadata import write_data_source_metadata
from .feature_panel_data import FeaturePanelData
from .preprocessing import InnerCleaningSummaryContext, summarize_inner_cleaning
from .registry import default_registries
from .tuning import assign_candidate_ids, build_candidates
from .tune_runner import init_ray_for_tuning
from .interrupt_cleanup import (
    cleanup_after_simulation_run,
    cleanup_before_simulation_run,
)
from .resume_manifest import SimulationResumeTracker
from .resume_flow import (
    RaySelectionContext,
    ResumeState,
    build_resume_tracker,
    load_best_config,
    load_global_best_config,
    load_completed_outer_fold,
    load_outer_result,
    mark_outer_complete_for_all,
    resolve_ray_selection_plan,
    resolve_ray_storage_path,
    validate_resume_request,
)
from .runner_helpers import (
    build_base_config,
    outer_fold_seed,
    outer_fold_payload,
    set_runtime_seed,
    should_stop_after,
    with_fold_seed,
    with_run_meta,
)
from .smoke_test import (
    apply_smoke_test_overrides,
    build_smoke_test_dataset,
    is_smoke_test_enabled,
)
from .selection import (
    BestConfigContext,
    BestConfigResources,
    run_global_diagnostics,
    select_best_config,
    select_global_best_config,
)
from .prebuild import PrebuildInputs, apply_prebuild, maybe_run_prebuild
from .simulation_context import (
    SimulationContext,
    build_simulation_context,
)
from .walkforward import (
    OuterEvaluationContext,
    PortfolioSpec,
    WalkforwardProgress,
    build_walkforward_progress,
    evaluate_outer_walk_forward,
    resolve_portfolio_base_dir,
    write_downstream_metrics,
    write_downstream_outputs,
    write_downstream_plots,
)
from .walkforward.seed_stability import run_seed_stability_study

logger = logging.getLogger(__name__)

_CUDA_REF_SAMPLE_LIMIT = 8

def _run_context(config_path: Path | None, resume: bool) -> Mapping[str, str]:
    return {
        "config": str(config_path or DEFAULT_CONFIG_PATH),
        "resume": str(resume),
    }


def _format_clock_time(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def _format_duration_hms(duration_seconds: float) -> str:
    total_seconds = max(int(round(duration_seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(frozen=True)
class _RunRuntimeOptions:
    use_ray: bool
    ray_address: str | None
    ray_logs_enabled: bool


@dataclass(frozen=True)
class _CudaRefSummary:
    count: int
    total_bytes: int
    examples: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedRunContext:
    config: SimulationConfig
    setup: RunSetup
    runtime: _RunRuntimeOptions
    resume_state: ResumeState
    resume_tracker: SimulationResumeTracker | None
    walkforward_progress: WalkforwardProgress | None


def _summarize_cuda_refs(named_values: Mapping[str, object]) -> _CudaRefSummary:
    examples: list[str] = []
    count, total_bytes = _walk_cuda_refs(
        value=named_values,
        path="root",
        seen=set(),
        examples=examples,
    )
    return _CudaRefSummary(
        count=count,
        total_bytes=total_bytes,
        examples=tuple(examples),
    )


def _resolve_runtime_options(config: SimulationConfig) -> _RunRuntimeOptions:
    use_ray = (
        config.flags.execution_mode in {"full", "model_research"}
        and config.modeling.tuning.engine == "ray"
    )
    return _RunRuntimeOptions(
        use_ray=use_ray,
        ray_address=config.modeling.tuning.ray.address,
        ray_logs_enabled=config.modeling.tuning.ray.logs_enabled,
    )


def _validate_execution_mode(config: SimulationConfig) -> None:
    return None


def _build_walkforward_progress(
    *,
    config: SimulationConfig,
    outer_folds: Sequence[OuterFold],
) -> WalkforwardProgress | None:
    if config.flags.execution_mode != "walkforward":
        return None
    return build_walkforward_progress(outer_folds)


def _is_walkforward_mode(flags: Any) -> bool:
    return getattr(flags, "execution_mode", None) == "walkforward"


def _outer_runtime_seed(
    *,
    config: Any,
    flags: Any,
    fold_id: int,
) -> int:
    if _is_walkforward_mode(flags):
        walkforward = getattr(config, "walkforward", None)
        seeds = getattr(walkforward, "seeds", (7,))
        base_seed = seeds[0] if len(seeds) > 0 else 7
        return outer_fold_seed(int(base_seed), fold_id)
    return outer_fold_seed(int(config.cv.cpcv.seed), fold_id)


def _close_walkforward_progress(
    progress: WalkforwardProgress | None,
) -> None:
    if progress is not None:
        progress.close()


def _should_run_walkforward_seed_stability(
    config: SimulationConfig,
) -> bool:
    return (
        config.flags.execution_mode == "walkforward"
        and config.walkforward.num_seeds > 1
    )


def _should_run_posterior_signal_seed_stability(
    config: SimulationConfig,
) -> bool:
    return (
        config.flags.execution_mode == "posterior_signal"
        and config.walkforward.num_seeds > 1
    )


def _maybe_run_posterior_signal(
    *,
    config: SimulationConfig,
    started_at: float,
) -> Mapping[str, Any] | None:
    if config.flags.execution_mode != "posterior_signal":
        return None
    if _should_run_posterior_signal_seed_stability(config):
        results = run_posterior_signal_seed_stability(config=config)
        return _complete_early_run(
            config=config,
            started_at=started_at,
            results=results,
        )
    cleanup_before_simulation_run(
        use_ray=False,
        ray_address=None,
        use_gpu=config.flags.use_gpu,
        log_cuda_clear=False,
    )
    try:
        results = run_posterior_signal_analysis(config=config)
    finally:
        cleanup_after_simulation_run(
            use_ray=False,
            ray_address=None,
            use_gpu=config.flags.use_gpu,
            interrupted=False,
            log_cuda_clear=False,
        )
    return _complete_early_run(
        config=config,
        started_at=started_at,
        results=results,
    )


def _maybe_run_walkforward_seed_stability(
    *,
    config: SimulationConfig,
    started_at: float,
) -> Mapping[str, Any] | None:
    if not _should_run_walkforward_seed_stability(config):
        return None
    results = run_seed_stability_study(config=config)
    return _complete_early_run(
        config=config,
        started_at=started_at,
        results=results,
    )


def _complete_early_run(
    *,
    config: SimulationConfig,
    started_at: float,
    results: Mapping[str, Any],
) -> Mapping[str, Any]:
    ended_at = time.time()
    logger.info(
        "simulation_timing start=%s end=%s duration=%s",
        _format_clock_time(started_at),
        _format_clock_time(ended_at),
        _format_duration_hms(ended_at - started_at),
    )
    return with_run_meta(results, config.flags)


def _walk_cuda_refs(
    *,
    value: object,
    path: str,
    seen: set[int],
    examples: list[str],
) -> tuple[int, int]:
    object_id = id(value)
    if object_id in seen:
        return 0, 0
    seen.add(object_id)
    if isinstance(value, torch.Tensor):
        return _summarize_cuda_tensor(value=value, path=path, examples=examples)
    if isinstance(value, Mapping):
        return _walk_mapping_cuda_refs(
            value=value, path=path, seen=seen, examples=examples
        )
    if isinstance(value, (list, tuple)):
        return _walk_sequence_cuda_refs(
            value=value, path=path, seen=seen, examples=examples
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _walk_dataclass_cuda_refs(
            value=value, path=path, seen=seen, examples=examples
        )
    return 0, 0


def _summarize_cuda_tensor(
    *, value: torch.Tensor, path: str, examples: list[str]
) -> tuple[int, int]:
    if value.device.type != "cuda":
        return 0, 0
    if len(examples) < _CUDA_REF_SAMPLE_LIMIT:
        examples.append(
            f"{path}: shape={tuple(value.shape)} dtype={value.dtype}"
        )
    return 1, int(value.numel() * value.element_size())


def _walk_mapping_cuda_refs(
    *,
    value: Mapping[str, object],
    path: str,
    seen: set[int],
    examples: list[str],
) -> tuple[int, int]:
    total_count = 0
    total_bytes = 0
    for key, item in value.items():
        count, size = _walk_cuda_refs(
            value=item,
            path=f"{path}.{key}",
            seen=seen,
            examples=examples,
        )
        total_count += count
        total_bytes += size
    return total_count, total_bytes


def _walk_sequence_cuda_refs(
    *,
    value: Sequence[object],
    path: str,
    seen: set[int],
    examples: list[str],
) -> tuple[int, int]:
    total_count = 0
    total_bytes = 0
    for index, item in enumerate(value):
        count, size = _walk_cuda_refs(
            value=item,
            path=f"{path}[{index}]",
            seen=seen,
            examples=examples,
        )
        total_count += count
        total_bytes += size
    return total_count, total_bytes


def _walk_dataclass_cuda_refs(
    *,
    value: object,
    path: str,
    seen: set[int],
    examples: list[str],
) -> tuple[int, int]:
    total_count = 0
    total_bytes = 0
    dataclass_value = cast(Any, value)
    for field in fields(dataclass_value):
        count, size = _walk_cuda_refs(
            value=getattr(dataclass_value, field.name),
            path=f"{path}.{field.name}",
            seen=seen,
            examples=examples,
        )
        total_count += count
        total_bytes += size
    return total_count, total_bytes


def _log_cuda_ref_summary(
    *,
    use_gpu: bool,
    stage: str,
    named_values: Mapping[str, object],
) -> None:
    if not use_gpu:
        return
    summary = _summarize_cuda_refs(named_values)
    logger.info(
        "event=simulation.cuda_refs stage=%s count=%s total_bytes=%s examples=%s",
        stage,
        summary.count,
        summary.total_bytes,
        list(summary.examples),
    )

@log_boundary("simulation.run", context=_run_context)
def run(
    *, config_path: Path | None = None, resume: bool = False
) -> Mapping[str, Any]:
    started_at = time.time()
    config = apply_smoke_test_overrides(load_config(config_path))
    _validate_execution_mode(config)
    validate_resume_request(config=config, resume=resume)
    posterior_results = _maybe_run_posterior_signal(
        config=config,
        started_at=started_at,
    )
    if posterior_results is not None:
        return posterior_results
    early_results = _maybe_run_walkforward_seed_stability(
        config=config,
        started_at=started_at,
    )
    if early_results is not None:
        return early_results
    results = _run_prepared_simulation(config=config, resume=resume)
    ended_at = time.time()
    logger.info(
        "simulation_timing start=%s end=%s duration=%s",
        _format_clock_time(started_at),
        _format_clock_time(ended_at),
        _format_duration_hms(ended_at - started_at),
    )
    return results


def _run_prepared_simulation(
    *,
    config: SimulationConfig,
    resume: bool,
) -> Mapping[str, Any]:
    device = _resolve_device(config.flags.use_gpu)
    setup: RunSetup | None = _prepare_run_state(config=config, device=device)
    if setup.early_results is not None:
        return setup.early_results
    runtime = _resolve_runtime_options(config)
    resume_state = ResumeState(enabled=resume)
    walkforward_progress = _build_walkforward_progress(
        config=config,
        outer_folds=setup.outer_context.context.cv.outer_folds,
    )
    resume_tracker = _initialize_run_runtime(
        config=config,
        setup=setup,
        runtime=runtime,
        resume=resume,
    )
    return _execute_run_loop(
        prepared=_PreparedRunContext(
            config=config,
            setup=setup,
            runtime=runtime,
            resume_state=resume_state,
            resume_tracker=resume_tracker,
            walkforward_progress=walkforward_progress,
        )
    )


def _initialize_run_runtime(
    *,
    config: SimulationConfig,
    setup: RunSetup,
    runtime: _RunRuntimeOptions,
    resume: bool,
) -> SimulationResumeTracker | None:
    cleanup_before_simulation_run(
        use_ray=runtime.use_ray,
        ray_address=runtime.ray_address,
        use_gpu=config.flags.use_gpu,
        log_cuda_clear=not _is_walkforward_mode(config.flags),
    )
    resume_tracker = build_resume_tracker(
        base_dir=setup.outer_context.artifacts.base_dir,
        outer_ids=setup.outer_context.context.cv.outer_ids,
        model_selection_enabled=(
            setup.outer_context.config.evaluation.model_selection.enable
        ),
        tuning_engine=setup.outer_context.config.modeling.tuning.engine,
        resume=resume,
    )
    if runtime.use_ray:
        init_ray_for_tuning(
            runtime.ray_address,
            logs_enabled=runtime.ray_logs_enabled,
        )
    return resume_tracker


def _execute_run_loop(
    *,
    prepared: _PreparedRunContext,
) -> Mapping[str, Any]:
    interrupted = False
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []
    results: Mapping[str, Any] | None = None
    try:
        chosen_configs, outer_results = _evaluate_outer_folds(
            outer_context=prepared.setup.outer_context,
            candidates=prepared.setup.candidates,
            resume_state=prepared.resume_state,
            resume_tracker=prepared.resume_tracker,
            week_progress=(
                None
                if prepared.walkforward_progress is None
                else prepared.walkforward_progress.update
            ),
        )
        _close_walkforward_progress(prepared.walkforward_progress)
        prepared = replace(prepared, walkforward_progress=None)
        if prepared.resume_tracker is not None:
            prepared.resume_tracker.mark_run_completed()
        results = _finalize_run_results(
            config=prepared.config,
            setup=prepared.setup,
            chosen_configs=chosen_configs,
            outer_results=outer_results,
        )
        _release_run_refs(
            use_gpu=prepared.config.flags.use_gpu,
            setup=prepared.setup,
            chosen_configs=chosen_configs,
            outer_results=outer_results,
            results=results,
        )
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if prepared.walkforward_progress is not None:
            prepared.walkforward_progress.close()
        cleanup_after_simulation_run(
            use_ray=prepared.runtime.use_ray,
            ray_address=prepared.runtime.ray_address,
            use_gpu=prepared.config.flags.use_gpu,
            interrupted=interrupted,
            log_cuda_clear=not _is_walkforward_mode(prepared.config.flags),
        )
    if results is None:
        raise SimulationError("Simulation completed without results")
    return results


def _finalize_run_results(
    *,
    config: SimulationConfig,
    setup: RunSetup,
    chosen_configs: Mapping[int, Mapping[str, Any]],
    outer_results: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    results = _build_results(
        config,
        setup.outer_context.context,
        chosen_configs,
        outer_results,
    )
    enriched = with_run_meta(results, config.flags)
    write_downstream_outputs(
        base_dir=setup.outer_context.artifacts.base_dir,
        outer_results=outer_results,
        assets=setup.outer_context.context.assets,
    )
    write_downstream_metrics(
        base_dir=setup.outer_context.artifacts.base_dir,
        dataset_params=config.data.dataset_params,
    )
    write_downstream_plots(
        base_dir=setup.outer_context.artifacts.base_dir,
        dataset_params=config.data.dataset_params,
    )
    setup.outer_context.artifacts.write_results(enriched)
    setup.outer_context.artifacts.write_cv_summary(enriched)
    return enriched


def _release_run_refs(
    *,
    use_gpu: bool,
    setup: RunSetup,
    chosen_configs: dict[int, Mapping[str, Any]],
    outer_results: list[Mapping[str, Any]],
    results: Mapping[str, Any],
) -> None:
    _log_cuda_ref_summary(
        use_gpu=use_gpu,
        stage="pre_release",
        named_values={
            "setup": setup,
            "chosen_configs": chosen_configs,
            "outer_results": outer_results,
            "results": results,
        },
    )
    emptied_setup = None
    chosen_configs.clear()
    outer_results.clear()
    _log_cuda_ref_summary(
        use_gpu=use_gpu,
        stage="post_release",
        named_values={
            "setup": emptied_setup,
            "chosen_configs": chosen_configs,
            "outer_results": outer_results,
            "results": results,
        },
    )

@dataclass(frozen=True)
class RunSetup:
    outer_context: OuterFoldContext
    candidates: Sequence[CandidateSpec]
    early_results: Mapping[str, Any] | None

def _prepare_run_state(
    *, config: SimulationConfig, device: str
) -> RunSetup:
    ray_storage_path = resolve_ray_storage_path(
        config.modeling.tuning.engine
    )
    source_dir, dataset, reused_inputs = _resolve_run_inputs(
        config=config, device=device
    )
    base_dir = resolve_portfolio_base_dir(
        source_dir=source_dir,
        portfolio_output_path=(
            config.data.portfolio_output_path
            if _is_walkforward_mode(config.flags)
            else None
        ),
        dataset_params=config.data.dataset_params,
    )
    context = build_simulation_context(config, dataset)
    artifacts = _build_artifacts(
        base_dir=base_dir,
        dataset=dataset,
        context=context,
        write_inputs=(not reused_inputs) or base_dir != source_dir,
        dataset_params=config.data.dataset_params,
    )
    _stage_walkforward_source_artifacts(
        config=config,
        source_dir=source_dir,
        target_dir=base_dir,
    )
    early_results = _maybe_finalize_inputs(
        config=config,
        context=context,
        artifacts=artifacts,
    )
    if early_results is not None:
        return RunSetup(
            outer_context=OuterFoldContext(
                config=config,
                context=context,
                base_config={},
                hooks=stub_hooks(),
                artifacts=artifacts,
                flags=config.flags,
                ray_storage_path=ray_storage_path,
            ),
            candidates=(),
            early_results=early_results,
        )
    candidates, base_config = _prepare_candidates(
        config=config,
        artifacts=artifacts,
        context=context,
        dataset=dataset,
    )
    early_results = _maybe_finalize_cv(
        config=config,
        context=context,
        artifacts=artifacts,
    )
    outer_context = OuterFoldContext(
        config=config,
        context=context,
        base_config=base_config,
        hooks=(
            stub_hooks()
            if config.flags.simulation_mode == "stub"
            else default_hooks()
        ),
        artifacts=artifacts,
        flags=config.flags,
        ray_storage_path=ray_storage_path,
    )
    return RunSetup(
        outer_context=outer_context,
        candidates=candidates,
        early_results=early_results,
    )


def _resolve_run_inputs(
    *, config: SimulationConfig, device: str
) -> tuple[Path, PanelDataset, bool]:
    base_dir = resolve_simulation_output_dir(
        simulation_output_path=config.data.simulation_output_path,
        dataset_params=config.data.dataset_params,
    )
    dataset, reused_inputs = _load_dataset_for_run(
        config=config, base_dir=base_dir, device=device
    )
    return base_dir, dataset, reused_inputs

def _maybe_finalize_inputs(
    *,
    config: SimulationConfig,
    context: SimulationContext,
    artifacts: SimulationArtifacts,
) -> Mapping[str, Any] | None:
    if not should_stop_after("inputs", config.flags):
        return None
    results = _build_results(
        config, context, chosen_configs={}, outer_results=[]
    )
    results = with_run_meta(results, config.flags)
    artifacts.write_results(results)
    return results

def _prepare_candidates(
    *,
    config: SimulationConfig,
    artifacts: SimulationArtifacts,
    context: SimulationContext,
    dataset: PanelDataset,
) -> tuple[Sequence[CandidateSpec], Mapping[str, Any]]:
    artifacts.write_cv_structure(
        inputs=CVStructureInputs(
            warmup_idx=context.cv.warmup_idx,
            groups=context.cv.groups,
            outer_ids=context.cv.outer_ids,
            outer_folds=[
                outer_fold_payload(fold)
                for fold in context.cv.outer_folds
            ],
            timestamps=dataset.dates,
        )
    )
    prebuild = maybe_run_prebuild(
        prebuild=config.modeling.model.prebuild,
        inputs=PrebuildInputs(
            X=context.X,
            X_global=context.X_global,
            y=context.y,
            M=context.M,
            M_global=context.M_global,
            outer_folds=context.cv.outer_folds,
            group_by_index=context.cv.group_by_index,
            feature_names=context.feature_names,
            global_feature_names=context.global_feature_names,
            assets=context.assets,
        ),
        artifacts=artifacts,
    )
    candidates = _write_candidate_artifacts(
        config=config, artifacts=artifacts
    )
    debug_output_dir = _debug_output_dir(
        artifacts=artifacts, flags=config.flags
    )
    base_config = build_base_config(
        config.modeling.model,
        config.modeling.training,
        config.flags,
        debug_output_dir,
    )
    if prebuild is not None:
        base_config = apply_prebuild(base_config, prebuild)
    return candidates, base_config

def _maybe_finalize_cv(
    *,
    config: SimulationConfig,
    context: SimulationContext,
    artifacts: SimulationArtifacts,
) -> Mapping[str, Any] | None:
    if not should_stop_after("cv", config.flags):
        return None
    results = _build_results(
        config, context, chosen_configs={}, outer_results=[]
    )
    results = with_run_meta(results, config.flags)
    artifacts.write_results(results)
    artifacts.write_cv_summary(results)
    return results

def _resolve_device(use_gpu: bool) -> str:
    if use_gpu and not torch.cuda.is_available():
        raise ConfigError("use_gpu is true but CUDA is not available")
    return "cuda" if use_gpu else "cpu"

def _debug_output_dir(
    *, artifacts: SimulationArtifacts, flags: SimulationFlags
) -> str | None:
    if not flags.smoke_test_debug:
        return None
    if flags.smoke_test_enabled:
        return str(artifacts.base_dir)
    return str(artifacts.base_dir / "smoke_test")

def _load_dataset(config: SimulationConfig, device: str) -> PanelDataset:
    registries = default_registries()
    return registries.datasets.build(
        "feature_store_split", config=config.data, device=device
    )

def _panel_tensor_path(base_dir: Path) -> Path:
    return base_dir / "inputs" / "panel_tensor.pt"

def _load_dataset_for_run(
    *,
    config: SimulationConfig,
    base_dir: Path,
    device: str,
) -> tuple[PanelDataset, bool]:
    if is_smoke_test_enabled(config):
        dataset = build_smoke_test_dataset(device)
        return dataset, False
    if base_dir.exists():
        if not base_dir.is_dir():
            raise SimulationError(
                "Simulation output path is not a directory",
                context={"path": str(base_dir)},
            )
        tensor_path = _panel_tensor_path(base_dir)
        if tensor_path.exists():
            dataset = load_panel_tensor_dataset(
                paths=DataPaths(tensor_path=str(tensor_path)),
                device=device,
            )
            logger.info(
                "Using existing simulation inputs path=%s", tensor_path
            )
            return dataset, True
        if _requires_existing_saved_inputs(config.flags.execution_mode):
            raise SimulationError(
                "panel_tensor.pt missing in simulation output directory",
                context={"path": str(tensor_path)},
            )
    dataset = _load_dataset(config, device)
    return dataset, False


def _requires_existing_saved_inputs(execution_mode: str) -> bool:
    return execution_mode in {
        "posterior_signal",
        "walkforward",
        "results_aggregation",
    }

@dataclass(frozen=True)
class OuterFoldContext:
    config: SimulationConfig
    context: SimulationContext
    base_config: Mapping[str, Any]
    hooks: SimulationHooks
    artifacts: SimulationArtifacts
    flags: SimulationFlags
    ray_storage_path: Path | None

def _evaluate_outer_folds(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
    week_progress: Callable[[int, Any], None] | None,
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    if getattr(outer_context.flags, "execution_mode", None) == "results_aggregation":
        return _load_saved_outer_outputs(outer_context=outer_context)
    if _is_walkforward_mode(outer_context.flags):
        return _evaluate_saved_best_configs_outer_only(
            outer_context=outer_context,
            resume_tracker=resume_tracker,
            week_progress=week_progress,
        )
    selection = outer_context.config.evaluation.model_selection
    if selection.enable:
        return _evaluate_outer_folds_global(
            outer_context=outer_context,
            candidates=candidates,
            resume_state=resume_state,
            resume_tracker=resume_tracker,
            week_progress=week_progress,
        )
    return _evaluate_outer_folds_per_outer(
        outer_context=outer_context,
        candidates=candidates,
        resume_state=resume_state,
        resume_tracker=resume_tracker,
        week_progress=week_progress,
    )

def _evaluate_outer_folds_per_outer(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
    week_progress: Callable[[int, Any], None] | None,
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []
    inner_only = should_stop_after("inner", outer_context.flags)
    for outer_fold in outer_context.context.cv.outer_folds:
        if (
            resume_tracker is not None
            and resume_tracker.is_outer_completed(outer_fold.k_test)
        ):
            skipped = load_completed_outer_fold(
                base_dir=outer_context.artifacts.base_dir,
                outer_k=outer_fold.k_test,
                inner_only=inner_only,
            )
            chosen_configs[outer_fold.k_test] = skipped.best_config
            if skipped.outer_result is not None:
                outer_results.append(skipped.outer_result)
            continue
        if (
            resume_tracker is not None
            and not inner_only
            and resume_tracker.is_inner_completed(outer_fold.k_test)
        ):
            resumed = _run_outer_from_saved_inner(
                outer_context=outer_context,
                outer_fold=outer_fold,
                resume_tracker=resume_tracker,
                week_progress=week_progress,
            )
            chosen_configs[outer_fold.k_test] = resumed.best_config
            if resumed.outer_result is None:
                raise SimulationError(
                    "Outer result missing for resumed outer evaluation",
                    context={"outer_k": str(outer_fold.k_test)},
                )
            outer_results.append(resumed.outer_result)
            continue
        fold_result = _run_outer_fold(
            outer_context=outer_context,
            outer_fold=outer_fold,
            candidates=candidates,
            controls=OuterRunControls(
                resume_state=resume_state,
                resume_tracker=resume_tracker,
                week_progress=week_progress,
            ),
        )
        chosen_configs[outer_fold.k_test] = fold_result.best_config
        if fold_result.outer_result is not None:
            outer_results.append(fold_result.outer_result)
    return chosen_configs, outer_results

def _evaluate_outer_folds_global(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
    week_progress: Callable[[int, Any], None] | None,
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []
    for outer_fold in outer_context.context.cv.outer_folds:
        if (
            resume_tracker is not None
            and resume_tracker.is_inner_completed(outer_fold.k_test)
        ):
            continue
        _run_outer_fold_inner_only(
            outer_context=outer_context,
            outer_fold=outer_fold,
            candidates=candidates,
            resume_state=resume_state,
            resume_tracker=resume_tracker,
        )
    global_best = select_global_best_config(
        artifacts=outer_context.artifacts,
        outer_ids=outer_context.context.cv.outer_ids,
        candidates=candidates,
        model_selection=outer_context.config.evaluation.model_selection,
        base_config=outer_context.base_config,
    )
    run_global_diagnostics(
        base_dir=outer_context.artifacts.base_dir,
        outer_ids=outer_context.context.cv.outer_ids,
        candidate_id=global_best.candidate_id,
        diagnostics=outer_context.config.evaluation.diagnostics,
        model_selection_enabled=(
            outer_context.config.evaluation.model_selection.enable
        ),
    )
    for outer_fold in outer_context.context.cv.outer_folds:
        chosen_configs[outer_fold.k_test] = global_best.best_config
    if should_stop_after("inner", outer_context.flags):
        if resume_tracker is not None:
            mark_outer_complete_for_all(
                resume_tracker=resume_tracker,
                outer_ids=outer_context.context.cv.outer_ids,
            )
        return chosen_configs, outer_results
    for outer_fold in outer_context.context.cv.outer_folds:
        if (
            resume_tracker is not None
            and resume_tracker.is_outer_completed(outer_fold.k_test)
        ):
            outer_results.append(
                load_outer_result(
                    base_dir=outer_context.artifacts.base_dir,
                    outer_k=outer_fold.k_test,
                )
            )
            continue
        if resume_tracker is not None:
            resume_tracker.mark_outer_started(outer_fold.k_test)
        started = time.perf_counter()
        result, cleaning_outer = _run_outer_evaluation(
            outer_context=outer_context,
            outer_fold=outer_fold,
            best_config=global_best.best_config,
            week_progress=week_progress,
        )
        if resume_tracker is not None:
            resume_tracker.mark_outer_completed(outer_fold.k_test)
        if not _is_walkforward_mode(outer_context.flags):
            _log_outer_complete(
                outer_k=outer_fold.k_test,
                started=started,
                phase="outer",
                result=result,
                cleaning=cleaning_outer,
            )
        outer_results.append(result)
    return chosen_configs, outer_results


def _evaluate_saved_best_configs_outer_only(
    *,
    outer_context: OuterFoldContext,
    resume_tracker: SimulationResumeTracker | None,
    week_progress: Callable[[int, Any], None] | None,
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs = _load_saved_chosen_configs(outer_context=outer_context)
    outer_results: list[Mapping[str, Any]] = []
    for outer_fold in outer_context.context.cv.outer_folds:
        if (
            resume_tracker is not None
            and resume_tracker.is_outer_completed(outer_fold.k_test)
        ):
            outer_results.append(
                load_outer_result(
                    base_dir=outer_context.artifacts.base_dir,
                    outer_k=outer_fold.k_test,
                )
            )
            continue
        if resume_tracker is not None:
            resume_tracker.mark_outer_started(outer_fold.k_test)
        started = time.perf_counter()
        result, cleaning_outer = _run_outer_evaluation(
            outer_context=outer_context,
            outer_fold=outer_fold,
            best_config=chosen_configs[outer_fold.k_test],
            week_progress=week_progress,
        )
        if resume_tracker is not None:
            resume_tracker.mark_outer_completed(outer_fold.k_test)
        if not _is_walkforward_mode(outer_context.flags):
            _log_outer_complete(
                outer_k=outer_fold.k_test,
                started=started,
                phase="outer",
                result=result,
                cleaning=cleaning_outer,
            )
        outer_results.append(result)
    return chosen_configs, outer_results


def _load_saved_outer_outputs(
    *,
    outer_context: OuterFoldContext,
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs = _load_saved_chosen_configs(outer_context=outer_context)
    outer_results = [
        load_outer_result(
            base_dir=outer_context.artifacts.base_dir,
            outer_k=outer_fold.k_test,
        )
        for outer_fold in outer_context.context.cv.outer_folds
    ]
    return chosen_configs, outer_results


def _load_saved_chosen_configs(
    *,
    outer_context: OuterFoldContext,
) -> dict[int, Mapping[str, Any]]:
    outer_ids = outer_context.context.cv.outer_ids
    if outer_context.config.evaluation.model_selection.enable:
        global_path = outer_context.artifacts.base_dir / "outer" / "best_config.json"
        if global_path.exists():
            best_config = load_global_best_config(
                base_dir=outer_context.artifacts.base_dir
            )
            return {int(outer_k): best_config for outer_k in outer_ids}
    return {
        int(outer_fold.k_test): load_best_config(
            base_dir=outer_context.artifacts.base_dir,
            outer_k=outer_fold.k_test,
        )
        for outer_fold in outer_context.context.cv.outer_folds
    }


@dataclass(frozen=True)
class OuterFoldRunResult:
    best_config: Mapping[str, Any]
    outer_result: Mapping[str, Any] | None


@dataclass(frozen=True)
class InnerObjectiveBundle:
    objective: Any
    context: InnerObjectiveContext
    hooks: SimulationHooks


@dataclass(frozen=True)
class OuterRunControls:
    resume_state: ResumeState
    resume_tracker: SimulationResumeTracker | None
    week_progress: Callable[[int, Any], None] | None


def _run_outer_fold(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    candidates: Sequence[CandidateSpec],
    controls: OuterRunControls,
) -> OuterFoldRunResult:
    started = time.perf_counter()
    inner_splits = _build_inner_splits(
        outer_context=outer_context,
        outer_fold=outer_fold,
    )
    inner_bundle = _build_inner_objective(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    objective = inner_bundle.objective
    _write_postprocess_metadata(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        candidates=candidates,
    )
    _write_inner_setup_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    if controls.resume_tracker is not None:
        controls.resume_tracker.mark_inner_started(outer_fold.k_test)
    ray_plan = resolve_ray_selection_plan(
        context=RaySelectionContext(
            tuning=outer_context.config.modeling.tuning,
            tuning_engine=outer_context.config.modeling.tuning.engine,
            ray_storage_path=outer_context.ray_storage_path,
            outer_k=outer_fold.k_test,
            resume_tracker=controls.resume_tracker,
        ),
        resume_state=controls.resume_state,
    )
    best_config = select_best_config(
        BestConfigContext(
            objective=objective,
            inner_context=inner_bundle.context,
            hooks=inner_bundle.hooks,
            base_config=outer_context.base_config,
            candidates=candidates,
            resources=BestConfigResources(
                tuning=outer_context.config.modeling.tuning,
                use_gpu=outer_context.flags.use_gpu,
                model_selection=(
                    outer_context.config.evaluation.model_selection
                ),
                artifacts=outer_context.artifacts,
                outer_k=outer_fold.k_test,
            ),
        ),
        resume_requested=ray_plan.resume_requested,
        ray_experiment_name=ray_plan.experiment_name,
        ray_storage_path=outer_context.ray_storage_path,
        ray_resume_experiment_dir=ray_plan.resume_experiment_dir,
    )
    _write_inner_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        best_config=best_config,
    )
    if controls.resume_tracker is not None:
        controls.resume_tracker.mark_inner_completed(outer_fold.k_test)
    if should_stop_after("inner", outer_context.flags):
        if controls.resume_tracker is not None:
            controls.resume_tracker.mark_outer_completed(outer_fold.k_test)
        if not _is_walkforward_mode(outer_context.flags):
            _log_outer_complete(
                outer_k=outer_fold.k_test,
                started=started,
                phase="inner",
                result=None,
                cleaning=None,
            )
        return OuterFoldRunResult(best_config=best_config, outer_result=None)
    if controls.resume_tracker is not None:
        controls.resume_tracker.mark_outer_started(outer_fold.k_test)
    result, cleaning_outer = _run_outer_evaluation(
        outer_context=outer_context,
        outer_fold=outer_fold,
        best_config=best_config,
        week_progress=controls.week_progress,
    )
    if controls.resume_tracker is not None:
        controls.resume_tracker.mark_outer_completed(outer_fold.k_test)
    if not _is_walkforward_mode(outer_context.flags):
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="outer",
            result=result,
            cleaning=cleaning_outer,
        )
    return OuterFoldRunResult(best_config=best_config, outer_result=result)

def _run_outer_fold_inner_only(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    inner_splits = _build_inner_splits(
        outer_context=outer_context,
        outer_fold=outer_fold,
    )
    inner_bundle = _build_inner_objective(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    objective = inner_bundle.objective
    _write_postprocess_metadata(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        candidates=candidates,
    )
    _write_inner_setup_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    if resume_tracker is not None:
        resume_tracker.mark_inner_started(outer_fold.k_test)
    ray_plan = resolve_ray_selection_plan(
        context=RaySelectionContext(
            tuning=outer_context.config.modeling.tuning,
            tuning_engine=outer_context.config.modeling.tuning.engine,
            ray_storage_path=outer_context.ray_storage_path,
            outer_k=outer_fold.k_test,
            resume_tracker=resume_tracker,
        ),
        resume_state=resume_state,
    )
    best_config = select_best_config(
        BestConfigContext(
            objective=objective,
            inner_context=inner_bundle.context,
            hooks=inner_bundle.hooks,
            base_config=outer_context.base_config,
            candidates=candidates,
            resources=BestConfigResources(
                tuning=outer_context.config.modeling.tuning,
                use_gpu=outer_context.flags.use_gpu,
                model_selection=(
                    outer_context.config.evaluation.model_selection
                ),
                artifacts=outer_context.artifacts,
                outer_k=outer_fold.k_test,
            ),
        ),
        resume_requested=ray_plan.resume_requested,
        ray_experiment_name=ray_plan.experiment_name,
        ray_storage_path=outer_context.ray_storage_path,
        ray_resume_experiment_dir=ray_plan.resume_experiment_dir,
    )
    _write_inner_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        best_config=best_config,
    )
    if resume_tracker is not None:
        resume_tracker.mark_inner_completed(outer_fold.k_test)
    if not _is_walkforward_mode(outer_context.flags):
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="inner",
            result=None,
            cleaning=None,
        )
    return best_config

def _build_inner_splits(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
) -> list[CPCVSplit]:
    inner_splits = make_cpcv_splits(
        warmup_idx=outer_context.context.cv.warmup_idx,
        groups=outer_context.context.cv.groups,
        inner_group_ids=outer_fold.inner_group_ids,
        params=with_fold_seed(
            outer_context.config.cv, outer_fold.k_test
        ),
    )
    if not inner_splits:
        raise SimulationError(
            "No inner CPCV splits available; check cv settings",
            context={"outer_group": str(outer_fold.k_test)},
        )
    return inner_splits

def _build_inner_objective(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    inner_splits: list[CPCVSplit],
):
    context = InnerObjectiveContext(
        data=InnerObjectiveData(
            panel=FeaturePanelData(
                X=outer_context.context.X,
                M=outer_context.context.M,
                X_global=outer_context.context.X_global,
                M_global=outer_context.context.M_global,
                global_feature_names=outer_context.context.global_feature_names,
            ),
            assets=outer_context.context.assets,
            y=outer_context.context.y,
        ),
        artifacts=outer_context.artifacts,
        group_by_index=outer_context.context.cv.group_by_index,
        inner_splits=inner_splits,
        params=InnerObjectiveParams(
            preprocess_spec=outer_context.context.preprocess_spec,
            score_spec=outer_context.config.evaluation.scoring.spec,
            num_pp_samples=(
                outer_context.config.evaluation.predictive.num_samples_inner
            ),
            outer_k=outer_fold.k_test,
            aggregate=outer_context.config.modeling.tuning.aggregate.method,
            aggregate_lambda=(
                outer_context.config.modeling.tuning.aggregate.penalty
            ),
        ),
        model_selection=outer_context.config.evaluation.model_selection,
    )
    return InnerObjectiveBundle(
        objective=make_inner_objective(
            context=context, hooks=outer_context.hooks
        ),
        context=context,
        hooks=outer_context.hooks,
    )

def _write_inner_outputs(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    best_config: Mapping[str, Any],
    inner_splits: list[CPCVSplit],
) -> None:
    outer_context.artifacts.write_inner(
        outer_k=outer_fold.k_test,
        inner_splits=inner_splits,
        warmup_idx=outer_context.context.cv.warmup_idx,
        best_config=best_config,
    )
    inner_summary = summarize_inner_cleaning(
        InnerCleaningSummaryContext(
            X=outer_context.context.X,
            M=outer_context.context.M,
            inner_splits=inner_splits,
            spec=outer_context.context.preprocess_spec,
            feature_names=outer_context.context.feature_names,
            outer_k=outer_fold.k_test,
        )
    )
    outer_context.artifacts.write_inner_cleaning_summary(
        outer_k=outer_fold.k_test,
        summary=inner_summary,
    )

def _write_inner_setup_outputs(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    inner_splits: list[CPCVSplit],
) -> None:
    outer_context.artifacts.write_inner(
        outer_k=outer_fold.k_test,
        inner_splits=inner_splits,
        warmup_idx=outer_context.context.cv.warmup_idx,
        best_config=None,
    )

def _write_postprocess_metadata(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    inner_splits: Sequence[CPCVSplit],
    candidates: Sequence[CandidateSpec],
) -> None:
    selection = outer_context.config.evaluation.model_selection
    if not selection.enable:
        return
    metadata = {
        "outer_k": int(outer_fold.k_test),
        "num_candidates": int(len(candidates)),
        "num_splits": int(len(inner_splits)),
        "num_pp_samples": int(
            outer_context.config.evaluation.predictive.num_samples_inner
        ),
        "assets": list(outer_context.context.assets),
        "phase_name": selection.phase_name,
        "candidates": {
            str(item.candidate_id): item.params for item in candidates
        },
        "splits_path": str(
            outer_context.artifacts.base_dir
            / "inner"
            / f"outer_{outer_fold.k_test}"
            / "splits.json"
        ),
    }
    outer_context.artifacts.write_postprocess_metadata(
        outer_k=outer_fold.k_test, metadata=metadata
    )

def _run_outer_evaluation(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    best_config: Mapping[str, Any],
    week_progress: Callable[[int, Any], None] | None,
) -> tuple[Mapping[str, Any], FeatureCleaningState | None]:
    set_runtime_seed(
        _outer_runtime_seed(
            config=outer_context.config,
            flags=outer_context.flags,
            fold_id=outer_fold.k_test,
        )
    )
    result, cleaning_outer = evaluate_outer_walk_forward(
        context=OuterEvaluationContext(
            panel=FeaturePanelData(
                X=outer_context.context.X,
                M=outer_context.context.M,
                X_global=outer_context.context.X_global,
                M_global=outer_context.context.M_global,
                global_feature_names=outer_context.context.global_feature_names,
            ),
            y=outer_context.context.y,
            timestamps=outer_context.context.timestamps,
            outer_fold=outer_fold,
            preprocess_spec=outer_context.context.preprocess_spec,
            num_pp_samples=(
                outer_context.config.evaluation.predictive.num_samples_outer
            ),
            portfolios=_build_portfolio_specs_for_outer(
                outer_context.config.evaluation.allocation,
                outer_context.config.evaluation.cost.spec,
            ),
            assets=outer_context.context.assets,
            execution_mode=outer_context.flags.execution_mode,
            week_progress=week_progress,
        ),
        best_config=best_config,
        hooks=outer_context.hooks,
    )
    if cleaning_outer is not None:
        outer_context.artifacts.write_cleaning_state(
            outer_k=outer_fold.k_test,
            cleaning=cleaning_outer,
            feature_names=outer_context.context.feature_names,
            spec=outer_context.context.preprocess_spec,
        )
    outer_context.artifacts.write_outer_result(
        outer_k=outer_fold.k_test,
        result=result,
    )
    return result, cleaning_outer


def _build_portfolio_specs_for_outer(
    allocation: AllocationConfig,
    cost_spec: Mapping[str, Any],
) -> tuple[PortfolioSpec, ...]:
    portfolios: list[PortfolioSpec] = [
        PortfolioSpec(
            name="primary",
            allocation=allocation.primary.to_spec(),
            cost=cost_spec,
        )
    ]
    used_names = {"primary"}
    for baseline in allocation.baselines:
        name = _unique_portfolio_name(baseline.family, used_names)
        used_names.add(name)
        portfolios.append(
            PortfolioSpec(
                name=name,
                allocation=baseline.to_spec(),
                cost=cost_spec,
            )
        )
    return tuple(portfolios)


def _unique_portfolio_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name
    suffix = 2
    while f"{name}_{suffix}" in used_names:
        suffix += 1
    return f"{name}_{suffix}"

def _log_outer_complete(
    *,
    outer_k: int,
    started: float,
    phase: str,
    result: Mapping[str, Any] | None,
    cleaning: FeatureCleaningState | None,
) -> None:
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "outer_k": outer_k,
        "phase": phase,
        "elapsed_s": round(elapsed, 4),
    }
    if cleaning is not None:
        payload["n_features_kept"] = int(cleaning.feature_idx.size)
    if result is not None:
        pnl = result.get("pnl")
        if isinstance(pnl, list) and pnl:
            pnl_arr = np.asarray(pnl, dtype=float)
            payload["pnl_mean"] = float(np.mean(pnl_arr))
            payload["pnl_std"] = float(np.std(pnl_arr))
    logger.info(
        "event=complete boundary=simulation.outer context=%s",
        payload,
    )


def _run_outer_from_saved_inner(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    resume_tracker: SimulationResumeTracker,
    week_progress: Callable[[int, Any], None] | None,
) -> OuterFoldRunResult:
    best_config = load_best_config(
        base_dir=outer_context.artifacts.base_dir,
        outer_k=outer_fold.k_test,
    )
    resume_tracker.mark_outer_started(outer_fold.k_test)
    started = time.perf_counter()
    result, cleaning_outer = _run_outer_evaluation(
        outer_context=outer_context,
        outer_fold=outer_fold,
        best_config=best_config,
        week_progress=week_progress,
    )
    resume_tracker.mark_outer_completed(outer_fold.k_test)
    if not _is_walkforward_mode(outer_context.flags):
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="outer",
            result=result,
            cleaning=cleaning_outer,
        )
    return OuterFoldRunResult(best_config=best_config, outer_result=result)

def _build_artifacts(
    *,
    base_dir: Path,
    dataset: PanelDataset,
    context: SimulationContext,
    write_inputs: bool,
    dataset_params: Mapping[str, Any],
) -> SimulationArtifacts:
    artifacts = SimulationArtifacts(base_dir)
    if write_inputs:
        artifacts.write_inputs(
            inputs=SimulationInputs(
                X=context.X,
                M=context.M,
                X_global=context.X_global,
                M_global=context.M_global,
                y=context.y,
                timestamps=dataset.dates,
                assets=dataset.assets,
                features=dataset.features,
                global_features=dataset.global_features,
            )
        )
        write_data_source_metadata(
            base_dir=base_dir,
            dataset_params=dataset_params,
        )
    return artifacts


def _stage_walkforward_source_artifacts(
    *,
    config: SimulationConfig,
    source_dir: Path,
    target_dir: Path,
) -> None:
    if not _is_walkforward_mode(config.flags):
        return
    if source_dir == target_dir:
        return
    _copy_optional_walkforward_artifact(
        source=source_dir / "outer" / "best_config.json",
        target=target_dir / "outer" / "best_config.json",
    )
    for source_path in source_dir.glob("inner/outer_*/best_config.json"):
        relative_path = source_path.relative_to(source_dir)
        _copy_optional_walkforward_artifact(
            source=source_path,
            target=target_dir / relative_path,
        )


def _copy_optional_walkforward_artifact(*, source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(source.read_bytes())
    except OSError as exc:
        raise SimulationError(
            "Failed to stage walkforward source artifact",
            context={"source": str(source), "target": str(target)},
        ) from exc

def _build_results(
    config: SimulationConfig,
    context: SimulationContext,
    chosen_configs: Mapping[int, Mapping[str, Any]],
    outer_results: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "cv_params": config.cv,
        "preprocess_spec": context.preprocess_spec,
        "n_groups": len(context.cv.groups),
        "outer_test_group_ids": context.cv.outer_ids,
        "chosen_configs_by_outer_group": chosen_configs,
        "outer_results": outer_results,
    }

def _write_candidate_artifacts(
    *,
    config: SimulationConfig,
    artifacts: SimulationArtifacts,
) -> list[CandidateSpec]:
    tuning = config.modeling.tuning
    candidates = build_candidates(
        space=tuning.space,
        num_samples=tuning.num_samples,
        seed=tuning.seed,
    )
    candidate_specs = assign_candidate_ids(candidates)
    artifacts.write_candidates(candidates=candidate_specs)
    return candidate_specs
