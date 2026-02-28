# pylint: disable=too-many-lines
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import (
    CandidateSpec,
    CPCVSplit,
    DataPaths,
    FeatureCleaningState,
    OuterFold,
    PanelDataset,
    PreprocessSpec,
    SimulationConfig,
    SimulationFlags,
)
from algo_trader.infrastructure import log_boundary
from algo_trader.infrastructure.data import load_panel_tensor_dataset

from .config import DEFAULT_CONFIG_PATH, load_config
from .cv_groups import build_equal_groups, make_cpcv_splits, make_outer_folds
from .hooks import SimulationHooks, default_hooks, stub_hooks
from .inner_objective import (
    InnerObjectiveContext,
    InnerObjectiveData,
    InnerObjectiveParams,
    make_inner_objective,
)
from .outer_walk_forward import (
    OuterEvaluationContext,
    PortfolioSpec,
    evaluate_outer_walk_forward,
)
from .artifacts import (
    CVStructureInputs,
    SimulationArtifacts,
    SimulationInputs,
    resolve_simulation_output_dir,
)
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
    load_completed_outer_fold,
    load_outer_result,
    mark_outer_complete_for_all,
    resolve_ray_selection_plan,
    resolve_ray_storage_path,
    validate_resume_request,
)
from .runner_helpers import (
    build_base_config,
    build_group_by_index,
    outer_fold_payload,
    resolve_outer_test_group_ids,
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

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CVStructure:
    warmup_idx: np.ndarray
    groups: list[np.ndarray]
    outer_ids: list[int]
    outer_folds: list
    group_by_index: np.ndarray

@dataclass(frozen=True)
class SimulationContext:
    X: torch.Tensor
    M: torch.Tensor
    y: torch.Tensor
    preprocess_spec: PreprocessSpec
    cv: CVStructure
    feature_names: Sequence[str]
    assets: Sequence[str]

def _run_context(config_path: Path | None, resume: bool) -> Mapping[str, str]:
    return {
        "config": str(config_path or DEFAULT_CONFIG_PATH),
        "resume": str(resume),
    }

@log_boundary("simulation.run", context=_run_context)
def run(
    *, config_path: Path | None = None, resume: bool = False
) -> Mapping[str, Any]:
    config = apply_smoke_test_overrides(load_config(config_path))
    validate_resume_request(config=config, resume=resume)
    device = _resolve_device(config.flags.use_gpu)
    setup = _prepare_run_state(config=config, device=device)
    if setup.early_results is not None:
        return setup.early_results
    resume_state = ResumeState(enabled=resume)
    use_ray = config.modeling.tuning.engine == "ray"
    ray_address = config.modeling.tuning.ray.address
    ray_logs_enabled = config.modeling.tuning.ray.logs_enabled
    cleanup_before_simulation_run(
        use_ray=use_ray,
        ray_address=ray_address,
        use_gpu=config.flags.use_gpu,
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
    if use_ray:
        init_ray_for_tuning(ray_address, logs_enabled=ray_logs_enabled)
    interrupted = False
    try:
        chosen_configs, outer_results = _evaluate_outer_folds(
            outer_context=setup.outer_context,
            candidates=setup.candidates,
            resume_state=resume_state,
            resume_tracker=resume_tracker,
        )
        if resume_tracker is not None:
            resume_tracker.mark_run_completed()
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        cleanup_after_simulation_run(
            use_ray=use_ray,
            ray_address=ray_address,
            use_gpu=config.flags.use_gpu,
            interrupted=interrupted,
        )
    results = _build_results(
        config,
        setup.outer_context.context,
        chosen_configs,
        outer_results,
    )
    results = with_run_meta(results, config.flags)
    setup.outer_context.artifacts.write_results(results)
    setup.outer_context.artifacts.write_cv_summary(results)
    return results

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
    base_dir, dataset, reused_inputs = _resolve_run_inputs(
        config=config, device=device
    )
    context = _build_context(config, dataset)
    artifacts = _build_artifacts(
        base_dir=base_dir,
        dataset=dataset,
        context=context,
        write_inputs=not reused_inputs,
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
            y=context.y,
            M=context.M,
            outer_folds=context.cv.outer_folds,
            group_by_index=context.cv.group_by_index,
            feature_names=context.feature_names,
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

def _resolve_preprocess_spec(
    spec: PreprocessSpec,
    dataset: PanelDataset,
    use_feature_names_for_scaling: bool,
) -> PreprocessSpec:
    if not use_feature_names_for_scaling:
        return spec
    if spec.scaling.inputs.feature_names is not None:
        return spec
    return replace(
        spec,
        scaling=replace(
            spec.scaling,
            inputs=replace(
                spec.scaling.inputs, feature_names=list(dataset.features)
            ),
        ),
    )


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
        "feature_store_panel", config=config.data, device=device
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
        if not tensor_path.exists():
            raise SimulationError(
                "panel_tensor.pt missing in simulation output directory",
                context={"path": str(tensor_path)},
            )
        dataset = load_panel_tensor_dataset(
            paths=DataPaths(tensor_path=str(tensor_path)),
            device=device,
        )
        logger.info(
            "Using existing simulation inputs path=%s", tensor_path
        )
        return dataset, True
    dataset = _load_dataset(config, device)
    return dataset, False

def _build_context(
    config: SimulationConfig, dataset: PanelDataset
) -> SimulationContext:
    preprocess_spec = _resolve_preprocess_spec(
        config.preprocessing, dataset, config.flags.use_feature_names_for_scaling
    )
    X, M, y = dataset.data, dataset.missing_mask, dataset.targets
    warmup_idx, groups = build_equal_groups(
        T=int(X.shape[0]),
        warmup_len=config.cv.window.warmup_len,
        group_len=config.cv.window.group_len,
    )
    if not groups:
        raise SimulationError("No groups available; check cv settings")
    outer_ids = resolve_outer_test_group_ids(
        config.outer.test_group_ids, config.outer.last_n, len(groups)
    )
    outer_folds = make_outer_folds(
        warmup_idx=warmup_idx,
        groups=groups,
        outer_test_group_ids=outer_ids,
        exclude_warmup=config.cv.exclude_warmup,
    )
    group_by_index = build_group_by_index(
        groups=groups, total_len=int(X.shape[0])
    )
    return SimulationContext(
        X=X,
        M=M,
        y=y,
        preprocess_spec=preprocess_spec,
        cv=CVStructure(
            warmup_idx=warmup_idx,
            groups=groups,
            outer_ids=outer_ids,
            outer_folds=outer_folds,
            group_by_index=group_by_index,
        ),
        feature_names=list(dataset.features),
        assets=list(dataset.assets),
    )

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
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    selection = outer_context.config.evaluation.model_selection
    if selection.enable:
        return _evaluate_outer_folds_global(
            outer_context=outer_context,
            candidates=candidates,
            resume_state=resume_state,
            resume_tracker=resume_tracker,
        )
    return _evaluate_outer_folds_per_outer(
        outer_context=outer_context,
        candidates=candidates,
        resume_state=resume_state,
        resume_tracker=resume_tracker,
    )

def _evaluate_outer_folds_per_outer(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
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
            resume_state=resume_state,
            resume_tracker=resume_tracker,
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
        )
        if resume_tracker is not None:
            resume_tracker.mark_outer_completed(outer_fold.k_test)
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="outer",
            result=result,
            cleaning=cleaning_outer,
        )
        outer_results.append(result)
    return chosen_configs, outer_results

@dataclass(frozen=True)
class OuterFoldRunResult:
    best_config: Mapping[str, Any]
    outer_result: Mapping[str, Any] | None


@dataclass(frozen=True)
class InnerObjectiveBundle:
    objective: Any
    context: InnerObjectiveContext
    hooks: SimulationHooks

def _run_outer_fold(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    candidates: Sequence[CandidateSpec],
    resume_state: ResumeState,
    resume_tracker: SimulationResumeTracker | None,
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
    if should_stop_after("inner", outer_context.flags):
        if resume_tracker is not None:
            resume_tracker.mark_outer_completed(outer_fold.k_test)
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="inner",
            result=None,
            cleaning=None,
        )
        return OuterFoldRunResult(best_config=best_config, outer_result=None)
    if resume_tracker is not None:
        resume_tracker.mark_outer_started(outer_fold.k_test)
    result, cleaning_outer = _run_outer_evaluation(
        outer_context=outer_context,
        outer_fold=outer_fold,
        best_config=best_config,
    )
    if resume_tracker is not None:
        resume_tracker.mark_outer_completed(outer_fold.k_test)
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
            X=outer_context.context.X,
            M=outer_context.context.M,
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
    inner_splits: list[CPCVSplit],
    best_config: Mapping[str, Any],
) -> None:
    outer_context.artifacts.write_inner(
        outer_k=outer_fold.k_test,
        inner_splits=inner_splits,
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
) -> tuple[Mapping[str, Any], FeatureCleaningState | None]:
    result, cleaning_outer = evaluate_outer_walk_forward(
        context=OuterEvaluationContext(
            X=outer_context.context.X,
            M=outer_context.context.M,
            y=outer_context.context.y,
            outer_fold=outer_fold,
            preprocess_spec=outer_context.context.preprocess_spec,
            num_pp_samples=(
                outer_context.config.evaluation.predictive.num_samples_outer
            ),
            portfolio=PortfolioSpec(
                allocation=outer_context.config.evaluation.allocation.spec,
                cost=outer_context.config.evaluation.cost.spec,
            ),
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
    )
    resume_tracker.mark_outer_completed(outer_fold.k_test)
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
) -> SimulationArtifacts:
    artifacts = SimulationArtifacts(base_dir)
    if write_inputs:
        artifacts.write_inputs(
            inputs=SimulationInputs(
                X=context.X,
                M=context.M,
                y=context.y,
                timestamps=dataset.dates,
                assets=dataset.assets,
                features=dataset.features,
            )
        )
    return artifacts

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
