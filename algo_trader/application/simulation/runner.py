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
    CVParams,
    DataPaths,
    FeatureCleaningState,
    ModelSelectionConfig,
    OuterFold,
    PanelDataset,
    PreprocessSpec,
    SimulationConfig,
    SimulationFlags,
    TrainingConfig,
    TuningConfig,
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
from .tuning import (
    apply_param_updates,
    assign_candidate_ids,
    build_candidates,
    with_candidate_context,
)
from .tune_runner import (
    RayTuneContext,
    init_ray_for_tuning,
    select_best_with_ray,
    shutdown_ray_for_tuning,
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
from .diagnostics import FanChartDiagnosticsContext, run_fan_chart_diagnostics
from .model_selection import (
    GlobalSelectionContext,
    PostTuneSelectionContext,
    select_best_candidate_global,
    select_best_candidate_post_tune,
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

def _run_context(config_path: Path | None) -> Mapping[str, str]:
    return {"config": str(config_path or DEFAULT_CONFIG_PATH)}

@log_boundary("simulation.run", context=_run_context)
def run(*, config_path: Path | None = None) -> Mapping[str, Any]:
    config = apply_smoke_test_overrides(load_config(config_path))
    device = _resolve_device(config.flags.use_gpu)
    setup = _prepare_run_state(config=config, device=device)
    if setup.early_results is not None:
        return setup.early_results
    use_ray = config.modeling.tuning.engine == "ray"
    ray_address = config.modeling.tuning.ray.address
    if use_ray:
        init_ray_for_tuning(ray_address)
    try:
        chosen_configs, outer_results = _evaluate_outer_folds(
            outer_context=setup.outer_context,
            candidates=setup.candidates,
        )
    finally:
        if use_ray and ray_address is None:
            shutdown_ray_for_tuning()
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

def _evaluate_outer_folds(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    selection = outer_context.config.evaluation.model_selection
    if selection.enable:
        return _evaluate_outer_folds_global(
            outer_context=outer_context,
            candidates=candidates,
        )
    return _evaluate_outer_folds_per_outer(
        outer_context=outer_context,
        candidates=candidates,
    )

def _evaluate_outer_folds_per_outer(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []
    for outer_fold in outer_context.context.cv.outer_folds:
        fold_result = _run_outer_fold(
            outer_context=outer_context,
            outer_fold=outer_fold,
            candidates=candidates,
        )
        chosen_configs[outer_fold.k_test] = fold_result.best_config
        if fold_result.outer_result is not None:
            outer_results.append(fold_result.outer_result)
    return chosen_configs, outer_results

def _evaluate_outer_folds_global(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []
    for outer_fold in outer_context.context.cv.outer_folds:
        _run_outer_fold_inner_only(
            outer_context=outer_context,
            outer_fold=outer_fold,
            candidates=candidates,
        )
    global_best = _select_global_best_config(
        outer_context=outer_context,
        candidates=candidates,
    )
    _run_global_diagnostics(
        outer_context=outer_context,
        global_best=global_best,
    )
    for outer_fold in outer_context.context.cv.outer_folds:
        chosen_configs[outer_fold.k_test] = global_best.best_config
    if should_stop_after("inner", outer_context.flags):
        return chosen_configs, outer_results
    for outer_fold in outer_context.context.cv.outer_folds:
        started = time.perf_counter()
        result, cleaning_outer = _run_outer_evaluation(
            outer_context=outer_context,
            outer_fold=outer_fold,
            best_config=global_best.best_config,
        )
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
class GlobalBestConfig:
    best_config: Mapping[str, Any]
    candidate_id: int

@dataclass(frozen=True)
class BestConfigContext:
    objective: Any
    base_config: Mapping[str, Any]
    candidates: Sequence[CandidateSpec]
    resources: "BestConfigResources"

@dataclass(frozen=True)
class BestConfigResources:
    tuning: TuningConfig
    use_gpu: bool
    model_selection: ModelSelectionConfig
    artifacts: SimulationArtifacts
    outer_k: int

def _run_outer_fold(
    *,
    outer_context: OuterFoldContext,
    outer_fold: OuterFold,
    candidates: Sequence[CandidateSpec],
) -> OuterFoldRunResult:
    started = time.perf_counter()
    inner_splits = _build_inner_splits(
        outer_context=outer_context,
        outer_fold=outer_fold,
    )
    objective = _build_inner_objective(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    _write_postprocess_metadata(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        candidates=candidates,
    )
    best_config = _select_best_config(
        BestConfigContext(
            objective=objective,
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
        )
    )
    _write_inner_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        best_config=best_config,
    )
    if should_stop_after("inner", outer_context.flags):
        _log_outer_complete(
            outer_k=outer_fold.k_test,
            started=started,
            phase="inner",
            result=None,
            cleaning=None,
        )
        return OuterFoldRunResult(best_config=best_config, outer_result=None)
    result, cleaning_outer = _run_outer_evaluation(
        outer_context=outer_context,
        outer_fold=outer_fold,
        best_config=best_config,
    )
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
) -> Mapping[str, Any]:
    started = time.perf_counter()
    inner_splits = _build_inner_splits(
        outer_context=outer_context,
        outer_fold=outer_fold,
    )
    objective = _build_inner_objective(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
    )
    _write_postprocess_metadata(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        candidates=candidates,
    )
    best_config = _select_best_config(
        BestConfigContext(
            objective=objective,
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
        )
    )
    _write_inner_outputs(
        outer_context=outer_context,
        outer_fold=outer_fold,
        inner_splits=inner_splits,
        best_config=best_config,
    )
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
    return make_inner_objective(
        context=InnerObjectiveContext(
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
        ),
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

def _select_best_config(
    context: BestConfigContext,
) -> Mapping[str, Any]:
    resources = context.resources
    if resources.tuning.engine == "ray":
        best_config = select_best_with_ray(
            RayTuneContext(
                objective=context.objective,
                base_config=context.base_config,
                candidates=context.candidates,
                resources=resources.tuning.ray.resources,
                use_gpu=resources.use_gpu,
                address=resources.tuning.ray.address,
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

def _select_global_best_config(
    *,
    outer_context: OuterFoldContext,
    candidates: Sequence[CandidateSpec],
) -> GlobalBestConfig:
    selection = select_best_candidate_global(
        GlobalSelectionContext(
            artifacts=outer_context.artifacts,
            outer_ids=outer_context.context.cv.outer_ids,
            candidates=candidates,
            model_selection=(
                outer_context.config.evaluation.model_selection
            ),
        )
    )
    params = _candidate_params_by_id(
        candidates=candidates, candidate_id=selection.best_candidate_id
    )
    best_config = apply_param_updates(outer_context.base_config, params)
    outer_context.artifacts.write_global_best_config(payload=best_config)
    return GlobalBestConfig(
        best_config=best_config,
        candidate_id=int(selection.best_candidate_id),
    )

def _run_global_diagnostics(
    *,
    outer_context: OuterFoldContext,
    global_best: GlobalBestConfig,
) -> None:
    diagnostics = outer_context.config.evaluation.diagnostics
    if not diagnostics.fan_charts.enable:
        return
    if not outer_context.config.evaluation.model_selection.enable:
        raise SimulationError(
            "Diagnostics require model_selection.enable"
        )
    run_fan_chart_diagnostics(
        FanChartDiagnosticsContext(
            base_dir=outer_context.artifacts.base_dir,
            outer_ids=outer_context.context.cv.outer_ids,
            candidate_id=global_best.candidate_id,
            config=diagnostics.fan_charts,
        )
    )
