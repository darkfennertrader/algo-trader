from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import (
    CVParams,
    ModelConfig,
    OuterFold,
    PanelDataset,
    PreprocessSpec,
    SimulationConfig,
    SimulationFlags,
    TrainingConfig,
    TuningConfig,
)
from algo_trader.infrastructure import log_boundary

from .config import DEFAULT_CONFIG_PATH, load_config
from .cv_groups import build_equal_groups, make_cpcv_splits, make_outer_folds
from .hooks import SimulationHooks, default_hooks
from .inner_objective import InnerObjectiveContext, make_inner_objective
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
from .registry import default_registries
from .tuning import expand_param_space, select_candidates

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVStructure:
    warmup_idx: np.ndarray
    groups: list[np.ndarray]
    outer_ids: list[int]
    outer_folds: list


@dataclass(frozen=True)
class SimulationContext:
    X: torch.Tensor
    M: torch.Tensor
    y: torch.Tensor
    preprocess_spec: PreprocessSpec
    cv: CVStructure


def _run_context(config_path: Path | None) -> Mapping[str, str]:
    return {"config": str(config_path or DEFAULT_CONFIG_PATH)}


@log_boundary("simulation.run", context=_run_context)
def run(*, config_path: Path | None = None) -> Mapping[str, Any]:
    config = load_config(config_path)
    device = _resolve_device(config.flags.use_gpu)
    hooks = default_hooks()

    dataset = _load_dataset(config, device)
    context = _build_context(config, dataset)
    artifacts = _build_artifacts(config, dataset, context)
    if _should_stop_after("inputs", config.flags):
        results = _build_results(
            config, context, chosen_configs={}, outer_results=[]
        )
        results = _with_run_meta(results, config.flags)
        artifacts.write_results(results)
        return results
    artifacts.write_cv_structure(
        inputs=CVStructureInputs(
            warmup_idx=context.cv.warmup_idx,
            groups=context.cv.groups,
            outer_ids=context.cv.outer_ids,
            outer_folds=[
                _outer_fold_payload(fold)
                for fold in context.cv.outer_folds
            ],
            timestamps=dataset.dates,
        )
    )
    base_config = _build_base_config(
        config.modeling.model, config.modeling.training
    )

    if _should_stop_after("cv", config.flags):
        results = _build_results(
            config, context, chosen_configs={}, outer_results=[]
        )
        results = _with_run_meta(results, config.flags)
        artifacts.write_results(results)
        artifacts.write_cv_summary(results)
        return results

    chosen_configs, outer_results = _evaluate_outer_folds(
        outer_context=OuterFoldContext(
            config=config,
            context=context,
            base_config=base_config,
            hooks=hooks,
            artifacts=artifacts,
            flags=config.flags,
        )
    )

    results = _build_results(config, context, chosen_configs, outer_results)
    results = _with_run_meta(results, config.flags)
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
    if spec.scaling.feature_names is not None:
        return spec
    return replace(
        spec,
        scaling=replace(
            spec.scaling, feature_names=list(dataset.features)
        ),
    )


def _load_dataset(config: SimulationConfig, device: str) -> PanelDataset:
    registries = default_registries()
    dataset = registries.datasets.build(
        config.data.dataset_name, config=config.data, device=device
    )
    return dataset.select_period_and_subsets(config.data)


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
    outer_ids = _resolve_outer_test_group_ids(
        config.outer.test_group_ids, config.outer.last_n, len(groups)
    )
    outer_folds = make_outer_folds(
        warmup_idx=warmup_idx, groups=groups, outer_test_group_ids=outer_ids
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
        ),
    )


def _with_fold_seed(cv: CVParams, fold_id: int) -> CVParams:
    return replace(
        cv,
        cpcv=replace(cv.cpcv, seed=_fold_seed(cv.cpcv.seed, fold_id)),
    )


def _fold_seed(base_seed: int, fold_id: int) -> int:
    return base_seed + 10_000 * fold_id


@dataclass(frozen=True)
class OuterFoldContext:
    config: SimulationConfig
    context: SimulationContext
    base_config: Mapping[str, Any]
    hooks: SimulationHooks
    artifacts: SimulationArtifacts
    flags: SimulationFlags


def _evaluate_outer_folds(
    *, outer_context: OuterFoldContext
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    chosen_configs: dict[int, Mapping[str, Any]] = {}
    outer_results: list[Mapping[str, Any]] = []

    config = outer_context.config
    context = outer_context.context
    hooks = outer_context.hooks
    artifacts = outer_context.artifacts
    flags = outer_context.flags

    for outer_fold in context.cv.outer_folds:
        inner_splits = make_cpcv_splits(
            warmup_idx=context.cv.warmup_idx,
            groups=context.cv.groups,
            inner_group_ids=outer_fold.inner_group_ids,
            params=_with_fold_seed(config.cv, outer_fold.k_test),
        )
        if not inner_splits:
            raise SimulationError(
                "No inner CPCV splits available; check cv settings",
                context={"outer_group": str(outer_fold.k_test)},
            )

        objective = make_inner_objective(
            context=InnerObjectiveContext(
                X=context.X,
                M=context.M,
                y=context.y,
                inner_splits=inner_splits,
                preprocess_spec=context.preprocess_spec,
                score_spec=config.evaluation.scoring.spec,
                num_pp_samples=config.evaluation.predictive.num_samples_inner,
            ),
            hooks=hooks,
        )

        best_config = _select_best_config(
            objective=objective,
            base_config=outer_context.base_config,
            tuning=config.modeling.tuning,
            seed=_fold_seed(config.cv.cpcv.seed, outer_fold.k_test),
        )
        chosen_configs[outer_fold.k_test] = best_config
        artifacts.write_inner(
            outer_k=outer_fold.k_test,
            inner_splits=inner_splits,
            best_config=best_config,
        )
        if _should_stop_after("inner", flags):
            continue

        result, cleaning_outer = evaluate_outer_walk_forward(
            context=OuterEvaluationContext(
                X=context.X,
                M=context.M,
                y=context.y,
                outer_fold=outer_fold,
                preprocess_spec=context.preprocess_spec,
                num_pp_samples=config.evaluation.predictive.num_samples_outer,
                portfolio=PortfolioSpec(
                    allocation=config.evaluation.allocation.spec,
                    cost=config.evaluation.cost.spec,
                ),
            ),
            best_config=best_config,
            hooks=hooks,
        )
        if cleaning_outer is not None:
            artifacts.write_cleaning_state(
                outer_k=outer_fold.k_test, cleaning=cleaning_outer
            )
        artifacts.write_outer_result(
            outer_k=outer_fold.k_test,
            result=result,
        )
        outer_results.append(result)

    return chosen_configs, outer_results


def _build_artifacts(
    config: SimulationConfig,
    dataset: PanelDataset,
    context: SimulationContext,
) -> SimulationArtifacts:
    base_dir = resolve_simulation_output_dir(
        dataset_name=config.data.dataset_name,
        dataset_params=config.data.dataset_params,
    )
    artifacts = SimulationArtifacts(base_dir)
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


def _outer_fold_payload(outer_fold: OuterFold) -> Mapping[str, Any]:
    return {
        "k_test": int(outer_fold.k_test),
        "train_idx": outer_fold.train_idx.tolist(),
        "test_idx": outer_fold.test_idx.tolist(),
        "inner_group_ids": list(outer_fold.inner_group_ids),
    }


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


def _should_stop_after(phase: str, flags: SimulationFlags) -> bool:
    if flags.stop_after is not None:
        return phase == flags.stop_after
    if flags.simulation_mode == "dry_run":
        return phase == "cv"
    return False


def _with_run_meta(
    results: Mapping[str, Any], flags: SimulationFlags
) -> Mapping[str, Any]:
    enriched = dict(results)
    enriched["run_mode"] = flags.simulation_mode
    enriched["stop_after"] = flags.stop_after
    return enriched


def _resolve_outer_test_group_ids(
    outer_ids: list[int] | None,
    outer_last_n: int | None,
    n_groups: int,
) -> list[int]:
    if outer_ids is not None:
        return outer_ids
    if outer_last_n is not None:
        if outer_last_n <= 0:
            raise ConfigError("outer.last_n must be positive")
        start = max(0, n_groups - outer_last_n)
        return list(range(start, n_groups))
    n_outer = min(5, n_groups)
    return list(range(n_groups - n_outer, n_groups))


def _build_base_config(
    model: ModelConfig,
    training: TrainingConfig,
) -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "guide_name": model.guide_name,
        "model_params": dict(model.params),
        "training": {
            "trainer_name": training.trainer_name,
            "num_steps": training.num_steps,
            "learning_rate": training.learning_rate,
            "batch_size": training.batch_size,
            "num_elbo_particles": training.num_elbo_particles,
        },
    }


def _select_best_config(
    *,
    objective,
    base_config: Mapping[str, Any],
    tuning: TuningConfig,
    seed: int,
) -> Mapping[str, Any]:
    configs = expand_param_space(tuning.param_space)
    candidates = select_candidates(configs, tuning.num_samples, seed)

    best_score = float("-inf")
    best_config: Mapping[str, Any] = dict(base_config)

    for candidate in candidates:
        merged = dict(base_config)
        merged.update(candidate)
        score = float(objective(merged))
        if score > best_score:
            best_score = score
            best_config = merged

    return best_config
