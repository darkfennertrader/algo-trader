from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import logging
import time

import numpy as np
import torch
from torch.linalg import cholesky, solve_triangular

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import (
    CPCVSplit,
    FeatureCleaningState,
    ModelSelectionConfig,
    PreprocessSpec,
)
from .artifacts import SimulationArtifacts
from .hooks import SimulationHooks
from .preprocessing import (
    GlobalBlockInputs,
    TransformState,
    fit_feature_cleaning,
    fit_global_feature_cleaning,
    fit_robust_scaler,
    fit_global_robust_scaler,
    transform_global_X,
    transform_X,
)
from .target_space_transform import TargetSpaceTransform


@dataclass(frozen=True)
class InnerObjectiveData:
    X: torch.Tensor
    M: torch.Tensor
    X_global: torch.Tensor | None
    M_global: torch.Tensor | None
    global_feature_names: Sequence[str]
    assets: Sequence[str]
    y: torch.Tensor


@dataclass(frozen=True)
class InnerObjectiveParams:
    preprocess_spec: PreprocessSpec
    score_spec: Mapping[str, Any]
    num_pp_samples: int
    outer_k: int
    aggregate: str
    aggregate_lambda: float


@dataclass(frozen=True)
class InnerObjectiveContext:
    data: InnerObjectiveData
    artifacts: SimulationArtifacts
    group_by_index: np.ndarray
    inner_splits: list[CPCVSplit]
    params: InnerObjectiveParams
    model_selection: ModelSelectionConfig


logger = logging.getLogger(__name__)


def make_inner_objective(
    *,
    context: InnerObjectiveContext,
    hooks: SimulationHooks,
) -> Callable[[Mapping[str, Any]], float]:
    def objective(config: Mapping[str, Any]) -> float:
        candidate_id = _require_candidate_id(config)
        fold_scores = evaluate_inner_splits(
            SplitEvaluationRequest(
                context=context,
                hooks=hooks,
                config=config,
                candidate_id=candidate_id,
            )
        )
        return aggregate_scores(
            fold_scores,
            method=context.params.aggregate,
            penalty=context.params.aggregate_lambda,
        )

    return objective


@dataclass(frozen=True)
class SplitEvaluationRequest:
    context: InnerObjectiveContext
    hooks: SimulationHooks
    config: Mapping[str, Any]
    candidate_id: int


@dataclass(frozen=True)
class SplitEvaluationResume:
    start_split_id: int = 0
    prior_scores: Sequence[float] | None = None
    on_split: Callable[[int, list[float]], None] | None = None


def evaluate_inner_splits(
    request: SplitEvaluationRequest,
    *,
    resume: SplitEvaluationResume | None = None,
) -> list[float]:
    effective_resume = resume or SplitEvaluationResume()
    fold_scores = list(effective_resume.prior_scores or [])
    for split_id, split in enumerate(request.context.inner_splits):
        if split_id < effective_resume.start_split_id:
            continue
        result = _evaluate_split(
            SplitContext(
                data=request.context.data,
                params=request.context.params,
                split=split,
                split_id=split_id,
                config=request.config,
                candidate_id=request.candidate_id,
                resources=SplitResources(
                    hooks=request.hooks,
                    artifacts=request.context.artifacts,
                    group_by_index=request.context.group_by_index,
                    model_selection=request.context.model_selection,
                ),
            )
        )
        if result is not None:
            fold_scores.append(result.score)
            logger.info(
                "event=complete boundary=simulation.inner_split context=%s",
                {
                    "outer_k": request.context.params.outer_k,
                    "split_id": split_id,
                    "train_size": result.train_size,
                    "test_size": result.test_size,
                    "score": result.score,
                    "elapsed_s": round(result.elapsed_s, 4),
                    "n_features_kept": result.n_features_kept,
                },
            )
        if effective_resume.on_split:
            effective_resume.on_split(split_id, fold_scores)
    return fold_scores


def _require_candidate_id(config: Mapping[str, Any]) -> int:
    run_context = config.get("run_context")
    if not isinstance(run_context, Mapping):
        return -1
    candidate_id = run_context.get("candidate_id", -1)
    try:
        return int(candidate_id)
    except (TypeError, ValueError) as exc:
        raise SimulationError(
            "candidate_id must be an int in run_context"
        ) from exc


@dataclass(frozen=True)
class SplitResult:
    score: float
    train_size: int
    test_size: int
    n_features_kept: int
    elapsed_s: float


@dataclass(frozen=True)
class SplitContext:
    data: InnerObjectiveData
    params: InnerObjectiveParams
    split: CPCVSplit
    split_id: int
    config: Mapping[str, Any]
    candidate_id: int
    resources: "SplitResources"


@dataclass(frozen=True)
class SplitResources:
    hooks: SimulationHooks
    artifacts: SimulationArtifacts
    group_by_index: np.ndarray
    model_selection: ModelSelectionConfig


@dataclass(frozen=True)
class SplitBatches:
    cleaning: FeatureCleaningState
    X_train: torch.Tensor
    X_train_global: torch.Tensor | None
    y_train: torch.Tensor
    X_test: torch.Tensor
    X_test_global: torch.Tensor | None
    y_test: torch.Tensor


@dataclass(frozen=True)
class _OnlineStepInputs:
    X_step: torch.Tensor
    X_step_global: torch.Tensor | None
    y_step: torch.Tensor


def _evaluate_split(context: SplitContext) -> SplitResult | None:
    started = time.perf_counter()
    split_config = _split_config(
        context.config,
        context.params.outer_k,
        context.split_id,
        context.data.assets,
    )
    batches = _prepare_split_batches(context)
    if batches is None:
        return None
    model_state = context.resources.hooks.fit_model(
        X_train=batches.X_train,
        X_train_global=batches.X_train_global,
        y_train=batches.y_train,
        config=split_config,
        init_state=None,
    )
    pred = _predict_validation_horizon(
        hooks=context.resources.hooks,
        batches=batches,
        config=split_config,
        num_samples=context.params.num_pp_samples,
        model_state=model_state,
    )
    pred = _maybe_prepare_energy_score_pred(
        pred=pred,
        y_train=batches.y_train,
        score_spec=context.params.score_spec,
    )
    _maybe_write_postprocess_inputs(
        context=context,
        pred=pred,
        y_train=batches.y_train,
        y_test=batches.y_test,
        model_state=model_state,
    )
    _maybe_write_postprocess_diagnostics(
        context=context,
        pred=pred,
        y_test=batches.y_test,
        model_state=model_state,
    )
    _maybe_write_postprocess_debug(
        context=context,
        model_state=model_state,
    )
    score = float(
        context.resources.hooks.score(
            y_true=batches.y_test,
            pred=pred,
            score_spec=context.params.score_spec,
        )
    )
    elapsed = time.perf_counter() - started
    return SplitResult(
        score=score,
        train_size=int(context.split.train_idx.size),
        test_size=int(context.split.test_idx.size),
        n_features_kept=int(batches.cleaning.feature_idx.size),
        elapsed_s=elapsed,
    )


def _predict_validation_horizon(
    *,
    hooks: SimulationHooks,
    batches: SplitBatches,
    config: Mapping[str, Any],
    num_samples: int,
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if _training_method(config) != "online_filtering":
        return hooks.predict(
            X_pred=batches.X_test,
            X_pred_global=batches.X_test_global,
            state=model_state,
            config=config,
            num_samples=num_samples,
        )
    return _predict_validation_sequential(
        hooks=hooks,
        batches=batches,
        config=config,
        num_samples=num_samples,
        model_state=model_state,
    )


def _training_method(config: Mapping[str, Any]) -> str:
    training = config.get("training")
    if not isinstance(training, Mapping):
        return "tbptt"
    method = str(training.get("method", "tbptt")).strip().lower()
    if method in {"tbptt", "online_filtering"}:
        return method
    raise SimulationError("training.method must be tbptt or online_filtering")


def _predict_validation_sequential(
    *,
    hooks: SimulationHooks,
    batches: SplitBatches,
    config: Mapping[str, Any],
    num_samples: int,
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    state = model_state
    predictions: list[Mapping[str, Any]] = []
    for step in range(int(batches.y_test.shape[0])):
        step_inputs = _online_step_inputs(batches, step)
        prediction = hooks.predict(
            X_pred=step_inputs.X_step,
            X_pred_global=step_inputs.X_step_global,
            state=state,
            config=config,
            num_samples=num_samples,
        )
        predictions.append(prediction)
        state = _maybe_update_online_filter_state(
            hooks=hooks,
            state=state,
            config=config,
            step_inputs=step_inputs,
        )
    return _merge_prediction_steps(predictions)


def _slice_time_step(values: torch.Tensor, step: int) -> torch.Tensor:
    return values[step : step + 1]


def _slice_optional_time_step(
    values: torch.Tensor | None, step: int
) -> torch.Tensor | None:
    if values is None:
        return None
    return values[step : step + 1]


def _online_step_inputs(
    batches: SplitBatches, step: int
) -> _OnlineStepInputs:
    return _OnlineStepInputs(
        X_step=_slice_time_step(batches.X_test, step),
        X_step_global=_slice_optional_time_step(batches.X_test_global, step),
        y_step=_slice_time_step(batches.y_test, step),
    )


def _maybe_update_online_filter_state(
    *,
    hooks: SimulationHooks,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    step_inputs: _OnlineStepInputs,
) -> Mapping[str, Any]:
    if not torch.isfinite(step_inputs.y_step).any():
        return state
    return hooks.fit_model(
        X_train=step_inputs.X_step,
        X_train_global=step_inputs.X_step_global,
        y_train=step_inputs.y_step,
        config=config,
        init_state=state,
    )


def _merge_prediction_steps(
    predictions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not predictions:
        raise SimulationError("Sequential validation requires predictions")
    samples = torch.cat(
        [_require_prediction_tensor(pred, key="samples") for pred in predictions],
        dim=1,
    )
    mean = torch.cat(
        [_require_prediction_tensor(pred, key="mean") for pred in predictions],
        dim=0,
    )
    merged: dict[str, Any] = {
        "samples": samples,
        "mean": mean,
    }
    covariance_steps = _collect_covariance_steps(predictions)
    if covariance_steps is not None:
        merged["covariance"] = covariance_steps
    return merged


def _collect_covariance_steps(
    predictions: Sequence[Mapping[str, Any]],
) -> torch.Tensor | None:
    covariances: list[torch.Tensor] = []
    for pred in predictions:
        covariance = pred.get("covariance")
        if covariance is None:
            return None
        if not isinstance(covariance, torch.Tensor):
            raise SimulationError(
                "Sequential prediction covariance must be a tensor"
            )
        covariances.append(_normalize_covariance_step(covariance))
    return torch.cat(covariances, dim=0)


def _normalize_covariance_step(covariance: torch.Tensor) -> torch.Tensor:
    if covariance.ndim == 2:
        return covariance.unsqueeze(0)
    if covariance.ndim != 3 or covariance.shape[0] != 1:
        raise SimulationError(
            "Sequential prediction covariance must be [A, A] or [1, A, A]"
        )
    return covariance


def _require_prediction_tensor(
    pred: Mapping[str, Any], *, key: str
) -> torch.Tensor:
    value = pred.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError(
            f"Sequential prediction output must include tensor {key!r}"
        )
    return value


def _prepare_split_batches(
    context: SplitContext,
) -> SplitBatches | None:
    cleaning = fit_feature_cleaning(
        X=context.data.X,
        M=context.data.M,
        train_idx=context.split.train_idx,
        spec=context.params.preprocess_spec,
        frozen_feature_idx=None,
    )
    if cleaning.feature_idx.size == 0:
        return None
    scaler = fit_robust_scaler(
        X=context.data.X,
        M=context.data.M,
        train_idx=context.split.train_idx,
        cleaning=cleaning,
        spec=context.params.preprocess_spec,
    )
    state = TransformState(
        cleaning=cleaning,
        scaler=scaler,
        spec=context.params.preprocess_spec,
    )
    X_train = transform_X(
        context.data.X,
        context.data.M,
        context.split.train_idx,
        state,
        validate=True,
    )
    X_test = transform_X(
        context.data.X,
        context.data.M,
        context.split.test_idx,
        state,
    )
    X_train_global, X_test_global = _prepare_global_batches(context)
    return SplitBatches(
        cleaning=cleaning,
        X_train=X_train,
        X_train_global=X_train_global,
        y_train=_select_targets(context.data.y, context.split.train_idx),
        X_test=X_test,
        X_test_global=X_test_global,
        y_test=_select_targets(context.data.y, context.split.test_idx),
    )


def _select_targets(y: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    safe_indices = np.array(indices, dtype=np.int64, copy=True)
    return y[torch.tensor(safe_indices, dtype=torch.long, device=y.device)]


def _prepare_global_batches(
    context: SplitContext,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if context.data.X_global is None or context.data.M_global is None:
        return None, None
    global_inputs = GlobalBlockInputs(
        X=context.data.X_global,
        M=context.data.M_global,
        feature_names=context.data.global_feature_names,
    )
    cleaning = fit_global_feature_cleaning(
        inputs=global_inputs,
        train_idx=context.split.train_idx,
        spec=context.params.preprocess_spec,
        frozen_feature_idx=None,
    )
    if cleaning.feature_idx.size == 0:
        return None, None
    scaler = fit_global_robust_scaler(
        inputs=global_inputs,
        train_idx=context.split.train_idx,
        cleaning=cleaning,
        spec=context.params.preprocess_spec,
    )
    state = TransformState(
        cleaning=cleaning,
        scaler=scaler,
        spec=context.params.preprocess_spec,
    )
    X_train = transform_global_X(
        global_inputs,
        context.split.train_idx,
        state,
        validate=True,
    )
    X_test = transform_global_X(
        global_inputs,
        context.split.test_idx,
        state,
    )
    return X_train, X_test


def aggregate_scores(
    scores: Sequence[float], *, method: str, penalty: float
) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return float("-inf")
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    if method == "mean_minus_std":
        return float(values.mean() - penalty * values.std())
    raise ValueError(f"Unknown aggregation method '{method}'")


def _split_config(
    config: Mapping[str, Any],
    outer_k: int,
    split_id: int,
    asset_names: Sequence[str],
) -> Mapping[str, Any]:
    run_context = config.get("run_context")
    if isinstance(run_context, Mapping):
        merged_context = dict(run_context)
    else:
        merged_context = {}
    merged_context["outer_k"] = outer_k
    merged_context["split_id"] = split_id
    merged_context["asset_names"] = list(asset_names)
    return {
        **config,
        "run_context": merged_context,
    }


def _maybe_prepare_energy_score_pred(
    *,
    pred: Mapping[str, Any],
    y_train: torch.Tensor,
    score_spec: Mapping[str, Any],
) -> Mapping[str, Any]:
    metric = str(score_spec.get("metric_name", "")).strip().lower()
    if metric != "energy_score":
        return pred
    transform = _build_energy_score_transform(
        y_train=y_train, score_spec=score_spec
    )
    enriched = dict(pred)
    enriched["energy_score"] = transform
    return enriched


def _maybe_write_postprocess_inputs(
    *,
    context: SplitContext,
    pred: Mapping[str, Any],
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    model_state: Mapping[str, Any],
) -> None:
    if not context.resources.model_selection.enable:
        return
    if context.candidate_id < 0:
        raise SimulationError(
            "Postprocess requires candidate_id in run_context"
        )
    samples = _require_samples_tensor(pred)
    transform = _resolve_energy_score_transform(
        pred=pred, y_train=y_train, score_spec=context.params.score_spec
    )
    if y_test.ndim != 2:
        raise SimulationError("Postprocess y_test must be [T, A]")
    if samples.shape[1] != y_test.shape[0]:
        raise SimulationError("Postprocess samples and y_test must align on T")
    if samples.shape[2] != y_test.shape[1]:
        raise SimulationError("Postprocess samples and y_test must align on A")
    z_true = y_test / transform["scale"]
    z_samples = samples / transform["scale"]
    target_transform = _resolve_target_space_transform(
        model_state=model_state,
        mad_scale=transform["scale"],
    )
    test_groups = context.resources.group_by_index[context.split.test_idx]
    if np.any(test_groups < 0):
        raise SimulationError("Postprocess test groups contain invalid ids")
    payload = {
        "z_true": z_true.detach().cpu(),
        "z_samples": z_samples.detach().cpu(),
        "scale": transform["scale"].detach().cpu(),
        "whitener": transform["whitener"].detach().cpu(),
        "target_space_transform": target_transform.to_payload(),
        "test_idx": context.split.test_idx.tolist(),
        "test_groups": test_groups.tolist(),
    }
    context.resources.artifacts.write_postprocess_candidate_split(
        outer_k=context.params.outer_k,
        candidate_id=context.candidate_id,
        split_id=context.split_id,
        payload=payload,
    )


def _maybe_write_postprocess_diagnostics(
    *,
    context: SplitContext,
    pred: Mapping[str, Any],
    y_test: torch.Tensor,
    model_state: Mapping[str, Any],
) -> None:
    if not context.resources.model_selection.enable:
        return
    if context.candidate_id < 0:
        raise SimulationError(
            "Diagnostics require candidate_id in run_context"
        )
    payload = _build_diagnostics_payload(
        context=context,
        pred=pred,
        y_true=y_test,
        model_state=model_state,
    )
    context.resources.artifacts.write_postprocess_diagnostics(
        outer_k=context.params.outer_k,
        candidate_id=context.candidate_id,
        split_id=context.split_id,
        payload=payload,
    )


def _maybe_write_postprocess_debug(
    *,
    context: SplitContext,
    model_state: Mapping[str, Any],
) -> None:
    if not context.resources.model_selection.enable:
        return
    if context.candidate_id < 0:
        raise SimulationError("Debug payload requires candidate_id in run_context")
    context.resources.artifacts.write_postprocess_debug(
        outer_k=context.params.outer_k,
        candidate_id=context.candidate_id,
        split_id=context.split_id,
        payload=_build_debug_payload(context=context, model_state=model_state),
    )


def _require_samples_tensor(pred: Mapping[str, Any]) -> torch.Tensor:
    samples = pred.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise SimulationError("Postprocess requires pred['samples'] tensor")
    if samples.ndim == 2:
        return samples.unsqueeze(0)
    if samples.ndim != 3:
        raise SimulationError("Postprocess samples must be [S, T, A]")
    return samples


def _build_diagnostics_payload(
    *,
    context: SplitContext,
    pred: Mapping[str, Any],
    y_true: torch.Tensor,
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    samples = _require_samples_tensor(pred)
    payload: dict[str, Any] = {
        "outer_k": context.params.outer_k,
        "candidate_id": context.candidate_id,
        "split_id": context.split_id,
        "posterior": _posterior_summary_payload(model_state),
        "state": _state_summary_payload(model_state),
        "training": _training_diagnostics_payload(model_state),
        "predictive": {
            "samples": _sample_stats(samples),
            "y_true": _value_stats(y_true),
        },
    }
    return payload


def _posterior_summary_payload(
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    summary = model_state.get("posterior_summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return {}


def _state_summary_payload(
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "filtering_state": _filtering_state_summary(model_state),
        "structural": _structural_state_summary(model_state),
    }


def _filtering_state_summary(
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    state = model_state.get("filtering_state")
    if not isinstance(state, Mapping):
        return {}
    payload: dict[str, Any] = {"steps_seen": state.get("steps_seen")}
    for key in ("h_loc", "h_scale"):
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = _value_stats(value.detach().reshape(-1))
    return payload


def _structural_state_summary(
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    structural = model_state.get("structural_posterior_means")
    if not isinstance(structural, Mapping):
        return {}
    payload: dict[str, Any] = {}
    for key, value in structural.items():
        if isinstance(value, torch.Tensor):
            payload[str(key)] = _value_stats(value.detach().reshape(-1))
    return payload


def _training_diagnostics_payload(
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    section = model_state.get("training_diagnostics")
    if not isinstance(section, Mapping):
        return {}
    history = section.get("svi_loss_history")
    if not isinstance(history, Sequence):
        return {}
    cleaned_history: list[float] = []
    for value in history:
        try:
            cleaned_history.append(float(value))
        except (TypeError, ValueError):
            continue
    if not cleaned_history:
        return {}
    return {"svi_loss_history": cleaned_history}


def _build_debug_payload(
    *,
    context: SplitContext,
    model_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "outer_k": context.params.outer_k,
        "candidate_id": context.candidate_id,
        "split_id": context.split_id,
        "filtering_state": model_state.get("filtering_state"),
        "structural_posterior_means": model_state.get(
            "structural_posterior_means"
        ),
        "posterior_summary": model_state.get("posterior_summary"),
        "training_diagnostics": model_state.get("training_diagnostics"),
    }


def _sample_stats(samples: torch.Tensor) -> Mapping[str, float]:
    flattened = samples.detach().reshape(-1)
    return _value_stats(
        flattened,
        quantiles=(0.01, 0.05, 0.50, 0.95, 0.99),
    )


def _value_stats(
    values: torch.Tensor,
    *,
    quantiles: Sequence[float] | None = None,
) -> Mapping[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return _nan_stats(quantiles)
    mean = float(finite.mean().item())
    std = float(finite.std(unbiased=False).item())
    min_val = float(finite.min().item())
    max_val = float(finite.max().item())
    stats: dict[str, float] = {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
    }
    if quantiles:
        stats.update(_quantile_stats(finite, quantiles))
    return stats


def _nan_stats(
    quantiles: Sequence[float] | None,
) -> Mapping[str, float]:
    stats: dict[str, float] = {
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }
    if quantiles:
        stats.update({f"q{int(q * 100):02d}": float("nan") for q in quantiles})
    return stats


def _quantile_stats(
    values: torch.Tensor, quantiles: Sequence[float]
) -> Mapping[str, float]:
    sorted_q = sorted(float(q) for q in quantiles)
    data = values.detach().cpu().numpy()
    result = np.quantile(data, sorted_q)
    return {
        f"q{int(q * 100):02d}": float(value)
        for q, value in zip(sorted_q, result)
    }


def _resolve_energy_score_transform(
    *,
    pred: Mapping[str, Any],
    y_train: torch.Tensor,
    score_spec: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    existing = pred.get("energy_score")
    if isinstance(existing, Mapping):
        scale = existing.get("scale")
        whitener = existing.get("whitener")
        if isinstance(scale, torch.Tensor) and isinstance(
            whitener, torch.Tensor
        ):
            return {"scale": scale, "whitener": whitener}
    return _build_energy_score_transform(
        y_train=y_train, score_spec=score_spec
    )


def _build_energy_score_transform(
    *, y_train: torch.Tensor, score_spec: Mapping[str, Any]
) -> Mapping[str, torch.Tensor]:
    if y_train.ndim != 2:
        raise SimulationError("Energy score requires y_train [T, A]")
    if not torch.isfinite(y_train).all():
        raise SimulationError(
            "Energy score training data must be fully observed"
        )
    n_eff = int(y_train.shape[0])
    if n_eff <= 1:
        raise SimulationError("Energy score requires at least 2 rows")
    scale = _mad_scale(y_train, score_spec)
    z_train = y_train / scale
    z_centered = z_train - z_train.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / float(n_eff - 1)
    shrinkage = _energy_score_shrinkage(n_eff)
    diag = torch.diag(torch.diag(cov))
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    var_floor = float(score_spec.get("var_floor", 0.0))
    if var_floor < 0.0:
        raise SimulationError("Energy score var_floor must be >= 0")
    if var_floor > 0.0:
        cov_shrunk = cov_shrunk + var_floor * torch.eye(
            cov_shrunk.shape[0],
            device=cov_shrunk.device,
            dtype=cov_shrunk.dtype,
        )
    eps = float(score_spec.get("eps", 1e-5))
    cov_shrunk = cov_shrunk + eps * torch.eye(
        cov_shrunk.shape[0],
        device=cov_shrunk.device,
        dtype=cov_shrunk.dtype,
    )
    whitener = _inverse_cholesky(cov_shrunk)
    return {"scale": scale, "whitener": whitener}


def _resolve_target_space_transform(
    *, model_state: Mapping[str, Any], mad_scale: torch.Tensor
) -> TargetSpaceTransform:
    target_norm = model_state.get("target_norm")
    if not isinstance(target_norm, Mapping):
        return TargetSpaceTransform(
            model_center=torch.tensor(0.0, device=mad_scale.device, dtype=mad_scale.dtype),
            model_scale=torch.tensor(1.0, device=mad_scale.device, dtype=mad_scale.dtype),
            mad_scale=mad_scale.to(dtype=mad_scale.dtype),
        )
    center = target_norm.get("center")
    scale = target_norm.get("scale")
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return TargetSpaceTransform(
            model_center=torch.tensor(0.0, device=mad_scale.device, dtype=mad_scale.dtype),
            model_scale=torch.tensor(1.0, device=mad_scale.device, dtype=mad_scale.dtype),
            mad_scale=mad_scale.to(dtype=mad_scale.dtype),
        )
    return TargetSpaceTransform(
        model_center=center.to(device=mad_scale.device, dtype=mad_scale.dtype),
        model_scale=scale.to(device=mad_scale.device, dtype=mad_scale.dtype),
        mad_scale=mad_scale.to(device=mad_scale.device, dtype=mad_scale.dtype),
    )


def _mad_scale(
    y_train: torch.Tensor, score_spec: Mapping[str, Any]
) -> torch.Tensor:
    median = y_train.median(dim=0).values
    mad = (y_train - median).abs().median(dim=0).values
    mad_eps = float(score_spec.get("mad_eps", 1e-12))
    return mad + mad_eps


def _energy_score_shrinkage(n_eff: int) -> float:
    if n_eff >= 300:
        return 0.075
    if n_eff >= 150:
        return 0.15
    return 0.30


def _inverse_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    try:
        chol = cholesky(matrix)  # pylint: disable=not-callable
        identity = torch.eye(
            chol.shape[0], device=chol.device, dtype=chol.dtype
        )
        return solve_triangular(  # pylint: disable=not-callable
            chol, identity, upper=False
        )
    except RuntimeError as exc:
        raise SimulationError(
            "Energy score covariance is not positive definite"
        ) from exc
