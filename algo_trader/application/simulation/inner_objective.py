from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import logging
import time

import numpy as np
import torch

from algo_trader.domain.simulation import (
    CPCVSplit,
    FeatureCleaningState,
    PreprocessSpec,
)
from .hooks import SimulationHooks
from .preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)


@dataclass(frozen=True)
class InnerObjectiveData:
    X: torch.Tensor
    M: torch.Tensor
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
    inner_splits: list[CPCVSplit]
    params: InnerObjectiveParams


logger = logging.getLogger(__name__)


def make_inner_objective(
    *,
    context: InnerObjectiveContext,
    hooks: SimulationHooks,
) -> Callable[[Mapping[str, Any]], float]:
    def objective(config: Mapping[str, Any]) -> float:
        fold_scores: list[float] = []
        for split_id, split in enumerate(context.inner_splits):
            result = _evaluate_split(
                SplitContext(
                    data=context.data,
                    params=context.params,
                    split=split,
                    split_id=split_id,
                    hooks=hooks,
                    config=config,
                )
            )
            if result is None:
                continue
            fold_scores.append(result.score)
            logger.info(
                "event=complete boundary=simulation.inner_split context=%s",
                {
                    "outer_k": context.params.outer_k,
                    "split_id": split_id,
                    "train_size": result.train_size,
                    "test_size": result.test_size,
                    "score": result.score,
                    "elapsed_s": round(result.elapsed_s, 4),
                    "n_features_kept": result.n_features_kept,
                },
            )

        if not fold_scores:
            return float("-inf")
        return _aggregate_scores(
            fold_scores,
            method=context.params.aggregate,
            penalty=context.params.aggregate_lambda,
        )

    return objective


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
    hooks: SimulationHooks
    config: Mapping[str, Any]


@dataclass(frozen=True)
class SplitBatches:
    cleaning: FeatureCleaningState
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def _evaluate_split(context: SplitContext) -> SplitResult | None:
    started = time.perf_counter()
    split_config = _split_config(
        context.config, context.params.outer_k, context.split_id
    )
    batches = _prepare_split_batches(context)
    if batches is None:
        return None
    model_state = context.hooks.fit_model(
        X_train=batches.X_train,
        y_train=batches.y_train,
        config=split_config,
        init_state=None,
    )
    pred = context.hooks.predict(
        X_pred=batches.X_test,
        state=model_state,
        config=split_config,
        num_samples=context.params.num_pp_samples,
    )
    score = float(
        context.hooks.score(
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
    )
    X_test = transform_X(
        context.data.X,
        context.data.M,
        context.split.test_idx,
        state,
    )
    return SplitBatches(
        cleaning=cleaning,
        X_train=X_train,
        y_train=_select_targets(context.data.y, context.split.train_idx),
        X_test=X_test,
        y_test=_select_targets(context.data.y, context.split.test_idx),
    )


def _select_targets(y: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    return y[
        torch.as_tensor(indices, dtype=torch.long, device=y.device)
    ]


def _aggregate_scores(
    scores: Sequence[float], *, method: str, penalty: float
) -> float:
    values = np.asarray(scores, dtype=float)
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
) -> Mapping[str, Any]:
    return {
        **config,
        "run_context": {
            "outer_k": outer_k,
            "split_id": split_id,
        },
    }
