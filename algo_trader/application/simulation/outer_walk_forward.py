from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain.simulation import FeatureCleaningState, OuterFold, PreprocessSpec
from .hooks import SimulationHooks
from .preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)


@dataclass(frozen=True)
class PortfolioSpec:
    allocation: Mapping[str, Any]
    cost: Mapping[str, Any]


@dataclass(frozen=True)
class OuterEvaluationContext:
    X: torch.Tensor
    M: torch.Tensor
    y: torch.Tensor
    outer_fold: OuterFold
    preprocess_spec: PreprocessSpec
    num_pp_samples: int
    portfolio: PortfolioSpec


@dataclass(frozen=True)
class WeeklyLoopContext:
    test_weeks: np.ndarray
    base_train: np.ndarray
    eval_context: OuterEvaluationContext
    best_config: Mapping[str, Any]
    hooks: SimulationHooks
    alloc_spec: Mapping[str, Any]
    cleaning_outer: FeatureCleaningState


def evaluate_outer_walk_forward(
    *,
    context: OuterEvaluationContext,
    best_config: Mapping[str, Any],
    hooks: SimulationHooks,
) -> tuple[Mapping[str, Any], FeatureCleaningState | None]:
    test_weeks = _sorted_indices(context.outer_fold.test_idx)
    base_train = _sorted_indices(context.outer_fold.train_idx)
    alloc_spec = _build_alloc_spec(context.portfolio, int(context.X.shape[1]))

    cleaning_outer = fit_feature_cleaning(
        X=context.X,
        M=context.M,
        train_idx=base_train,
        spec=context.preprocess_spec,
        frozen_feature_idx=None,
    )

    if cleaning_outer.feature_idx.size == 0:
        return _empty_result(context.outer_fold), cleaning_outer

    loop_context = WeeklyLoopContext(
        test_weeks=test_weeks,
        base_train=base_train,
        eval_context=context,
        best_config=best_config,
        hooks=hooks,
        alloc_spec=alloc_spec,
        cleaning_outer=cleaning_outer,
    )
    pnl, weights = _run_weekly_loop(loop_context)

    return (
        _build_result(
            outer_fold=context.outer_fold,
            cleaning_outer=cleaning_outer,
            pnl=pnl,
            weights=weights,
        ),
        cleaning_outer,
    )


def _sorted_indices(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(indices, dtype=int))


def _build_alloc_spec(portfolio: PortfolioSpec, n_assets: int) -> dict[str, Any]:
    alloc_spec = dict(portfolio.allocation)
    alloc_spec.setdefault("n_assets", n_assets)
    return alloc_spec


def _empty_result(outer_fold: OuterFold) -> Mapping[str, Any]:
    return {
        "outer_k_test": outer_fold.k_test,
        "f_clean_outer_size": 0,
        "pnl": [],
        "notes": "No features survived outer-fold cleaning (1)(2).",
        "metrics": {},
    }


def _run_weekly_loop(
    context: WeeklyLoopContext,
) -> tuple[list[float], list[np.ndarray]]:
    state: Mapping[str, Any] | None = None
    pnl: list[float] = []
    weights: list[np.ndarray] = []
    w_prev: torch.Tensor | None = None

    for t in context.test_weeks:
        state, w_prev, pnl_t = _evaluate_week(
            loop_context=context,
            current_t=int(t),
            state=state,
            w_prev=w_prev,
        )
        pnl.append(float(pnl_t.detach().cpu()))
        weights.append(w_prev.detach().cpu().numpy())

    return pnl, weights


def _evaluate_week(
    *,
    loop_context: WeeklyLoopContext,
    current_t: int,
    state: Mapping[str, Any] | None,
    w_prev: torch.Tensor | None,
) -> tuple[Mapping[str, Any] | None, torch.Tensor, torch.Tensor]:
    train_idx_t = _expanding_train(
        loop_context.base_train, loop_context.test_weeks, current_t
    )
    X_train_t, y_train_t, X_pred_t = _prepare_batches(
        eval_context=loop_context.eval_context,
        train_idx=train_idx_t,
        pred_t=current_t,
        cleaning_outer=loop_context.cleaning_outer,
    )

    state = loop_context.hooks.fit_model(
        X_train=X_train_t,
        y_train=y_train_t,
        config=loop_context.best_config,
        init_state=state,
    )

    pred = loop_context.hooks.predict(
        X_pred=X_pred_t,
        state=state,
        config=loop_context.best_config,
        num_samples=loop_context.eval_context.num_pp_samples,
    )

    w = loop_context.hooks.allocate(pred=pred, alloc_spec=loop_context.alloc_spec)
    if w.device != loop_context.eval_context.y.device:
        w = w.to(device=loop_context.eval_context.y.device)

    y_t = loop_context.eval_context.y[current_t]
    pnl_t = loop_context.hooks.compute_pnl(
        w=w, y_t=y_t, w_prev=w_prev, cost_spec=loop_context.eval_context.portfolio.cost
    )

    return state, w, pnl_t


def _prepare_batches(
    *,
    eval_context: OuterEvaluationContext,
    train_idx: np.ndarray,
    pred_t: int,
    cleaning_outer: FeatureCleaningState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scaler_t = fit_robust_scaler(
        X=eval_context.X,
        M=eval_context.M,
        train_idx=train_idx,
        cleaning=cleaning_outer,
        spec=eval_context.preprocess_spec,
    )
    state_t = TransformState(
        cleaning=cleaning_outer,
        scaler=scaler_t,
        spec=eval_context.preprocess_spec,
    )
    X_train_t = transform_X(eval_context.X, eval_context.M, train_idx, state_t)
    y_train_t = eval_context.y[
        torch.as_tensor(train_idx, dtype=torch.long, device=eval_context.y.device)
    ]
    X_pred_t = transform_X(
        eval_context.X,
        eval_context.M,
        np.array([pred_t], dtype=int),
        state_t,
    )
    return X_train_t, y_train_t, X_pred_t


def _expanding_train(
    base_train: np.ndarray,
    test_weeks: np.ndarray,
    current_t: int,
) -> np.ndarray:
    realized_test = test_weeks[test_weeks < current_t]
    return np.unique(np.concatenate([base_train, realized_test]))


def _build_result(
    *,
    outer_fold: OuterFold,
    cleaning_outer: FeatureCleaningState,
    pnl: list[float],
    weights: list[np.ndarray],
) -> Mapping[str, Any]:
    return {
        "outer_k_test": outer_fold.k_test,
        "f_clean_outer_size": int(cleaning_outer.feature_idx.size),
        "pnl": pnl,
        "weights": weights,
        "cleaning_diagnostics": {
            "dropped_low_usable": cleaning_outer.dropped_low_usable,
            "dropped_low_var": cleaning_outer.dropped_low_var,
            "dropped_duplicates": cleaning_outer.dropped_duplicates,
            "n_duplicate_pairs": len(cleaning_outer.duplicate_pairs),
        },
        "metrics": {
            "sharpe": None,
            "max_drawdown": None,
        },
    }
