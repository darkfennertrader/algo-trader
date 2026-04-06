from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain.simulation import (
    AllocationRequest,
    FeatureCleaningState,
    OuterFold,
    PredictionPacket,
    PreprocessSpec,
)
from .feature_panel_data import (
    FeaturePanelData,
    prepare_global_feature_batches,
    with_run_context_updates,
)
from .hooks import SimulationHooks
from .prediction_handoff import build_prediction_packet
from .preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    allocation: Mapping[str, Any]
    cost: Mapping[str, Any]


@dataclass(frozen=True)
class OuterEvaluationContext:  # pylint: disable=too-many-instance-attributes
    panel: FeaturePanelData
    y: torch.Tensor
    timestamps: Sequence[Any]
    outer_fold: OuterFold
    preprocess_spec: PreprocessSpec
    num_pp_samples: int
    portfolios: tuple[PortfolioSpec, ...]
    assets: Sequence[str]
    execution_mode: str
    week_progress: Callable[[int, Any], None] | None = None


@dataclass(frozen=True)
class WeeklyLoopContext:
    test_weeks: np.ndarray
    base_train: np.ndarray
    eval_context: OuterEvaluationContext
    best_config: Mapping[str, Any]
    hooks: SimulationHooks
    portfolio_specs: tuple[PortfolioSpec, ...]
    cleaning_outer: FeatureCleaningState


@dataclass(frozen=True)
class PreparedOuterBatches:
    X_train: torch.Tensor
    X_train_global: torch.Tensor | None
    y_train: torch.Tensor
    X_pred: torch.Tensor
    X_pred_global: torch.Tensor | None


@dataclass(frozen=True)
class WeeklyPortfolioResult:
    timestamp: Any
    weights: torch.Tensor
    gross_return: torch.Tensor
    net_return: torch.Tensor
    cost: torch.Tensor
    turnover: torch.Tensor


def evaluate_outer_walk_forward(
    *,
    context: OuterEvaluationContext,
    best_config: Mapping[str, Any],
    hooks: SimulationHooks,
) -> tuple[Mapping[str, Any], FeatureCleaningState | None]:
    test_weeks = _sorted_indices(context.outer_fold.test_idx)
    base_train = _sorted_indices(context.outer_fold.train_idx)
    portfolio_specs = _build_portfolio_specs(
        context.portfolios, int(context.panel.X.shape[1])
    )

    cleaning_outer = fit_feature_cleaning(
        X=context.panel.X,
        M=context.panel.M,
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
        portfolio_specs=portfolio_specs,
        cleaning_outer=cleaning_outer,
    )
    portfolio_results = _run_weekly_loop(loop_context)

    return (
        _build_result(
            outer_fold=context.outer_fold,
            cleaning_outer=cleaning_outer,
            portfolio_results=portfolio_results,
        ),
        cleaning_outer,
    )


def _sorted_indices(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(indices, dtype=int))


def _build_portfolio_specs(
    portfolios: Sequence[PortfolioSpec], n_assets: int
) -> tuple[PortfolioSpec, ...]:
    resolved: list[PortfolioSpec] = []
    for portfolio in portfolios:
        alloc_spec = dict(portfolio.allocation)
        alloc_spec.setdefault("n_assets", n_assets)
        resolved.append(
            PortfolioSpec(
                name=portfolio.name,
                allocation=alloc_spec,
                cost=portfolio.cost,
            )
        )
    return tuple(resolved)


def _empty_result(outer_fold: OuterFold) -> Mapping[str, Any]:
    return {
        "outer_k_test": outer_fold.k_test,
        "f_clean_outer_size": 0,
        "portfolio_primary": "primary",
        "timestamps": [],
        "gross_returns": [],
        "net_returns": [],
        "costs": [],
        "turnover": [],
        "pnl": [],
        "weights": [],
        "portfolios": {},
        "notes": "No features survived outer-fold cleaning (1)(2).",
        "metrics": {},
    }


def _run_weekly_loop(
    context: WeeklyLoopContext,
) -> dict[str, dict[str, list[Any]]]:
    state: Mapping[str, Any] | None = None
    results = _initialize_portfolio_results(context.portfolio_specs)
    previous_weights: dict[str, torch.Tensor | None] = {
        portfolio.name: None for portfolio in context.portfolio_specs
    }

    for t in context.test_weeks:
        state, week_results = _evaluate_week(
            loop_context=context,
            current_t=int(t),
            state=state,
            previous_weights=previous_weights,
        )
        _append_week_results(
            aggregate=results,
            week_results=week_results,
            previous_weights=previous_weights,
        )
        _report_week_progress(context.eval_context, int(t))

    return results


def _evaluate_week(
    *,
    loop_context: WeeklyLoopContext,
    current_t: int,
    state: Mapping[str, Any] | None,
    previous_weights: Mapping[str, torch.Tensor | None],
) -> tuple[Mapping[str, Any] | None, dict[str, WeeklyPortfolioResult]]:
    config = _with_asset_names(
        loop_context.best_config,
        loop_context.eval_context.assets,
        loop_context.eval_context.execution_mode,
    )
    train_idx_t = _expanding_train(
        loop_context.base_train, loop_context.test_weeks, current_t
    )
    batches = _prepare_batches(
        eval_context=loop_context.eval_context,
        train_idx=train_idx_t,
        pred_t=current_t,
        cleaning_outer=loop_context.cleaning_outer,
    )

    state = loop_context.hooks.fit_model(
        X_train=batches.X_train,
        X_train_global=batches.X_train_global,
        y_train=batches.y_train,
        config=config,
        init_state=state,
    )
    week_results = _allocate_for_week(
        loop_context=loop_context,
        batches=batches,
        state=state,
        current_t=current_t,
        previous_weights=previous_weights,
    )
    return state, week_results


def _with_asset_names(
    config: Mapping[str, Any],
    asset_names: Sequence[str],
    execution_mode: str,
) -> Mapping[str, Any]:
    return with_run_context_updates(
        config,
        asset_names=list(asset_names),
        execution_mode=execution_mode,
    )


def _prepare_batches(
    *,
    eval_context: OuterEvaluationContext,
    train_idx: np.ndarray,
    pred_t: int,
    cleaning_outer: FeatureCleaningState,
) -> PreparedOuterBatches:
    scaler_t = fit_robust_scaler(
        X=eval_context.panel.X,
        M=eval_context.panel.M,
        train_idx=train_idx,
        cleaning=cleaning_outer,
        spec=eval_context.preprocess_spec,
    )
    state_t = TransformState(
        cleaning=cleaning_outer,
        scaler=scaler_t,
        spec=eval_context.preprocess_spec,
    )
    X_train_t = transform_X(
        eval_context.panel.X,
        eval_context.panel.M,
        train_idx,
        state_t,
        validate=True,
    )
    y_train_t = eval_context.y[
        torch.tensor(
            np.array(train_idx, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=eval_context.y.device,
        )
    ]
    X_pred_t = transform_X(
        eval_context.panel.X,
        eval_context.panel.M,
        np.array([pred_t], dtype=int),
        state_t,
    )
    X_train_global_t, X_pred_global_t = _prepare_global_batches(
        eval_context=eval_context,
        train_idx=train_idx,
        pred_t=pred_t,
    )
    return PreparedOuterBatches(
        X_train=X_train_t,
        X_train_global=X_train_global_t,
        y_train=y_train_t,
        X_pred=X_pred_t,
        X_pred_global=X_pred_global_t,
    )


def _allocate_for_week(
    *,
    loop_context: WeeklyLoopContext,
    batches: PreparedOuterBatches,
    state: Mapping[str, Any] | None,
    current_t: int,
    previous_weights: Mapping[str, torch.Tensor | None],
) -> dict[str, WeeklyPortfolioResult]:
    config = _with_asset_names(
        loop_context.best_config,
        loop_context.eval_context.assets,
        loop_context.eval_context.execution_mode,
    )
    prediction = _build_weekly_prediction(
        loop_context=loop_context,
        batches=batches,
        state=state,
        current_t=current_t,
        config=config,
    )
    y_t = loop_context.eval_context.y[current_t]
    results: dict[str, WeeklyPortfolioResult] = {}
    for portfolio in loop_context.portfolio_specs:
        results[portfolio.name] = _evaluate_portfolio_week(
            loop_context=loop_context,
            portfolio=portfolio,
            prediction=prediction,
            y_t=y_t,
            previous_weights=previous_weights[portfolio.name],
        )
    return results


def _build_weekly_prediction(
    *,
    loop_context: WeeklyLoopContext,
    batches: PreparedOuterBatches,
    state: Mapping[str, Any] | None,
    current_t: int,
    config: Mapping[str, Any],
) -> PredictionPacket:
    pred = loop_context.hooks.predict(
        X_pred=batches.X_pred,
        X_pred_global=batches.X_pred_global,
        state=state or {},
        config=config,
        num_samples=loop_context.eval_context.num_pp_samples,
    )
    return build_prediction_packet(
        pred=pred,
        asset_names=loop_context.eval_context.assets,
        rebalance_index=current_t,
        rebalance_timestamp=loop_context.eval_context.timestamps[current_t],
    )


def _evaluate_portfolio_week(
    *,
    loop_context: WeeklyLoopContext,
    portfolio: PortfolioSpec,
    prediction: PredictionPacket,
    y_t: torch.Tensor,
    previous_weights: torch.Tensor | None,
) -> WeeklyPortfolioResult:
    request = AllocationRequest(
        prediction=prediction,
        allocation_spec=portfolio.allocation,
        previous_weights=previous_weights,
    )
    allocation = loop_context.hooks.allocate(request=request)
    weights = _resolve_weights_device(
        weights=allocation.weights,
        target_device=loop_context.eval_context.y.device,
    )
    gross = _gross_return(weights=weights, y_t=y_t)
    pnl = loop_context.hooks.compute_pnl(
        w=weights,
        y_t=y_t,
        w_prev=previous_weights,
        cost_spec=portfolio.cost,
    )
    return WeeklyPortfolioResult(
        timestamp=prediction.rebalance_timestamp,
        weights=weights,
        gross_return=gross,
        net_return=pnl,
        cost=gross - pnl,
        turnover=_coerce_turnover(
            turnover=allocation.turnover,
            weights=weights,
            previous_weights=previous_weights,
        ),
    )


def _resolve_weights_device(
    *,
    weights: torch.Tensor,
    target_device: torch.device,
) -> torch.Tensor:
    if weights.device == target_device:
        return weights
    return weights.to(device=target_device)


def _initialize_portfolio_results(
    portfolio_specs: Sequence[PortfolioSpec],
) -> dict[str, dict[str, list[Any]]]:
    return {
        portfolio.name: {
            "timestamps": [],
            "gross_returns": [],
            "net_returns": [],
            "costs": [],
            "turnover": [],
            "pnl": [],
            "weights": [],
        }
        for portfolio in portfolio_specs
    }


def _append_week_results(
    *,
    aggregate: dict[str, dict[str, list[Any]]],
    week_results: Mapping[str, WeeklyPortfolioResult],
    previous_weights: dict[str, torch.Tensor | None],
) -> None:
    for name, result in week_results.items():
        aggregate[name]["timestamps"].append(result.timestamp)
        aggregate[name]["gross_returns"].append(
            float(result.gross_return.detach().cpu())
        )
        aggregate[name]["net_returns"].append(
            float(result.net_return.detach().cpu())
        )
        aggregate[name]["costs"].append(float(result.cost.detach().cpu()))
        aggregate[name]["turnover"].append(
            float(result.turnover.detach().cpu())
        )
        aggregate[name]["pnl"].append(float(result.net_return.detach().cpu()))
        aggregate[name]["weights"].append(
            result.weights.detach().cpu().numpy()
        )
        previous_weights[name] = result.weights


def _report_week_progress(
    eval_context: OuterEvaluationContext, current_t: int
) -> None:
    if eval_context.week_progress is None:
        return
    eval_context.week_progress(
        eval_context.outer_fold.k_test,
        eval_context.timestamps[current_t],
    )


def _prepare_global_batches(
    *,
    eval_context: OuterEvaluationContext,
    train_idx: np.ndarray,
    pred_t: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return prepare_global_feature_batches(
        panel=eval_context.panel,
        train_idx=train_idx,
        test_idx=np.array([pred_t], dtype=int),
        preprocess_spec=eval_context.preprocess_spec,
        validate_train=True,
    )


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
    portfolio_results: Mapping[str, Mapping[str, list[Any]]],
) -> Mapping[str, Any]:
    primary_name = next(iter(portfolio_results))
    primary = portfolio_results[primary_name]
    return {
        "outer_k_test": outer_fold.k_test,
        "f_clean_outer_size": int(cleaning_outer.feature_idx.size),
        "portfolio_primary": primary_name,
        "timestamps": primary["timestamps"],
        "gross_returns": primary["gross_returns"],
        "net_returns": primary["net_returns"],
        "costs": primary["costs"],
        "turnover": primary["turnover"],
        "pnl": primary["pnl"],
        "weights": primary["weights"],
        "portfolios": portfolio_results,
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


def _gross_return(*, weights: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return (weights * y_t).sum()


def _coerce_turnover(
    *,
    turnover: torch.Tensor | None,
    weights: torch.Tensor,
    previous_weights: torch.Tensor | None,
) -> torch.Tensor:
    if turnover is not None:
        return turnover
    if previous_weights is None:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    return torch.abs(weights - previous_weights).sum()
