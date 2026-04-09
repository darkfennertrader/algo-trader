from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from algo_trader.domain.simulation import (
    AllocationRequest,
    FeatureCleaningState,
    OuterFold,
    PreprocessSpec,
)
from .feature_panel_data import FeaturePanelData
from .hooks import SimulationHooks
from .outer_prediction import (
    WeeklyPredictionResult,
    build_outer_prediction_context,
    evaluate_outer_predictions,
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
class WeeklyPortfolioResult:
    timestamp: Any
    weights: torch.Tensor
    gross_return: torch.Tensor
    net_return: torch.Tensor
    cost: torch.Tensor
    turnover: torch.Tensor


@dataclass(frozen=True)
class PortfolioEvaluationRuntime:
    hooks: SimulationHooks
    target_device: torch.device


def evaluate_outer_walk_forward(
    *,
    context: OuterEvaluationContext,
    best_config: Mapping[str, Any],
    hooks: SimulationHooks,
) -> tuple[Mapping[str, Any], FeatureCleaningState | None]:
    portfolio_specs = _build_portfolio_specs(
        context.portfolios, int(context.panel.X.shape[1])
    )
    weekly_predictions, cleaning_outer = evaluate_outer_predictions(
        context=build_outer_prediction_context(
            owner=context,
            outer_fold=context.outer_fold,
            week_progress=context.week_progress,
        ),
        best_config=best_config,
        hooks=hooks,
    )
    if cleaning_outer is None or cleaning_outer.feature_idx.size == 0:
        return _empty_result(context.outer_fold), cleaning_outer
    portfolio_results = _evaluate_prediction_series(
        portfolio_specs=portfolio_specs,
        weekly_predictions=weekly_predictions,
        runtime=PortfolioEvaluationRuntime(
            hooks=hooks,
            target_device=context.y.device,
        ),
    )

    return (
        _build_result(
            outer_fold=context.outer_fold,
            cleaning_outer=cleaning_outer,
            portfolio_results=portfolio_results,
        ),
        cleaning_outer,
    )


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


def _evaluate_prediction_series(
    *,
    portfolio_specs: Sequence[PortfolioSpec],
    weekly_predictions: Sequence[WeeklyPredictionResult],
    runtime: PortfolioEvaluationRuntime,
) -> dict[str, dict[str, list[Any]]]:
    results = _initialize_portfolio_results(portfolio_specs)
    previous_weights: dict[str, torch.Tensor | None] = {
        portfolio.name: None for portfolio in portfolio_specs
    }
    for weekly_prediction in weekly_predictions:
        week_results = _evaluate_prediction_week(
            portfolio_specs=portfolio_specs,
            weekly_prediction=weekly_prediction,
            previous_weights=previous_weights,
            runtime=runtime,
        )
        _append_week_results(
            aggregate=results,
            week_results=week_results,
            previous_weights=previous_weights,
        )
    return results


def _evaluate_prediction_week(
    *,
    portfolio_specs: Sequence[PortfolioSpec],
    weekly_prediction: WeeklyPredictionResult,
    previous_weights: Mapping[str, torch.Tensor | None],
    runtime: PortfolioEvaluationRuntime,
) -> dict[str, WeeklyPortfolioResult]:
    results: dict[str, WeeklyPortfolioResult] = {}
    for portfolio in portfolio_specs:
        results[portfolio.name] = _evaluate_portfolio_week(
            portfolio=portfolio,
            weekly_prediction=weekly_prediction,
            previous_weights=previous_weights[portfolio.name],
            runtime=runtime,
        )
    return results


def _evaluate_portfolio_week(
    *,
    portfolio: PortfolioSpec,
    weekly_prediction: WeeklyPredictionResult,
    previous_weights: torch.Tensor | None,
    runtime: PortfolioEvaluationRuntime,
) -> WeeklyPortfolioResult:
    request = AllocationRequest(
        prediction=weekly_prediction.prediction,
        allocation_spec=portfolio.allocation,
        previous_weights=previous_weights,
    )
    allocation = runtime.hooks.allocate(request=request)
    weights = _resolve_weights_device(
        weights=allocation.weights,
        target_device=runtime.target_device,
    )
    gross = _gross_return(
        weights=weights,
        y_t=weekly_prediction.realized_returns,
    )
    pnl = runtime.hooks.compute_pnl(
        w=weights,
        y_t=weekly_prediction.realized_returns,
        w_prev=previous_weights,
        cost_spec=portfolio.cost,
    )
    return WeeklyPortfolioResult(
        timestamp=weekly_prediction.prediction.rebalance_timestamp,
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
