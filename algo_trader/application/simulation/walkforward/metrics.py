from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, TypedDict

import numpy as np
import pandas as pd

from algo_trader.domain import SimulationError

from ..data_source_metadata import load_data_source_metadata
from ..io_utils import write_json_file
from .pathing import walkforward_dir

_INITIAL_CAPITAL = 100.0
_PERIODS_PER_YEAR = 52.0
_RISK_FREE_RATE = 0.0
_DOWNSIDE_TARGET = 0.0
_SUPPORTED_RETURN_TYPES = {"log", "simple"}
_SUPPORTED_RETURN_FREQUENCY = "weekly"


class PortfolioMetrics(TypedDict):
    portfolio_name: str
    return_type: str
    return_frequency: str
    initial_capital: float
    n_periods: int
    start_timestamp: str
    end_timestamp: str
    cumulative_gross_return: float
    cumulative_net_return: float
    final_gross_wealth: float
    final_net_wealth: float
    annualized_return_geometric: float
    annualized_log_mean: float
    annualized_volatility: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    calmar: float | None
    total_cost: float
    total_turnover: float
    mean_turnover: float
    cost_drag: float
    gross_to_net_retention: float | None


class _ReturnStatistics(TypedDict):
    cumulative_gross_return: float
    cumulative_net_return: float
    final_gross_wealth: float
    final_net_wealth: float
    annualized_return_geometric: float
    annualized_log_mean: float
    annualized_volatility: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    calmar: float | None


class _TradingStatistics(TypedDict):
    total_cost: float
    total_turnover: float
    mean_turnover: float
    cost_drag: float
    gross_to_net_retention: float | None


def write_downstream_metrics(
    *,
    base_dir: Path,
    dataset_params: Mapping[str, Any],
) -> None:
    output_dir = walkforward_dir(base_dir)
    if not output_dir.exists():
        return
    stitched_returns_path = output_dir / "stitched_returns.csv"
    if not stitched_returns_path.exists():
        return
    metadata = load_data_source_metadata(
        base_dir=base_dir,
        dataset_params=dataset_params,
    )
    _validate_return_metadata(metadata.return_type, metadata.return_frequency)
    portfolios_dir = output_dir / "portfolios"
    if not portfolios_dir.exists():
        raise SimulationError(
            "Downstream portfolios directory is missing",
            context={"path": str(portfolios_dir)},
        )
    metrics_dir = output_dir / "metrics"
    by_portfolio_dir = metrics_dir / "by_portfolio"
    by_portfolio_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[Mapping[str, Any]] = []
    for portfolio_dir in sorted(
        path for path in portfolios_dir.iterdir() if path.is_dir()
    ):
        metrics = _build_portfolio_metrics(
            portfolio_name=portfolio_dir.name,
            returns_frame=_load_returns_frame(portfolio_dir),
            return_type=metadata.return_type,
            return_frequency=metadata.return_frequency,
        )
        summary_rows.append(metrics)
        _write_json(
            by_portfolio_dir / f"{portfolio_dir.name}.json",
            metrics,
            message="Failed to write downstream portfolio metrics",
        )
    summary_frame = pd.DataFrame(summary_rows)
    _write_csv(metrics_dir / "summary.csv", summary_frame)
    _write_json(
        metrics_dir / "summary.json",
        {
            "metadata": {
                "return_type": metadata.return_type,
                "return_frequency": metadata.return_frequency,
                "initial_capital": _INITIAL_CAPITAL,
                "periods_per_year": _PERIODS_PER_YEAR,
                "risk_free_rate": _RISK_FREE_RATE,
                "downside_target": _DOWNSIDE_TARGET,
            },
            "portfolios": summary_rows,
        },
        message="Failed to write downstream metrics summary",
    )


def _build_portfolio_metrics(
    *,
    portfolio_name: str,
    returns_frame: pd.DataFrame,
    return_type: str,
    return_frequency: str,
) -> PortfolioMetrics:
    return_statistics = _build_return_statistics(
        returns_frame=returns_frame,
        return_type=return_type,
    )
    trading_statistics = _build_trading_statistics(
        returns_frame=returns_frame,
        return_statistics=return_statistics,
    )
    return PortfolioMetrics(
        portfolio_name=portfolio_name,
        return_type=return_type,
        return_frequency=return_frequency,
        initial_capital=_INITIAL_CAPITAL,
        n_periods=int(len(returns_frame)),
        start_timestamp=str(returns_frame["timestamp"].iloc[0]),
        end_timestamp=str(returns_frame["timestamp"].iloc[-1]),
        cumulative_gross_return=return_statistics["cumulative_gross_return"],
        cumulative_net_return=return_statistics["cumulative_net_return"],
        final_gross_wealth=return_statistics["final_gross_wealth"],
        final_net_wealth=return_statistics["final_net_wealth"],
        annualized_return_geometric=return_statistics["annualized_return_geometric"],
        annualized_log_mean=return_statistics["annualized_log_mean"],
        annualized_volatility=return_statistics["annualized_volatility"],
        sharpe=return_statistics["sharpe"],
        sortino=return_statistics["sortino"],
        max_drawdown=return_statistics["max_drawdown"],
        calmar=return_statistics["calmar"],
        total_cost=trading_statistics["total_cost"],
        total_turnover=trading_statistics["total_turnover"],
        mean_turnover=trading_statistics["mean_turnover"],
        cost_drag=trading_statistics["cost_drag"],
        gross_to_net_retention=trading_statistics["gross_to_net_retention"],
    )


def _build_return_statistics(
    *,
    returns_frame: pd.DataFrame,
    return_type: str,
) -> _ReturnStatistics:
    net_returns = returns_frame["net_return"].to_numpy(dtype=float)
    gross_returns = returns_frame["gross_return"].to_numpy(dtype=float)
    net_wealth = _wealth_path(net_returns, return_type)
    gross_wealth = _wealth_path(gross_returns, return_type)
    annualized_log_mean = _annualized_log_mean(net_returns, return_type)
    annualized_volatility = _annualized_volatility(net_returns, return_type)
    annualized_downside = _annualized_downside_deviation(
        net_returns,
        return_type,
    )
    max_drawdown = _max_drawdown(net_wealth)
    annualized_return_geometric = _annualized_geometric_return(
        net_wealth[-1],
        len(net_returns),
    )
    return {
        "cumulative_gross_return": float(gross_wealth[-1] / _INITIAL_CAPITAL - 1.0),
        "cumulative_net_return": float(net_wealth[-1] / _INITIAL_CAPITAL - 1.0),
        "final_gross_wealth": float(gross_wealth[-1]),
        "final_net_wealth": float(net_wealth[-1]),
        "annualized_return_geometric": annualized_return_geometric,
        "annualized_log_mean": annualized_log_mean,
        "annualized_volatility": annualized_volatility,
        "sharpe": _ratio_or_none(
            annualized_log_mean - _RISK_FREE_RATE,
            annualized_volatility,
        ),
        "sortino": _ratio_or_none(
            annualized_log_mean - _DOWNSIDE_TARGET,
            annualized_downside,
        ),
        "max_drawdown": max_drawdown,
        "calmar": _ratio_or_none(annualized_return_geometric, max_drawdown),
    }


def _build_trading_statistics(
    *,
    returns_frame: pd.DataFrame,
    return_statistics: _ReturnStatistics,
) -> _TradingStatistics:
    return {
        "total_cost": float(returns_frame["cost"].sum()),
        "total_turnover": float(returns_frame["turnover"].sum()),
        "mean_turnover": float(returns_frame["turnover"].mean()),
        "cost_drag": float(
            return_statistics["cumulative_gross_return"]
            - return_statistics["cumulative_net_return"]
        ),
        "gross_to_net_retention": _ratio_or_none(
            return_statistics["cumulative_net_return"],
            return_statistics["cumulative_gross_return"],
        ),
    }


def _load_returns_frame(portfolio_dir: Path) -> pd.DataFrame:
    path = portfolio_dir / "weekly_returns.csv"
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise SimulationError(
            "Failed to read downstream portfolio returns",
            context={"path": str(path)},
        ) from exc


def _validate_return_metadata(
    return_type: str,
    return_frequency: str,
) -> None:
    if return_type not in _SUPPORTED_RETURN_TYPES:
        raise SimulationError(
            "Unsupported downstream return type",
            context={"return_type": return_type},
        )
    if return_frequency != _SUPPORTED_RETURN_FREQUENCY:
        raise SimulationError(
            "Unsupported downstream return frequency",
            context={"return_frequency": return_frequency},
        )


def _wealth_path(returns: np.ndarray, return_type: str) -> np.ndarray:
    if return_type == "log":
        return _INITIAL_CAPITAL * np.exp(np.cumsum(returns))
    if return_type == "simple":
        return _INITIAL_CAPITAL * np.cumprod(1.0 + returns)
    raise SimulationError(
        "Unsupported return type for wealth path",
        context={"return_type": return_type},
    )


def _annualized_log_mean(returns: np.ndarray, return_type: str) -> float:
    if return_type == "log":
        return float(np.mean(returns) * _PERIODS_PER_YEAR)
    if return_type == "simple":
        return float(np.mean(np.log1p(returns)) * _PERIODS_PER_YEAR)
    raise SimulationError(
        "Unsupported return type for annualized log mean",
        context={"return_type": return_type},
    )


def _annualized_volatility(returns: np.ndarray, return_type: str) -> float:
    log_returns = returns if return_type == "log" else np.log1p(returns)
    if len(log_returns) < 2:
        return 0.0
    return float(np.std(log_returns, ddof=1) * math.sqrt(_PERIODS_PER_YEAR))


def _annualized_downside_deviation(
    returns: np.ndarray,
    return_type: str,
) -> float:
    log_returns = returns if return_type == "log" else np.log1p(returns)
    downside = np.minimum(log_returns - _DOWNSIDE_TARGET, 0.0)
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(_PERIODS_PER_YEAR))


def _annualized_geometric_return(
    final_wealth: float,
    n_periods: int,
) -> float:
    if n_periods <= 0:
        return 0.0
    growth = final_wealth / _INITIAL_CAPITAL
    return float(growth ** (_PERIODS_PER_YEAR / n_periods) - 1.0)


def _max_drawdown(wealth: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - (wealth / running_peak)
    return float(np.max(drawdowns, initial=0.0))


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    try:
        frame.to_csv(path, index=False)
    except Exception as exc:
        raise SimulationError(
            "Failed to write downstream metrics CSV",
            context={"path": str(path)},
        ) from exc


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    message: str,
) -> None:
    write_json_file(path=path, payload=payload, message=message)
