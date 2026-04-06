from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from algo_trader.domain import SimulationError

from ..data_source_metadata import load_data_source_metadata
from .pathing import walkforward_dir


def write_downstream_plots(
    *,
    base_dir: Path,
    dataset_params: Mapping[str, Any],
) -> None:
    output_dir = walkforward_dir(base_dir)
    returns_path = output_dir / "stitched_returns.csv"
    if not returns_path.exists():
        return
    metadata = load_data_source_metadata(
        base_dir=base_dir,
        dataset_params=dataset_params,
    )
    returns_frame = _load_stitched_returns(returns_path)
    cumulative_frame = _build_cumulative_returns_frame(
        returns_frame=returns_frame,
        return_type=metadata.return_type,
    )
    drawdown_frame = _build_underwater_drawdown_frame(
        returns_frame=returns_frame,
        return_type=metadata.return_type,
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_cumulative_returns_plot(
        path=plots_dir / "cumulative_returns.png",
        cumulative_frame=cumulative_frame,
    )
    _write_underwater_drawdown_plot(
        path=plots_dir / "underwater_drawdown.png",
        drawdown_frame=drawdown_frame,
    )


def _load_stitched_returns(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise SimulationError(
            "Failed to read stitched downstream returns",
            context={"path": str(path)},
        ) from exc
    if "timestamp" not in frame.columns:
        raise SimulationError(
            "Stitched downstream returns are missing timestamp",
            context={"path": str(path)},
        )
    return frame


def _build_cumulative_returns_frame(
    *,
    returns_frame: pd.DataFrame,
    return_type: str,
) -> pd.DataFrame:
    frame = returns_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    result = pd.DataFrame({"timestamp": frame["timestamp"]})
    for column in _portfolio_columns(frame):
        returns = frame[column].to_numpy(dtype=float)
        result[column] = _cumulative_returns_pct(
            returns=returns,
            return_type=return_type,
        )
    return result


def _portfolio_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "timestamp"]


def _cumulative_returns_pct(
    *,
    returns: np.ndarray,
    return_type: str,
) -> np.ndarray:
    if return_type == "log":
        return (np.exp(np.cumsum(returns)) - 1.0) * 100.0
    if return_type == "simple":
        return (np.cumprod(1.0 + returns) - 1.0) * 100.0
    raise SimulationError(
        "Unsupported return type for cumulative return plot",
        context={"return_type": return_type},
    )


def _write_cumulative_returns_plot(
    *,
    path: Path,
    cumulative_frame: pd.DataFrame,
) -> None:
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for column in _portfolio_columns(cumulative_frame):
        axis.plot(
            cumulative_frame["timestamp"],
            cumulative_frame[column],
            label=column,
        )
    axis.set_title("Cumulative Net Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Return (%)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=150)
    except Exception as exc:
        raise SimulationError(
            "Failed to write cumulative return plot",
            context={"path": str(path)},
        ) from exc


def _build_underwater_drawdown_frame(
    *,
    returns_frame: pd.DataFrame,
    return_type: str,
) -> pd.DataFrame:
    frame = returns_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    result = pd.DataFrame({"timestamp": frame["timestamp"]})
    for column in _portfolio_columns(frame):
        returns = frame[column].to_numpy(dtype=float)
        result[column] = _underwater_drawdown_pct(
            returns=returns,
            return_type=return_type,
        )
    return result


def _underwater_drawdown_pct(
    *,
    returns: np.ndarray,
    return_type: str,
) -> np.ndarray:
    wealth = _wealth_index(returns=returns, return_type=return_type)
    running_peak = np.maximum.accumulate(wealth)
    return ((wealth / running_peak) - 1.0) * 100.0


def _wealth_index(
    *,
    returns: np.ndarray,
    return_type: str,
) -> np.ndarray:
    if return_type == "log":
        return np.exp(np.cumsum(returns))
    if return_type == "simple":
        return np.cumprod(1.0 + returns)
    raise SimulationError(
        "Unsupported return type for downstream plot",
        context={"return_type": return_type},
    )


def _write_underwater_drawdown_plot(
    *,
    path: Path,
    drawdown_frame: pd.DataFrame,
) -> None:
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for column in _portfolio_columns(drawdown_frame):
        axis.plot(
            drawdown_frame["timestamp"],
            drawdown_frame[column],
            label=column,
        )
    axis.set_title("Underwater Drawdown")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown (%)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=150)
    except Exception as exc:
        raise SimulationError(
            "Failed to write underwater drawdown plot",
            context={"path": str(path)},
        ) from exc
