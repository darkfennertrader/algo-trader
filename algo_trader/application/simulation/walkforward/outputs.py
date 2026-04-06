from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import SimulationError

from ..io_utils import format_timestamp_dates, write_json_file
from .pathing import walkforward_dir


def write_downstream_outputs(
    *,
    base_dir: Path,
    outer_results: Sequence[Mapping[str, Any]],
    assets: Sequence[str],
) -> None:
    if not outer_results:
        return
    output_dir = walkforward_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stitched = _stitch_portfolio_outputs(outer_results)
    stitched_returns: dict[str, pd.DataFrame] = {}
    for portfolio_name, frame in stitched.items():
        portfolio_dir = output_dir / "portfolios" / portfolio_name
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        returns_frame = _build_portfolio_returns_frame(frame)
        weights_frame = _build_portfolio_weights_frame(frame, assets)
        summary = _build_portfolio_summary(
            portfolio_name=portfolio_name,
            returns_frame=returns_frame,
            frame=frame,
        )
        _write_csv(portfolio_dir / "weekly_returns.csv", returns_frame)
        _write_csv(portfolio_dir / "weights.csv", weights_frame)
        _write_json(
            portfolio_dir / "summary.json",
            summary,
            message="Failed to write stitched portfolio summary",
        )
        stitched_returns[portfolio_name] = returns_frame
    _write_json(
        output_dir / "portfolio_manifest.json",
        _build_downstream_manifest(stitched_returns),
        message="Failed to write downstream portfolio manifest",
    )
    _write_csv(
        output_dir / "stitched_returns.csv",
        _build_cross_portfolio_returns_frame(stitched_returns),
    )


def _stitch_portfolio_outputs(
    outer_results: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    by_portfolio: dict[str, list[pd.DataFrame]] = {}
    for outer_result in outer_results:
        outer_k = int(outer_result["outer_k_test"])
        portfolios = _portfolio_payloads(outer_result)
        for portfolio_name, payload in portfolios.items():
            frame = pd.DataFrame(
                {
                    "timestamp": format_timestamp_dates(payload["timestamps"]),
                    "outer_k": outer_k,
                    "gross_return": payload["gross_returns"],
                    "net_return": payload["net_returns"],
                    "cost": payload["costs"],
                    "turnover": payload["turnover"],
                    "weights": payload["weights"],
                }
            )
            by_portfolio.setdefault(portfolio_name, []).append(frame)
    return {
        portfolio_name: _stitch_single_portfolio(
            portfolio_name=portfolio_name,
            frames=frames,
        )
        for portfolio_name, frames in by_portfolio.items()
    }


def _stitch_single_portfolio(
    *,
    portfolio_name: str,
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(list(frames), ignore_index=True)
    stitched = combined.sort_values(
        by=["timestamp", "outer_k"],
        kind="stable",
    ).reset_index(drop=True)
    _validate_stitched_portfolio(portfolio_name=portfolio_name, frame=stitched)
    return stitched


def _portfolio_payloads(
    outer_result: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Sequence[Any]]]:
    raw = outer_result.get("portfolios")
    if not isinstance(raw, Mapping):
        raise SimulationError("Outer result missing portfolios payload")
    return raw


def _validate_stitched_portfolio(
    *,
    portfolio_name: str,
    frame: pd.DataFrame,
) -> None:
    duplicated = frame.duplicated(subset=["timestamp"])
    if bool(duplicated.any()):
        duplicates = frame.loc[duplicated, "timestamp"].astype(str).tolist()
        raise SimulationError(
            "Duplicate stitched portfolio timestamps",
            context={
                "portfolio": portfolio_name,
                "timestamps": ", ".join(duplicates[:5]),
            },
        )


def _build_portfolio_returns_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        ["timestamp", "outer_k", "gross_return", "net_return", "cost", "turnover"]
    ].copy()


def _build_portfolio_weights_frame(
    frame: pd.DataFrame,
    assets: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        _append_weight_rows(
            rows=rows,
            timestamp=item.timestamp,
            outer_k=item.outer_k,
            weights=item.weights,
            assets=assets,
        )
    return pd.DataFrame(rows)


def _append_weight_rows(
    *,
    rows: list[dict[str, Any]],
    timestamp: Any,
    outer_k: Any,
    weights: Any,
    assets: Sequence[str],
) -> None:
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape[0] != len(assets):
        raise SimulationError(
            "Stitched portfolio weights do not match asset count",
            context={
                "timestamp": str(timestamp),
                "expected_assets": str(len(assets)),
                "actual_assets": str(weight_array.shape[0]),
            },
        )
    outer_k_value = int(np.asarray(outer_k).item())
    for asset, weight in zip(assets, weight_array, strict=True):
        rows.append(
            {
                "timestamp": timestamp,
                "outer_k": outer_k_value,
                "asset": asset,
                "weight": float(weight),
            }
        )


def _build_portfolio_summary(
    *,
    portfolio_name: str,
    returns_frame: pd.DataFrame,
    frame: pd.DataFrame,
) -> Mapping[str, Any]:
    timestamps = returns_frame["timestamp"].astype(str).tolist()
    outer_folds = sorted(int(value) for value in frame["outer_k"].unique().tolist())
    return {
        "portfolio_name": portfolio_name,
        "n_periods": int(len(returns_frame)),
        "start_timestamp": timestamps[0],
        "end_timestamp": timestamps[-1],
        "total_cost": float(returns_frame["cost"].sum()),
        "mean_turnover": float(returns_frame["turnover"].mean()),
        "gross_return_sum": float(returns_frame["gross_return"].sum()),
        "net_return_sum": float(returns_frame["net_return"].sum()),
        "outer_folds": outer_folds,
    }


def _build_downstream_manifest(
    stitched_returns: Mapping[str, pd.DataFrame],
) -> Mapping[str, Any]:
    portfolios = {
        portfolio_name: {
            "n_periods": int(len(frame)),
            "start_timestamp": str(frame["timestamp"].iloc[0]),
            "end_timestamp": str(frame["timestamp"].iloc[-1]),
        }
        for portfolio_name, frame in stitched_returns.items()
    }
    return {
        "portfolio_names": sorted(stitched_returns),
        "portfolios": portfolios,
    }


def _build_cross_portfolio_returns_frame(
    stitched_returns: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for portfolio_name in sorted(stitched_returns):
        frame = stitched_returns[portfolio_name][["timestamp", "net_return"]].rename(
            columns={"net_return": portfolio_name}
        )
        merged = frame if merged is None else merged.merge(frame, on="timestamp", how="inner")
    if merged is None:
        raise SimulationError("No stitched portfolio returns available")
    return merged


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    try:
        frame.to_csv(path, index=False)
    except Exception as exc:
        raise SimulationError(
            "Failed to write stitched downstream CSV",
            context={"path": str(path)},
        ) from exc


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    message: str,
) -> None:
    write_json_file(path=path, payload=payload, message=message)
