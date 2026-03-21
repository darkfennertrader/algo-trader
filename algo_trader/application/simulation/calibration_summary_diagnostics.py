from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from algo_trader.domain import SimulationError


@dataclass(frozen=True)
class CalibrationSummary:
    coverage_by_level: Mapping[float, float]
    abs_error_by_level: Mapping[float, float]
    mean_abs_coverage_error: float
    max_abs_coverage_error: float
    pit_uniform_rmse: float
    pit_sample_count: int
    pit_bin_count: int


def build_calibration_summary(
    *, curve: Mapping[float, float], pit_values: np.ndarray
) -> CalibrationSummary:
    coverage_by_level: dict[float, float] = {}
    abs_error_by_level: dict[float, float] = {}
    finite_errors: list[float] = []
    for level, empirical in curve.items():
        coverage_by_level[float(level)] = float(empirical)
        abs_error = abs(float(empirical) - float(level))
        abs_error_by_level[float(level)] = abs_error
        if np.isfinite(abs_error):
            finite_errors.append(abs_error)
    pit_score, sample_count, bin_count = pit_uniform_rmse(pit_values)
    mean_error, max_error = _coverage_error_summary(finite_errors)
    return CalibrationSummary(
        coverage_by_level=coverage_by_level,
        abs_error_by_level=abs_error_by_level,
        mean_abs_coverage_error=mean_error,
        max_abs_coverage_error=max_error,
        pit_uniform_rmse=pit_score,
        pit_sample_count=sample_count,
        pit_bin_count=bin_count,
    )


def pit_uniform_rmse(
    values: np.ndarray, *, bin_count: int = 20
) -> tuple[float, int, int]:
    pit_values = values[np.isfinite(values)]
    if pit_values.size == 0:
        raise SimulationError("PIT deviation score has no valid values")
    clipped = np.clip(pit_values, 0.0, np.nextafter(1.0, 0.0))
    scaled = np.floor(clipped * float(bin_count)).astype(np.int64)
    counts = np.bincount(
        scaled,
        minlength=int(bin_count),
    )
    actual = counts.astype(float) / float(pit_values.size)
    target = np.full(actual.shape, 1.0 / float(bin_count), dtype=float)
    rmse = float(np.sqrt(np.mean((actual - target) ** 2)))
    return rmse, int(pit_values.size), int(bin_count)


def write_calibration_summary(
    *, summary: CalibrationSummary, output_dir: Path
) -> None:
    _write_calibration_summary_csv(summary=summary, output_dir=output_dir)
    _write_calibration_summary_json(summary=summary, output_dir=output_dir)


def plot_calibration_gap(
    *, summary: CalibrationSummary, output_dir: Path
) -> None:
    plt, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    levels = sorted(summary.abs_error_by_level.keys())
    labels = [coverage_level_tag(level) for level in levels]
    values = [summary.abs_error_by_level[level] for level in levels]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=labels, y=values, ax=ax, color="tab:blue")
    ax.set_title("Calibration absolute error by level")
    ax.set_xlabel("coverage level")
    ax.set_ylabel("absolute error")
    fig.tight_layout()
    fig.savefig(
        output_dir / "calibration_gap.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _coverage_error_summary(
    finite_errors: list[float],
) -> tuple[float, float]:
    if not finite_errors:
        return float("nan"), float("nan")
    mean_error = float(np.mean(finite_errors))
    max_error = float(np.max(finite_errors))
    return mean_error, max_error


def _write_calibration_summary_csv(
    *, summary: CalibrationSummary, output_dir: Path
) -> None:
    levels = sorted(summary.coverage_by_level.keys())
    frame = pd.DataFrame(
        {
            "nominal_level": levels,
            "empirical_coverage": [
                summary.coverage_by_level[level] for level in levels
            ],
            "abs_error": [
                summary.abs_error_by_level[level] for level in levels
            ],
        }
    )
    frame.to_csv(output_dir / "calibration_summary.csv", index=False)


def _write_calibration_summary_json(
    *, summary: CalibrationSummary, output_dir: Path
) -> None:
    payload = {
        "coverage_by_level": {
            coverage_level_tag(level): summary.coverage_by_level[level]
            for level in sorted(summary.coverage_by_level.keys())
        },
        "abs_error_by_level": {
            coverage_level_tag(level): summary.abs_error_by_level[level]
            for level in sorted(summary.abs_error_by_level.keys())
        },
        "mean_abs_coverage_error": summary.mean_abs_coverage_error,
        "max_abs_coverage_error": summary.max_abs_coverage_error,
        "pit_uniform_rmse": summary.pit_uniform_rmse,
        "pit_sample_count": summary.pit_sample_count,
        "pit_bin_count": summary.pit_bin_count,
    }
    path = output_dir / "calibration_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def coverage_level_tag(level: float) -> str:
    pct = float(level) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) <= 1e-6:
        return f"p{int(rounded):02d}"
    as_text = f"{pct:.3f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"p{as_text}"


def _require_plotting():
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        import seaborn as sns  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover
        raise SimulationError(
            "Plotting dependencies missing for calibration diagnostics"
        ) from exc
    return plt, sns
