from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from algo_trader.domain import SimulationError
from algo_trader.infrastructure import ensure_directory

from . import calibration_summary_diagnostics as calibration_diags
from .plotting_backend import require_pyplot_and_seaborn


def resolve_diagnostics_root(
    *, base_dir: Path, diagnostics_root: Path | None
) -> Path:
    if diagnostics_root is not None:
        return diagnostics_root
    return base_dir / "outer" / "diagnostics"


def ensure_fan_output_dir(diagnostics_root: Path) -> Path:
    target_dir = diagnostics_root / "fan_charts"
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="Diagnostics path is not a directory",
        create_message="Failed to create diagnostics output",
        context={"path": str(target_dir)},
    )
    return target_dir


def ensure_calibration_output_dir(diagnostics_root: Path) -> Path:
    target_dir = diagnostics_root / "calibration_cpcv_ensemble"
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="Calibration path is not a directory",
        create_message="Failed to create calibration output",
        context={"path": str(target_dir)},
    )
    return target_dir


def write_aggregate_calibration_diagnostics(
    *, base_dir: Path, outer_ids: Sequence[int]
) -> None:
    summaries = _load_outer_calibration_summaries(
        base_dir=base_dir,
        outer_ids=outer_ids,
    )
    aggregate = calibration_diags.aggregate_calibration_summaries(summaries)
    diagnostics_root = resolve_diagnostics_root(
        base_dir=base_dir,
        diagnostics_root=None,
    )
    calibration_dir = ensure_calibration_output_dir(diagnostics_root)
    _cleanup_root_calibration_outputs(calibration_dir)
    calibration_diags.write_calibration_summary(
        summary=aggregate,
        output_dir=calibration_dir,
    )
    calibration_diags.plot_calibration_gap(
        summary=aggregate,
        output_dir=calibration_dir,
    )
    plot_coverage_curve(
        curve=aggregate.coverage_by_level,
        output_dir=calibration_dir,
    )
    _write_aggregate_manifest(
        output_dir=calibration_dir,
        outer_ids=outer_ids,
    )


def cleanup_global_diagnostics_root(*, base_dir: Path) -> None:
    diagnostics_root = resolve_diagnostics_root(
        base_dir=base_dir,
        diagnostics_root=None,
    )
    fan_dir = diagnostics_root / "fan_charts"
    if fan_dir.exists():
        for path in fan_dir.glob("fan_*.png"):
            path.unlink()
    calibration_dir = diagnostics_root / "calibration_cpcv_ensemble"
    if calibration_dir.exists():
        _cleanup_root_calibration_outputs(calibration_dir)


def _load_outer_calibration_summaries(
    *, base_dir: Path, outer_ids: Sequence[int]
) -> list[calibration_diags.CalibrationSummary]:
    summaries: list[calibration_diags.CalibrationSummary] = []
    for outer_k in outer_ids:
        path = (
            base_dir
            / "outer"
            / "diagnostics"
            / f"outer_{int(outer_k)}"
            / "calibration_cpcv_ensemble"
            / "calibration_summary.json"
        )
        if not path.exists():
            raise SimulationError(
                "Missing per-outer calibration summary",
                context={"path": str(path)},
            )
        summaries.append(calibration_diags.read_calibration_summary(path))
    return summaries


def _cleanup_root_calibration_outputs(output_dir: Path) -> None:
    patterns = (
        "calibration_fan_*.png",
        "calibration_fan_*.csv",
        "pit_histogram.png",
        "coverage_curve.png",
        "calibration_gap.png",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def _write_aggregate_manifest(
    *, output_dir: Path, outer_ids: Sequence[int]
) -> None:
    payload = {
        "aggregation": "median_over_outer_folds",
        "outer_ids": [int(item) for item in outer_ids],
        "scope": "inner_cpcv_postprocess",
    }
    (output_dir / "aggregate_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def plot_coverage_curve(
    *, curve: Mapping[float, float], output_dir: Path
) -> None:
    plt, sns = require_pyplot_and_seaborn()
    sns.set_theme(style="whitegrid")
    levels = sorted(curve.keys())
    coverage = [curve[level] for level in levels]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=levels, y=coverage, marker="o", ax=ax)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Coverage vs nominal")
    ax.set_xlabel("nominal p")
    ax.set_ylabel("coverage")
    fig.tight_layout()
    fig.savefig(
        output_dir / "coverage_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
