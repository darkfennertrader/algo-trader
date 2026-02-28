from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import SimulationError
from algo_trader.infrastructure import ensure_directory
from .diagnostics import _require_plotting


@dataclass(frozen=True)
class SviLossSummary:
    steps: np.ndarray
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    counts: np.ndarray


def run_svi_loss_diagnostics(
    *, base_dir: Path, outer_ids: Sequence[int]
) -> None:
    output_dir = _ensure_svi_loss_output_dir(base_dir)
    for outer_k in outer_ids:
        curves = _collect_outer_svi_loss_curves(
            base_dir=base_dir, outer_k=int(outer_k)
        )
        if not curves:
            continue
        summary = _aggregate_svi_loss_curves(curves)
        _write_svi_loss_csv(
            output_dir=output_dir, outer_k=int(outer_k), summary=summary
        )
        _render_svi_loss_plot(
            output_dir=output_dir,
            outer_k=int(outer_k),
            curves=curves,
            summary=summary,
        )


def _collect_outer_svi_loss_curves(
    *, base_dir: Path, outer_k: int
) -> list[np.ndarray]:
    curves: list[np.ndarray] = []
    for path in _iter_postprocess_diagnostic_paths(base_dir, outer_k):
        payload = _load_postprocess_diagnostics(path)
        curve = _extract_svi_loss_curve(payload)
        if curve is not None:
            curves.append(curve)
    return curves


def _iter_postprocess_diagnostic_paths(
    base_dir: Path, outer_k: int
) -> list[Path]:
    diagnostics_dir = (
        base_dir
        / "inner"
        / f"outer_{outer_k}"
        / "postprocessing"
        / "diagnostics"
    )
    return sorted(diagnostics_dir.glob("candidate_*_split_*.json"))


def _load_postprocess_diagnostics(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise SimulationError(
            "Failed to read postprocess diagnostics payload",
            context={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise SimulationError(
            "Postprocess diagnostics payload must be a mapping",
            context={"path": str(path)},
        )
    return payload


def _extract_svi_loss_curve(payload: Mapping[str, Any]) -> np.ndarray | None:
    training = payload.get("training")
    if not isinstance(training, Mapping):
        return None
    history = training.get("svi_loss_history")
    if not isinstance(history, Sequence):
        return None
    values: list[float] = []
    for item in history:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(float("nan"))
    if not values:
        return None
    curve = np.asarray(values, dtype=float)
    if not np.isfinite(curve).any():
        return None
    return curve


def _aggregate_svi_loss_curves(curves: Sequence[np.ndarray]) -> SviLossSummary:
    max_len = max(int(curve.shape[0]) for curve in curves)
    matrix = np.vstack(
        [_pad_curve(curve, max_len) for curve in curves]
    )
    steps = np.arange(1, max_len + 1, dtype=int)
    counts = np.sum(np.isfinite(matrix), axis=0).astype(int)
    median = _column_stat(matrix, mode="median")
    p10 = _column_stat(matrix, mode="p10")
    p90 = _column_stat(matrix, mode="p90")
    return SviLossSummary(
        steps=steps,
        median=median,
        p10=p10,
        p90=p90,
        counts=counts,
    )


def _pad_curve(curve: np.ndarray, width: int) -> np.ndarray:
    pad = max(0, width - int(curve.shape[0]))
    if pad == 0:
        return curve
    return np.pad(curve, (0, pad), mode="constant", constant_values=np.nan)


def _column_stat(matrix: np.ndarray, *, mode: str) -> np.ndarray:
    values_by_col: list[float] = []
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            values_by_col.append(float("nan"))
            continue
        if mode == "median":
            values_by_col.append(float(np.median(finite)))
        elif mode == "p10":
            values_by_col.append(float(np.quantile(finite, 0.10)))
        elif mode == "p90":
            values_by_col.append(float(np.quantile(finite, 0.90)))
        else:
            raise SimulationError("Unknown SVI loss aggregation mode")
    return np.asarray(values_by_col, dtype=float)


def _write_svi_loss_csv(
    *, output_dir: Path, outer_k: int, summary: SviLossSummary
) -> None:
    frame = pd.DataFrame(
        {
            "step": summary.steps.astype(int),
            "median_loss": summary.median,
            "p10_loss": summary.p10,
            "p90_loss": summary.p90,
            "n_curves": summary.counts.astype(int),
        }
    )
    frame.to_csv(output_dir / f"outer_{outer_k}_svi_loss.csv", index=False)


def _render_svi_loss_plot(
    *,
    output_dir: Path,
    outer_k: int,
    curves: Sequence[np.ndarray],
    summary: SviLossSummary,
) -> None:
    plt, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    for curve in _sample_curves_for_plot(curves):
        x = np.arange(1, curve.shape[0] + 1, dtype=int)
        ax.plot(x, curve, color="tab:blue", alpha=0.08, linewidth=1)
    ax.fill_between(
        summary.steps, summary.p10, summary.p90, alpha=0.20, color="tab:blue"
    )
    ax.plot(
        summary.steps, summary.median, color="tab:blue", linewidth=2, label="median"
    )
    ax.set_title(f"Outer {outer_k} inner SVI loss")
    ax.set_xlabel("SVI step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"outer_{outer_k}_svi_loss.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _sample_curves_for_plot(
    curves: Sequence[np.ndarray], *, max_curves: int = 40
) -> list[np.ndarray]:
    if len(curves) <= max_curves:
        return list(curves)
    stride = max(1, len(curves) // max_curves)
    sampled = list(curves[::stride])
    return sampled[:max_curves]


def _ensure_svi_loss_output_dir(base_dir: Path) -> Path:
    target_dir = base_dir / "outer" / "diagnostics" / "svi_loss"
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="SVI loss diagnostics path is not a directory",
        create_message="Failed to create SVI loss diagnostics output",
        context={"path": str(target_dir)},
    )
    return target_dir
