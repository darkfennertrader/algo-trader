from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import FanChartsConfig
from algo_trader.infrastructure import ensure_directory


@dataclass(frozen=True)
class FanChartDiagnosticsContext:
    base_dir: Path
    outer_ids: Sequence[int]
    candidate_id: int
    config: FanChartsConfig


@dataclass(frozen=True)
class FanChartData:
    timestamps: list[pd.Timestamp]
    asset_names: list[str]
    z_true: np.ndarray
    quantiles: Mapping[float, np.ndarray]


@dataclass(frozen=True)
class TimeSampleRecord:
    source: str
    z_true: np.ndarray
    z_samples: np.ndarray


TRUE_TOL_ABS = 1e-8
TRUE_TOL_REL = 1e-6


def run_fan_chart_diagnostics(
    context: FanChartDiagnosticsContext,
) -> None:
    if not context.config.enable:
        return
    timestamps = _load_timestamps(context.base_dir)
    assets = _load_asset_names(context.base_dir)
    selected_assets = _resolve_assets(assets, context.config)
    asset_indices = _resolve_asset_indices(assets, selected_assets)
    global_scale = _collect_global_scale(
        base_dir=context.base_dir,
        outer_ids=context.outer_ids,
        candidate_id=context.candidate_id,
        asset_indices=asset_indices,
    )
    required_quantiles = _merge_quantiles(
        context.config.quantiles, context.config.coverage_levels
    )
    samples = _collect_candidate_samples(
        base_dir=context.base_dir,
        outer_ids=context.outer_ids,
        candidate_id=context.candidate_id,
        asset_indices=asset_indices,
        global_scale=global_scale,
    )
    data = _build_fan_chart_data(
        samples=samples,
        timestamps=timestamps,
        asset_names=selected_assets,
        quantiles=required_quantiles,
    )
    output_dir = _ensure_output_dir(context.base_dir)
    _render_asset_fan_charts(
        data=data,
        quantile_levels=context.config.quantiles,
        output_dir=output_dir,
    )
    calibration_dir = _ensure_calibration_output_dir(context.base_dir)
    _render_calibration_charts(
        data=data,
        coverage_levels=context.config.coverage_levels,
        rolling_windows=context.config.rolling_mean,
        output_dir=calibration_dir,
    )
    _run_calibration_diagnostics(
        samples=samples,
        asset_count=len(selected_assets),
        fan_chart_data=data,
        coverage_levels=context.config.coverage_levels,
        output_dir=calibration_dir,
    )


def _load_timestamps(base_dir: Path) -> list[pd.Timestamp]:
    path = base_dir / "inputs" / "timestamps.csv"
    if not path.exists():
        raise SimulationError(
            "Missing timestamps for diagnostics",
            context={"path": str(path)},
        )
    frame = pd.read_csv(path, usecols=["datetime_utc"])
    stamps = pd.to_datetime(frame["datetime_utc"], utc=True)
    return list(stamps.to_list())


def _load_asset_names(base_dir: Path) -> list[str]:
    path = base_dir / "inputs" / "targets.csv"
    if not path.exists():
        raise SimulationError(
            "Missing targets for diagnostics",
            context={"path": str(path)},
        )
    frame = pd.read_csv(path, nrows=0)
    columns = [str(col) for col in frame.columns]
    if "timestamp" in columns:
        columns.remove("timestamp")
    if not columns:
        raise SimulationError(
            "Targets CSV has no asset columns",
            context={"path": str(path)},
        )
    return columns


def _resolve_assets(
    assets: Sequence[str], config: FanChartsConfig
) -> list[str]:
    if config.assets_mode == "all":
        return list(assets)
    selected = list(config.assets)
    missing = [name for name in selected if name not in assets]
    if missing:
        raise SimulationError(
            "Diagnostics assets missing from inputs",
            context={"missing": ",".join(missing)},
        )
    return selected


def _resolve_asset_indices(
    assets: Sequence[str], selected: Sequence[str]
) -> list[int]:
    index_map = {name: idx for idx, name in enumerate(assets)}
    return [index_map[name] for name in selected]


def _merge_quantiles(
    quantiles: Sequence[float], coverage_levels: Sequence[float]
) -> list[float]:
    levels = set(float(q) for q in quantiles)
    for level in coverage_levels:
        lo = (1.0 - float(level)) / 2.0
        hi = 1.0 - lo
        levels.add(lo)
        levels.add(hi)
    return sorted(levels)


@dataclass(frozen=True)
class CandidateSamples:
    records_by_time: Mapping[int, list[TimeSampleRecord]]


def _collect_global_scale(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
    asset_indices: Sequence[int],
) -> np.ndarray:
    scales: list[np.ndarray] = []
    for path in _iter_payload_paths(base_dir, outer_ids, candidate_id):
        payload = _load_payload(path)
        scale = _require_tensor(payload, "scale").detach().cpu().numpy()
        scales.append(scale[asset_indices])
    if not scales:
        raise SimulationError(
            "No postprocess scales found for diagnostics",
            context={"candidate_id": str(candidate_id)},
        )
    stacked = np.vstack(scales)
    global_scale = np.nanmedian(stacked, axis=0)
    if not np.all(np.isfinite(global_scale)):
        raise SimulationError("Diagnostics scale contains non-finite values")
    if np.any(global_scale <= 0):
        raise SimulationError("Diagnostics scale must be positive")
    return global_scale


def _collect_candidate_samples(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
    asset_indices: Sequence[int],
    global_scale: np.ndarray,
) -> CandidateSamples:
    records_by_time: dict[int, list[TimeSampleRecord]] = {}
    total = 0
    for path in _iter_payload_paths(base_dir, outer_ids, candidate_id):
        payload = _load_payload(path)
        _append_payload(
            payload=payload,
            asset_indices=asset_indices,
            global_scale=global_scale,
            source=str(path),
            records_by_time=records_by_time,
        )
        total += 1
    if total == 0:
        raise SimulationError(
            "No postprocess payloads found for diagnostics",
            context={"candidate_id": str(candidate_id)},
        )
    return CandidateSamples(
        records_by_time=records_by_time,
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise SimulationError(
            "Failed to read postprocess payload",
            context={"path": str(path)},
        ) from exc


def _append_payload(
    *,
    payload: Mapping[str, Any],
    asset_indices: Sequence[int],
    global_scale: np.ndarray,
    source: str,
    records_by_time: dict[int, list[TimeSampleRecord]],
) -> None:
    z_true_np, z_samples_np, test_idx = _extract_scaled_payload_arrays(
        payload=payload,
        asset_indices=asset_indices,
        global_scale=global_scale,
    )
    for pos, time_idx in enumerate(test_idx):
        key = int(time_idx)
        records = records_by_time.setdefault(key, [])
        records.append(
            TimeSampleRecord(
                source=source,
                z_true=z_true_np[pos],
                z_samples=z_samples_np[:, pos, :],
            )
        )


def _extract_scaled_payload_arrays(
    *,
    payload: Mapping[str, Any],
    asset_indices: Sequence[int],
    global_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Sequence[Any]]:
    z_true = _require_tensor(payload, "z_true")
    z_samples = _require_tensor(payload, "z_samples")
    scale = _require_tensor(payload, "scale")
    test_idx = payload.get("test_idx")
    if not isinstance(test_idx, Sequence):
        raise SimulationError("Postprocess payload missing test_idx")
    selected_scale = scale.detach().cpu().numpy()[asset_indices]
    z_true_np = z_true[:, asset_indices].detach().cpu().numpy()
    z_samples_np = z_samples[:, :, asset_indices].detach().cpu().numpy()
    z_true_scaled = (z_true_np * selected_scale) / global_scale
    z_samples_scaled = (z_samples_np * selected_scale) / global_scale
    if z_true_scaled.shape[0] != len(test_idx):
        raise SimulationError("Postprocess payload misaligned on T")
    return z_true_scaled, z_samples_scaled, test_idx


def _iter_payload_paths(
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
) -> list[Path]:
    paths: list[Path] = []
    for outer_k in outer_ids:
        candidates_dir = (
            base_dir
            / "inner"
            / f"outer_{outer_k}"
            / "postprocessing"
            / "candidates"
        )
        pattern = f"candidate_{candidate_id:04d}_split_*.pt"
        paths.extend(sorted(candidates_dir.glob(pattern)))
    return paths


def _require_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError(f"Postprocess payload missing {key}")
    return value


def _build_fan_chart_data(
    *,
    samples: CandidateSamples,
    timestamps: Sequence[pd.Timestamp],
    asset_names: Sequence[str],
    quantiles: Sequence[float],
) -> FanChartData:
    time_indices = sorted(samples.records_by_time.keys())
    if not time_indices:
        raise SimulationError("No test weeks available for diagnostics")
    z_true = _build_true_matrix(
        time_indices=time_indices,
        records_by_time=samples.records_by_time,
        asset_count=len(asset_names),
    )
    quantiles_map = _build_quantiles_matrix(
        time_indices=time_indices,
        records_by_time=samples.records_by_time,
        quantiles=quantiles,
        asset_count=len(asset_names),
    )
    resolved_times = _resolve_timestamps(timestamps, time_indices)
    return FanChartData(
        timestamps=resolved_times,
        asset_names=list(asset_names),
        z_true=z_true,
        quantiles=quantiles_map,
    )


def _build_true_matrix(
    *,
    time_indices: Sequence[int],
    records_by_time: Mapping[int, Sequence[TimeSampleRecord]],
    asset_count: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for time_idx in time_indices:
        records = records_by_time.get(time_idx, [])
        if not records:
            rows.append(np.full(asset_count, np.nan))
        else:
            rows.append(
                _resolve_true_row(
                    records=records,
                    time_idx=int(time_idx),
                    asset_count=asset_count,
                )
            )
    return np.vstack(rows)


def _build_quantiles_matrix(
    *,
    time_indices: Sequence[int],
    records_by_time: Mapping[int, Sequence[TimeSampleRecord]],
    quantiles: Sequence[float],
    asset_count: int,
) -> dict[float, np.ndarray]:
    quantile_map: dict[float, np.ndarray] = {
        q: np.full((len(time_indices), asset_count), np.nan)
        for q in quantiles
    }
    for row, time_idx in enumerate(time_indices):
        records = records_by_time.get(time_idx, [])
        if not records:
            continue
        for asset_idx in range(asset_count):
            pooled = _pool_equal_split_samples(
                records=records,
                asset_idx=asset_idx,
            )
            if pooled.size == 0:
                continue
            for q in quantiles:
                quantile_map[q][row, asset_idx] = float(
                    np.nanquantile(pooled, float(q))
                )
    return quantile_map


def _resolve_true_row(
    *,
    records: Sequence[TimeSampleRecord],
    time_idx: int,
    asset_count: int,
) -> np.ndarray:
    resolved = [
        _resolve_true_value(
            values=np.asarray(
                [
                    float(np.take(record.z_true, asset_idx))
                    for record in records
                ],
                dtype=float,
            ),
            records=records,
            time_idx=time_idx,
            asset_idx=asset_idx,
        )
        for asset_idx in range(asset_count)
    ]
    return np.asarray(resolved, dtype=float)


def _resolve_true_value(
    *,
    values: np.ndarray,
    records: Sequence[TimeSampleRecord],
    time_idx: int,
    asset_idx: int,
) -> float:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float("nan")
    finite_values = values[finite_mask]
    z_ref = float(np.median(finite_values))
    max_abs_diff = float(np.max(np.abs(finite_values - z_ref)))
    tolerance = TRUE_TOL_ABS + TRUE_TOL_REL * max(1.0, abs(z_ref))
    if max_abs_diff > tolerance:
        sources = [
            record.source
            for row_idx, record in enumerate(records)
            if bool(finite_mask[row_idx])
        ]
        raise SimulationError(
            "Inconsistent z_true across duplicated diagnostics keys",
            context={
                "time_idx": str(time_idx),
                "asset_idx": str(asset_idx),
                "z_ref": str(z_ref),
                "max_abs_diff": str(max_abs_diff),
                "tolerance": str(tolerance),
                "sources": ";".join(sources),
                "values": ",".join(f"{val:.12g}" for val in finite_values),
            },
        )
    return z_ref


def _pool_equal_split_samples(
    *, records: Sequence[TimeSampleRecord], asset_idx: int
) -> np.ndarray:
    split_samples: list[np.ndarray] = []
    for record in records:
        values = record.z_samples[:, asset_idx]
        finite = values[np.isfinite(values)]
        if finite.size:
            split_samples.append(finite)
    if not split_samples:
        return np.asarray([], dtype=float)
    target = min(int(values.size) for values in split_samples)
    pooled = [
        _deterministic_subsample(values=values, target=target)
        for values in split_samples
    ]
    return np.concatenate(pooled)


def _deterministic_subsample(*, values: np.ndarray, target: int) -> np.ndarray:
    if target <= 0:
        return np.asarray([], dtype=float)
    if target >= values.size:
        return values
    indices = (np.arange(target, dtype=int) * values.size) // target
    return values[indices]


def _resolve_timestamps(
    timestamps: Sequence[pd.Timestamp], time_indices: Sequence[int]
) -> list[pd.Timestamp]:
    resolved: list[pd.Timestamp] = []
    total = len(timestamps)
    for idx in time_indices:
        if idx < 0 or idx >= total:
            raise SimulationError(
                "Diagnostics time index out of range",
                context={"index": str(idx)},
            )
        resolved.append(timestamps[idx])
    return resolved


def _ensure_output_dir(base_dir: Path) -> Path:
    target_dir = base_dir / "outer" / "diagnostics" / "fan_charts"
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="Diagnostics path is not a directory",
        create_message="Failed to create diagnostics output",
        context={"path": str(target_dir)},
    )
    return target_dir


def _ensure_calibration_output_dir(base_dir: Path) -> Path:
    target_dir = base_dir / "outer" / "diagnostics" / "calibration_cpcv_ensemble"
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="Calibration path is not a directory",
        create_message="Failed to create calibration output",
        context={"path": str(target_dir)},
    )
    return target_dir


def _render_asset_fan_charts(
    *,
    data: FanChartData,
    quantile_levels: Sequence[float],
    output_dir: Path,
) -> None:
    plt, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    bands = _fan_bands(quantile_levels)
    median = 0.5 if 0.5 in quantile_levels else None
    for idx, asset in enumerate(data.asset_names):
        fig, ax = plt.subplots(figsize=(10, 5))
        for low, high, alpha, color in bands:
            if low in data.quantiles and high in data.quantiles:
                ax.fill_between(
                    data.timestamps,
                    data.quantiles[low][:, idx],
                    data.quantiles[high][:, idx],
                    alpha=alpha,
                    color=color,
                    label=_fan_band_label(low, high),
                )
        if median is not None and median in data.quantiles:
            sns.lineplot(
                x=data.timestamps,
                y=data.quantiles[median][:, idx],
                ax=ax,
                label="median",
                color="tab:blue",
            )
        sns.lineplot(
            x=data.timestamps,
            y=data.z_true[:, idx],
            ax=ax,
            label="realized",
            color="black",
        )
        ax.set_title(f"{asset} fan chart (z-space)")
        ax.set_xlabel("test week")
        ax.set_ylabel("z-score")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"fan_{asset}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _render_calibration_charts(
    *,
    data: FanChartData,
    coverage_levels: Sequence[float],
    rolling_windows: Sequence[int],
    output_dir: Path,
) -> None:
    series = _collect_calibration_level_series(
        data=data,
        coverage_levels=coverage_levels,
        rolling_windows=rolling_windows,
    )
    _cleanup_legacy_calibration_outputs(output_dir)
    for item in series:
        _write_calibration_level_csv(
            data=data,
            series=item,
            output_dir=output_dir,
        )
        _plot_calibration_level(
            series=item,
            timestamps=data.timestamps,
            output_dir=output_dir,
        )


def _run_calibration_diagnostics(
    *,
    samples: CandidateSamples,
    asset_count: int,
    fan_chart_data: FanChartData,
    coverage_levels: Sequence[float],
    output_dir: Path,
) -> None:
    if not samples.records_by_time:
        return
    pit_values = _pool_raw_pit_values(
        records_by_time=samples.records_by_time,
        asset_count=asset_count,
    )
    _plot_pit_histogram(pit_values, output_dir)
    curve = _coverage_curve_from_raw(
        data=fan_chart_data,
        coverage_levels=coverage_levels,
    )
    _plot_coverage_curve(curve, output_dir)


@dataclass(frozen=True)
class CalibrationLevelSeries:
    level: float
    weekly_coverage: np.ndarray
    weekly_band: tuple[np.ndarray, np.ndarray]
    cumulative_coverage: np.ndarray
    rolling_coverage: Mapping[int, np.ndarray]


def _collect_calibration_level_series(
    *,
    data: FanChartData,
    coverage_levels: Sequence[float],
    rolling_windows: Sequence[int],
) -> list[CalibrationLevelSeries]:
    series: list[CalibrationLevelSeries] = []
    for level in coverage_levels:
        item = _build_calibration_level_series(
            data=data,
            level=float(level),
            rolling_windows=rolling_windows,
        )
        series.append(item)
    return series


def _build_calibration_level_series(
    *,
    data: FanChartData,
    level: float,
    rolling_windows: Sequence[int],
) -> CalibrationLevelSeries:
    q_lo, q_hi = _coverage_quantile_bounds(data, level)
    indicator, mask = _compute_coverage_indicator_matrix(
        data=data, q_lo=q_lo, q_hi=q_hi
    )
    weekly_coverage, counts = _coverage_from_indicator(indicator, mask)
    weekly_band = _compute_coverage_band(weekly_coverage, counts)
    cumulative_coverage = _cumulative_coverage_from_indicator(indicator, mask)
    rolling_coverage = _rolling_coverage(
        coverage=weekly_coverage, windows=rolling_windows
    )
    return CalibrationLevelSeries(
        level=level,
        weekly_coverage=weekly_coverage,
        weekly_band=weekly_band,
        cumulative_coverage=cumulative_coverage,
        rolling_coverage=rolling_coverage,
    )


def _rolling_coverage(
    *, coverage: np.ndarray, windows: Sequence[int]
) -> dict[int, np.ndarray]:
    series = pd.Series(coverage, dtype=float)
    rolling: dict[int, np.ndarray] = {}
    for window in windows:
        values = series.rolling(
            window=int(window), min_periods=int(window)
        ).mean()
        rolling[int(window)] = values.to_numpy(dtype=float)
    return rolling


def _coverage_from_indicator(
    indicator: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    covered_counts = np.nansum(indicator, axis=1)
    counts = mask.sum(axis=1)
    coverage = np.divide(
        covered_counts,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    return coverage, counts


def _cumulative_coverage_from_indicator(
    indicator: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    covered_counts = np.nansum(indicator, axis=1)
    counts = mask.sum(axis=1).astype(float)
    cumulative_hits = np.cumsum(covered_counts)
    cumulative_counts = np.cumsum(counts)
    return np.divide(
        cumulative_hits,
        cumulative_counts,
        out=np.full(cumulative_counts.shape, np.nan, dtype=float),
        where=cumulative_counts > 0,
    )


def _write_calibration_level_csv(
    *,
    data: FanChartData,
    series: CalibrationLevelSeries,
    output_dir: Path,
) -> None:
    frame = _build_calibration_level_frame(data, series)
    frame.to_csv(
        output_dir / f"calibration_fan_{_coverage_level_tag(series.level)}.csv",
        index=False,
    )


def _build_calibration_level_frame(
    data: FanChartData, series: CalibrationLevelSeries
) -> pd.DataFrame:
    rows: dict[str, list[float] | list[str]] = {
        "timestamp": [
            stamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            for stamp in data.timestamps
        ],
        "nominal_level": [series.level] * len(data.timestamps),
        "weekly_coverage": list(series.weekly_coverage),
        "weekly_band_low": list(series.weekly_band[0]),
        "weekly_band_high": list(series.weekly_band[1]),
        "cumulative_coverage": list(series.cumulative_coverage),
    }
    for window, values in series.rolling_coverage.items():
        rows[f"rolling_mean_{int(window)}w"] = list(values)
    return pd.DataFrame(rows)


def _plot_calibration_level(
    *,
    series: CalibrationLevelSeries,
    timestamps: Sequence[pd.Timestamp],
    output_dir: Path,
) -> None:
    plt, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x=timestamps,
        y=series.weekly_coverage,
        ax=ax,
        label="weekly coverage",
    )
    ax.fill_between(
        timestamps,
        series.weekly_band[0],
        series.weekly_band[1],
        alpha=0.1,
    )
    sns.lineplot(
        x=timestamps,
        y=series.cumulative_coverage,
        ax=ax,
        label="cumulative coverage",
        color="tab:orange",
    )
    for window, values in series.rolling_coverage.items():
        sns.lineplot(
            x=timestamps,
            y=values,
            ax=ax,
            label=f"rolling mean {int(window)}w",
        )
    ax.axhline(series.level, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"Calibration coverage ({_coverage_level_tag(series.level)})"
    )
    ax.set_xlabel("test week")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"calibration_fan_{_coverage_level_tag(series.level)}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _cleanup_legacy_calibration_outputs(output_dir: Path) -> None:
    for name in ("calibration_fan.png", "calibration_fan.csv"):
        legacy = output_dir / name
        if legacy.exists():
            legacy.unlink()


def _coverage_level_tag(level: float) -> str:
    pct = float(level) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) <= 1e-6:
        return f"p{int(rounded):02d}"
    as_text = f"{pct:.3f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"p{as_text}"


def _fan_bands(quantiles: Sequence[float]) -> list[tuple[float, float, float, str]]:
    bands = [
        (0.05, 0.95, 0.15, "tab:blue"),
        (0.10, 0.90, 0.25, "tab:blue"),
        (0.25, 0.75, 0.35, "tab:blue"),
    ]
    available = set(float(q) for q in quantiles)
    return [band for band in bands if band[0] in available and band[1] in available]


def _fan_band_label(low: float, high: float) -> str:
    return f"q{_quantile_percent_label(low)}-q{_quantile_percent_label(high)}"


def _quantile_percent_label(level: float) -> str:
    pct = float(level) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) <= 1e-6:
        return f"{int(rounded):02d}"
    return f"{pct:.3f}".rstrip("0").rstrip(".")


def _coverage_quantile_bounds(
    data: FanChartData, level: float
) -> tuple[np.ndarray, np.ndarray]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    lo_key = _match_quantile_key(data.quantiles, lo)
    hi_key = _match_quantile_key(data.quantiles, hi)
    if lo_key is None or hi_key is None:
        raise SimulationError(
            "Coverage quantiles missing",
            context={"level": str(level)},
        )
    return data.quantiles[lo_key], data.quantiles[hi_key]


def _compute_coverage_indicator_matrix(
    *, data: FanChartData, q_lo: np.ndarray, q_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    z_true = data.z_true
    valid_mask = np.isfinite(z_true) & np.isfinite(q_lo) & np.isfinite(q_hi)
    covered_mask = (z_true >= q_lo) & (z_true <= q_hi) & valid_mask
    indicator = np.where(valid_mask, covered_mask.astype(float), np.nan)
    return indicator, valid_mask


def _compute_coverage_band(
    coverage: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    safe_cov = np.nan_to_num(coverage)
    se = np.sqrt(
        safe_cov * (1.0 - safe_cov) / np.maximum(counts, 1)
    )
    lower = np.clip(coverage - 2.0 * se, 0.0, 1.0)
    upper = np.clip(coverage + 2.0 * se, 0.0, 1.0)
    return lower, upper


def _require_plotting():
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        import seaborn as sns  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SimulationError(
            "Diagnostics require matplotlib and seaborn"
        ) from exc
    return plt, sns


def _match_quantile_key(
    quantiles: Mapping[float, np.ndarray], target: float
) -> float | None:
    if target in quantiles:
        return target
    tolerance = 1e-6
    closest = None
    closest_diff = float("inf")
    for key in quantiles.keys():
        diff = abs(float(key) - target)
        if diff < closest_diff:
            closest = float(key)
            closest_diff = diff
    if closest is not None and closest_diff <= tolerance:
        return closest
    return None


def _pool_raw_pit_values(
    *,
    records_by_time: Mapping[int, Sequence[TimeSampleRecord]],
    asset_count: int,
) -> np.ndarray:
    pooled: list[float] = []
    for time_idx in sorted(records_by_time.keys()):
        records = records_by_time[time_idx]
        true_row = _resolve_true_row(
            records=records,
            time_idx=int(time_idx),
            asset_count=asset_count,
        )
        for asset_idx in range(asset_count):
            z_true_value = float(np.take(true_row, asset_idx))
            if not np.isfinite(z_true_value):
                continue
            values = _pool_equal_split_samples(
                records=records,
                asset_idx=asset_idx,
            )
            if values.size == 0:
                continue
            pooled.append(float(np.mean(values <= z_true_value)))
    return np.asarray(pooled, dtype=float)


def _compute_pit(z_samples: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    valid = np.isfinite(z_true) & np.isfinite(z_samples).all(axis=0)
    pit = (z_samples <= z_true[None, :, :]).mean(axis=0)
    pit[~valid] = np.nan
    return pit


def _plot_pit_histogram(values: np.ndarray, output_dir: Path) -> None:
    plt, sns = _require_plotting()
    pit_values = values[np.isfinite(values)]
    if pit_values.size == 0:
        raise SimulationError("PIT histogram has no valid values")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(pit_values, bins=20, stat="density", ax=ax)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("PIT histogram (pooled)")
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(
        output_dir / "pit_histogram.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _coverage_curve_from_raw(
    *,
    data: FanChartData,
    coverage_levels: Sequence[float],
) -> Mapping[float, float]:
    curve: dict[float, float] = {}
    for level in coverage_levels:
        lo = (1.0 - float(level)) / 2.0
        hi = 1.0 - lo
        lo_key = _match_quantile_key(data.quantiles, lo)
        hi_key = _match_quantile_key(data.quantiles, hi)
        if lo_key is None or hi_key is None:
            raise SimulationError(
                "Coverage quantiles missing",
                context={"level": str(level)},
            )
        q_lo = data.quantiles[lo_key]
        q_hi = data.quantiles[hi_key]
        valid = (
            np.isfinite(data.z_true)
            & np.isfinite(q_lo)
            & np.isfinite(q_hi)
        )
        covered = (data.z_true >= q_lo) & (data.z_true <= q_hi) & valid
        finite = covered[valid]
        if finite.size == 0:
            curve[float(level)] = float("nan")
        else:
            curve[float(level)] = float(np.mean(finite.astype(float)))
    return curve


def _plot_coverage_curve(
    curve: Mapping[float, float], output_dir: Path
) -> None:
    plt, sns = _require_plotting()
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
