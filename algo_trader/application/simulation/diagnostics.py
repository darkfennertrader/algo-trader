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
class SplitPayload:
    z_true: np.ndarray
    z_samples: np.ndarray
    test_groups: np.ndarray


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
    split_payloads = _collect_split_payloads(
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
        output_dir=calibration_dir,
    )
    _run_calibration_diagnostics(
        split_payloads=split_payloads,
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
    samples_by_time: Mapping[int, list[np.ndarray]]
    true_by_time: Mapping[int, np.ndarray]


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
    samples_by_time: dict[int, list[np.ndarray]] = {}
    true_by_time: dict[int, np.ndarray] = {}
    total = 0
    for path in _iter_payload_paths(base_dir, outer_ids, candidate_id):
        payload = _load_payload(path)
        _append_payload(
            payload=payload,
            asset_indices=asset_indices,
            global_scale=global_scale,
            samples_by_time=samples_by_time,
            true_by_time=true_by_time,
        )
        total += 1
    if total == 0:
        raise SimulationError(
            "No postprocess payloads found for diagnostics",
            context={"candidate_id": str(candidate_id)},
        )
    return CandidateSamples(
        samples_by_time=samples_by_time,
        true_by_time=true_by_time,
    )


def _collect_split_payloads(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
    asset_indices: Sequence[int],
    global_scale: np.ndarray,
) -> list[SplitPayload]:
    payloads: list[SplitPayload] = []
    for path in _iter_payload_paths(base_dir, outer_ids, candidate_id):
        raw = _load_payload(path)
        payloads.append(
            _build_split_payload(
                raw=raw,
                asset_indices=asset_indices,
                global_scale=global_scale,
            )
        )
    if not payloads:
        raise SimulationError(
            "No postprocess payloads found for diagnostics",
            context={"candidate_id": str(candidate_id)},
        )
    return payloads


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
    samples_by_time: dict[int, list[np.ndarray]],
    true_by_time: dict[int, np.ndarray],
) -> None:
    z_true = _require_tensor(payload, "z_true")
    z_samples = _require_tensor(payload, "z_samples")
    scale = _require_tensor(payload, "scale")
    test_idx = payload.get("test_idx")
    if not isinstance(test_idx, Sequence):
        raise SimulationError("Postprocess payload missing test_idx")
    scale_np = scale.detach().cpu().numpy()[asset_indices]
    z_true_np = z_true[:, asset_indices].detach().cpu().numpy()
    z_samples_np = z_samples[:, :, asset_indices].detach().cpu().numpy()
    z_true_np = (z_true_np * scale_np) / global_scale
    z_samples_np = (z_samples_np * scale_np) / global_scale
    if z_true_np.shape[0] != len(test_idx):
        raise SimulationError("Postprocess payload misaligned on T")
    for pos, time_idx in enumerate(test_idx):
        key = int(time_idx)
        samples_by_time.setdefault(key, []).append(z_samples_np[:, pos, :])
        if key not in true_by_time:
            true_by_time[key] = z_true_np[pos]


def _build_split_payload(
    *,
    raw: Mapping[str, Any],
    asset_indices: Sequence[int],
    global_scale: np.ndarray,
) -> SplitPayload:
    z_true = _require_tensor(raw, "z_true")
    z_samples = _require_tensor(raw, "z_samples")
    scale = _require_tensor(raw, "scale")
    test_groups = raw.get("test_groups")
    if not isinstance(test_groups, Sequence):
        raise SimulationError("Postprocess payload missing test_groups")
    scale_np = scale.detach().cpu().numpy()[asset_indices]
    z_true_np = z_true[:, asset_indices].detach().cpu().numpy()
    z_samples_np = z_samples[:, :, asset_indices].detach().cpu().numpy()
    z_true_np = (z_true_np * scale_np) / global_scale
    z_samples_np = (z_samples_np * scale_np) / global_scale
    groups_np = np.asarray(test_groups, dtype=int)
    return SplitPayload(
        z_true=z_true_np,
        z_samples=z_samples_np,
        test_groups=groups_np,
    )


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
    time_indices = sorted(samples.samples_by_time.keys())
    if not time_indices:
        raise SimulationError("No test weeks available for diagnostics")
    z_true = _build_true_matrix(
        time_indices=time_indices,
        true_by_time=samples.true_by_time,
        asset_count=len(asset_names),
    )
    quantiles_map = _build_quantiles_matrix(
        time_indices=time_indices,
        samples_by_time=samples.samples_by_time,
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
    true_by_time: Mapping[int, np.ndarray],
    asset_count: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for time_idx in time_indices:
        row = true_by_time.get(time_idx)
        if row is None:
            rows.append(np.full(asset_count, np.nan))
        else:
            rows.append(row)
    return np.vstack(rows)


def _build_quantiles_matrix(
    *,
    time_indices: Sequence[int],
    samples_by_time: Mapping[int, list[np.ndarray]],
    quantiles: Sequence[float],
    asset_count: int,
) -> dict[float, np.ndarray]:
    quantile_map: dict[float, np.ndarray] = {
        q: np.full((len(time_indices), asset_count), np.nan)
        for q in quantiles
    }
    for row, time_idx in enumerate(time_indices):
        samples_list = samples_by_time.get(time_idx, [])
        if not samples_list:
            continue
        samples = np.concatenate(samples_list, axis=0)
        for q in quantiles:
            quantile_map[q][row] = np.nanquantile(
                samples, float(q), axis=0
            )
    return quantile_map


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
    target_dir = base_dir / "outer" / "diagnostics" / "calibration"
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
    output_dir: Path,
) -> None:
    plt, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    series = _collect_coverage_series(data, coverage_levels)
    _write_calibration_csv(
        data=data,
        series=series,
        output_dir=output_dir,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for item in series:
        sns.lineplot(
            x=data.timestamps,
            y=item.coverage,
            ax=ax,
            label=f"coverage {item.level:.2f}",
        )
        ax.fill_between(
            data.timestamps,
            item.band[0],
            item.band[1],
            alpha=0.1,
        )
        ax.axhline(
            item.level, color="black", linestyle="--", linewidth=1
        )
    ax.set_title("Calibration coverage")
    ax.set_xlabel("test week")
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        output_dir / "calibration_fan.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def _run_calibration_diagnostics(
    *,
    split_payloads: Sequence[SplitPayload],
    coverage_levels: Sequence[float],
    output_dir: Path,
) -> None:
    if not split_payloads:
        return
    group_count = _infer_group_count(split_payloads)
    asset_count = split_payloads[0].z_true.shape[1]
    pit_group = _aggregate_pit_groups(
        split_payloads=split_payloads,
        group_count=group_count,
        asset_count=asset_count,
    )
    _plot_pit_histogram(pit_group, output_dir)
    coverage_groups = _aggregate_coverage_groups(
        split_payloads=split_payloads,
        group_count=group_count,
        asset_count=asset_count,
        coverage_levels=coverage_levels,
    )
    curve = _coverage_curve_from_groups(coverage_groups)
    _plot_coverage_curve(curve, output_dir)


@dataclass(frozen=True)
class CoverageSeries:
    level: float
    coverage: np.ndarray
    band: tuple[np.ndarray, np.ndarray]


def _collect_coverage_series(
    data: FanChartData, coverage_levels: Sequence[float]
) -> list[CoverageSeries]:
    series: list[CoverageSeries] = []
    for level in coverage_levels:
        coverage, band = _coverage_series(data, float(level))
        series.append(
            CoverageSeries(
                level=float(level),
                coverage=coverage,
                band=band,
            )
        )
    return series


def _write_calibration_csv(
    *,
    data: FanChartData,
    series: Sequence[CoverageSeries],
    output_dir: Path,
) -> None:
    frame = _build_calibration_frame(data, series)
    frame.to_csv(output_dir / "calibration_fan.csv", index=False)


def _build_calibration_frame(
    data: FanChartData, series: Sequence[CoverageSeries]
) -> pd.DataFrame:
    rows: dict[str, list[float] | list[str]] = {
        "timestamp": [
            stamp.strftime("%Y-%m-%d %H:%M:%S UTC")
            for stamp in data.timestamps
        ]
    }
    for item in series:
        label = f"p{int(round(item.level * 100)):02d}"
        rows[f"coverage_{label}"] = list(item.coverage)
        rows[f"band_low_{label}"] = list(item.band[0])
        rows[f"band_high_{label}"] = list(item.band[1])
    return pd.DataFrame(rows)


def _fan_bands(quantiles: Sequence[float]) -> list[tuple[float, float, float, str]]:
    bands = [
        (0.05, 0.95, 0.15, "tab:blue"),
        (0.10, 0.90, 0.25, "tab:blue"),
        (0.25, 0.75, 0.35, "tab:blue"),
    ]
    available = set(float(q) for q in quantiles)
    return [band for band in bands if band[0] in available and band[1] in available]


def _coverage_series(
    data: FanChartData, level: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    lo_key = _match_quantile_key(data.quantiles, lo)
    hi_key = _match_quantile_key(data.quantiles, hi)
    if lo_key is None or hi_key is None:
        raise SimulationError(
            "Coverage quantiles missing",
            context={"level": str(level)},
        )
    coverage, counts = _compute_coverage(
        data=data,
        q_lo=data.quantiles[lo_key],
        q_hi=data.quantiles[hi_key],
    )
    band = _compute_coverage_band(coverage, counts)
    return coverage, band


def _compute_coverage(
    *, data: FanChartData, q_lo: np.ndarray, q_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    z_true = data.z_true
    mask = np.isfinite(z_true) & np.isfinite(q_lo) & np.isfinite(q_hi)
    covered = (z_true >= q_lo) & (z_true <= q_hi) & mask
    counts = mask.sum(axis=1)
    covered_counts = covered.sum(axis=1)
    coverage = np.divide(
        covered_counts,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    return coverage, counts


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


def _infer_group_count(split_payloads: Sequence[SplitPayload]) -> int:
    max_group = -1
    for payload in split_payloads:
        if payload.test_groups.size:
            max_group = max(max_group, int(payload.test_groups.max()))
    if max_group < 0:
        raise SimulationError("Calibration payloads missing groups")
    return max_group + 1


def _aggregate_pit_groups(
    *,
    split_payloads: Sequence[SplitPayload],
    group_count: int,
    asset_count: int,
) -> np.ndarray:
    sums = np.zeros((group_count, asset_count), dtype=float)
    counts = np.zeros((group_count, asset_count), dtype=float)
    for payload in split_payloads:
        pit = _compute_pit(payload.z_samples, payload.z_true)
        group_means, present = _mean_by_group(
            values=pit,
            group_ids=payload.test_groups,
            group_count=group_count,
        )
        sums = np.where(present, sums + group_means, sums)
        counts = np.where(present, counts + 1.0, counts)
    return _safe_divide(sums, counts)


def _aggregate_coverage_groups(
    *,
    split_payloads: Sequence[SplitPayload],
    group_count: int,
    asset_count: int,
    coverage_levels: Sequence[float],
) -> Mapping[float, np.ndarray]:
    sums: dict[float, np.ndarray] = {}
    counts: dict[float, np.ndarray] = {}
    for level in coverage_levels:
        sums[level] = np.zeros((group_count, asset_count), dtype=float)
        counts[level] = np.zeros((group_count, asset_count), dtype=float)
    for payload in split_payloads:
        for level in coverage_levels:
            indicator = _compute_coverage_indicator(
                payload.z_samples, payload.z_true, float(level)
            )
            group_means, present = _mean_by_group(
                values=indicator,
                group_ids=payload.test_groups,
                group_count=group_count,
            )
            sums[level] = np.where(
                present, sums[level] + group_means, sums[level]
            )
            counts[level] = np.where(
                present, counts[level] + 1.0, counts[level]
            )
    return {
        level: _safe_divide(sums[level], counts[level])
        for level in coverage_levels
    }


def _compute_pit(z_samples: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    valid = np.isfinite(z_true) & np.isfinite(z_samples).all(axis=0)
    pit = (z_samples <= z_true[None, :, :]).mean(axis=0)
    pit[~valid] = np.nan
    return pit


def _compute_coverage_indicator(
    z_samples: np.ndarray, z_true: np.ndarray, level: float
) -> np.ndarray:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    lower = np.nanquantile(z_samples, lo, axis=0)
    upper = np.nanquantile(z_samples, hi, axis=0)
    valid = np.isfinite(z_true) & np.isfinite(lower) & np.isfinite(upper)
    indicator = ((z_true >= lower) & (z_true <= upper)).astype(float)
    indicator[~valid] = np.nan
    return indicator


def _mean_by_group(
    *,
    values: np.ndarray,
    group_ids: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    asset_count = values.shape[1]
    means: list[np.ndarray] = []
    present_rows: list[np.ndarray] = []
    for group in range(group_count):
        idx = group_ids == group
        if not np.any(idx):
            mean = np.full(asset_count, np.nan, dtype=float)
        else:
            mean = np.nanmean(values[idx], axis=0)
        means.append(mean)
        present_rows.append(np.isfinite(mean))
    return np.vstack(means), np.vstack(present_rows)


def _safe_divide(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )


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


def _coverage_curve_from_groups(
    coverage_groups: Mapping[float, np.ndarray],
) -> Mapping[float, float]:
    curve: dict[float, float] = {}
    for level, group_vals in coverage_groups.items():
        asset_means = np.nanmean(group_vals, axis=0)
        curve[level] = float(np.nanmean(asset_means))
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
