from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.domain import INDICES_BLOCK, SimulationError, build_asset_block_index_map
from algo_trader.infrastructure import ensure_directory
from algo_trader.infrastructure.data import symbol_directory

from . import calibration_summary_diagnostics as calibration_diags
from .diagnostics import _iter_payload_paths, _load_asset_names, _load_payload
from .residual_dependence_diagnostics import _deterministic_sample_indices

BASKET_ORDER = (
    "index_equal_weight",
    "us_index",
    "europe_index",
    "swiss_index",
    "us_minus_europe",
    "broad_mixed",
)
BASKET_FIELDS = (
    "crps",
    "quantile_loss_p05",
    "quantile_loss_p25",
    "quantile_loss_p75",
    "quantile_loss_p95",
    "coverage_p50",
    "coverage_p90",
    "coverage_p95",
    "pit_uniform_rmse",
    "sharpness_p50",
    "sharpness_p90",
    "sharpness_p95",
    "n_assets",
    "n_time",
)
_QUANTILE_LEVELS = (0.05, 0.25, 0.75, 0.95)
_COVERAGE_LEVELS = (0.5, 0.9, 0.95)
_PIT_BIN_COUNT = 20
_TRUE_TOL_ABS = 1e-8
_TRUE_TOL_REL = 1e-6
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TICKERS_CONFIG_PATH = _REPO_ROOT / "config" / "tickers.yml"
_US_INDEX_SYMBOLS = ("IBUS30", "IBUS500", "IBUST100")
_EUROPE_INDEX_SYMBOLS = (
    "IBDE40",
    "IBES35",
    "IBEU50",
    "IBFR40",
    "IBGB100",
    "IBNL25",
)
_SWISS_INDEX_SYMBOLS = ("IBCH20",)
_BROAD_MIXED_SYMBOLS = (
    "IBUS500",
    "IBEU50",
    "IBCH20",
    "EUR.USD",
    "USD.JPY",
    "XAUUSD",
)


@dataclass(frozen=True)
class BasketSpec:
    name: str
    weights: np.ndarray
    requested_assets: tuple[str, ...]
    assets_present: tuple[str, ...]
    assets_missing: tuple[str, ...]
    weight_by_asset: Mapping[str, float]
    status: str


@dataclass(frozen=True)
class BasketRecord:
    source: str
    true_value: float
    sample_values: np.ndarray


@dataclass(frozen=True)
class BasketHistogram:
    bin_left: np.ndarray
    bin_right: np.ndarray
    probability: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class BasketDiagnosticsResult:
    scores: Mapping[str, Mapping[str, float]]
    basket_specs: Mapping[str, BasketSpec]
    histograms: Mapping[str, BasketHistogram]


@dataclass
class BasketSeriesBuffers:
    pit_values: list[float]
    crps_values: list[float]
    ql_values: dict[float, list[float]]
    coverage_values: dict[float, list[float]]
    sharpness_values: dict[float, list[float]]


def compute_candidate_basket_diagnostics(
    *,
    asset_names: Sequence[str],
    payloads: Sequence[Mapping[str, Any]],
) -> BasketDiagnosticsResult:
    specs = _build_basket_specs(asset_names)
    scores: dict[str, Mapping[str, float]] = {}
    histograms: dict[str, BasketHistogram] = {}
    for basket_name in BASKET_ORDER:
        spec = specs[basket_name]
        pooled = _pooled_basket_series(payloads=payloads, spec=spec)
        scores[basket_name] = _summarize_basket_values(spec=spec, pooled=pooled)
        histograms[basket_name] = _build_histogram(pooled["pit_values"])
    return BasketDiagnosticsResult(
        scores=scores,
        basket_specs=specs,
        histograms=histograms,
    )


def write_postprocess_basket_diagnostics(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> None:
    result = _compute_outer_basket_diagnostics(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=candidate_id,
    )
    output_dir = _ensure_output_dir(
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "basket_diagnostics"
    )
    _write_outputs(
        output_dir=output_dir,
        payload=_selection_payload(
            result=result,
            candidate_id=candidate_id,
            outer_k=outer_k,
        ),
        result=result,
        global_scope=False,
    )


def write_global_basket_diagnostics(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
) -> None:
    results = [
        _compute_outer_basket_diagnostics(
            base_dir=base_dir,
            outer_k=int(outer_k),
            candidate_id=candidate_id,
        )
        for outer_k in outer_ids
    ]
    aggregate = _aggregate_results(results)
    output_dir = _ensure_output_dir(base_dir / "outer" / "diagnostics" / "basket_diagnostics")
    _write_outputs(
        output_dir=output_dir,
        payload=_selection_payload(
            result=aggregate,
            candidate_id=candidate_id,
            outer_ids=outer_ids,
        ),
        result=aggregate,
        global_scope=True,
        outer_ids=outer_ids,
    )


def _compute_outer_basket_diagnostics(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> BasketDiagnosticsResult:
    asset_names = _load_asset_names(base_dir)
    payloads = [
        _load_payload(path)
        for path in _iter_payload_paths(base_dir, [int(outer_k)], int(candidate_id))
    ]
    if not payloads:
        raise SimulationError(
            "No postprocess payloads found for basket diagnostics",
            context={"candidate_id": str(candidate_id), "outer_k": str(outer_k)},
        )
    return compute_candidate_basket_diagnostics(
        asset_names=asset_names,
        payloads=payloads,
    )


def _selection_payload(
    *,
    result: BasketDiagnosticsResult,
    candidate_id: int,
    outer_k: int | None = None,
    outer_ids: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "best_candidate_id": int(candidate_id),
        "baskets": _serialize_basket_entries(result),
        "basket_definitions": _serialize_basket_specs(result.basket_specs),
        "basket_diagnostics": _serialize_scores(result.scores),
    }
    if outer_k is not None:
        payload["scope"] = "outer_postprocess_selection"
        payload["outer_k"] = int(outer_k)
        return payload
    payload["scope"] = "global_selection"
    payload["aggregation"] = "median_over_outer_folds"
    payload["outer_ids"] = [int(item) for item in (outer_ids or ())]
    return payload


def _build_basket_specs(asset_names: Sequence[str]) -> Mapping[str, BasketSpec]:
    names = tuple(str(name) for name in asset_names)
    block_map = build_asset_block_index_map(names)
    us_index_names = _canonicalize_requested_assets(_US_INDEX_SYMBOLS)
    europe_index_names = _canonicalize_requested_assets(_EUROPE_INDEX_SYMBOLS)
    swiss_index_names = _canonicalize_requested_assets(_SWISS_INDEX_SYMBOLS)
    broad_mixed_names = _canonicalize_requested_assets(_BROAD_MIXED_SYMBOLS)
    specs = {
        "index_equal_weight": _equal_weight_spec(
            name="index_equal_weight",
            asset_names=names,
            requested=tuple(names[idx] for idx in block_map[INDICES_BLOCK]),
        ),
        "us_index": _equal_weight_spec(
            name="us_index",
            asset_names=names,
            requested=us_index_names,
        ),
        "europe_index": _equal_weight_spec(
            name="europe_index",
            asset_names=names,
            requested=europe_index_names,
        ),
        "swiss_index": _equal_weight_spec(
            name="swiss_index",
            asset_names=names,
            requested=swiss_index_names,
        ),
        "us_minus_europe": _spread_spec(
            name="us_minus_europe",
            asset_names=names,
            long_assets=us_index_names,
            short_assets=europe_index_names,
        ),
        "broad_mixed": _equal_weight_spec(
            name="broad_mixed",
            asset_names=names,
            requested=broad_mixed_names,
        ),
    }
    return specs


@lru_cache(maxsize=1)
def _ticker_symbol_aliases() -> Mapping[str, str]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG_PATH)
    aliases: dict[str, str] = {}
    for ticker in config.tickers:
        canonical_name = symbol_directory(ticker)
        aliases[ticker.symbol] = canonical_name
        aliases[canonical_name] = canonical_name
    return aliases


def _canonicalize_requested_assets(requested: Sequence[str]) -> tuple[str, ...]:
    aliases = _ticker_symbol_aliases()
    return tuple(str(aliases.get(asset, asset)) for asset in requested)


def _equal_weight_spec(
    *,
    name: str,
    asset_names: Sequence[str],
    requested: Sequence[str],
) -> BasketSpec:
    present, missing = _resolve_requested_assets(asset_names, requested)
    weights = np.zeros(len(asset_names), dtype=float)
    weight = 0.0
    is_available = _is_available_equal_weight(requested=requested, missing=missing)
    if is_available:
        weight = 1.0 / float(len(present))
        for asset in present:
            weights[asset_names.index(asset)] = weight
    return BasketSpec(
        name=name,
        weights=weights,
        requested_assets=tuple(str(asset) for asset in requested),
        assets_present=present,
        assets_missing=missing,
        weight_by_asset=(
            {asset: float(weight) for asset in present}
            if is_available
            else {}
        ),
        status=_basket_status(is_available),
    )


def _spread_spec(
    *,
    name: str,
    asset_names: Sequence[str],
    long_assets: Sequence[str],
    short_assets: Sequence[str],
) -> BasketSpec:
    long_present, long_missing = _resolve_requested_assets(asset_names, long_assets)
    short_present, short_missing = _resolve_requested_assets(asset_names, short_assets)
    weights = np.zeros(len(asset_names), dtype=float)
    is_available = _is_available_spread(
        long_assets=long_assets,
        short_assets=short_assets,
        long_missing=long_missing,
        short_missing=short_missing,
    )
    if is_available:
        long_weight = 0.5 / float(len(long_present))
        short_weight = -0.5 / float(len(short_present))
        for asset in long_present:
            weights[asset_names.index(asset)] = long_weight
        for asset in short_present:
            weights[asset_names.index(asset)] = short_weight
    return BasketSpec(
        name=name,
        weights=weights,
        requested_assets=tuple([*long_assets, *short_assets]),
        assets_present=long_present + short_present,
        assets_missing=long_missing + short_missing,
        weight_by_asset=(
            _spread_weight_map(
                long_assets=long_present,
                short_assets=short_present,
            )
            if is_available
            else {}
        ),
        status=_basket_status(is_available),
    )


def _is_available_equal_weight(
    *, requested: Sequence[str], missing: Sequence[str]
) -> bool:
    return bool(requested) and not missing


def _is_available_spread(
    *,
    long_assets: Sequence[str],
    short_assets: Sequence[str],
    long_missing: Sequence[str],
    short_missing: Sequence[str],
) -> bool:
    return (
        bool(long_assets)
        and bool(short_assets)
        and not long_missing
        and not short_missing
    )


def _basket_status(is_available: bool) -> str:
    return "ok" if is_available else "unavailable"


def _spread_weight_map(
    *, long_assets: Sequence[str], short_assets: Sequence[str]
) -> Mapping[str, float]:
    if not long_assets or not short_assets:
        return {}
    return {
        **{asset: float(0.5 / float(len(long_assets))) for asset in long_assets},
        **{asset: float(-0.5 / float(len(short_assets))) for asset in short_assets},
    }


def _resolve_requested_assets(
    asset_names: Sequence[str], requested: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = set(asset_names)
    present = tuple(asset for asset in requested if asset in available)
    missing = tuple(asset for asset in requested if asset not in available)
    return present, missing


def _pooled_basket_series(
    *, payloads: Sequence[Mapping[str, Any]], spec: BasketSpec
) -> Mapping[str, np.ndarray]:
    if spec.status != "ok" or not np.any(np.abs(spec.weights) > 0.0):
        return _empty_pooled_basket_series()
    records_by_time = _collect_records_by_time(payloads=payloads, weights=spec.weights)
    truth: list[float] = []
    sample_mean: list[float] = []
    buffers = _init_series_buffers()
    for time_idx in sorted(records_by_time):
        pooled_true, pooled_samples = _pool_records(records_by_time[time_idx])
        truth.append(float(pooled_true))
        sample_mean.append(float(np.mean(pooled_samples)))
        _append_time_metrics(buffers=buffers, true_value=pooled_true, samples=pooled_samples)
    return {
        "truth": np.asarray(truth, dtype=float),
        "samples_mean": np.asarray(sample_mean, dtype=float),
        "pit_values": np.asarray(buffers.pit_values, dtype=float),
        "crps_values": np.asarray(buffers.crps_values, dtype=float),
        "ql_p05": np.asarray(buffers.ql_values[0.05], dtype=float),
        "ql_p25": np.asarray(buffers.ql_values[0.25], dtype=float),
        "ql_p75": np.asarray(buffers.ql_values[0.75], dtype=float),
        "ql_p95": np.asarray(buffers.ql_values[0.95], dtype=float),
        "coverage_p50": np.asarray(buffers.coverage_values[0.5], dtype=float),
        "coverage_p90": np.asarray(buffers.coverage_values[0.9], dtype=float),
        "coverage_p95": np.asarray(buffers.coverage_values[0.95], dtype=float),
        "sharpness_p50": np.asarray(buffers.sharpness_values[0.5], dtype=float),
        "sharpness_p90": np.asarray(buffers.sharpness_values[0.9], dtype=float),
        "sharpness_p95": np.asarray(buffers.sharpness_values[0.95], dtype=float),
    }


def _empty_pooled_basket_series() -> Mapping[str, np.ndarray]:
    empty = np.asarray([], dtype=float)
    return {
        "truth": empty,
        "samples_mean": empty,
        "pit_values": empty,
        "crps_values": empty,
        "ql_p05": empty,
        "ql_p25": empty,
        "ql_p75": empty,
        "ql_p95": empty,
        "coverage_p50": empty,
        "coverage_p90": empty,
        "coverage_p95": empty,
        "sharpness_p50": empty,
        "sharpness_p90": empty,
        "sharpness_p95": empty,
    }


def _init_series_buffers() -> BasketSeriesBuffers:
    return BasketSeriesBuffers(
        pit_values=[],
        crps_values=[],
        ql_values={alpha: [] for alpha in _QUANTILE_LEVELS},
        coverage_values={level: [] for level in _COVERAGE_LEVELS},
        sharpness_values={level: [] for level in _COVERAGE_LEVELS},
    )


def _append_time_metrics(
    *, buffers: BasketSeriesBuffers, true_value: float, samples: np.ndarray
) -> None:
    buffers.pit_values.append(_pit_value(true_value, samples))
    buffers.crps_values.append(_scalar_crps(true_value, samples))
    for alpha in _QUANTILE_LEVELS:
        buffers.ql_values[alpha].append(
            _scalar_quantile_loss(true_value, samples, alpha)
        )
    for level in _COVERAGE_LEVELS:
        lower_q, upper_q = _coverage_bounds(samples, level)
        buffers.coverage_values[level].append(float(lower_q <= true_value <= upper_q))
        buffers.sharpness_values[level].append(float(upper_q - lower_q))


def _collect_records_by_time(
    *, payloads: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> dict[int, list[BasketRecord]]:
    records_by_time: dict[int, list[BasketRecord]] = {}
    for index, payload in enumerate(payloads):
        basket_true, basket_samples, test_idx = _basket_payload(payload=payload, weights=weights)
        for pos, time_idx in enumerate(test_idx):
            records_by_time.setdefault(int(time_idx), []).append(
                BasketRecord(
                    source=f"payload_{index}",
                    true_value=float(basket_true[pos]),
                    sample_values=np.take(basket_samples, pos, axis=1),
                )
            )
    return records_by_time


def _basket_payload(
    *, payload: Mapping[str, Any], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Sequence[Any]]:
    z_true = _require_tensor_2d(payload, "z_true")
    z_samples = _require_tensor_3d(payload, "z_samples")
    scale = _require_tensor_1d(payload, "scale")
    test_idx = payload.get("test_idx")
    if not isinstance(test_idx, Sequence):
        raise SimulationError("Postprocess payload missing test_idx")
    y_true = z_true * np.expand_dims(scale, axis=0)
    y_samples = z_samples * np.expand_dims(np.expand_dims(scale, axis=0), axis=0)
    basket_true = y_true @ weights
    basket_samples = np.asarray(
        np.tensordot(y_samples, weights, axes=([2], [0])),
        dtype=float,
    )
    return basket_true, basket_samples, test_idx


def _require_tensor_1d(payload: Mapping[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError("Postprocess payload missing tensor", context={"key": key})
    array = value.detach().cpu().numpy()
    if array.ndim != 1:
        raise SimulationError("Postprocess tensor must be 1D", context={"key": key})
    return np.asarray(array, dtype=float)


def _require_tensor_2d(payload: Mapping[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError("Postprocess payload missing tensor", context={"key": key})
    array = value.detach().cpu().numpy()
    if array.ndim != 2:
        raise SimulationError("Postprocess tensor must be 2D", context={"key": key})
    return np.asarray(array, dtype=float)


def _require_tensor_3d(payload: Mapping[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError("Postprocess payload missing tensor", context={"key": key})
    array = value.detach().cpu().numpy()
    if array.ndim != 3:
        raise SimulationError("Postprocess tensor must be 3D", context={"key": key})
    return np.asarray(array, dtype=float)


def _pool_records(records: Sequence[BasketRecord]) -> tuple[float, np.ndarray]:
    truth = _resolve_consistent_truth(records)
    target_samples = min(int(record.sample_values.shape[0]) for record in records)
    pooled = [
        record.sample_values[_deterministic_sample_indices(record.sample_values.shape[0], target_samples)]
        for record in records
    ]
    return truth, np.concatenate(pooled, axis=0)


def _resolve_consistent_truth(records: Sequence[BasketRecord]) -> float:
    values = np.asarray([record.true_value for record in records], dtype=float)
    reference = float(np.nanmedian(values))
    tolerance = _TRUE_TOL_ABS + _TRUE_TOL_REL * max(1.0, abs(reference))
    if float(np.nanmax(np.abs(values - reference))) > tolerance:
        raise SimulationError(
            "Inconsistent basket truth across diagnostic records",
            context={"sources": ", ".join(record.source for record in records)},
        )
    return reference


def _pit_value(true_value: float, samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0 or not np.isfinite(true_value):
        return float("nan")
    return float(np.mean(finite <= true_value))


def _scalar_crps(true_value: float, samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0 or not np.isfinite(true_value):
        return float("nan")
    term1 = np.mean(np.abs(finite - true_value))
    diff = finite[:, None] - finite[None, :]
    term2 = 0.5 * np.mean(np.abs(diff))
    return float(term1 - term2)


def _scalar_quantile_loss(true_value: float, samples: np.ndarray, alpha: float) -> float:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0 or not np.isfinite(true_value):
        return float("nan")
    q_hat = float(np.quantile(finite, alpha))
    error = true_value - q_hat
    indicator = 1.0 if error < 0.0 else 0.0
    return float((alpha - indicator) * error)


def _coverage_bounds(samples: np.ndarray, level: float) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - float(level)) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def _summarize_basket_values(
    *, spec: BasketSpec, pooled: Mapping[str, np.ndarray]
) -> Mapping[str, float]:
    if spec.status != "ok":
        return {field: float("nan") for field in BASKET_FIELDS}
    pit_values = pooled["pit_values"]
    return {
        "crps": _mean_or_nan(pooled["crps_values"]),
        "quantile_loss_p05": _mean_or_nan(pooled["ql_p05"]),
        "quantile_loss_p25": _mean_or_nan(pooled["ql_p25"]),
        "quantile_loss_p75": _mean_or_nan(pooled["ql_p75"]),
        "quantile_loss_p95": _mean_or_nan(pooled["ql_p95"]),
        "coverage_p50": _mean_or_nan(pooled["coverage_p50"]),
        "coverage_p90": _mean_or_nan(pooled["coverage_p90"]),
        "coverage_p95": _mean_or_nan(pooled["coverage_p95"]),
        "pit_uniform_rmse": _pit_rmse_or_nan(pit_values),
        "sharpness_p50": _median_or_nan(pooled["sharpness_p50"]),
        "sharpness_p90": _median_or_nan(pooled["sharpness_p90"]),
        "sharpness_p95": _median_or_nan(pooled["sharpness_p95"]),
        "n_assets": float(len(spec.assets_present)),
        "n_time": float(np.isfinite(pit_values).sum()),
    }


def _mean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _median_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _pit_rmse_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    summary = calibration_diags.pit_uniform_rmse(finite, bin_count=_PIT_BIN_COUNT)
    return float(summary[0])


def _build_histogram(values: np.ndarray) -> BasketHistogram:
    finite = values[np.isfinite(values)]
    edges = np.linspace(0.0, 1.0, num=_PIT_BIN_COUNT + 1, dtype=float)
    if finite.size == 0:
        zeros = np.zeros(_PIT_BIN_COUNT, dtype=float)
        return BasketHistogram(
            bin_left=edges[:-1],
            bin_right=edges[1:],
            probability=zeros,
            count=zeros,
        )
    clipped = np.clip(finite, 0.0, np.nextafter(1.0, 0.0))
    scaled = np.floor(clipped * float(_PIT_BIN_COUNT)).astype(np.int64)
    counts = np.bincount(scaled, minlength=_PIT_BIN_COUNT).astype(float)
    probability = counts.astype(float) / float(finite.size)
    return BasketHistogram(
        bin_left=edges[:-1],
        bin_right=edges[1:],
        probability=probability,
        count=counts,
    )


def _aggregate_results(results: Sequence[BasketDiagnosticsResult]) -> BasketDiagnosticsResult:
    first = results[0]
    return BasketDiagnosticsResult(
        scores=_aggregate_scores([result.scores for result in results]),
        basket_specs=first.basket_specs,
        histograms=_aggregate_histograms([result.histograms for result in results]),
    )


def _aggregate_scores(
    scores_list: Sequence[Mapping[str, Mapping[str, float]]]
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for basket_name in BASKET_ORDER:
        result[basket_name] = {
            field: _median_float(
                [
                    float(scores.get(basket_name, {}).get(field, float("nan")))
                    for scores in scores_list
                ]
            )
            for field in BASKET_FIELDS
        }
    return result


def _aggregate_histograms(
    histograms_list: Sequence[Mapping[str, BasketHistogram]]
) -> Mapping[str, BasketHistogram]:
    result: dict[str, BasketHistogram] = {}
    for basket_name in BASKET_ORDER:
        source = [histograms[basket_name] for histograms in histograms_list]
        result[basket_name] = BasketHistogram(
            bin_left=source[0].bin_left,
            bin_right=source[0].bin_right,
            probability=np.nanmedian(
                np.stack([item.probability for item in source], axis=0),
                axis=0,
            ),
            count=np.nanmedian(
                np.stack([item.count for item in source], axis=0),
                axis=0,
            ),
        )
    return result


def _median_float(values: Sequence[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.median(np.asarray(finite, dtype=float)))


def _serialize_scores(
    scores: Mapping[str, Mapping[str, float]]
) -> Mapping[str, Mapping[str, float]]:
    return {
        basket_name: {
            field: float(scores.get(basket_name, {}).get(field, float("nan")))
            for field in BASKET_FIELDS
        }
        for basket_name in BASKET_ORDER
    }


def _serialize_basket_specs(
    basket_specs: Mapping[str, BasketSpec]
) -> Mapping[str, Mapping[str, Any]]:
    return {
        basket_name: {
            "status": spec.status,
            "requested_assets": list(spec.requested_assets),
            "assets_present": list(spec.assets_present),
            "assets_missing": list(spec.assets_missing),
            "weights": {
                asset_name: float(weight)
                for asset_name, weight in spec.weight_by_asset.items()
            },
        }
        for basket_name, spec in basket_specs.items()
    }


def _serialize_basket_entries(
    result: BasketDiagnosticsResult,
) -> Mapping[str, Mapping[str, Any]]:
    basket_specs = _serialize_basket_specs(result.basket_specs)
    basket_scores = _serialize_scores(result.scores)
    return {
        basket_name: {
            **basket_specs[basket_name],
            **basket_scores[basket_name],
        }
        for basket_name in BASKET_ORDER
    }


def _ensure_output_dir(path: Path) -> Path:
    ensure_directory(
        path,
        error_type=SimulationError,
        invalid_message="Basket diagnostics output path is not a directory",
        create_message="Failed to create basket diagnostics output",
        context={"path": str(path)},
    )
    return path


def _write_outputs(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    result: BasketDiagnosticsResult,
    global_scope: bool,
    outer_ids: Sequence[int] | None = None,
) -> None:
    _write_json(output_dir / "basket_scores.json", payload)
    _write_score_csv(
        output_dir / "basket_scores.csv",
        result.scores,
        result.basket_specs,
    )
    for basket_name in BASKET_ORDER:
        if result.basket_specs[basket_name].status != "ok":
            continue
        _write_histogram_csv(
            path=output_dir / f"pit_histogram_{basket_name}.csv",
            histogram=result.histograms[basket_name],
            global_scope=global_scope,
            outer_count=(len(outer_ids) if outer_ids is not None else None),
        )
    if global_scope:
        _write_json(
            output_dir / "aggregate_manifest.json",
            {
                "scope": "selected_candidate_basket_diagnostics",
                "aggregation": "median_over_outer_folds",
                "outer_ids": [int(item) for item in (outer_ids or ())],
                "pit_bin_count": _PIT_BIN_COUNT,
                "baskets": list(BASKET_ORDER),
            },
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_score_csv(
    path: Path,
    scores: Mapping[str, Mapping[str, float]],
    basket_specs: Mapping[str, BasketSpec],
) -> None:
    rows = ["basket,status," + ",".join(BASKET_FIELDS)]
    for basket_name in BASKET_ORDER:
        values = scores.get(basket_name, {})
        status = basket_specs[basket_name].status
        fields = [str(float(values.get(field, float("nan")))) for field in BASKET_FIELDS]
        rows.append(",".join([basket_name, status, *fields]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_histogram_csv(
    *,
    path: Path,
    histogram: BasketHistogram,
    global_scope: bool,
    outer_count: int | None,
) -> None:
    if global_scope:
        header = "bin_left,bin_right,median_probability,median_count,n_outer_folds"
        rows = [
            ",".join(
                [
                    str(float(left)),
                    str(float(right)),
                    str(float(probability)),
                    str(float(count)),
                    str(int(outer_count or 0)),
                ]
            )
            for left, right, probability, count in zip(
                histogram.bin_left,
                histogram.bin_right,
                histogram.probability,
                histogram.count,
                strict=True,
            )
        ]
    else:
        header = "bin_left,bin_right,probability,count"
        rows = [
            ",".join(
                [
                    str(float(left)),
                    str(float(right)),
                    str(float(probability)),
                    str(int(count)),
                ]
            )
            for left, right, probability, count in zip(
                histogram.bin_left,
                histogram.bin_right,
                histogram.probability,
                histogram.count,
                strict=True,
            )
        ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
