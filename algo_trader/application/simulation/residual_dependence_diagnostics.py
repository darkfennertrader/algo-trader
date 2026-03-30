from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from algo_trader.domain import (
    FULL_BLOCK,
    INDICES_BLOCK,
    SimulationError,
    build_asset_block_index_map,
)
from algo_trader.infrastructure import ensure_directory

from .diagnostics import (
    TRUE_TOL_ABS,
    TRUE_TOL_REL,
    _extract_scaled_payload_arrays,
    _iter_payload_paths,
    _load_asset_names,
    _load_payload,
)

_HARD_INDEX_NAMES = (
    "IBEU50",
    "IBUS30",
    "IBUS500",
    "IBCH20",
    "IBFR40",
    "IBGB100",
)
_SUMMARY_FIELDS = (
    "residual_corr_mean_abs_offdiag",
    "residual_corr_max_abs_offdiag",
    "residual_corr_fro_identity",
    "whitened_corr_mean_abs_offdiag",
    "whitened_corr_max_abs_offdiag",
    "whitened_corr_fro_identity",
    "n_assets",
    "n_time",
    "n_hard_assets_present",
    "n_hard_assets_missing",
)
_COV_EPS = 1e-8


@dataclass(frozen=True)
class BlockSampleRecord:
    source: str
    y_true: np.ndarray
    y_samples: np.ndarray


@dataclass(frozen=True)
class CorrelationMatrices:
    residual_corr: np.ndarray
    residual_counts: np.ndarray
    whitened_corr: np.ndarray
    whitened_counts: np.ndarray


@dataclass(frozen=True)
class HardAssetDetails:
    hard_assets_requested: tuple[str, ...]
    hard_assets_present: tuple[str, ...]
    hard_assets_missing: tuple[str, ...]
    pairwise_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class BlockResidualDiagnostics:
    block: str
    asset_names: tuple[str, ...]
    summary: Mapping[str, float]
    matrices: CorrelationMatrices
    hard_assets: HardAssetDetails


@dataclass(frozen=True)
class OuterResidualDependence:
    outer_k: int
    candidate_id: int
    block_results: Mapping[str, BlockResidualDiagnostics]


@dataclass(frozen=True)
class BlockResidualRequest:
    base_dir: Path
    outer_k: int
    candidate_id: int
    asset_names: tuple[str, ...]
    asset_indices: tuple[int, ...]
    block: str
    hard_assets: tuple[str, ...]


def write_postprocess_residual_dependence(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> None:
    result = _compute_outer_residual_dependence(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=candidate_id,
    )
    output_dir = _ensure_dir(
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "residual_dependence"
    )
    _write_summary_outputs(
        output_dir=output_dir,
        payload={
            "scope": "outer_postprocess_selection",
            "outer_k": int(outer_k),
            "best_candidate_id": int(candidate_id),
            "blocks": _serialize_block_results(result.block_results),
        },
        block_results=result.block_results,
    )


def write_global_residual_dependence(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    candidate_id: int,
) -> None:
    outer_results = [
        _compute_outer_residual_dependence(
            base_dir=base_dir,
            outer_k=int(outer_k),
            candidate_id=candidate_id,
        )
        for outer_k in outer_ids
    ]
    aggregate = _aggregate_outer_residual_dependence(
        outer_results=outer_results,
        candidate_id=candidate_id,
    )
    output_dir = _ensure_dir(base_dir / "outer" / "diagnostics" / "residual_dependence")
    _write_summary_outputs(
        output_dir=output_dir,
        payload={
            "scope": "global_selection",
            "aggregation": "median_over_outer_folds",
            "outer_ids": [int(item) for item in outer_ids],
            "best_candidate_id": int(candidate_id),
            "blocks": _serialize_block_results(aggregate.block_results),
        },
        block_results=aggregate.block_results,
    )
    _write_json(
        output_dir / "aggregate_manifest.json",
        {
            "scope": "selected_candidate_residual_dependence",
            "aggregation": "median_over_outer_folds",
            "outer_ids": [int(item) for item in outer_ids],
            "hard_index_names": list(_HARD_INDEX_NAMES),
            "blocks": [INDICES_BLOCK, FULL_BLOCK],
        },
    )


def _compute_outer_residual_dependence(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> OuterResidualDependence:
    asset_names = tuple(_load_asset_names(base_dir))
    block_map = build_asset_block_index_map(asset_names)
    indices_result = _compute_block_residual_dependence(
        BlockResidualRequest(
            base_dir=base_dir,
            outer_k=int(outer_k),
            candidate_id=int(candidate_id),
            asset_names=asset_names,
            asset_indices=block_map[INDICES_BLOCK],
            block=INDICES_BLOCK,
            hard_assets=_HARD_INDEX_NAMES,
        )
    )
    full_result = _compute_block_residual_dependence(
        BlockResidualRequest(
            base_dir=base_dir,
            outer_k=int(outer_k),
            candidate_id=int(candidate_id),
            asset_names=asset_names,
            asset_indices=block_map[FULL_BLOCK],
            block=FULL_BLOCK,
            hard_assets=tuple(),
        )
    )
    return OuterResidualDependence(
        outer_k=int(outer_k),
        candidate_id=int(candidate_id),
        block_results={
            INDICES_BLOCK: indices_result,
            FULL_BLOCK: full_result,
        },
    )


def _compute_block_residual_dependence(
    request: BlockResidualRequest,
) -> BlockResidualDiagnostics:
    block_asset_names = tuple(
        request.asset_names[idx] for idx in request.asset_indices
    )
    if not request.asset_indices:
        return _empty_block_residual_diagnostics(
            block=request.block,
            hard_assets=request.hard_assets,
        )
    standardized, whitened = _block_residual_series(
        request=request,
        asset_count=len(block_asset_names),
    )
    matrices = _build_correlation_matrices(
        standardized=standardized,
        whitened=whitened,
    )
    hard_assets = _build_hard_asset_details(
        block_asset_names=block_asset_names,
        matrices=matrices,
        hard_assets=request.hard_assets,
        outer_k=request.outer_k,
    )
    return BlockResidualDiagnostics(
        block=request.block,
        asset_names=block_asset_names,
        summary=_build_block_summary(
            block=request.block,
            matrices=matrices,
            n_time=int(standardized.shape[0]),
            n_assets=len(block_asset_names),
            hard_assets=hard_assets,
        ),
        matrices=matrices,
        hard_assets=hard_assets,
    )


def _collect_block_records_by_time(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    asset_indices: Sequence[int],
) -> dict[int, list[BlockSampleRecord]]:
    records_by_time: dict[int, list[BlockSampleRecord]] = {}
    for path in _iter_payload_paths(base_dir, [outer_k], candidate_id):
        payload = _load_payload(path)
        y_true, y_samples, test_idx = _extract_block_payload(
            payload=payload,
            asset_indices=asset_indices,
        )
        for pos, time_idx in enumerate(test_idx):
            records_by_time.setdefault(int(time_idx), []).append(
                BlockSampleRecord(
                    source=str(path),
                    y_true=y_true[pos],
                    y_samples=y_samples[:, pos, :],
                )
            )
    if not records_by_time:
        raise SimulationError(
            "No postprocess payloads found for residual dependence diagnostics",
            context={"candidate_id": str(candidate_id), "outer_k": str(outer_k)},
        )
    return records_by_time


def _extract_block_payload(
    *,
    payload: Mapping[str, Any],
    asset_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, Sequence[Any]]:
    selected_indices = list(asset_indices)
    ones = np.ones(len(selected_indices), dtype=float)
    return _extract_scaled_payload_arrays(
        payload=payload,
        asset_indices=selected_indices,
        global_scale=ones,
    )


@dataclass(frozen=True)
class PooledBlockSamples:
    y_true: np.ndarray
    y_samples: np.ndarray


def _block_residual_series(
    *,
    request: BlockResidualRequest,
    asset_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    records_by_time = _collect_block_records_by_time(
        base_dir=request.base_dir,
        outer_k=request.outer_k,
        candidate_id=request.candidate_id,
        asset_indices=request.asset_indices,
    )
    standardized_rows: list[np.ndarray] = []
    whitened_rows: list[np.ndarray] = []
    for time_idx in sorted(records_by_time):
        pooled = _pool_block_records(records_by_time[time_idx])
        standardized, whitened = _residual_vectors_from_pooled_samples(pooled)
        standardized_rows.append(standardized)
        whitened_rows.append(whitened)
    return (
        _stack_rows(standardized_rows, asset_count=asset_count),
        _stack_rows(whitened_rows, asset_count=asset_count),
    )


def _pool_block_records(records: Sequence[BlockSampleRecord]) -> PooledBlockSamples:
    y_true = _resolve_consistent_y_true(records)
    target_samples = min(int(record.y_samples.shape[0]) for record in records)
    pooled = [
        record.y_samples[_deterministic_sample_indices(record.y_samples.shape[0], target_samples)]
        for record in records
    ]
    return PooledBlockSamples(y_true=y_true, y_samples=np.concatenate(pooled, axis=0))


def _resolve_consistent_y_true(records: Sequence[BlockSampleRecord]) -> np.ndarray:
    stacked = np.vstack([record.y_true[None, :] for record in records])
    reference = np.asarray(np.nanmedian(stacked, axis=0), dtype=float)
    expanded = np.expand_dims(reference, axis=0)
    max_abs = np.nanmax(np.abs(stacked - expanded), axis=0)
    tolerance = TRUE_TOL_ABS + TRUE_TOL_REL * np.maximum(1.0, np.abs(reference))
    if np.any(max_abs > tolerance):
        raise SimulationError(
            "Inconsistent y_true across residual dependence diagnostic records",
            context={"sources": ", ".join(record.source for record in records)},
        )
    return reference


def _deterministic_sample_indices(num_samples: int, target_samples: int) -> np.ndarray:
    if target_samples >= num_samples:
        return np.arange(num_samples, dtype=int)
    return np.rint(np.linspace(0, num_samples - 1, num=target_samples)).astype(int)


def _residual_vectors_from_pooled_samples(
    pooled: PooledBlockSamples,
) -> tuple[np.ndarray, np.ndarray]:
    residual = pooled.y_true - np.mean(pooled.y_samples, axis=0)
    std = np.std(pooled.y_samples, axis=0, ddof=0)
    standardized = np.divide(
        residual,
        std,
        out=np.full_like(residual, np.nan, dtype=float),
        where=np.isfinite(std) & (std > 0.0),
    )
    return standardized, _whiten_residual(
        residual=residual,
        covariance=_predictive_covariance(pooled.y_samples),
    )


def _predictive_covariance(samples: np.ndarray) -> np.ndarray:
    valid_rows = np.all(np.isfinite(samples), axis=1)
    finite = samples[valid_rows]
    if finite.shape[0] < 2:
        return np.full((samples.shape[1], samples.shape[1]), np.nan, dtype=float)
    centered = finite - np.mean(finite, axis=0, keepdims=True)
    return (centered.T @ centered) / float(finite.shape[0])


def _whiten_residual(*, residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(covariance)):
        return np.full_like(residual, np.nan, dtype=float)
    inverse_sqrt = _inverse_sqrt_covariance(covariance)
    return residual @ inverse_sqrt.T


def _inverse_sqrt_covariance(covariance: np.ndarray) -> np.ndarray:
    sym = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.clip(eigvals, _COV_EPS, None)
    inv_sqrt = eigvecs @ np.diag(clipped ** -0.5) @ eigvecs.T
    return inv_sqrt


def _stack_rows(rows: Sequence[np.ndarray], *, asset_count: int) -> np.ndarray:
    if not rows:
        return np.empty((0, asset_count), dtype=float)
    return np.vstack(rows)


def _build_correlation_matrices(
    *,
    standardized: np.ndarray,
    whitened: np.ndarray,
) -> CorrelationMatrices:
    residual_corr, residual_counts = _pairwise_corr_and_counts(standardized)
    whitened_corr, whitened_counts = _pairwise_corr_and_counts(whitened)
    return CorrelationMatrices(
        residual_corr=residual_corr,
        residual_counts=residual_counts,
        whitened_corr=whitened_corr,
        whitened_counts=whitened_counts,
    )


def _pairwise_corr_and_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    asset_count = int(values.shape[1]) if values.ndim == 2 else 0
    corr_rows: list[list[float]] = [
        [float("nan")] * asset_count for _ in range(asset_count)
    ]
    counts = np.zeros((asset_count, asset_count), dtype=int)
    for left, right in itertools.product(range(asset_count), repeat=2):
        left_values = values[:, left]
        right_values = values[:, right]
        mask = np.isfinite(left_values) & np.isfinite(right_values)
        count = int(np.sum(mask))
        counts[left, right] = count
        if left == right:
            corr_rows[left][right] = 1.0 if count > 0 else float("nan")
            continue
        if count < 2:
            continue
        corr_rows[left][right] = _safe_corrcoef(left_values[mask], right_values[mask])
    return np.asarray(corr_rows, dtype=float), counts


def _safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    corr = np.asarray(np.corrcoef(left, right), dtype=float)
    return float(corr.item((0, 1)))


def _resolve_hard_assets(
    *, block_asset_names: Sequence[str], hard_assets: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present = tuple(name for name in hard_assets if name in block_asset_names)
    missing = tuple(name for name in hard_assets if name not in block_asset_names)
    return present, missing


def _build_hard_asset_details(
    *,
    block_asset_names: Sequence[str],
    matrices: CorrelationMatrices,
    hard_assets: Sequence[str],
    outer_k: int,
) -> HardAssetDetails:
    present, missing = _resolve_hard_assets(
        block_asset_names=block_asset_names,
        hard_assets=hard_assets,
    )
    return HardAssetDetails(
        hard_assets_requested=tuple(str(item) for item in hard_assets),
        hard_assets_present=present,
        hard_assets_missing=missing,
        pairwise_rows=_build_pairwise_rows(
            asset_names=block_asset_names,
            matrices=matrices,
            hard_assets=present,
            outer_k=outer_k,
        ),
    )


def _build_block_summary(
    *,
    block: str,
    matrices: CorrelationMatrices,
    n_time: int,
    n_assets: int,
    hard_assets: HardAssetDetails,
) -> Mapping[str, float]:
    return {
        "residual_corr_mean_abs_offdiag": _mean_abs_offdiag(matrices.residual_corr),
        "residual_corr_max_abs_offdiag": _max_abs_offdiag(matrices.residual_corr),
        "residual_corr_fro_identity": _fro_identity_distance(matrices.residual_corr),
        "whitened_corr_mean_abs_offdiag": _mean_abs_offdiag(matrices.whitened_corr),
        "whitened_corr_max_abs_offdiag": _max_abs_offdiag(matrices.whitened_corr),
        "whitened_corr_fro_identity": _fro_identity_distance(
            matrices.whitened_corr
        ),
        "n_assets": float(n_assets),
        "n_time": float(n_time),
        "n_hard_assets_present": (
            float(len(hard_assets.hard_assets_present))
            if block == INDICES_BLOCK
            else float("nan")
        ),
        "n_hard_assets_missing": (
            float(len(hard_assets.hard_assets_missing))
            if block == INDICES_BLOCK
            else float("nan")
        ),
    }


def _mean_abs_offdiag(matrix: np.ndarray) -> float:
    values = _finite_offdiag(matrix)
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values)))


def _max_abs_offdiag(matrix: np.ndarray) -> float:
    values = _finite_offdiag(matrix)
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _fro_identity_distance(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return float("nan")
    diff = matrix - np.eye(matrix.shape[0], dtype=float)
    if not np.isfinite(diff).any():
        return float("nan")
    return float(np.sqrt(np.nansum(diff * diff)))


def _finite_offdiag(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    values: list[float] = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[1]):
            value = float(matrix[left, right])
            if np.isfinite(value):
                values.append(value)
    return np.asarray(values, dtype=float)


def _build_pairwise_rows(
    *,
    asset_names: Sequence[str],
    matrices: CorrelationMatrices,
    hard_assets: Sequence[str],
    outer_k: int,
) -> tuple[Mapping[str, Any], ...]:
    index_map = {name: idx for idx, name in enumerate(asset_names)}
    rows: list[Mapping[str, Any]] = []
    for left_asset, right_asset in itertools.combinations(hard_assets, 2):
        left = index_map[left_asset]
        right = index_map[right_asset]
        rows.append(
            {
                "outer_k": int(outer_k),
                "left_asset": left_asset,
                "right_asset": right_asset,
                "residual_corr": float(matrices.residual_corr[left, right]),
                "abs_residual_corr": float(abs(matrices.residual_corr[left, right])),
                "whitened_residual_corr": float(matrices.whitened_corr[left, right]),
                "abs_whitened_residual_corr": float(
                    abs(matrices.whitened_corr[left, right])
                ),
                "n_obs": int(matrices.residual_counts[left, right]),
                "n_obs_whitened": int(matrices.whitened_counts[left, right]),
            }
        )
    return tuple(rows)


def _empty_block_residual_diagnostics(
    *, block: str, hard_assets: Sequence[str]
) -> BlockResidualDiagnostics:
    summary = {
        field: float("nan")
        for field in _SUMMARY_FIELDS
    }
    summary["n_assets"] = 0.0
    summary["n_time"] = 0.0
    return BlockResidualDiagnostics(
        block=block,
        asset_names=tuple(),
        summary=summary,
        matrices=CorrelationMatrices(
            residual_corr=np.empty((0, 0), dtype=float),
            residual_counts=np.empty((0, 0), dtype=int),
            whitened_corr=np.empty((0, 0), dtype=float),
            whitened_counts=np.empty((0, 0), dtype=int),
        ),
        hard_assets=HardAssetDetails(
            hard_assets_requested=tuple(str(item) for item in hard_assets),
            hard_assets_present=tuple(),
            hard_assets_missing=tuple(str(item) for item in hard_assets),
            pairwise_rows=tuple(),
        ),
    )


def _aggregate_outer_residual_dependence(
    *,
    outer_results: Sequence[OuterResidualDependence],
    candidate_id: int,
) -> OuterResidualDependence:
    block_results = {
        block: _aggregate_block_results(
            [result.block_results[block] for result in outer_results],
        )
        for block in (INDICES_BLOCK, FULL_BLOCK)
    }
    return OuterResidualDependence(
        outer_k=-1,
        candidate_id=int(candidate_id),
        block_results=block_results,
    )


def _aggregate_block_results(
    block_results: Sequence[BlockResidualDiagnostics],
) -> BlockResidualDiagnostics:
    first = block_results[0]
    matrices = _aggregate_matrices(block_results)
    hard_assets = _aggregate_hard_assets(block_results)
    return BlockResidualDiagnostics(
        block=first.block,
        asset_names=first.asset_names,
        summary=_aggregate_summary(block_results),
        matrices=matrices,
        hard_assets=hard_assets,
    )


def _aggregate_matrices(
    block_results: Sequence[BlockResidualDiagnostics],
) -> CorrelationMatrices:
    return CorrelationMatrices(
        residual_corr=_median_stack(
            [item.matrices.residual_corr for item in block_results]
        ),
        residual_counts=_median_count_stack(
            [item.matrices.residual_counts for item in block_results]
        ),
        whitened_corr=_median_stack(
            [item.matrices.whitened_corr for item in block_results]
        ),
        whitened_counts=_median_count_stack(
            [item.matrices.whitened_counts for item in block_results]
        ),
    )


def _aggregate_hard_assets(
    block_results: Sequence[BlockResidualDiagnostics],
) -> HardAssetDetails:
    first = block_results[0].hard_assets
    return HardAssetDetails(
        hard_assets_requested=first.hard_assets_requested,
        hard_assets_present=first.hard_assets_present,
        hard_assets_missing=first.hard_assets_missing,
        pairwise_rows=_aggregate_pairwise_rows(block_results),
    )


def _aggregate_summary(
    block_results: Sequence[BlockResidualDiagnostics],
) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for field in _SUMMARY_FIELDS:
        values = [float(item.summary.get(field, float("nan"))) for item in block_results]
        result[field] = _median_or_nan(values)
    return result


def _median_stack(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.empty((0, 0), dtype=float)
    return np.nanmedian(np.stack(matrices, axis=0), axis=0)


def _median_count_stack(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.empty((0, 0), dtype=int)
    stacked = np.stack([matrix.astype(float) for matrix in matrices], axis=0)
    return np.rint(np.nanmedian(stacked, axis=0)).astype(int)


def _aggregate_pairwise_rows(
    block_results: Sequence[BlockResidualDiagnostics],
) -> tuple[Mapping[str, Any], ...]:
    rows_by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in block_results:
        for row in result.hard_assets.pairwise_rows:
            key = (str(row["left_asset"]), str(row["right_asset"]))
            rows_by_pair.setdefault(key, []).append(row)
    aggregated: list[Mapping[str, Any]] = []
    for key in sorted(rows_by_pair):
        rows = rows_by_pair[key]
        aggregated.append(
            {
                "left_asset": key[0],
                "right_asset": key[1],
                "residual_corr": _median_or_nan([float(row["residual_corr"]) for row in rows]),
                "abs_residual_corr": _median_or_nan(
                    [float(row["abs_residual_corr"]) for row in rows]
                ),
                "whitened_residual_corr": _median_or_nan(
                    [float(row["whitened_residual_corr"]) for row in rows]
                ),
                "abs_whitened_residual_corr": _median_or_nan(
                    [float(row["abs_whitened_residual_corr"]) for row in rows]
                ),
                "median_n_obs": _median_or_nan([float(row["n_obs"]) for row in rows]),
                "median_n_obs_whitened": _median_or_nan(
                    [float(row["n_obs_whitened"]) for row in rows]
                ),
                "n_outer_folds": float(len(rows)),
            }
        )
    return tuple(aggregated)


def _median_or_nan(values: Sequence[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.median(np.asarray(finite, dtype=float)))


def _write_summary_outputs(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    block_results: Mapping[str, BlockResidualDiagnostics],
) -> None:
    full_payload = dict(payload)
    blocks_payload = payload.get("blocks")
    if isinstance(blocks_payload, Mapping):
        full_payload.update(
            {str(block): value for block, value in blocks_payload.items()}
        )
    _write_json(output_dir / "residual_dependence_summary.json", full_payload)
    _write_summary_csv(
        output_dir / "residual_dependence_summary.csv",
        block_results=block_results,
    )
    indices = block_results[INDICES_BLOCK]
    _write_matrix_csv(
        output_dir / "index_residual_corr_matrix.csv",
        asset_names=indices.asset_names,
        matrix=indices.matrices.residual_corr,
    )
    _write_matrix_csv(
        output_dir / "index_whitened_residual_corr_matrix.csv",
        asset_names=indices.asset_names,
        matrix=indices.matrices.whitened_corr,
    )
    _write_pairwise_csv(
        output_dir / "index_pairwise_residual_structure.csv",
        rows=indices.hard_assets.pairwise_rows,
    )


def _serialize_block_results(
    block_results: Mapping[str, BlockResidualDiagnostics]
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {}
    for block, result in block_results.items():
        payload[block] = {
            "summary": {
                key: float(value)
                for key, value in result.summary.items()
            },
            "asset_names": list(result.asset_names),
            "hard_assets_requested": list(result.hard_assets.hard_assets_requested),
            "hard_assets_present": list(result.hard_assets.hard_assets_present),
            "hard_assets_missing": list(result.hard_assets.hard_assets_missing),
        }
    return payload


def _ensure_dir(path: Path) -> Path:
    ensure_directory(
        path,
        error_type=SimulationError,
        invalid_message="Residual dependence output path is not a directory",
        create_message="Failed to create residual dependence output",
        context={"path": str(path)},
    )
    return path


def _write_summary_csv(
    path: Path,
    *,
    block_results: Mapping[str, BlockResidualDiagnostics],
) -> None:
    rows: list[dict[str, Any]] = []
    for block in (INDICES_BLOCK, FULL_BLOCK):
        summary: dict[str, Any] = dict(block_results[block].summary)
        summary["block"] = block
        rows.append(summary)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_matrix_csv(
    path: Path,
    *,
    asset_names: Sequence[str],
    matrix: np.ndarray,
) -> None:
    frame = pd.DataFrame(matrix, index=list(asset_names), columns=list(asset_names))
    frame.index.name = "asset"
    frame.to_csv(path)


def _write_pairwise_csv(path: Path, *, rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
