from __future__ import annotations
# pylint: disable=too-many-lines,duplicate-code

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.application.signal_metrics import (
    brier_score,
    calibration_rmse,
    hit_rate,
    mean_spread,
    pearson_correlation,
    rest_indices,
    spearman_correlation,
    top_indices,
    top_k_count,
)
from algo_trader.domain import (
    BLOCK_ORDER,
    COMMODITIES_BLOCK,
    FULL_BLOCK,
    FX_BLOCK,
    INDICES_BLOCK,
    SimulationError,
    build_asset_block_index_map,
)
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionComplexity,
    ModelSelectionConfig,
)
from .artifacts import SimulationArtifacts
from . import calibration_summary_diagnostics as calibration_diags
from .block_scoring_output import (
    write_global_block_scores,
    write_postprocess_block_scores,
)
from .dependence_scoring_output import (
    write_global_dependence_scores,
    write_postprocess_dependence_scores,
)
from .basket_diagnostics import (
    BASKET_ORDER,
    compute_candidate_basket_diagnostics,
    write_global_basket_diagnostics,
    write_postprocess_basket_diagnostics,
)
from .index_ranges import decode_indices_field
from .metrics.inner import energy_score_terms
from .residual_dependence_diagnostics import write_postprocess_residual_dependence


@dataclass(frozen=True)
class PostTuneSelectionContext:
    artifacts: SimulationArtifacts
    outer_k: int
    candidates: Sequence[CandidateSpec]
    model_selection: ModelSelectionConfig
    score_spec: Mapping[str, Any]
    use_gpu: bool


@dataclass(frozen=True)
class PostTuneSelectionResult:
    best_candidate_id: int
    metrics: Mapping[str, Any]
    selection: Mapping[str, Any]


@dataclass(frozen=True)
class GlobalSelectionContext:
    artifacts: SimulationArtifacts
    outer_ids: Sequence[int]
    candidates: Sequence[CandidateSpec]
    model_selection: ModelSelectionConfig


@dataclass(frozen=True)
class SecondaryEvalContext:
    base_dir: Path
    outer_k: int
    alpha: float
    batch_splits: int
    device: torch.device


@dataclass(frozen=True)
class BlockMetricContext:
    alpha: float
    batch_splits: int
    score_spec: Mapping[str, Any]
    device: torch.device
    block_indices: Mapping[str, tuple[int, ...]]
    panel_targets: torch.Tensor
    splits: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class BlockScoreInputs:
    es_metrics: Mapping[int, Mapping[str, float]]
    calibration_metrics: Mapping[int, Mapping[str, float]]
    crps_metrics: Mapping[int, float]
    ql_metrics: Mapping[int, float]


@dataclass(frozen=True)
class CandidateBlockMetricValues:
    es_metrics: Mapping[str, float]
    calibration_metrics: Mapping[str, float]
    crps_model: float
    ql_model: float
    secondary_available: bool


@dataclass
class SecondaryAccumulators:
    crps_sum: torch.Tensor
    crps_count: torch.Tensor
    ql_sum: torch.Tensor
    ql_count: torch.Tensor


@dataclass
class CalibrationAccumulators:
    coverage_sum: dict[float, torch.Tensor]
    coverage_count: dict[float, torch.Tensor]
    pit_sum: torch.Tensor
    pit_count: torch.Tensor


@dataclass(frozen=True)
class CandidatePayloadEntry:
    split_id: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class FinalSelectionInputs:
    es_metrics: Mapping[int, Mapping[str, float]]
    calibration_metrics: Mapping[int, Mapping[str, float]]
    crps_metrics: Mapping[int, float]
    ql_metrics: Mapping[int, float]
    signal_metrics: Mapping[int, Mapping[str, float]]
    basket_diagnostics: Mapping[int, Mapping[str, Mapping[str, float]]]
    complexity: Mapping[int, float]


@dataclass(frozen=True)
class CoverageErrorSummary:
    coverage_by_level: Mapping[float, float]
    abs_error_by_level: Mapping[float, float]
    mean_error: float
    max_error: float


@dataclass
class SignalAccumulators:
    metric_sum: dict[str, torch.Tensor]
    metric_count: dict[str, torch.Tensor]
    calibration_values: dict[int, list[tuple[np.ndarray, np.ndarray]]]


def reselect_best_candidate(
    *,
    inputs: FinalSelectionInputs,
    model_selection: ModelSelectionConfig,
) -> Mapping[str, Any]:
    return _select_final_candidate(
        inputs=inputs,
        model_selection=model_selection,
    )


_BLOCK_COVERAGE_LEVELS = (0.5, 0.9, 0.95)
_DEPENDENCE_VARIOGRAM_P = 0.5


def select_best_candidate_post_tune(
    context: PostTuneSelectionContext,
) -> PostTuneSelectionResult:
    device = _resolve_device(context.use_gpu)
    candidate_ids = [int(candidate.candidate_id) for candidate in context.candidates]
    es_metrics = _compute_es_metrics(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidates=context.candidates,
        model_selection=context.model_selection,
        device=device,
    )
    calibration_metrics = _compute_calibration_metrics(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidates=context.candidates,
        model_selection=context.model_selection,
        device=device,
    )
    crps_metrics, ql_metrics = _compute_secondary_metrics(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidate_ids=candidate_ids,
        model_selection=context.model_selection,
        device=device,
    )
    signal_metrics = _compute_signal_metrics(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidate_ids=candidate_ids,
        model_selection=context.model_selection,
        device=device,
    )
    block_scores, dependence_scores, basket_diagnostics = _compute_diagnostic_scores(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        context=_build_block_metric_context(
            base_dir=context.artifacts.base_dir,
            outer_k=context.outer_k,
            model_selection=context.model_selection,
            score_spec=context.score_spec,
            device=device,
        ),
        candidate_ids=candidate_ids,
        inputs=BlockScoreInputs(
            es_metrics=es_metrics,
            calibration_metrics=calibration_metrics,
            crps_metrics=crps_metrics,
            ql_metrics=ql_metrics,
        ),
    )
    complexity = _complexity_scores_post_tune(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidates=context.candidates,
        model_selection=context.model_selection,
    )
    selection = _select_final_candidate(
        inputs=FinalSelectionInputs(
            es_metrics=es_metrics,
            calibration_metrics=calibration_metrics,
            crps_metrics=crps_metrics,
            ql_metrics=ql_metrics,
            signal_metrics=signal_metrics,
            basket_diagnostics=basket_diagnostics,
            complexity=complexity,
        ),
        model_selection=context.model_selection,
    )
    metrics_payload = _build_metrics_payload(
        es_metrics=es_metrics,
        calibration_metrics=calibration_metrics,
        crps_metrics=crps_metrics,
        ql_metrics=ql_metrics,
        signal_metrics=signal_metrics,
        block_scores=block_scores,
        dependence_scores=dependence_scores,
        basket_diagnostics=basket_diagnostics,
        complexity=complexity,
    )
    selection = _with_best_candidate_block_scores(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    selection = _with_best_candidate_dependence_scores(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    selection = _with_best_candidate_basket_diagnostics(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    context.artifacts.write_postprocess_metrics(
        outer_k=context.outer_k, metrics=metrics_payload
    )
    context.artifacts.write_postprocess_selection(
        outer_k=context.outer_k, selection=selection
    )
    _write_postprocess_block_scores_report(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        selection=selection,
    )
    _write_postprocess_dependence_scores_report(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        selection=selection,
    )
    _write_postprocess_basket_diagnostics_report(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        selection=selection,
    )
    _write_postprocess_residual_dependence_report(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        selection=selection,
    )
    return PostTuneSelectionResult(
        best_candidate_id=int(selection["best_candidate_id"]),
        metrics=metrics_payload,
        selection=selection,
    )


def select_best_candidate_global(
    context: GlobalSelectionContext,
) -> PostTuneSelectionResult:
    outer_metrics = _load_outer_metrics(
        base_dir=context.artifacts.base_dir,
        outer_ids=context.outer_ids,
    )
    (
        es_metrics,
        calibration_metrics,
        crps_metrics,
        ql_metrics,
        signal_metrics,
        block_scores,
        dependence_scores,
        basket_diagnostics,
        complexity,
    ) = _aggregate_global_metrics(outer_metrics)
    selection = _select_final_candidate(
        inputs=FinalSelectionInputs(
            es_metrics=es_metrics,
            calibration_metrics=calibration_metrics,
            crps_metrics=crps_metrics,
            ql_metrics=ql_metrics,
            signal_metrics=signal_metrics,
            basket_diagnostics=basket_diagnostics,
            complexity=complexity,
        ),
        model_selection=context.model_selection,
    )
    metrics_payload = _build_metrics_payload(
        es_metrics=es_metrics,
        calibration_metrics=calibration_metrics,
        crps_metrics=crps_metrics,
        ql_metrics=ql_metrics,
        signal_metrics=signal_metrics,
        block_scores=block_scores,
        dependence_scores=dependence_scores,
        basket_diagnostics=basket_diagnostics,
        complexity=complexity,
    )
    selection = _with_best_candidate_block_scores(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    selection = _with_best_candidate_dependence_scores(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    selection = _with_best_candidate_basket_diagnostics(
        selection=selection,
        metrics_payload=metrics_payload,
    )
    context.artifacts.write_global_metrics(payload=metrics_payload)
    context.artifacts.write_global_selection(payload=selection)
    _write_global_block_scores_report(
        base_dir=context.artifacts.base_dir,
        outer_ids=context.outer_ids,
        selection=selection,
    )
    _write_global_dependence_scores_report(
        base_dir=context.artifacts.base_dir,
        outer_ids=context.outer_ids,
        selection=selection,
    )
    _write_global_basket_diagnostics_report(
        base_dir=context.artifacts.base_dir,
        outer_ids=context.outer_ids,
        selection=selection,
    )
    return PostTuneSelectionResult(
        best_candidate_id=int(selection["best_candidate_id"]),
        metrics=metrics_payload,
        selection=selection,
    )


def _resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_es_metrics(
    *,
    base_dir: Path,
    outer_k: int,
    candidates: Sequence[CandidateSpec],
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> Mapping[int, Mapping[str, float]]:
    metrics: dict[int, Mapping[str, float]] = {}
    batch_size = max(1, int(model_selection.batching.candidates))
    for batch in _iter_candidate_batches(candidates, batch_size):
        for candidate in batch:
            es_model, se_es = _evaluate_es_for_candidate(
                base_dir=base_dir,
                outer_k=outer_k,
                candidate_id=candidate.candidate_id,
                model_selection=model_selection,
                device=device,
            )
            metrics[int(candidate.candidate_id)] = {
                "es_model": float(es_model),
                "se_es": float(se_es),
            }
    return metrics


def _compute_secondary_metrics(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_ids: Sequence[int],
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> tuple[Mapping[int, float], Mapping[int, float]]:
    crps: dict[int, float] = {}
    ql: dict[int, float] = {}
    batch_size = max(1, int(model_selection.batching.candidates))
    batch_splits = max(1, int(model_selection.batching.splits))
    eval_context = SecondaryEvalContext(
        base_dir=base_dir,
        outer_k=outer_k,
        alpha=float(model_selection.tail.alpha),
        batch_splits=batch_splits,
        device=device,
    )
    for batch in _iter_id_batches(candidate_ids, batch_size):
        for candidate_id in batch:
            crps_model, ql_model = _evaluate_secondary_for_candidate(
                eval_context, candidate_id
            )
            crps[int(candidate_id)] = float(crps_model)
            ql[int(candidate_id)] = float(ql_model)
    return crps, ql


def _compute_calibration_metrics(
    *,
    base_dir: Path,
    outer_k: int,
    candidates: Sequence[CandidateSpec],
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> Mapping[int, Mapping[str, float]]:
    metrics: dict[int, Mapping[str, float]] = {}
    batch_size = max(1, int(model_selection.batching.candidates))
    for batch in _iter_candidate_batches(candidates, batch_size):
        for candidate in batch:
            metrics[int(candidate.candidate_id)] = _evaluate_calibration_for_candidate(
                base_dir=base_dir,
                outer_k=outer_k,
                candidate_id=int(candidate.candidate_id),
                model_selection=model_selection,
                device=device,
            )
    return metrics


def _compute_signal_metrics(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_ids: Sequence[int],
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> Mapping[int, Mapping[str, float]]:
    metrics: dict[int, Mapping[str, float]] = {}
    batch_size = max(1, int(model_selection.batching.candidates))
    batch_splits = max(1, int(model_selection.batching.splits))
    for batch in _iter_id_batches(candidate_ids, batch_size):
        for candidate_id in batch:
            metrics[int(candidate_id)] = _evaluate_signal_for_candidate(
                base_dir=base_dir,
                outer_k=outer_k,
                candidate_id=int(candidate_id),
                device=device,
                batch_splits=batch_splits,
            )
    return metrics


def _build_block_metric_context(
    *,
    base_dir: Path,
    outer_k: int,
    model_selection: ModelSelectionConfig,
    score_spec: Mapping[str, Any],
    device: torch.device,
) -> BlockMetricContext:
    asset_names = _load_asset_names(base_dir)
    return BlockMetricContext(
        alpha=float(model_selection.tail.alpha),
        batch_splits=max(1, int(model_selection.batching.splits)),
        score_spec=score_spec,
        device=device,
        block_indices=build_asset_block_index_map(asset_names),
        panel_targets=_load_panel_targets(base_dir),
        splits=_load_outer_splits(base_dir=base_dir, outer_k=outer_k),
    )


def _compute_diagnostic_scores(
    *,
    base_dir: Path,
    outer_k: int,
    context: BlockMetricContext,
    candidate_ids: Sequence[int],
    inputs: BlockScoreInputs,
) -> tuple[
    Mapping[int, Mapping[str, Mapping[str, float]]],
    Mapping[int, Mapping[str, Mapping[str, float]]],
    Mapping[int, Mapping[str, Mapping[str, float]]],
]:
    block_scores: dict[int, Mapping[str, Mapping[str, float]]] = {}
    dependence_scores: dict[int, Mapping[str, Mapping[str, float]]] = {}
    basket_diagnostics: dict[int, Mapping[str, Mapping[str, float]]] = {}
    asset_names = _load_asset_names(base_dir)
    for candidate_id in candidate_ids:
        entries = _load_candidate_payload_entries(
            base_dir=base_dir,
            outer_k=outer_k,
            candidate_id=int(candidate_id),
        )
        candidate_block_scores = _evaluate_block_scores_for_candidate(
            context=context,
            entries=entries,
            values=_candidate_block_metric_values(
                candidate_id=int(candidate_id),
                inputs=inputs,
            ),
        )
        block_scores[int(candidate_id)] = candidate_block_scores
        dependence_scores[int(candidate_id)] = _evaluate_dependence_scores_for_candidate(
            context=context,
            entries=entries,
            block_scores=candidate_block_scores,
            variogram_p=_DEPENDENCE_VARIOGRAM_P,
        )
        basket_diagnostics[int(candidate_id)] = compute_candidate_basket_diagnostics(
            asset_names=asset_names,
            payloads=[entry.payload for entry in entries],
        ).scores
    return block_scores, dependence_scores, basket_diagnostics


def _candidate_block_metric_values(
    *, candidate_id: int, inputs: BlockScoreInputs
) -> CandidateBlockMetricValues:
    return CandidateBlockMetricValues(
        es_metrics=inputs.es_metrics[candidate_id],
        calibration_metrics=inputs.calibration_metrics[candidate_id],
        crps_model=inputs.crps_metrics.get(candidate_id, float("nan")),
        ql_model=inputs.ql_metrics.get(candidate_id, float("nan")),
        secondary_available=(
            candidate_id in inputs.crps_metrics and candidate_id in inputs.ql_metrics
        ),
    )


def _evaluate_block_scores_for_candidate(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    values: CandidateBlockMetricValues,
) -> Mapping[str, Mapping[str, float]]:
    block_scores = dict(
        _evaluate_non_full_block_scores(
            context=context,
            entries=entries,
            secondary_available=values.secondary_available,
        )
    )
    block_scores[FULL_BLOCK] = _full_block_scores(
        block_indices=context.block_indices,
        es_metrics=values.es_metrics,
        calibration_metrics=values.calibration_metrics,
        crps_model=values.crps_model,
        ql_model=values.ql_model,
    )
    return block_scores


def _evaluate_non_full_block_scores(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    secondary_available: bool,
) -> Mapping[str, Mapping[str, float]]:
    block_names = (FX_BLOCK, INDICES_BLOCK, COMMODITIES_BLOCK)
    scores: dict[str, dict[str, float]] = {
        block: _empty_block_scores(context.block_indices[block])
        for block in block_names
    }
    if not entries:
        return scores
    es_values = {
        block: _aggregate_block_es_groups(context=context, entries=entries, block=block)
        for block in block_names
    }
    coverage_values = {
        block: _aggregate_block_coverage_groups(
            context=context,
            entries=entries,
            block=block,
            coverage_levels=_BLOCK_COVERAGE_LEVELS,
        )
        for block in block_names
    }
    crps_group: torch.Tensor | None = None
    ql_group: torch.Tensor | None = None
    if secondary_available:
        crps_group, ql_group = _aggregate_secondary_groups_from_entries(
            entries=entries,
            alpha=context.alpha,
            device=context.device,
            batch_splits=context.batch_splits,
        )
    for block in block_names:
        block_scores = scores[block]
        indices = context.block_indices[block]
        if not indices:
            continue
        block_scores["es"] = _median_or_nan(es_values[block].tolist())
        if crps_group is not None and ql_group is not None:
            block_scores["crps"] = _median_over_asset_indices(crps_group, indices)
            block_scores["ql"] = _median_over_asset_indices(ql_group, indices)
        for level in _BLOCK_COVERAGE_LEVELS:
            tag = calibration_diags.coverage_level_tag(level)
            block_scores[f"coverage_{tag}"] = _median_over_defined(
                coverage_values[block][float(level)]
            )
    return scores


def _empty_block_scores(indices: Sequence[int]) -> dict[str, float]:
    values = {
        "es": float("nan"),
        "crps": float("nan"),
        "ql": float("nan"),
        "coverage_p50": float("nan"),
        "coverage_p90": float("nan"),
        "coverage_p95": float("nan"),
        "n_assets": float(len(indices)),
    }
    return values


def _full_block_scores(
    *,
    block_indices: Mapping[str, tuple[int, ...]],
    es_metrics: Mapping[str, float],
    calibration_metrics: Mapping[str, float],
    crps_model: float,
    ql_model: float,
) -> Mapping[str, float]:
    return {
        "es": float(es_metrics["es_model"]),
        "crps": float(crps_model),
        "ql": float(ql_model),
        "coverage_p50": float(calibration_metrics.get("coverage_p50", float("nan"))),
        "coverage_p90": float(calibration_metrics.get("coverage_p90", float("nan"))),
        "coverage_p95": float(calibration_metrics.get("coverage_p95", float("nan"))),
        "n_assets": float(len(block_indices[FULL_BLOCK])),
    }


def _evaluate_dependence_scores_for_candidate(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    block_scores: Mapping[str, Mapping[str, float]],
    variogram_p: float,
) -> Mapping[str, Mapping[str, float]]:
    scores: dict[str, dict[str, float]] = {
        block: _empty_dependence_scores(
            indices=context.block_indices[block],
            variogram_p=variogram_p,
        )
        for block in BLOCK_ORDER
    }
    for block in BLOCK_ORDER:
        scores[block]["energy_score"] = float(
            block_scores.get(block, {}).get("es", float("nan"))
        )
    if not entries:
        return scores
    variogram_values = {
        block: _aggregate_block_variogram_groups(
            context=context,
            entries=entries,
            block=block,
            variogram_p=variogram_p,
        )
        for block in BLOCK_ORDER
    }
    for block in BLOCK_ORDER:
        scores[block]["variogram_score"] = _median_or_nan(
            variogram_values[block].tolist()
        )
    return scores


def _empty_dependence_scores(
    *, indices: Sequence[int], variogram_p: float
) -> dict[str, float]:
    return {
        "energy_score": float("nan"),
        "variogram_score": float("nan"),
        "variogram_p": float(variogram_p),
        "n_assets": float(len(indices)),
    }


def _aggregate_block_variogram_groups(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    block: str,
    variogram_p: float,
) -> np.ndarray:
    _validate_variogram_p(variogram_p)
    index_tensor = _block_pair_index_tensor(
        indices=context.block_indices[block],
        device=context.device,
    )
    if index_tensor is None:
        return np.asarray([], dtype=float)
    group_count, variogram_sum, variogram_count = _init_scalar_group_totals(
        payloads=[entry.payload for entry in entries],
        device=context.device,
    )
    for batch in _iter_entry_batches(entries, context.batch_splits):
        for entry in batch:
            variogram_week, group_ids = _prepare_block_variogram_payload(
                context=context,
                entry=entry,
                index_tensor=index_tensor,
                variogram_p=variogram_p,
            )
            _accumulate_scalar_group_mean(
                total_sum=variogram_sum,
                total_count=variogram_count,
                values=variogram_week,
                group_ids=group_ids,
                group_count=group_count,
            )
    values = _safe_divide(variogram_sum, variogram_count)
    return values.detach().cpu().numpy()


def _validate_variogram_p(variogram_p: float) -> None:
    if variogram_p <= 0.0:
        raise SimulationError("Variogram exponent must be > 0")


def _block_pair_index_tensor(
    *, indices: Sequence[int], device: torch.device
) -> torch.Tensor | None:
    if len(indices) < 2:
        return None
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _init_scalar_group_totals(
    *, payloads: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[int, torch.Tensor, torch.Tensor]:
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    zeros = torch.zeros(group_count, device=device, dtype=dtype)
    return group_count, zeros.clone(), zeros


def _accumulate_scalar_group_mean(
    *,
    total_sum: torch.Tensor,
    total_count: torch.Tensor,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> None:
    group_mean, present = _group_scalar_mean(
        values=values,
        group_ids=group_ids,
        group_count=group_count,
    )
    total_sum[present] = total_sum[present] + group_mean[present]
    total_count[present] = total_count[present] + 1


def _prepare_block_variogram_payload(
    *,
    context: BlockMetricContext,
    entry: CandidatePayloadEntry,
    index_tensor: torch.Tensor,
    variogram_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_true, z_samples, group_ids = _prepare_z_payload(entry.payload, context.device)
    return (
        _variogram_weekly(
            z_true=z_true.index_select(1, index_tensor),
            z_samples=z_samples.index_select(2, index_tensor),
            variogram_p=variogram_p,
        ),
        group_ids,
    )


def _variogram_weekly(
    *,
    z_true: torch.Tensor,
    z_samples: torch.Tensor,
    variogram_p: float,
) -> torch.Tensor:
    pair_indices = _resolve_variogram_pair_indices(z_true=z_true, z_samples=z_samples)
    if pair_indices is None:
        return torch.full(
            (z_true.shape[0],),
            float("nan"),
            device=z_true.device,
            dtype=z_true.dtype,
        )
    pair_i, pair_j = pair_indices
    true_pairs, true_valid = _variogram_true_pairs(
        z_true=z_true,
        pair_i=pair_i,
        pair_j=pair_j,
        variogram_p=variogram_p,
    )
    sample_mean = _variogram_sample_pair_mean(
        z_samples=z_samples,
        pair_i=pair_i,
        pair_j=pair_j,
        variogram_p=variogram_p,
    )
    return _mean_variogram_sq_error(
        true_pairs=true_pairs,
        true_valid=true_valid,
        sample_mean=sample_mean,
    )


def _resolve_variogram_pair_indices(
    *,
    z_true: torch.Tensor,
    z_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if z_true.ndim != 2:
        raise SimulationError("Variogram score requires z_true [T, A]")
    if z_samples.ndim != 3:
        raise SimulationError("Variogram score requires z_samples [S, T, A]")
    if z_samples.shape[1] != z_true.shape[0] or z_samples.shape[2] != z_true.shape[1]:
        raise SimulationError("Variogram score requires aligned z_true/z_samples")
    asset_count = int(z_true.shape[1])
    if asset_count < 2:
        return None
    pair_indices = torch.triu_indices(
        asset_count,
        asset_count,
        offset=1,
        device=z_true.device,
    )
    return pair_indices[0], pair_indices[1]


def _variogram_true_pairs(
    *,
    z_true: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    variogram_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = z_true.index_select(1, pair_i)
    right = z_true.index_select(1, pair_j)
    return (left - right).abs().pow(variogram_p), torch.isfinite(left) & torch.isfinite(
        right
    )


def _variogram_sample_pair_mean(
    *,
    z_samples: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    variogram_p: float,
) -> torch.Tensor:
    left = z_samples.index_select(2, pair_i)
    right = z_samples.index_select(2, pair_j)
    pair_values = (left - right).abs().pow(variogram_p)
    valid = torch.isfinite(left) & torch.isfinite(right)
    sample_sum = torch.where(valid, pair_values, torch.zeros_like(pair_values)).sum(dim=0)
    sample_count = valid.to(z_samples.dtype).sum(dim=0)
    return _safe_divide(sample_sum, sample_count)


def _mean_variogram_sq_error(
    *,
    true_pairs: torch.Tensor,
    true_valid: torch.Tensor,
    sample_mean: torch.Tensor,
) -> torch.Tensor:
    pair_valid = true_valid & torch.isfinite(sample_mean)
    sq_error = (sample_mean - true_pairs).pow(2)
    pair_sum = torch.where(pair_valid, sq_error, torch.zeros_like(sq_error)).sum(dim=1)
    pair_count = pair_valid.to(true_pairs.dtype).sum(dim=1)
    return _safe_divide(pair_sum, pair_count)

# pylint: disable=too-many-locals
def _aggregate_block_es_groups(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    block: str,
) -> np.ndarray:  # pylint: disable=too-many-locals
    indices = context.block_indices[block]
    if not indices:
        return np.asarray([], dtype=float)
    payloads = [entry.payload for entry in entries]
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    es_sum = torch.zeros(group_count, device=context.device, dtype=dtype)
    es_count = torch.zeros(group_count, device=context.device, dtype=dtype)
    index_tensor = torch.as_tensor(indices, device=context.device, dtype=torch.long)
    for batch in _iter_entry_batches(entries, context.batch_splits):
        for entry in batch:
            z_true, z_samples, group_ids = _prepare_z_payload(entry.payload, context.device)
            z_true_block = z_true.index_select(1, index_tensor)
            z_samples_block = z_samples.index_select(2, index_tensor)
            u_true, u_samples = _block_whitened_payload(
                context=context,
                entry=entry,
                z_true=z_true_block,
                z_samples=z_samples_block,
                index_tensor=index_tensor,
            )
            es_week = energy_score_terms(u_true, u_samples)
            group_mean, present = _group_scalar_mean(
                values=es_week,
                group_ids=group_ids,
                group_count=group_count,
            )
            es_sum[present] = es_sum[present] + group_mean[present]
            es_count[present] = es_count[present] + 1
    values = _safe_divide(es_sum, es_count)
    return values.detach().cpu().numpy()

# pylint: disable=too-many-locals
def _aggregate_block_coverage_groups(
    *,
    context: BlockMetricContext,
    entries: Sequence[CandidatePayloadEntry],
    block: str,
    coverage_levels: Sequence[float],
) -> Mapping[float, torch.Tensor]:  # pylint: disable=too-many-locals
    indices = context.block_indices[block]
    payloads = [entry.payload for entry in entries]
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    sums = {
        float(level): torch.zeros(group_count, device=context.device, dtype=dtype)
        for level in coverage_levels
    }
    counts = {
        float(level): torch.zeros(group_count, device=context.device, dtype=dtype)
        for level in coverage_levels
    }
    if not indices:
        return {
            float(level): torch.full(
                (group_count,), float("nan"), device=context.device, dtype=dtype
            )
            for level in coverage_levels
        }
    index_tensor = torch.as_tensor(indices, device=context.device, dtype=torch.long)
    for batch in _iter_entry_batches(entries, context.batch_splits):
        for entry in batch:
            z_true, z_samples, group_ids = _prepare_z_payload(entry.payload, context.device)
            z_true_block = z_true.index_select(1, index_tensor)
            z_samples_block = z_samples.index_select(2, index_tensor)
            for level in coverage_levels:
                coverage_group, present = _coverage_group_values(
                    z_true=z_true_block,
                    z_samples=z_samples_block,
                    group_ids=group_ids,
                    group_count=group_count,
                    level=float(level),
                )
                sums[float(level)][present] = sums[float(level)][present] + coverage_group[present]
                counts[float(level)][present] = counts[float(level)][present] + 1
    return {
        float(level): _safe_divide(sums[float(level)], counts[float(level)])
        for level in coverage_levels
    }


def _aggregate_secondary_groups_from_entries(
    *,
    entries: Sequence[CandidatePayloadEntry],
    alpha: float,
    device: torch.device,
    batch_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    payloads = [entry.payload for entry in entries]
    return _aggregate_secondary_groups(
        payloads=payloads,
        alpha=alpha,
        device=device,
        batch_splits=batch_splits,
    )


def _block_whitened_payload(
    *,
    context: BlockMetricContext,
    entry: CandidatePayloadEntry,
    z_true: torch.Tensor,
    z_samples: torch.Tensor,
    index_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _require_tensor(entry.payload, "scale", context.device).index_select(
        0, index_tensor
    )
    y_train = _train_targets_for_split(
        context=context,
        split_id=entry.split_id,
    ).index_select(1, index_tensor)
    whitener = _block_whitener(
        y_train=y_train,
        scale=scale,
        score_spec=context.score_spec,
    )
    return (
        torch.matmul(z_true, whitener.T),
        torch.matmul(z_samples, whitener.T),
    )


def _block_whitener(
    *,
    y_train: torch.Tensor,
    scale: torch.Tensor,
    score_spec: Mapping[str, Any],
) -> torch.Tensor:
    if y_train.ndim != 2:
        raise SimulationError("Block ES requires y_train [T, A]")
    if not torch.isfinite(y_train).all():
        raise SimulationError("Block ES requires fully observed training targets")
    n_eff = int(y_train.shape[0])
    if n_eff <= 1:
        raise SimulationError("Block ES requires at least 2 training rows")
    z_train = y_train / scale
    z_centered = z_train - z_train.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / float(n_eff - 1)
    shrinkage = _energy_score_shrinkage(n_eff)
    diag = torch.diag(torch.diag(cov))
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    var_floor = float(score_spec.get("var_floor", 0.0))
    if var_floor < 0.0:
        raise SimulationError("Energy score var_floor must be >= 0")
    if var_floor > 0.0:
        cov_shrunk = cov_shrunk + var_floor * torch.eye(
            cov_shrunk.shape[0],
            device=cov_shrunk.device,
            dtype=cov_shrunk.dtype,
        )
    eps = float(score_spec.get("eps", 1e-5))
    cov_shrunk = cov_shrunk + eps * torch.eye(
        cov_shrunk.shape[0],
        device=cov_shrunk.device,
        dtype=cov_shrunk.dtype,
    )
    return _inverse_cholesky(cov_shrunk)


def _train_targets_for_split(
    *, context: BlockMetricContext, split_id: int
) -> torch.Tensor:
    split = _split_entry(context.splits, split_id)
    train_idx = decode_indices_field(
        split,
        idx_key="train_idx",
        ranges_key="train_ranges",
        field="train",
    )
    return context.panel_targets.index_select(
        0,
        torch.as_tensor(train_idx, dtype=torch.long),
    ).to(device=context.device)


def _split_entry(
    splits: Sequence[Mapping[str, Any]], split_id: int
) -> Mapping[str, Any]:
    if split_id < 0 or split_id >= len(splits):
        raise SimulationError(
            "Split id out of range for block scoring",
            context={"split_id": str(split_id), "num_splits": str(len(splits))},
        )
    return splits[split_id]


def _load_asset_names(base_dir: Path) -> tuple[str, ...]:
    path = base_dir / "inputs" / "targets.csv"
    payload = _read_json_headerless_csv_columns(path)
    return tuple(payload)


def _read_json_headerless_csv_columns(path: Path) -> list[str]:
    if not path.exists():
        raise SimulationError("Missing targets.csv for block scoring", context={"path": str(path)})
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    columns = [str(column) for column in header if str(column) != "timestamp"]
    if not columns:
        raise SimulationError("Targets CSV has no asset columns", context={"path": str(path)})
    return columns


def _load_panel_targets(base_dir: Path) -> torch.Tensor:
    path = base_dir / "inputs" / "panel_tensor.pt"
    if not path.exists():
        raise SimulationError(
            "Missing panel_tensor.pt for block scoring",
            context={"path": str(path)},
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise SimulationError("panel_tensor.pt payload is invalid")
    targets = payload.get("targets")
    if not isinstance(targets, torch.Tensor):
        raise SimulationError("panel_tensor.pt missing targets tensor")
    return targets


def _load_outer_splits(
    *, base_dir: Path, outer_k: int
) -> Sequence[Mapping[str, Any]]:
    path = base_dir / "inner" / f"outer_{outer_k}" / "splits.json"
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise SimulationError(
            "splits.json payload must be a list",
            context={"path": str(path)},
        )
    return payload


def _iter_entry_batches(
    entries: Sequence[CandidatePayloadEntry], batch_size: int
) -> Sequence[Sequence[CandidatePayloadEntry]]:
    batches: list[Sequence[CandidatePayloadEntry]] = []
    for start in range(0, len(entries), batch_size):
        batches.append(entries[start : start + batch_size])
    return batches


def _iter_candidate_batches(
    candidates: Sequence[CandidateSpec], batch_size: int
) -> Sequence[Sequence[CandidateSpec]]:
    batches: list[Sequence[CandidateSpec]] = []
    for start in range(0, len(candidates), batch_size):
        batches.append(candidates[start : start + batch_size])
    return batches


def _iter_id_batches(
    candidate_ids: Sequence[int], batch_size: int
) -> Sequence[Sequence[int]]:
    batches: list[Sequence[int]] = []
    for start in range(0, len(candidate_ids), batch_size):
        batches.append(candidate_ids[start : start + batch_size])
    return batches


def _iter_payload_batches(
    payloads: Sequence[Mapping[str, Any]], batch_size: int
) -> Sequence[Sequence[Mapping[str, Any]]]:
    batches: list[Sequence[Mapping[str, Any]]] = []
    for start in range(0, len(payloads), batch_size):
        batches.append(payloads[start : start + batch_size])
    return batches


def _evaluate_es_for_candidate(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> tuple[float, float]:
    payloads = _load_candidate_payloads(
        base_dir=base_dir, outer_k=outer_k, candidate_id=candidate_id
    )
    es_group = _aggregate_es_groups(
        payloads, device, int(model_selection.batching.splits)
    )
    es_model = _median_over_defined(es_group)
    se_es = _bootstrap_es(
        es_group, model_selection.bootstrap.num_samples, model_selection.bootstrap.seed
    )
    return es_model, se_es


def _evaluate_secondary_for_candidate(
    context: SecondaryEvalContext, candidate_id: int
) -> tuple[float, float]:
    payloads = _load_candidate_payloads(
        base_dir=context.base_dir,
        outer_k=context.outer_k,
        candidate_id=candidate_id,
    )
    crps_group, ql_group = _aggregate_secondary_groups(
        payloads,
        context.alpha,
        context.device,
        context.batch_splits,
    )
    crps_model = _median_over_assets(crps_group)
    ql_model = _median_over_assets(ql_group)
    return crps_model, ql_model


def _evaluate_calibration_for_candidate(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    model_selection: ModelSelectionConfig,
    device: torch.device,
) -> Mapping[str, float]:
    payloads = _load_candidate_payloads(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=candidate_id,
    )
    coverage_group, pit_group = _aggregate_calibration_groups(
        payloads=payloads,
        coverage_levels=model_selection.calibration.coverage_levels,
        device=device,
        batch_splits=int(model_selection.batching.splits),
    )
    return _summarize_calibration_metrics(
        coverage_group=coverage_group,
        pit_group=pit_group,
        model_selection=model_selection,
    )


def _evaluate_signal_for_candidate(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
    device: torch.device,
    batch_splits: int,
) -> Mapping[str, float]:
    payloads = _load_candidate_payloads(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=candidate_id,
    )
    group_metrics = _aggregate_signal_groups(
        payloads=payloads,
        device=device,
        batch_splits=batch_splits,
    )
    return _summarize_signal_metrics(group_metrics)


def _load_candidate_payloads(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> list[Mapping[str, Any]]:
    candidates_dir = (
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "candidates"
    )
    pattern = f"candidate_{candidate_id:04d}_split_*.pt"
    paths = sorted(candidates_dir.glob(pattern))
    if not paths:
        raise SimulationError(
            "Missing postprocess candidate splits",
            context={"candidate_id": str(candidate_id)},
        )
    payloads: list[Mapping[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise SimulationError(
                "Postprocess payload is invalid",
                context={"path": str(path)},
            )
        payloads.append(payload)
    return payloads


def _load_candidate_payload_entries(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> list[CandidatePayloadEntry]:
    candidates_dir = (
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "candidates"
    )
    pattern = f"candidate_{candidate_id:04d}_split_*.pt"
    paths = sorted(candidates_dir.glob(pattern))
    if not paths:
        raise SimulationError(
            "Missing postprocess candidate splits",
            context={"candidate_id": str(candidate_id)},
        )
    entries: list[CandidatePayloadEntry] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise SimulationError(
                "Postprocess payload is invalid",
                context={"path": str(path)},
            )
        entries.append(
            CandidatePayloadEntry(
                split_id=_split_id_from_candidate_payload_path(path),
                payload=payload,
            )
        )
    return entries


def _split_id_from_candidate_payload_path(path: Path) -> int:
    name = path.stem
    parts = name.split("_")
    if len(parts) < 4 or parts[-2] != "split":
        raise SimulationError(
            "Unable to parse split id from candidate payload path",
            context={"path": str(path)},
        )
    return int(parts[-1])


def _aggregate_es_groups(
    payloads: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_splits: int,
) -> torch.Tensor:
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    es_sum = torch.zeros(group_count, device=device, dtype=dtype)
    es_count = torch.zeros(group_count, device=device, dtype=dtype)
    for batch in _iter_payload_batches(payloads, batch_splits):
        for payload in batch:
            u_true, u_samples, group_ids = _prepare_u_payload(
                payload, device
            )
            es_week = energy_score_terms(u_true, u_samples)
            group_mean, present = _group_mean(
                es_week, group_ids, group_count
            )
            es_sum[present] = es_sum[present] + group_mean[present]
            es_count[present] = es_count[present] + 1
    return _safe_divide(es_sum, es_count)


def _aggregate_secondary_groups(
    payloads: Sequence[Mapping[str, Any]],
    alpha: float,
    device: torch.device,
    batch_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    accumulators, group_count = _init_secondary_accumulators(
        payloads, device
    )
    for batch in _iter_payload_batches(payloads, batch_splits):
        for payload in batch:
            _update_secondary_accumulators(
                accumulators,
                payload,
                alpha,
                device,
                group_count,
            )
    crps_group = _safe_divide(
        accumulators.crps_sum, accumulators.crps_count
    )
    ql_group = _safe_divide(
        accumulators.ql_sum, accumulators.ql_count
    )
    return crps_group, ql_group


def _aggregate_calibration_groups(
    *,
    payloads: Sequence[Mapping[str, Any]],
    coverage_levels: Sequence[float],
    device: torch.device,
    batch_splits: int,
) -> tuple[Mapping[float, torch.Tensor], torch.Tensor]:
    accumulators, group_count = _init_calibration_accumulators(
        payloads=payloads,
        coverage_levels=coverage_levels,
        device=device,
    )
    for batch in _iter_payload_batches(payloads, batch_splits):
        for payload in batch:
            _update_calibration_accumulators(
                accumulators=accumulators,
                payload=payload,
                coverage_levels=coverage_levels,
                device=device,
                group_count=group_count,
            )
    coverage_group = {
        level: _safe_divide(
            accumulators.coverage_sum[level],
            accumulators.coverage_count[level],
        )
        for level in coverage_levels
    }
    pit_group = _safe_divide(accumulators.pit_sum, accumulators.pit_count)
    return coverage_group, pit_group


def _aggregate_signal_groups(
    *,
    payloads: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_splits: int,
) -> Mapping[str, torch.Tensor]:
    accumulators, group_count = _init_signal_accumulators(
        payloads=payloads,
        device=device,
    )
    for batch in _iter_payload_batches(payloads, batch_splits):
        for payload in batch:
            _update_signal_accumulators(
                accumulators=accumulators,
                payload=payload,
                device=device,
                group_count=group_count,
            )
    result = {
        metric: _safe_divide(
            accumulators.metric_sum[metric],
            accumulators.metric_count[metric],
        )
        for metric in accumulators.metric_sum
    }
    result["calibration_rmse"] = _signal_calibration_by_group(
        accumulators.calibration_values,
        device=device,
        group_count=group_count,
        dtype=_infer_payload_dtype(payloads),
    )
    return result


def _init_signal_accumulators(
    *,
    payloads: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[SignalAccumulators, int]:
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    metric_names = (
        "mean_rank_ic",
        "positive_rank_ic_fraction",
        "mean_linear_ic",
        "mean_top_k_spread",
        "mean_top_k_hit_rate",
        "mean_confidence_top_k_spread",
        "mean_brier_score",
        "mean_posterior_std",
    )
    return (
        SignalAccumulators(
            metric_sum={
                name: torch.zeros(group_count, device=device, dtype=dtype)
                for name in metric_names
            },
            metric_count={
                name: torch.zeros(group_count, device=device, dtype=dtype)
                for name in metric_names
            },
            calibration_values={group_id: [] for group_id in range(group_count)},
        ),
        group_count,
    )


def _update_signal_accumulators(
    *,
    accumulators: SignalAccumulators,
    payload: Mapping[str, Any],
    device: torch.device,
    group_count: int,
) -> None:
    z_true, z_samples, group_ids = _prepare_z_payload(payload, device)
    weekly_metrics = _signal_weekly_metrics(
        z_true=z_true.detach().cpu().numpy(),
        z_samples=z_samples.detach().cpu().numpy(),
    )
    for metric, values in weekly_metrics.items():
        group_values, present = _group_scalar_mean(
            values=torch.as_tensor(values, device=device, dtype=z_true.dtype),
            group_ids=group_ids,
            group_count=group_count,
        )
        accumulators.metric_sum[metric][present] = (
            accumulators.metric_sum[metric][present] + group_values[present]
        )
        accumulators.metric_count[metric][present] = (
            accumulators.metric_count[metric][present] + 1
        )
    p_positive = (z_samples > 0.0).to(z_true.dtype).mean(dim=0).detach().cpu().numpy()
    actual_positive = (z_true > 0.0).to(z_true.dtype).detach().cpu().numpy()
    groups = group_ids.detach().cpu().numpy()
    for group_id in range(group_count):
        mask = groups == group_id
        if not np.any(mask):
            continue
        accumulators.calibration_values[group_id].append(
            (
                p_positive[mask].reshape(-1),
                actual_positive[mask].reshape(-1),
            )
        )


def _signal_weekly_metrics(
    *,
    z_true: np.ndarray,
    z_samples: np.ndarray,
) -> Mapping[str, np.ndarray]:
    posterior_mean = np.mean(z_samples, axis=0)
    posterior_std = np.std(
        z_samples,
        axis=0,
        ddof=1 if z_samples.shape[0] > 1 else 0,
    )
    p_positive = np.mean(z_samples > 0.0, axis=0)
    top_k = top_k_count(z_true.shape[1])
    rank_ic: list[float] = []
    positive_rank_ic: list[float] = []
    linear_ic: list[float] = []
    top_k_spread: list[float] = []
    top_k_hit_rate: list[float] = []
    confidence_top_k_spread: list[float] = []
    brier_values: list[float] = []
    posterior_std_values: list[float] = []
    for mean_row, std_row, p_row, realized_row in zip(
        posterior_mean,
        posterior_std,
        p_positive,
        z_true,
        strict=True,
    ):
        top_idx = top_indices(mean_row, top_k)
        confidence_idx = top_indices(p_row, top_k)
        rest_idx = rest_indices(realized_row.shape[0], top_idx)
        confidence_rest_idx = rest_indices(realized_row.shape[0], confidence_idx)
        rank_ic_value = spearman_correlation(mean_row, realized_row)
        rank_ic.append(rank_ic_value)
        positive_rank_ic.append(float(rank_ic_value > 0.0))
        linear_ic.append(pearson_correlation(mean_row, realized_row))
        top_k_spread.append(mean_spread(realized_row, top_idx, rest_idx))
        top_k_hit_rate.append(hit_rate(realized_row, top_idx))
        confidence_top_k_spread.append(
            mean_spread(realized_row, confidence_idx, confidence_rest_idx)
        )
        brier_values.append(brier_score(p_row, realized_row > 0.0))
        posterior_std_values.append(float(np.mean(std_row)))
    return {
        "mean_rank_ic": np.asarray(rank_ic, dtype=float),
        "positive_rank_ic_fraction": np.asarray(positive_rank_ic, dtype=float),
        "mean_linear_ic": np.asarray(linear_ic, dtype=float),
        "mean_top_k_spread": np.asarray(top_k_spread, dtype=float),
        "mean_top_k_hit_rate": np.asarray(top_k_hit_rate, dtype=float),
        "mean_confidence_top_k_spread": np.asarray(
            confidence_top_k_spread,
            dtype=float,
        ),
        "mean_brier_score": np.asarray(brier_values, dtype=float),
        "mean_posterior_std": np.asarray(posterior_std_values, dtype=float),
    }


def _signal_calibration_by_group(
    calibration_values: Mapping[int, list[tuple[np.ndarray, np.ndarray]]],
    *,
    device: torch.device,
    group_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.full(
        (group_count,),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    for group_id, pairs in calibration_values.items():
        if not pairs:
            continue
        predicted = np.concatenate([pair[0] for pair in pairs])
        actual = np.concatenate([pair[1] for pair in pairs])
        values[group_id] = float(calibration_rmse(predicted, actual))
    return values


def _summarize_signal_metrics(
    signal_group: Mapping[str, torch.Tensor],
) -> Mapping[str, float]:
    return {
        metric: _median_over_defined(values)
        for metric, values in signal_group.items()
    }


def _prepare_u_payload(
    payload: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_true, z_samples, group_ids = _prepare_z_payload(payload, device)
    scale = _require_tensor(payload, "scale", device)
    whitener = _require_tensor(payload, "whitener", device)
    u_true = torch.matmul(z_true, whitener.T)
    u_samples = torch.matmul(z_samples, whitener.T)
    if scale.numel() == 0 or whitener.numel() == 0:
        raise SimulationError("Postprocess transform is empty")
    return u_true, u_samples, group_ids


def _prepare_z_payload(
    payload: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_true = _require_tensor(payload, "z_true", device)
    z_samples = _require_tensor(payload, "z_samples", device)
    if z_true.ndim != 2:
        raise SimulationError("Postprocess z_true must be [T, A]")
    if z_samples.ndim != 3:
        raise SimulationError("Postprocess z_samples must be [S, T, A]")
    group_ids = _require_group_ids(payload, device)
    if z_samples.shape[1] != z_true.shape[0]:
        raise SimulationError("Postprocess z_samples and z_true must align on T")
    if z_samples.shape[2] != z_true.shape[1]:
        raise SimulationError("Postprocess z_samples and z_true must align on A")
    return z_true, z_samples, group_ids


def _require_group_ids(
    payload: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    groups = payload.get("test_groups")
    if not isinstance(groups, list):
        raise SimulationError("Postprocess payload missing test_groups")
    group_ids = torch.as_tensor(groups, device=device, dtype=torch.long)
    if group_ids.ndim != 1:
        raise SimulationError("Postprocess test_groups must be 1D")
    return group_ids


def _require_tensor(
    payload: Mapping[str, Any], key: str, device: torch.device
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise SimulationError(
            "Postprocess payload missing tensor",
            context={"key": key},
        )
    return value.to(device)


def _infer_group_count(payloads: Sequence[Mapping[str, Any]]) -> int:
    max_group = -1
    for payload in payloads:
        groups = payload.get("test_groups")
        if isinstance(groups, list) and groups:
            max_group = max([max_group, *groups])
    if max_group < 0:
        raise SimulationError("Postprocess payload missing group ids")
    return int(max_group + 1)


def _infer_asset_count(payloads: Sequence[Mapping[str, Any]]) -> int:
    first = payloads[0]
    z_true = first.get("z_true")
    if not isinstance(z_true, torch.Tensor):
        raise SimulationError("Postprocess payload missing z_true")
    if z_true.ndim != 2:
        raise SimulationError("Postprocess z_true must be [T, A]")
    return int(z_true.shape[1])


def _infer_payload_dtype(payloads: Sequence[Mapping[str, Any]]) -> torch.dtype:
    first = payloads[0]
    z_true = first.get("z_true")
    if not isinstance(z_true, torch.Tensor):
        raise SimulationError("Postprocess payload missing z_true")
    return z_true.dtype


def _init_secondary_accumulators(
    payloads: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[SecondaryAccumulators, int]:
    group_count = _infer_group_count(payloads)
    asset_count = _infer_asset_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    crps_sum = torch.zeros((group_count, asset_count), device=device, dtype=dtype)
    crps_count = torch.zeros(group_count, device=device, dtype=dtype)
    ql_sum = torch.zeros((group_count, asset_count), device=device, dtype=dtype)
    ql_count = torch.zeros(group_count, device=device, dtype=dtype)
    return (
        SecondaryAccumulators(
            crps_sum=crps_sum,
            crps_count=crps_count,
            ql_sum=ql_sum,
            ql_count=ql_count,
        ),
        group_count,
    )


def _init_calibration_accumulators(
    *,
    payloads: Sequence[Mapping[str, Any]],
    coverage_levels: Sequence[float],
    device: torch.device,
) -> tuple[CalibrationAccumulators, int]:
    group_count = _infer_group_count(payloads)
    dtype = _infer_payload_dtype(payloads)
    return (
        CalibrationAccumulators(
            coverage_sum={
                float(level): torch.zeros(
                    group_count,
                    device=device,
                    dtype=dtype,
                )
                for level in coverage_levels
            },
            coverage_count={
                float(level): torch.zeros(
                    group_count,
                    device=device,
                    dtype=dtype,
                )
                for level in coverage_levels
            },
            pit_sum=torch.zeros(group_count, device=device, dtype=dtype),
            pit_count=torch.zeros(group_count, device=device, dtype=dtype),
        ),
        group_count,
    )


def _update_secondary_accumulators(
    accumulators: SecondaryAccumulators,
    payload: Mapping[str, Any],
    alpha: float,
    device: torch.device,
    group_count: int,
) -> None:
    z_true, z_samples, group_ids = _prepare_z_payload(payload, device)
    crps_week = _crps_weekly(z_true, z_samples)
    ql_week = _quantile_loss_weekly(z_true, z_samples, alpha)
    crps_group, crps_present = _group_mean(
        crps_week, group_ids, group_count
    )
    ql_group, ql_present = _group_mean(
        ql_week, group_ids, group_count
    )
    accumulators.crps_sum[crps_present] = (
        accumulators.crps_sum[crps_present] + crps_group[crps_present]
    )
    accumulators.crps_count[crps_present] = (
        accumulators.crps_count[crps_present] + 1
    )
    accumulators.ql_sum[ql_present] = (
        accumulators.ql_sum[ql_present] + ql_group[ql_present]
    )
    accumulators.ql_count[ql_present] = (
        accumulators.ql_count[ql_present] + 1
    )


def _update_calibration_accumulators(
    *,
    accumulators: CalibrationAccumulators,
    payload: Mapping[str, Any],
    coverage_levels: Sequence[float],
    device: torch.device,
    group_count: int,
) -> None:
    z_true, z_samples, group_ids = _prepare_z_payload(payload, device)
    pit_values = _compute_pit_values(z_true=z_true, z_samples=z_samples)
    pit_group, pit_present = _group_pit_rmse(
        pit_values=pit_values,
        group_ids=group_ids,
        group_count=group_count,
    )
    accumulators.pit_sum[pit_present] = (
        accumulators.pit_sum[pit_present] + pit_group[pit_present]
    )
    accumulators.pit_count[pit_present] = (
        accumulators.pit_count[pit_present] + 1
    )
    for level in coverage_levels:
        coverage_group, coverage_present = _coverage_group_values(
            z_true=z_true,
            z_samples=z_samples,
            group_ids=group_ids,
            group_count=group_count,
            level=float(level),
        )
        accumulators.coverage_sum[level][coverage_present] = (
            accumulators.coverage_sum[level][coverage_present]
            + coverage_group[coverage_present]
        )
        accumulators.coverage_count[level][coverage_present] = (
            accumulators.coverage_count[level][coverage_present] + 1
        )


def _crps_weekly(
    z_true: torch.Tensor, z_samples: torch.Tensor
) -> torch.Tensor:
    term1 = (z_samples - z_true.unsqueeze(0)).abs().mean(dim=0)
    diff = z_samples.unsqueeze(0) - z_samples.unsqueeze(1)
    term2 = 0.5 * diff.abs().mean(dim=(0, 1))
    return term1 - term2


def _quantile_loss_weekly(
    z_true: torch.Tensor, z_samples: torch.Tensor, alpha: float
) -> torch.Tensor:
    q_hat = torch.quantile(z_samples, alpha, dim=0)
    error = z_true - q_hat
    indicator = (error < 0).to(error.dtype)
    return (alpha - indicator) * error


def _coverage_group_values(
    *,
    z_true: torch.Tensor,
    z_samples: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
    level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_q, upper_q = _coverage_quantile_bounds(
        z_samples=z_samples,
        level=level,
    )
    valid = (
        torch.isfinite(z_true)
        & torch.isfinite(lower_q)
        & torch.isfinite(upper_q)
    )
    indicator = (
        (z_true >= lower_q) & (z_true <= upper_q) & valid
    ).to(z_true.dtype)
    row_count = valid.to(z_true.dtype).sum(dim=1)
    row_mean = _safe_divide(indicator.sum(dim=1), row_count)
    return _group_scalar_mean(
        values=row_mean,
        group_ids=group_ids,
        group_count=group_count,
    )


def _coverage_quantile_bounds(
    *,
    z_samples: torch.Tensor,
    level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = (1.0 - float(level)) / 2.0
    lower_q = torch.quantile(z_samples, alpha, dim=0)
    upper_q = torch.quantile(z_samples, 1.0 - alpha, dim=0)
    return lower_q, upper_q


def _group_scalar_mean(
    *,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_values = torch.zeros(
        group_count,
        device=values.device,
        dtype=values.dtype,
    )
    count = torch.zeros(
        group_count,
        device=values.device,
        dtype=values.dtype,
    )
    mask = torch.isfinite(values)
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    sum_values.index_add_(0, group_ids, safe_values)
    count.index_add_(0, group_ids, mask.to(values.dtype))
    present = count > 0
    mean = _safe_divide(sum_values, count)
    return mean, present


def _compute_pit_values(
    *, z_true: torch.Tensor, z_samples: torch.Tensor
) -> torch.Tensor:
    sample_valid = torch.isfinite(z_samples)
    truth_valid = torch.isfinite(z_true)
    indicator = ((z_samples <= z_true.unsqueeze(0)) & sample_valid).to(
        z_samples.dtype
    )
    counts = sample_valid.to(z_samples.dtype).sum(dim=0)
    pit = _safe_divide(indicator.sum(dim=0), counts)
    missing = torch.full_like(pit, float("nan"))
    return torch.where(truth_valid, pit, missing)


def _group_pit_rmse(
    *,
    pit_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.full_like(
        torch.zeros(group_count, device=pit_values.device, dtype=pit_values.dtype),
        float("nan"),
    )
    present = torch.zeros(group_count, device=pit_values.device, dtype=torch.bool)
    pit_np = pit_values.detach().cpu().numpy()
    groups_np = group_ids.detach().cpu().numpy()
    for group_id in range(group_count):
        rows = pit_np[groups_np == group_id]
        if rows.size == 0:
            continue
        values = rows.reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        score, _, _ = calibration_diags.pit_uniform_rmse(finite)
        result[group_id] = float(score)
        present[group_id] = True
    return result, present


def _group_mean(
    values: torch.Tensor, group_ids: torch.Tensor, group_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_values = torch.zeros(
        (group_count,) + values.shape[1:],
        device=values.device,
        dtype=values.dtype,
    )
    sum_values.index_add_(0, group_ids, values)
    ones = torch.ones_like(group_ids, dtype=values.dtype)
    count = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    count.index_add_(0, group_ids, ones)
    present = count > 0
    mean = _safe_divide(sum_values, count)
    return mean, present


def _safe_divide(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    denom = denominator
    if denom.ndim < numerator.ndim:
        expand_shape = (denom.shape[0],) + (1,) * (
            numerator.ndim - denom.ndim
        )
        denom = denom.view(expand_shape)
    mask = denom > 0
    safe = numerator / torch.where(mask, denom, torch.ones_like(denom))
    return torch.where(mask, safe, torch.full_like(safe, float("nan")))


def _median_over_defined(values: torch.Tensor) -> float:
    mask = values.isfinite()
    if not mask.any():
        raise SimulationError("Postprocess ES has no defined groups")
    return float(values[mask].median().item())


def _bootstrap_es(
    values: torch.Tensor, num_samples: int, seed: int
) -> float:
    mask = values.isfinite()
    defined = values[mask]
    if defined.numel() == 0:
        raise SimulationError("Postprocess ES has no defined groups")
    rng = np.random.default_rng(seed)
    medians: list[float] = []
    for _ in range(num_samples):
        indices = rng.integers(0, defined.numel(), size=defined.numel())
        resample = defined[torch.as_tensor(indices, device=defined.device)]
        medians.append(float(resample.median().item()))
    return float(np.std(medians, ddof=0))


def _median_over_assets(group_values: torch.Tensor) -> float:
    if group_values.ndim != 2:
        raise SimulationError("Postprocess group values must be [G, A]")
    mask = torch.isfinite(group_values)
    sums = torch.where(mask, group_values, torch.zeros_like(group_values)).sum(dim=0)
    counts = mask.sum(dim=0).to(group_values.dtype)
    means = _safe_divide(sums, counts)
    valid = torch.isfinite(means)
    if not valid.any():
        raise SimulationError("Postprocess metrics missing defined assets")
    return float(means[valid].median().item())


def _median_over_asset_indices(
    group_values: torch.Tensor, indices: Sequence[int]
) -> float:
    if not indices:
        return float("nan")
    index_tensor = torch.as_tensor(indices, device=group_values.device, dtype=torch.long)
    selected = group_values.index_select(1, index_tensor)
    return _median_over_assets(selected)


def _select_es_survivors(
    es_metrics: Mapping[int, Mapping[str, float]],
    model_selection: ModelSelectionConfig,
    *,
    candidate_ids: Sequence[int] | None = None,
) -> list[int]:
    if not es_metrics:
        raise SimulationError("Postprocess ES metrics missing")
    allowed = set(candidate_ids) if candidate_ids is not None else None
    filtered = [
        (candidate_id, values)
        for candidate_id, values in es_metrics.items()
        if allowed is None or candidate_id in allowed
    ]
    if not filtered:
        raise SimulationError("Postprocess ES survivors received no candidates")
    ordered = sorted(filtered, key=lambda item: item[1]["es_model"])
    best_vals = ordered[0][1]
    threshold = (
        best_vals["es_model"]
        + model_selection.es_band.c * best_vals["se_es"]
    )
    survivors = [
        candidate_id
        for candidate_id, values in ordered
        if values["es_model"] <= threshold
    ]
    if len(survivors) < model_selection.es_band.min_keep:
        survivors = [
            candidate_id
            for candidate_id, _ in ordered[: model_selection.es_band.min_keep]
        ]
    if len(survivors) > model_selection.es_band.max_keep:
        survivors = [
            candidate_id
            for candidate_id, _ in ordered[: model_selection.es_band.max_keep]
        ]
    return survivors


def _select_final_candidate(
    *,
    inputs: FinalSelectionInputs,
    model_selection: ModelSelectionConfig,
) -> Mapping[str, Any]:
    calibration_survivors = _select_calibration_survivors(
        inputs.calibration_metrics,
        model_selection,
    )
    survivors = _select_es_survivors(
        inputs.es_metrics,
        model_selection,
        candidate_ids=calibration_survivors,
    )
    secondary = _secondary_scores(survivors, inputs.crps_metrics, inputs.ql_metrics)
    signal_scores = _signal_scores(
        survivors=survivors,
        signal_metrics=inputs.signal_metrics,
        model_selection=model_selection,
    )
    basket_scores = _basket_scores(
        survivors=survivors,
        basket_diagnostics=inputs.basket_diagnostics,
        model_selection=model_selection,
    )
    survivors2, adjusted_scores = _final_survivors(
        survivors=survivors,
        secondary=secondary,
        signal_scores=signal_scores,
        basket_scores=basket_scores,
        model_selection=model_selection,
    )
    best_id = _pick_best_by_complexity(survivors2, inputs.complexity, inputs.es_metrics)
    selection: dict[str, Any] = {
        "best_candidate_id": int(best_id),
        "survivors_calibration": calibration_survivors,
        "survivors_es": survivors,
        "survivors_secondary": survivors2,
        "calibration_scores": {
            candidate_id: inputs.calibration_metrics[candidate_id][
                "calibration_score"
            ]
            for candidate_id in calibration_survivors
        },
        "secondary_scores": secondary,
        "complexity": inputs.complexity,
    }
    if basket_scores:
        selection["basket_scores"] = basket_scores
    if signal_scores:
        selection["signal_scores"] = signal_scores
    if adjusted_scores:
        selection["adjusted_scores"] = adjusted_scores
    return selection


def _with_best_candidate_block_scores(
    *,
    selection: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    best_candidate_id = int(selection["best_candidate_id"])
    entry = metrics_payload.get(str(best_candidate_id))
    if not isinstance(entry, Mapping):
        raise SimulationError(
            "Best candidate metrics missing for block scoring",
            context={"candidate_id": str(best_candidate_id)},
        )
    block_scores = entry.get("block_scores")
    if not isinstance(block_scores, Mapping):
        raise SimulationError(
            "Best candidate block scores missing",
            context={"candidate_id": str(best_candidate_id)},
        )
    updated = dict(selection)
    updated["best_candidate_block_scores"] = block_scores
    return updated


def _with_best_candidate_dependence_scores(
    *,
    selection: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    best_candidate_id = int(selection["best_candidate_id"])
    entry = metrics_payload.get(str(best_candidate_id))
    if not isinstance(entry, Mapping):
        raise SimulationError(
            "Best candidate metrics missing for dependence scoring",
            context={"candidate_id": str(best_candidate_id)},
        )
    dependence_scores = entry.get("dependence_scores")
    if not isinstance(dependence_scores, Mapping):
        raise SimulationError(
            "Best candidate dependence scores missing",
            context={"candidate_id": str(best_candidate_id)},
        )
    updated = dict(selection)
    updated["best_candidate_dependence_scores"] = dependence_scores
    return updated


def _with_best_candidate_basket_diagnostics(
    *,
    selection: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    best_candidate_id = int(selection["best_candidate_id"])
    entry = metrics_payload.get(str(best_candidate_id))
    if not isinstance(entry, Mapping):
        raise SimulationError(
            "Best candidate metrics missing for basket diagnostics",
            context={"candidate_id": str(best_candidate_id)},
        )
    basket_diagnostics = entry.get("basket_diagnostics")
    if not isinstance(basket_diagnostics, Mapping):
        raise SimulationError(
            "Best candidate basket diagnostics missing",
            context={"candidate_id": str(best_candidate_id)},
        )
    updated = dict(selection)
    updated["best_candidate_basket_diagnostics"] = basket_diagnostics
    return updated


def _write_postprocess_block_scores_report(
    *,
    base_dir: Path,
    outer_k: int,
    selection: Mapping[str, Any],
) -> None:
    best_candidate_id = int(selection["best_candidate_id"])
    block_scores = selection.get("best_candidate_block_scores")
    if not isinstance(block_scores, Mapping):
        raise SimulationError(
            "Postprocess selection missing best_candidate_block_scores",
            context={"outer_k": str(outer_k)},
        )
    write_postprocess_block_scores(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=best_candidate_id,
        block_scores=block_scores,
    )


def _write_postprocess_dependence_scores_report(
    *,
    base_dir: Path,
    outer_k: int,
    selection: Mapping[str, Any],
) -> None:
    best_candidate_id = int(selection["best_candidate_id"])
    dependence_scores = selection.get("best_candidate_dependence_scores")
    if not isinstance(dependence_scores, Mapping):
        raise SimulationError(
            "Postprocess selection missing best_candidate_dependence_scores",
            context={"outer_k": str(outer_k)},
        )
    write_postprocess_dependence_scores(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=best_candidate_id,
        dependence_scores=dependence_scores,
    )


def _write_postprocess_basket_diagnostics_report(
    *,
    base_dir: Path,
    outer_k: int,
    selection: Mapping[str, Any],
) -> None:
    if not isinstance(selection.get("best_candidate_basket_diagnostics"), Mapping):
        raise SimulationError(
            "Postprocess selection missing best_candidate_basket_diagnostics",
            context={"outer_k": str(outer_k)},
        )
    write_postprocess_basket_diagnostics(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=int(selection["best_candidate_id"]),
    )


def _write_global_block_scores_report(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    selection: Mapping[str, Any],
) -> None:
    best_candidate_id = int(selection["best_candidate_id"])
    block_scores = selection.get("best_candidate_block_scores")
    if not isinstance(block_scores, Mapping):
        raise SimulationError("Global selection missing best_candidate_block_scores")
    write_global_block_scores(
        base_dir=base_dir,
        outer_ids=outer_ids,
        candidate_id=best_candidate_id,
        block_scores=block_scores,
    )


def _write_global_dependence_scores_report(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    selection: Mapping[str, Any],
) -> None:
    best_candidate_id = int(selection["best_candidate_id"])
    dependence_scores = selection.get("best_candidate_dependence_scores")
    if not isinstance(dependence_scores, Mapping):
        raise SimulationError("Global selection missing best_candidate_dependence_scores")
    write_global_dependence_scores(
        base_dir=base_dir,
        outer_ids=outer_ids,
        candidate_id=best_candidate_id,
        dependence_scores=dependence_scores,
    )


def _write_global_basket_diagnostics_report(
    *,
    base_dir: Path,
    outer_ids: Sequence[int],
    selection: Mapping[str, Any],
) -> None:
    if not isinstance(selection.get("best_candidate_basket_diagnostics"), Mapping):
        raise SimulationError("Global selection missing best_candidate_basket_diagnostics")
    write_global_basket_diagnostics(
        base_dir=base_dir,
        outer_ids=outer_ids,
        candidate_id=int(selection["best_candidate_id"]),
    )


def _write_postprocess_residual_dependence_report(
    *,
    base_dir: Path,
    outer_k: int,
    selection: Mapping[str, Any],
) -> None:
    write_postprocess_residual_dependence(
        base_dir=base_dir,
        outer_k=outer_k,
        candidate_id=int(selection["best_candidate_id"]),
    )


def _select_calibration_survivors(
    calibration_metrics: Mapping[int, Mapping[str, float]],
    model_selection: ModelSelectionConfig,
) -> list[int]:
    if not calibration_metrics:
        raise SimulationError("Postprocess calibration metrics missing")
    ordered = sorted(
        calibration_metrics.items(),
        key=lambda item: (
            _sort_metric(item[1], "calibration_score"),
            _sort_metric(item[1], "max_abs_coverage_error"),
            _sort_metric(item[1], "mean_abs_coverage_error"),
            _sort_metric(item[1], "pit_uniform_rmse"),
            item[0],
        ),
    )
    top_k = min(int(model_selection.calibration.top_k), len(ordered))
    return [candidate_id for candidate_id, _ in ordered[:top_k]]


def _sort_metric(values: Mapping[str, float], key: str) -> float:
    value = float(values.get(key, float("inf")))
    return value if np.isfinite(value) else float("inf")


def _secondary_scores(
    survivors: Sequence[int],
    crps_metrics: Mapping[int, float],
    ql_metrics: Mapping[int, float],
) -> Mapping[int, int]:
    rank_crps = _rank_values(crps_metrics, survivors)
    rank_ql = _rank_values(ql_metrics, survivors)
    return {
        candidate_id: rank_crps[candidate_id] + rank_ql[candidate_id]
        for candidate_id in survivors
    }


def _secondary_survivors(
    secondary: Mapping[int, int]
) -> list[int]:
    min_secondary = min(secondary.values())
    return [
        candidate_id
        for candidate_id, score in secondary.items()
        if score == min_secondary
    ]


def _final_survivors(
    *,
    survivors: Sequence[int],
    secondary: Mapping[int, int],
    signal_scores: Mapping[int, int],
    basket_scores: Mapping[int, float],
    model_selection: ModelSelectionConfig,
) -> tuple[list[int], Mapping[int, int]]:
    if model_selection.mode == "global_calibrated":
        return _secondary_survivors(secondary), {}
    if model_selection.mode == "basket_aware":
        adjusted = _adjusted_scores(
            survivors=survivors,
            secondary=secondary,
            basket_scores=basket_scores,
        )
        min_score = min(adjusted.values())
        survivors2 = [
            candidate_id
            for candidate_id, score in adjusted.items()
            if score == min_score
        ]
        return survivors2, adjusted
    if model_selection.mode == "signal_aware":
        adjusted = _adjusted_scores(
            survivors=survivors,
            secondary=secondary,
            basket_scores=signal_scores,
        )
        min_score = min(adjusted.values())
        survivors2 = [
            candidate_id
            for candidate_id, score in adjusted.items()
            if score == min_score
        ]
        return survivors2, adjusted
    raise SimulationError(
        "Unknown model selection mode",
        context={"mode": str(model_selection.mode)},
    )


def _adjusted_scores(
    *,
    survivors: Sequence[int],
    secondary: Mapping[int, int],
    basket_scores: Mapping[int, float],
) -> Mapping[int, int]:
    basket_rank = _rank_values(basket_scores, survivors)
    return {
        candidate_id: secondary[candidate_id] + basket_rank[candidate_id]
        for candidate_id in survivors
    }


def _basket_scores(
    *,
    survivors: Sequence[int],
    basket_diagnostics: Mapping[int, Mapping[str, Mapping[str, float]]],
    model_selection: ModelSelectionConfig,
) -> Mapping[int, float]:
    if model_selection.mode != "basket_aware":
        return {}
    scores: dict[int, float] = {}
    for candidate_id in survivors:
        diagnostics = basket_diagnostics.get(candidate_id)
        if diagnostics is None:
            raise SimulationError(
                "Basket diagnostics missing for candidate selection",
                context={"candidate_id": str(candidate_id)},
            )
        scores[candidate_id] = _basket_score_for_candidate(
            diagnostics=diagnostics,
            model_selection=model_selection,
        )
    return scores


def _signal_scores(
    *,
    survivors: Sequence[int],
    signal_metrics: Mapping[int, Mapping[str, float]],
    model_selection: ModelSelectionConfig,
) -> Mapping[int, int]:
    if model_selection.mode != "signal_aware":
        return {}
    required = (
        "mean_rank_ic",
        "positive_rank_ic_fraction",
        "mean_linear_ic",
        "mean_top_k_spread",
        "mean_top_k_hit_rate",
        "mean_brier_score",
        "calibration_rmse",
    )
    descending_ranks = (
        "mean_rank_ic",
        "positive_rank_ic_fraction",
        "mean_linear_ic",
        "mean_top_k_spread",
        "mean_top_k_hit_rate",
    )
    signal_ranks = {
        key: _rank_values(
            {
                candidate_id: _signal_metric_value(
                    signal_metrics,
                    candidate_id,
                    key,
                    descending=key in descending_ranks,
                )
                for candidate_id in survivors
            },
            survivors,
        )
        for key in required
    }
    return {
        candidate_id: sum(
            metric_ranks[candidate_id]
            for metric_ranks in signal_ranks.values()
        )
        for candidate_id in survivors
    }


def _signal_metric_value(
    signal_metrics: Mapping[int, Mapping[str, float]],
    candidate_id: int,
    key: str,
    *,
    descending: bool,
) -> float:
    metrics = signal_metrics.get(candidate_id)
    if metrics is None:
        raise SimulationError(
            "Signal metrics missing for candidate selection",
            context={"candidate_id": str(candidate_id)},
        )
    value = float(metrics.get(key, float("nan")))
    if not np.isfinite(value):
        raise SimulationError(
            "Signal metric is missing or invalid",
            context={"candidate_id": str(candidate_id), "metric": key},
        )
    return -value if descending else value


def _basket_score_for_candidate(
    *,
    diagnostics: Mapping[str, Mapping[str, float]],
    model_selection: ModelSelectionConfig,
) -> float:
    basket_errors = _basket_coverage_errors(
        diagnostics=diagnostics,
        basket_names=model_selection.basket.baskets,
        coverage_levels=model_selection.calibration.coverage_levels,
    )
    basket_pit = _basket_mean_pit_rmse(
        diagnostics=diagnostics,
        basket_names=model_selection.basket.baskets,
    )
    return (
        model_selection.basket.mean_abs_weight * float(np.mean(basket_errors))
        + model_selection.basket.max_abs_weight * float(np.max(basket_errors))
        + model_selection.basket.pit_weight * basket_pit
    )


def _basket_coverage_errors(
    *,
    diagnostics: Mapping[str, Mapping[str, float]],
    basket_names: Sequence[str],
    coverage_levels: Sequence[float],
) -> list[float]:
    errors: list[float] = []
    for basket_name in basket_names:
        values = diagnostics.get(basket_name)
        if values is None:
            raise SimulationError(
                "Configured basket missing from diagnostics",
                context={"basket": basket_name},
            )
        for level in coverage_levels:
            tag = calibration_diags.coverage_level_tag(float(level))
            field = f"coverage_{tag}"
            observed = float(values.get(field, float("nan")))
            if not np.isfinite(observed):
                raise SimulationError(
                    "Basket coverage metric missing for candidate selection",
                    context={"basket": basket_name, "field": field},
                )
            errors.append(abs(observed - float(level)))
    return errors


def _basket_mean_pit_rmse(
    *,
    diagnostics: Mapping[str, Mapping[str, float]],
    basket_names: Sequence[str],
) -> float:
    values: list[float] = []
    for basket_name in basket_names:
        basket_values = diagnostics.get(basket_name)
        if basket_values is None:
            raise SimulationError(
                "Configured basket missing from diagnostics",
                context={"basket": basket_name},
            )
        pit = float(basket_values.get("pit_uniform_rmse", float("nan")))
        if not np.isfinite(pit):
            raise SimulationError(
                "Basket PIT metric missing for candidate selection",
                context={"basket": basket_name},
            )
        values.append(pit)
    return float(np.mean(np.asarray(values, dtype=float)))


def _pick_best_by_complexity(
    survivors: Sequence[int],
    complexity: Mapping[int, float],
    es_metrics: Mapping[int, Mapping[str, float]],
) -> int:
    ordered = sorted(
        survivors,
        key=lambda candidate_id: (
            complexity[candidate_id],
            es_metrics[candidate_id]["es_model"],
        ),
    )
    return ordered[0]


def _rank_values(
    values: Mapping[int, float], candidate_ids: Sequence[int]
) -> Mapping[int, int]:
    ordered = sorted(
        ((candidate_id, values[candidate_id]) for candidate_id in candidate_ids),
        key=lambda item: item[1],
    )
    ranks: dict[int, int] = {}
    current_rank = 1
    last_value: float | None = None
    for candidate_id, value in ordered:
        if last_value is None:
            current_rank = 1
        elif value > last_value:
            current_rank += 1
        ranks[candidate_id] = current_rank
        last_value = value
    return ranks


def _complexity_scores(
    *,
    values_by_candidate: Mapping[int, Sequence[float]],
    candidates: Sequence[CandidateSpec],
    method: ModelSelectionComplexity,
) -> Mapping[int, float]:
    if method.method == "random":
        rng = np.random.default_rng(method.seed)
        return {
            int(candidate.candidate_id): float(rng.random())
            for candidate in candidates
        }
    return _posterior_l1_complexity(
        values_by_candidate=values_by_candidate,
        candidates=candidates,
    )

# pylint: disable=too-many-arguments
def _build_metrics_payload(
    *,
    es_metrics: Mapping[int, Mapping[str, float]],
    calibration_metrics: Mapping[int, Mapping[str, float]],
    crps_metrics: Mapping[int, float],
    ql_metrics: Mapping[int, float],
    signal_metrics: Mapping[int, Mapping[str, float]],
    block_scores: Mapping[int, Mapping[str, Mapping[str, float]]],
    dependence_scores: Mapping[int, Mapping[str, Mapping[str, float]]],
    basket_diagnostics: Mapping[int, Mapping[str, Mapping[str, float]]],
    complexity: Mapping[int, float],
) -> Mapping[str, Any]:  # pylint: disable=too-many-arguments
    payload: dict[str, Any] = {}
    for candidate_id, values in es_metrics.items():
        calibration = calibration_metrics.get(candidate_id, {})
        payload[str(candidate_id)] = {
            "es_model": float(values["es_model"]),
            "se_es": float(values["se_es"]),
            **{
                key: float(val)
                for key, val in calibration.items()
            },
            "crps_model": float(crps_metrics.get(candidate_id, float("nan"))),
            "ql_model": float(ql_metrics.get(candidate_id, float("nan"))),
            **{
                key: float(val)
                for key, val in signal_metrics.get(candidate_id, {}).items()
            },
            "block_scores": _serialize_candidate_block_scores(
                block_scores.get(candidate_id, {})
            ),
            "dependence_scores": _serialize_candidate_block_scores(
                dependence_scores.get(candidate_id, {})
            ),
            "basket_diagnostics": _serialize_candidate_nested_scores(
                basket_diagnostics.get(candidate_id, {})
            ),
            "complexity": float(
                complexity.get(candidate_id, float("nan"))
            ),
        }
    return payload


def _serialize_candidate_block_scores(
    block_scores: Mapping[str, Mapping[str, float]]
) -> Mapping[str, Mapping[str, float]]:
    return _serialize_candidate_nested_scores(block_scores)


def _serialize_candidate_nested_scores(
    nested_scores: Mapping[str, Mapping[str, float]]
) -> Mapping[str, Mapping[str, float]]:
    return {
        block: {key: float(value) for key, value in values.items()}
        for block, values in nested_scores.items()
    }


def _load_outer_metrics(
    *, base_dir: Path, outer_ids: Sequence[int]
) -> list[Mapping[str, Mapping[str, Any]]]:
    metrics_list: list[Mapping[str, Mapping[str, Any]]] = []
    for outer_k in outer_ids:
        path = (
            base_dir
            / "inner"
            / f"outer_{outer_k}"
            / "postprocessing"
            / "metrics.json"
        )
        if not path.exists():
            raise SimulationError(
                "Missing postprocess metrics",
                context={"path": str(path)},
            )
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise SimulationError(
                "Postprocess metrics payload invalid",
                context={"path": str(path)},
            )
        metrics_list.append(payload)
    return metrics_list


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SimulationError(
            "Failed to read postprocess metrics",
            context={"path": str(path)},
        ) from exc


def _aggregate_global_metrics(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
) -> tuple[
    Mapping[int, Mapping[str, float]],
    Mapping[int, Mapping[str, float]],
    Mapping[int, float],
    Mapping[int, float],
    Mapping[int, Mapping[str, float]],
    Mapping[int, Mapping[str, Mapping[str, float]]],
    Mapping[int, Mapping[str, Mapping[str, float]]],
    Mapping[int, Mapping[str, Mapping[str, float]]],
    Mapping[int, float],
]:
    candidate_ids = _collect_candidate_ids(metrics_list)
    es_metrics: dict[int, Mapping[str, float]] = {}
    calibration_metrics: dict[int, Mapping[str, float]] = {}
    crps_metrics: dict[int, float] = {}
    ql_metrics: dict[int, float] = {}
    signal_metrics: dict[int, Mapping[str, float]] = {}
    block_score_metrics: dict[int, Mapping[str, Mapping[str, float]]] = {}
    dependence_score_metrics: dict[int, Mapping[str, Mapping[str, float]]] = {}
    basket_diagnostic_metrics: dict[int, Mapping[str, Mapping[str, float]]] = {}
    complexity_metrics: dict[int, float] = {}
    calibration_keys = _collect_calibration_metric_keys(metrics_list)
    for candidate_id in candidate_ids:
        es_vals = _collect_metric(metrics_list, candidate_id, "es_model")
        se_vals = _collect_metric(metrics_list, candidate_id, "se_es")
        crps_vals = _collect_metric(metrics_list, candidate_id, "crps_model")
        ql_vals = _collect_metric(metrics_list, candidate_id, "ql_model")
        complexity_vals = _collect_metric(
            metrics_list, candidate_id, "complexity"
        )
        es_metrics[candidate_id] = {
            "es_model": _median_or_nan(es_vals),
            "se_es": _median_or_nan(se_vals),
        }
        calibration_metrics[candidate_id] = {
            key: _median_or_nan(_collect_metric(metrics_list, candidate_id, key))
            for key in calibration_keys
        }
        crps_metrics[candidate_id] = _median_or_nan(crps_vals)
        ql_metrics[candidate_id] = _median_or_nan(ql_vals)
        signal_metrics[candidate_id] = {
            key: _median_or_nan(_collect_metric(metrics_list, candidate_id, key))
            for key in _collect_signal_metric_keys(metrics_list)
        }
        block_score_metrics[candidate_id] = _aggregate_block_scores_for_candidate(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
        )
        dependence_score_metrics[candidate_id] = _aggregate_dependence_scores_for_candidate(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
        )
        basket_diagnostic_metrics[candidate_id] = _aggregate_basket_diagnostics_for_candidate(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
        )
        complexity_metrics[candidate_id] = _median_or_nan(complexity_vals)
    return (
        es_metrics,
        calibration_metrics,
        crps_metrics,
        ql_metrics,
        signal_metrics,
        block_score_metrics,
        dependence_score_metrics,
        basket_diagnostic_metrics,
        complexity_metrics,
    )


def _collect_calibration_metric_keys(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
) -> tuple[str, ...]:
    fixed = {
        "calibration_score",
        "mean_abs_coverage_error",
        "max_abs_coverage_error",
        "pit_uniform_rmse",
    }
    dynamic: set[str] = set()
    for metrics in metrics_list:
        for entry in metrics.values():
            if not isinstance(entry, Mapping):
                continue
            for key in entry.keys():
                if (
                    str(key) in fixed
                    or str(key).startswith("coverage_")
                    or str(key).startswith("abs_error_")
                ):
                    dynamic.add(str(key))
    ordered = sorted(fixed | dynamic)
    return tuple(ordered)


def _collect_signal_metric_keys(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
) -> tuple[str, ...]:
    keys = {
        "mean_rank_ic",
        "positive_rank_ic_fraction",
        "mean_linear_ic",
        "mean_top_k_spread",
        "mean_top_k_hit_rate",
        "mean_confidence_top_k_spread",
        "mean_brier_score",
        "calibration_rmse",
        "mean_posterior_std",
    }
    present: set[str] = set()
    for metrics in metrics_list:
        for entry in metrics.values():
            if not isinstance(entry, Mapping):
                continue
            for key in keys:
                if key in entry:
                    present.add(key)
    return tuple(sorted(present))


def _aggregate_block_scores_for_candidate(
    *,
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
    candidate_id: int,
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for block in BLOCK_ORDER:
        block_metrics = _collect_nested_block_metrics(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
            nested_key="block_scores",
            block=block,
        )
        if not block_metrics:
            continue
        result[block] = {
            key: _median_or_nan(values)
            for key, values in block_metrics.items()
        }
    return result


def _aggregate_dependence_scores_for_candidate(
    *,
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
    candidate_id: int,
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for block in BLOCK_ORDER:
        block_metrics = _collect_nested_block_metrics(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
            nested_key="dependence_scores",
            block=block,
        )
        if not block_metrics:
            continue
        result[block] = {
            key: _median_or_nan(values)
            for key, values in block_metrics.items()
        }
    return result


def _aggregate_basket_diagnostics_for_candidate(
    *,
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
    candidate_id: int,
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for basket_name in BASKET_ORDER:
        basket_metrics = _collect_nested_block_metrics(
            metrics_list=metrics_list,
            candidate_id=candidate_id,
            nested_key="basket_diagnostics",
            block=basket_name,
        )
        if not basket_metrics:
            continue
        result[basket_name] = {
            key: _median_or_nan(values)
            for key, values in basket_metrics.items()
        }
    return result


def _collect_nested_block_metrics(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
    candidate_id: int,
    nested_key: str,
    block: str,
) -> Mapping[str, list[float]]:
    result: dict[str, list[float]] = {}
    for metrics in metrics_list:
        entry = metrics.get(str(candidate_id))
        if not isinstance(entry, Mapping):
            continue
        raw_block_scores = entry.get(nested_key)
        if not isinstance(raw_block_scores, Mapping):
            continue
        block_values = raw_block_scores.get(block)
        if not isinstance(block_values, Mapping):
            continue
        for key, raw_value in block_values.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            result.setdefault(str(key), []).append(value)
    return result


def _summarize_calibration_metrics(
    *,
    coverage_group: Mapping[float, torch.Tensor],
    pit_group: torch.Tensor,
    model_selection: ModelSelectionConfig,
) -> Mapping[str, float]:
    coverage_summary = _summarize_coverage_errors(
        coverage_group=coverage_group,
        coverage_levels=model_selection.calibration.coverage_levels,
    )
    pit_values = pit_group[torch.isfinite(pit_group)]
    if pit_values.numel() == 0:
        raise SimulationError("Postprocess PIT has no defined groups")
    pit_rmse = float(pit_values.median().item())
    calibration_score = _calibration_score(
        mean_error=coverage_summary.mean_error,
        max_error=coverage_summary.max_error,
        pit_rmse=pit_rmse,
        model_selection=model_selection,
    )
    payload: dict[str, float] = {
        "calibration_score": calibration_score,
        "mean_abs_coverage_error": coverage_summary.mean_error,
        "max_abs_coverage_error": coverage_summary.max_error,
        "pit_uniform_rmse": pit_rmse,
    }
    for level, empirical in coverage_summary.coverage_by_level.items():
        tag = calibration_diags.coverage_level_tag(level)
        payload[f"coverage_{tag}"] = empirical
        payload[f"abs_error_{tag}"] = (
            coverage_summary.abs_error_by_level[level]
        )
    return payload


def _summarize_coverage_errors(
    *,
    coverage_group: Mapping[float, torch.Tensor],
    coverage_levels: Sequence[float],
) -> CoverageErrorSummary:
    coverage_by_level: dict[float, float] = {}
    abs_error_by_level: dict[float, float] = {}
    finite_errors: list[float] = []
    for level in coverage_levels:
        level_value = float(level)
        empirical = _median_over_defined(coverage_group[level_value])
        coverage_by_level[level_value] = empirical
        abs_error = abs(empirical - level_value)
        abs_error_by_level[level_value] = abs_error
        if np.isfinite(abs_error):
            finite_errors.append(abs_error)
    return CoverageErrorSummary(
        coverage_by_level=coverage_by_level,
        abs_error_by_level=abs_error_by_level,
        mean_error=float(np.mean(np.asarray(finite_errors, dtype=float))),
        max_error=float(np.max(np.asarray(finite_errors, dtype=float))),
    )


def _calibration_score(
    *,
    mean_error: float,
    max_error: float,
    pit_rmse: float,
    model_selection: ModelSelectionConfig,
) -> float:
    calibration = model_selection.calibration
    return (
        calibration.mean_abs_weight * mean_error
        + calibration.max_abs_weight * max_error
        + calibration.pit_weight * pit_rmse
    )


def _complexity_scores_post_tune(
    *,
    base_dir: Path,
    outer_k: int,
    candidates: Sequence[CandidateSpec],
    model_selection: ModelSelectionConfig,
) -> Mapping[int, float]:
    values_by_candidate = {
        int(candidate.candidate_id): _load_complexity_values(
            base_dir=base_dir,
            outer_k=outer_k,
            candidate_id=int(candidate.candidate_id),
        )
        for candidate in candidates
    }
    return _complexity_scores(
        values_by_candidate=values_by_candidate,
        candidates=candidates,
        method=model_selection.complexity,
    )


def _load_complexity_values(
    *,
    base_dir: Path,
    outer_k: int,
    candidate_id: int,
) -> tuple[float, ...]:
    pattern = (
        base_dir
        / "inner"
        / f"outer_{outer_k}"
        / "postprocessing"
        / "debug"
    )
    paths = sorted(
        pattern.glob(f"candidate_{candidate_id:04d}_split_*_state.pt")
    )
    values: list[float] = []
    for path in paths:
        values.append(_load_complexity_value(path))
    return tuple(values)


def _load_complexity_value(path: Path) -> float:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        return float("inf")
    structural = payload.get("structural_posterior_means")
    if not isinstance(structural, Mapping):
        return float("inf")
    tensors = _structural_complexity_tensors(structural)
    if not tensors:
        return float("inf")
    total_abs = 0.0
    total_numel = 0
    for tensor in tensors:
        total_abs += float(tensor.abs().sum().item())
        total_numel += int(tensor.numel())
    if total_numel <= 0:
        return float("inf")
    return total_abs / float(total_numel)


def _structural_complexity_tensors(
    structural: Mapping[str, Any],
) -> list[torch.Tensor]:
    excluded = {"alpha", "sigma_idio", "s_u_mean"}
    tensors: list[torch.Tensor] = []
    for key, value in structural.items():
        if key in excluded or not isinstance(value, torch.Tensor):
            continue
        if value.numel() <= 0:
            continue
        tensors.append(value.detach().reshape(-1))
    return tensors


def _posterior_l1_complexity(
    *,
    values_by_candidate: Mapping[int, Sequence[float]],
    candidates: Sequence[CandidateSpec],
) -> Mapping[int, float]:
    scores: dict[int, float] = {}
    for candidate in candidates:
        candidate_id = int(candidate.candidate_id)
        values = [
            float(value)
            for value in values_by_candidate.get(candidate_id, ())
            if np.isfinite(value)
        ]
        if values:
            scores[candidate_id] = float(np.median(values))
        else:
            scores[candidate_id] = float("inf")
    return scores


def _energy_score_shrinkage(n_eff: int) -> float:
    if n_eff >= 300:
        return 0.075
    if n_eff >= 150:
        return 0.15
    return 0.30


def _inverse_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(matrix)  # pylint: disable=not-callable
    identity = torch.eye(
        chol.shape[0], device=chol.device, dtype=chol.dtype
    )
    return torch.linalg.solve_triangular(  # pylint: disable=not-callable
        chol, identity, upper=False
    )


def _collect_candidate_ids(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
) -> list[int]:
    ids: set[int] = set()
    for metrics in metrics_list:
        ids.update({int(key) for key in metrics.keys()})
    if not ids:
        raise SimulationError("Postprocess metrics missing candidates")
    return sorted(ids)


def _collect_metric(
    metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
    candidate_id: int,
    key: str,
) -> list[float]:
    values: list[float] = []
    for metrics in metrics_list:
        entry = metrics.get(str(candidate_id))
        if not isinstance(entry, Mapping):
            continue
        value = entry.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _median_or_nan(values: Sequence[float]) -> float:
    filtered = [value for value in values if np.isfinite(value)]
    if not filtered:
        return float("nan")
    return float(np.median(np.asarray(filtered, dtype=float)))
