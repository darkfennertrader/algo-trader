from __future__ import annotations
# pylint: disable=too-many-lines

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionComplexity,
    ModelSelectionConfig,
)
from .artifacts import SimulationArtifacts
from . import calibration_summary_diagnostics as calibration_diags
from .metrics.inner import energy_score_terms


@dataclass(frozen=True)
class PostTuneSelectionContext:
    artifacts: SimulationArtifacts
    outer_k: int
    candidates: Sequence[CandidateSpec]
    model_selection: ModelSelectionConfig
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
class FinalSelectionInputs:
    es_metrics: Mapping[int, Mapping[str, float]]
    calibration_metrics: Mapping[int, Mapping[str, float]]
    crps_metrics: Mapping[int, float]
    ql_metrics: Mapping[int, float]
    complexity: Mapping[int, float]


@dataclass(frozen=True)
class CoverageErrorSummary:
    coverage_by_level: Mapping[float, float]
    abs_error_by_level: Mapping[float, float]
    mean_error: float
    max_error: float


def select_best_candidate_post_tune(
    context: PostTuneSelectionContext,
) -> PostTuneSelectionResult:
    device = _resolve_device(context.use_gpu)
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
    calibration_survivors = _select_calibration_survivors(
        calibration_metrics, context.model_selection
    )
    survivors = _select_es_survivors(
        es_metrics,
        context.model_selection,
        candidate_ids=calibration_survivors,
    )
    crps_metrics, ql_metrics = _compute_secondary_metrics(
        base_dir=context.artifacts.base_dir,
        outer_k=context.outer_k,
        candidate_ids=survivors,
        model_selection=context.model_selection,
        device=device,
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
            complexity=complexity,
        ),
        model_selection=context.model_selection,
    )
    metrics_payload = _build_metrics_payload(
        es_metrics=es_metrics,
        calibration_metrics=calibration_metrics,
        crps_metrics=crps_metrics,
        ql_metrics=ql_metrics,
        complexity=complexity,
    )
    context.artifacts.write_postprocess_metrics(
        outer_k=context.outer_k, metrics=metrics_payload
    )
    context.artifacts.write_postprocess_selection(
        outer_k=context.outer_k, selection=selection
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
    es_metrics, calibration_metrics, crps_metrics, ql_metrics, complexity = (
        _aggregate_global_metrics(outer_metrics)
    )
    selection = _select_final_candidate(
        inputs=FinalSelectionInputs(
            es_metrics=es_metrics,
            calibration_metrics=calibration_metrics,
            crps_metrics=crps_metrics,
            ql_metrics=ql_metrics,
            complexity=complexity,
        ),
        model_selection=context.model_selection,
    )
    metrics_payload = _build_metrics_payload(
        es_metrics=es_metrics,
        calibration_metrics=calibration_metrics,
        crps_metrics=crps_metrics,
        ql_metrics=ql_metrics,
        complexity=complexity,
    )
    context.artifacts.write_global_metrics(payload=metrics_payload)
    context.artifacts.write_global_selection(payload=selection)
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
        for candidate_id, values in filtered
        if values["es_model"] <= threshold
    ]
    if not survivors:
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
    secondary = _secondary_scores(
        survivors, inputs.crps_metrics, inputs.ql_metrics
    )
    survivors2 = _secondary_survivors(secondary)
    best_id = _pick_best_by_complexity(
        survivors2, inputs.complexity, inputs.es_metrics
    )
    return {
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


def _build_metrics_payload(
    *,
    es_metrics: Mapping[int, Mapping[str, float]],
    calibration_metrics: Mapping[int, Mapping[str, float]],
    crps_metrics: Mapping[int, float],
    ql_metrics: Mapping[int, float],
    complexity: Mapping[int, float],
) -> Mapping[str, Any]:
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
            "complexity": float(
                complexity.get(candidate_id, float("nan"))
            ),
        }
    return payload


def _load_outer_metrics(
    *, base_dir: Path, outer_ids: Sequence[int]
) -> list[Mapping[str, Mapping[str, float]]]:
    metrics_list: list[Mapping[str, Mapping[str, float]]] = []
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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SimulationError(
            "Failed to read postprocess metrics",
            context={"path": str(path)},
        ) from exc


def _aggregate_global_metrics(
    metrics_list: Sequence[Mapping[str, Mapping[str, float]]],
) -> tuple[
    Mapping[int, Mapping[str, float]],
    Mapping[int, Mapping[str, float]],
    Mapping[int, float],
    Mapping[int, float],
    Mapping[int, float],
]:
    candidate_ids = _collect_candidate_ids(metrics_list)
    es_metrics: dict[int, Mapping[str, float]] = {}
    calibration_metrics: dict[int, Mapping[str, float]] = {}
    crps_metrics: dict[int, float] = {}
    ql_metrics: dict[int, float] = {}
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
        complexity_metrics[candidate_id] = _median_or_nan(complexity_vals)
    return (
        es_metrics,
        calibration_metrics,
        crps_metrics,
        ql_metrics,
        complexity_metrics,
    )


def _collect_calibration_metric_keys(
    metrics_list: Sequence[Mapping[str, Mapping[str, float]]],
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


def _collect_candidate_ids(
    metrics_list: Sequence[Mapping[str, Mapping[str, float]]],
) -> list[int]:
    ids: set[int] = set()
    for metrics in metrics_list:
        ids.update({int(key) for key in metrics.keys()})
    if not ids:
        raise SimulationError("Postprocess metrics missing candidates")
    return sorted(ids)


def _collect_metric(
    metrics_list: Sequence[Mapping[str, Mapping[str, float]]],
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
