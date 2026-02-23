from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from algo_trader.domain.simulation import (
    CPCVSplit,
    FeatureCleaningState,
    PreprocessSpec,
    RobustScalerState,
)
from .preprocessing_scaling import (
    _apply_breakout_transform,
    _apply_constant_guardrail,
    _apply_winsorization_2d,
    _apply_winsorization_3d,
    _clip_scaled_features,
    _compute_breakout_params,
    _compute_near_constant_mask,
    _compute_scale_params,
    _compute_winsor_params,
    _empty_breakout_state,
    _empty_winsor_state,
    _resolve_feature_names,
    _validate_scaled_features,
)
from .preprocessing_stats import _nanmedian
from .summary_utils import build_cleaning_thresholds


@dataclass(frozen=True)
class TransformState:
    cleaning: FeatureCleaningState
    scaler: RobustScalerState
    spec: PreprocessSpec


@dataclass(frozen=True)
class CleaningDrops:
    dropped_low_usable: np.ndarray
    dropped_low_var: np.ndarray
    dropped_duplicates: np.ndarray
    duplicate_pairs: list[tuple[int, int, float]]



def _masked_variance_1d(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n = m.sum(dim=0).clamp_min(0)
    x0 = torch.where(m, x, torch.zeros_like(x))
    mean = x0.sum(dim=0) / n.clamp_min(1)
    dev2 = torch.where(m, (x - mean) ** 2, torch.zeros_like(x)).sum(dim=0)
    denom = (n - 1).clamp_min(1)
    var = dev2 / denom
    return torch.where(n >= 2, var, torch.zeros_like(var))


def _flatten_training_slice(
    X: torch.Tensor,
    M: torch.Tensor,
    train_idx: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tidx = torch.as_tensor(train_idx, dtype=torch.long, device=X.device)
    Xtr = X.index_select(dim=0, index=tidx)
    Mtr = M.index_select(dim=0, index=tidx)
    F = X.shape[-1]
    return Xtr.reshape(-1, F), Mtr.reshape(-1, F)


def _compute_basic_stats(
    X2: torch.Tensor, M2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    usable_ratio = M2.float().mean(dim=0)
    variance = _masked_variance_1d(X2, M2)
    return usable_ratio, variance


def _validate_inputs(X: torch.Tensor, M: torch.Tensor) -> None:
    if X.ndim != 3:
        raise ValueError("X must be [T, A, F]")
    if M.shape != X.shape:
        raise ValueError("M must have same shape as X")
    if M.dtype != torch.bool:
        raise ValueError("M must be boolean")


def _observed_mask(
    missing_mask: torch.Tensor, values: torch.Tensor | None = None
) -> torch.Tensor:
    observed = ~missing_mask
    if values is None:
        return observed
    return observed & torch.isfinite(values)


@dataclass(frozen=True)
class InnerCleaningSummaryContext:
    X: torch.Tensor
    M: torch.Tensor
    inner_splits: Sequence[CPCVSplit]
    spec: PreprocessSpec
    feature_names: Sequence[str]
    outer_k: int


def summarize_inner_cleaning(
    context: InnerCleaningSummaryContext,
) -> dict[str, Any]:
    cleanings = [
        fit_feature_cleaning(
            X=context.X,
            M=context.M,
            train_idx=split.train_idx,
            spec=context.spec,
            frozen_feature_idx=None,
        )
        for split in context.inner_splits
    ]
    return _build_inner_cleaning_summary(
        cleanings=cleanings,
        feature_names=context.feature_names,
        spec=context.spec,
        outer_k=context.outer_k,
    )


def _build_inner_cleaning_summary(
    *,
    cleanings: Sequence[FeatureCleaningState],
    feature_names: Sequence[str],
    spec: PreprocessSpec,
    outer_k: int,
) -> dict[str, Any]:
    total_features = len(feature_names)
    usable_matrix = _stack_feature_stats(cleanings, "usable_ratio", total_features)
    variance_matrix = _stack_feature_stats(cleanings, "variance", total_features)
    drop_counts = _aggregate_drop_counts(cleanings, total_features)
    kept_counts = _aggregate_kept_counts(cleanings, total_features)
    features = _build_feature_summaries(
        feature_names=feature_names,
        usable_matrix=usable_matrix,
        variance_matrix=variance_matrix,
        drop_counts=drop_counts,
        kept_counts=kept_counts,
    )
    split_stats = _summarize_split_counts(cleanings)
    return {
        "outer_k": outer_k,
        "split_count": len(cleanings),
        "n_features_total": total_features,
        "split_stats": split_stats,
        "thresholds": build_cleaning_thresholds(spec),
        "features": features,
    }


def _stack_feature_stats(
    cleanings: Sequence[FeatureCleaningState],
    field: str,
    total_features: int,
) -> np.ndarray:
    if not cleanings:
        return np.empty((0, total_features), dtype=float)
    rows = [np.asarray(getattr(cleaning, field), dtype=float) for cleaning in cleanings]
    return np.vstack(rows)


def _aggregate_drop_counts(
    cleanings: Sequence[FeatureCleaningState], total_features: int
) -> dict[str, np.ndarray]:
    low_usable = np.zeros(total_features, dtype=int)
    low_var = np.zeros(total_features, dtype=int)
    duplicates = np.zeros(total_features, dtype=int)
    for cleaning in cleanings:
        _increment_counts(low_usable, cleaning.dropped_low_usable)
        _increment_counts(low_var, cleaning.dropped_low_var)
        _increment_counts(duplicates, cleaning.dropped_duplicates)
    return {
        "low_usable": low_usable,
        "low_variance": low_var,
        "duplicate": duplicates,
    }


def _aggregate_kept_counts(
    cleanings: Sequence[FeatureCleaningState], total_features: int
) -> np.ndarray:
    kept = np.zeros(total_features, dtype=int)
    for cleaning in cleanings:
        _increment_counts(kept, cleaning.feature_idx)
    return kept


def _increment_counts(counts: np.ndarray, indices: np.ndarray) -> None:
    if indices.size == 0:
        return
    counts[np.asarray(indices, dtype=int)] += 1


def _build_feature_summaries(
    *,
    feature_names: Sequence[str],
    usable_matrix: np.ndarray,
    variance_matrix: np.ndarray,
    drop_counts: dict[str, np.ndarray],
    kept_counts: np.ndarray,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        summaries.append(
            {
                "name": name,
                "usable_ratio": _summarize_vector(usable_matrix[:, idx]),
                "variance": _summarize_vector(variance_matrix[:, idx]),
                "kept_count": int(kept_counts[idx]),
                "dropped_low_usable_count": int(drop_counts["low_usable"][idx]),
                "dropped_low_variance_count": int(drop_counts["low_variance"][idx]),
                "dropped_duplicate_count": int(drop_counts["duplicate"][idx]),
            }
        )
    return summaries


def _summarize_vector(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0 or np.all(np.isnan(values)):
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
    }


def _summarize_split_counts(
    cleanings: Sequence[FeatureCleaningState],
) -> dict[str, float | None]:
    if not cleanings:
        return {"min_kept": None, "median_kept": None, "max_kept": None, "mean_kept": None}
    kept_counts = np.asarray([cleaning.feature_idx.size for cleaning in cleanings])
    return {
        "min_kept": float(np.min(kept_counts)),
        "median_kept": float(np.median(kept_counts)),
        "max_kept": float(np.max(kept_counts)),
        "mean_kept": float(np.mean(kept_counts)),
    }


def _apply_usability_filter(
    usable_ratio: torch.Tensor, min_usable_ratio: float
) -> Tuple[torch.Tensor, np.ndarray]:
    keep = usable_ratio >= min_usable_ratio
    dropped = torch.where(~keep)[0].detach().cpu().numpy()
    return keep, dropped


def _apply_variance_filter(
    variance: torch.Tensor, keep_usable: torch.Tensor, min_variance: float
) -> Tuple[torch.Tensor, np.ndarray]:
    keep_var = variance >= min_variance
    keep = keep_usable & keep_var
    dropped = torch.where(keep_usable & ~keep_var)[0].detach().cpu().numpy()
    return keep, dropped


def _prepare_corr_inputs(
    X2: torch.Tensor,
    M2: torch.Tensor,
    cand: torch.Tensor,
    corr_subsample: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = X2.shape[0]
    if corr_subsample is not None and corr_subsample < N:
        idx_lin = torch.linspace(0, N - 1, steps=corr_subsample, device=X2.device)
        sub = idx_lin.round().long().clamp(0, N - 1)
        Xc = X2.index_select(dim=0, index=sub)
        Mc = M2.index_select(dim=0, index=sub)
    else:
        Xc, Mc = X2, M2
    return Xc[:, cand], Mc[:, cand]


def _impute_and_standardize(
    Xc: torch.Tensor, Mc: torch.Tensor
) -> torch.Tensor:
    Xc_nan = Xc.masked_fill(~Mc, float("nan"))
    med = _nanmedian(Xc_nan, dim=0)
    Xc_imp = torch.where(Mc, Xc, med)
    mu = Xc_imp.mean(dim=0)
    sd = Xc_imp.std(dim=0, unbiased=False).clamp_min(1e-12)
    return (Xc_imp - mu) / sd


def _compute_abs_corr(Z: torch.Tensor) -> np.ndarray:
    denom = max(int(Z.shape[0] - 1), 1)
    corr = (Z.transpose(0, 1) @ Z) / float(denom)
    return corr.abs().detach().cpu().numpy()


def _select_features_with_duplicates(
    X2: torch.Tensor,
    M2: torch.Tensor,
    cand: torch.Tensor,
    spec: PreprocessSpec,
    usable_ratio: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    Xc, Mc = _prepare_corr_inputs(
        X2, M2, cand, spec.cleaning.corr_subsample
    )
    Z = _impute_and_standardize(Xc, Mc)
    abs_corr = _compute_abs_corr(Z)
    cand_np = cand.detach().cpu().numpy()
    usable_np = usable_ratio.detach().cpu().numpy()
    return _prune_duplicates(
        cand_np, usable_np, abs_corr, float(spec.cleaning.max_abs_corr)
    )


def _prune_duplicates(
    cand: np.ndarray,
    usable_ratio: np.ndarray,
    abs_corr: np.ndarray,
    max_abs_corr: float,
) -> Tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    order = sorted(
        list(range(len(cand))),
        key=lambda k: (-usable_ratio[cand[k]], cand[k]),
    )
    dropped = np.zeros(len(cand), dtype=bool)
    duplicate_pairs: list[tuple[int, int, float]] = []
    for ii in order:
        if dropped[ii]:
            continue
        row = abs_corr[ii]
        high = np.where((row > max_abs_corr) & (~dropped))[0]
        high = high[high != ii]
        for jj in high:
            dropped[jj] = True
            duplicate_pairs.append(
                (int(cand[ii]), int(cand[jj]), float(row[jj]))
            )
    kept_mask = ~dropped
    return cand[kept_mask], cand[dropped], duplicate_pairs


def fit_feature_cleaning(
    X: torch.Tensor,
    M: torch.Tensor,
    train_idx: np.ndarray,
    spec: PreprocessSpec,
    frozen_feature_idx: Optional[np.ndarray] = None,
) -> FeatureCleaningState:
    _validate_inputs(X, M)

    observed = _observed_mask(M, X)
    X2, M2 = _flatten_training_slice(X, observed, train_idx)
    usable_ratio, variance = _compute_basic_stats(X2, M2)

    if frozen_feature_idx is not None:
        return _frozen_cleaning_state(frozen_feature_idx, usable_ratio, variance)

    return _compute_cleaning_state(X2, M2, usable_ratio, variance, spec)


def _frozen_cleaning_state(
    frozen_feature_idx: np.ndarray | list[int],
    usable_ratio: torch.Tensor,
    variance: torch.Tensor,
) -> FeatureCleaningState:
    fidx = np.asarray(frozen_feature_idx, dtype=int)
    return FeatureCleaningState(
        feature_idx=fidx,
        usable_ratio=usable_ratio.detach().cpu().numpy(),
        variance=variance.detach().cpu().numpy(),
        dropped_low_usable=np.array([], dtype=int),
        dropped_low_var=np.array([], dtype=int),
        dropped_duplicates=np.array([], dtype=int),
        duplicate_pairs=[],
    )


def _compute_cleaning_state(
    X2: torch.Tensor,
    M2: torch.Tensor,
    usable_ratio: torch.Tensor,
    variance: torch.Tensor,
    spec: PreprocessSpec,
) -> FeatureCleaningState:
    keep_usable, dropped_low_usable = _apply_usability_filter(
        usable_ratio, spec.cleaning.min_usable_ratio
    )
    keep_final, dropped_low_var = _apply_variance_filter(
        variance, keep_usable, spec.cleaning.min_variance
    )
    cand = torch.where(keep_final)[0]
    if cand.numel() == 0:
        return _empty_cleaning_state(usable_ratio, variance, dropped_low_usable, dropped_low_var)

    feature_idx, dropped_duplicates, duplicate_pairs = (
        _select_features_with_duplicates(X2, M2, cand, spec, usable_ratio)
    )
    drops = CleaningDrops(
        dropped_low_usable=dropped_low_usable,
        dropped_low_var=dropped_low_var,
        dropped_duplicates=dropped_duplicates,
        duplicate_pairs=duplicate_pairs,
    )
    return _final_cleaning_state(
        feature_idx=feature_idx,
        usable_ratio=usable_ratio,
        variance=variance,
        drops=drops,
    )


def _empty_cleaning_state(
    usable_ratio: torch.Tensor,
    variance: torch.Tensor,
    dropped_low_usable: np.ndarray,
    dropped_low_var: np.ndarray,
) -> FeatureCleaningState:
    return FeatureCleaningState(
        feature_idx=np.array([], dtype=int),
        usable_ratio=usable_ratio.detach().cpu().numpy(),
        variance=variance.detach().cpu().numpy(),
        dropped_low_usable=dropped_low_usable,
        dropped_low_var=dropped_low_var,
        dropped_duplicates=np.array([], dtype=int),
        duplicate_pairs=[],
    )


def _final_cleaning_state(
    *,
    feature_idx: np.ndarray,
    usable_ratio: torch.Tensor,
    variance: torch.Tensor,
    drops: CleaningDrops,
) -> FeatureCleaningState:
    return FeatureCleaningState(
        feature_idx=np.asarray(feature_idx, dtype=int),
        usable_ratio=usable_ratio.detach().cpu().numpy(),
        variance=variance.detach().cpu().numpy(),
        dropped_low_usable=np.asarray(drops.dropped_low_usable, dtype=int),
        dropped_low_var=np.asarray(drops.dropped_low_var, dtype=int),
        dropped_duplicates=np.asarray(drops.dropped_duplicates, dtype=int),
        duplicate_pairs=drops.duplicate_pairs,
    )


def fit_robust_scaler(
    X: torch.Tensor,
    M: torch.Tensor,
    train_idx: np.ndarray,
    cleaning: FeatureCleaningState,
    spec: PreprocessSpec,
) -> RobustScalerState:
    device = X.device
    fidx_np = cleaning.feature_idx
    if fidx_np.size == 0:
        return _empty_robust_scaler(fidx_np, device, spec)

    observed = _observed_mask(M, X)
    Xtr, Mtr = _select_training_matrix(
        X=X, observed=observed, train_idx=train_idx, feature_idx=fidx_np
    )
    winsor = _compute_winsor_params(
        Xtr=Xtr,
        Mtr=Mtr,
        feature_idx=fidx_np,
        spec=spec,
    )
    Xtr = _apply_winsorization_2d(Xtr=Xtr, Mtr=Mtr, winsor=winsor)

    shift, scale = _compute_scale_params(
        Xtr=Xtr,
        Mtr=Mtr,
        feature_idx=fidx_np,
        spec=spec,
        total_features=X.shape[-1],
    )
    breakout = _compute_breakout_params(
        X=X,
        observed=observed,
        train_idx=train_idx,
        feature_idx=fidx_np,
        spec=spec,
    )
    near_constant_mask = _compute_near_constant_mask(
        X=X,
        observed=observed,
        train_idx=train_idx,
        feature_idx=fidx_np,
        spec=spec,
    )

    return RobustScalerState(
        feature_idx=fidx_np,
        shift=shift,
        scale=scale,
        mad_eps=spec.scaling.mad_eps,
        breakout=breakout,
        winsor=winsor,
        near_constant_mask=near_constant_mask,
    )


def _empty_robust_scaler(
    feature_idx: np.ndarray, device: torch.device, spec: PreprocessSpec
) -> RobustScalerState:
    return RobustScalerState(
        feature_idx=feature_idx,
        shift=torch.zeros(0, device=device),
        scale=torch.ones(0, device=device),
        mad_eps=spec.scaling.mad_eps,
        breakout=_empty_breakout_state(device, spec.scaling.breakout_var_floor),
        winsor=_empty_winsor_state(device),
        near_constant_mask=torch.zeros((0, 0), dtype=torch.bool, device=device),
    )


def _select_training_matrix(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    train_idx: np.ndarray,
    feature_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = X.device
    fidx = torch.as_tensor(feature_idx, dtype=torch.long, device=device)
    tidx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    Xtr = X.index_select(dim=0, index=tidx).reshape(-1, X.shape[-1])[:, fidx]
    Mtr = (
        observed.index_select(dim=0, index=tidx)
        .reshape(-1, X.shape[-1])[:, fidx]
    )
    return Xtr, Mtr


def transform_X(
    X: torch.Tensor,
    M: torch.Tensor,
    idx: np.ndarray,
    state: TransformState,
    validate: bool = False,
) -> torch.Tensor:
    device = X.device
    fidx_np = state.cleaning.feature_idx
    if fidx_np.size == 0:
        n = len(idx)
        A = X.shape[1]
        return torch.zeros((n, A, 0), device=device, dtype=X.dtype)

    observed = _observed_mask(M, X)
    X_sel, M_sel = _select_transform_tensors(
        X=X, observed=observed, idx=idx, feature_idx=fidx_np
    )
    X_sel = _apply_winsorization_3d(
        X_sel=X_sel, M_sel=M_sel, winsor=state.scaler.winsor
    )

    X_scaled = (X_sel - state.scaler.shift) / state.scaler.scale
    X_scaled = _apply_breakout_transform(
        X_scaled=X_scaled, X_sel=X_sel, breakout=state.scaler.breakout
    )
    X_scaled = _apply_constant_guardrail(
        X_scaled=X_scaled, mask=state.scaler.near_constant_mask
    )
    if validate:
        feature_names = _resolve_feature_names(
            state.spec.scaling.inputs.feature_names, fidx_np
        )
        _validate_scaled_features(
            X_scaled=X_scaled,
            observed=M_sel,
            feature_names=feature_names,
            max_abs_fail=state.spec.scaling.clip.max_abs_fail,
        )
    X_scaled = _clip_scaled_features(X_scaled, state.spec)
    if state.spec.scaling.inputs.impute_missing_to_zero:
        X_scaled = torch.where(M_sel, X_scaled, torch.zeros_like(X_scaled))

    if state.spec.scaling.inputs.append_mask_as_features:
        mask_f = M_sel.to(dtype=X_scaled.dtype)
        X_scaled = torch.cat([X_scaled, mask_f], dim=-1)

    return X_scaled


def _select_transform_tensors(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    idx: np.ndarray,
    feature_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = X.device
    fidx = torch.as_tensor(feature_idx, dtype=torch.long, device=device)
    iidx = torch.as_tensor(idx, dtype=torch.long, device=device)
    X_sel = X.index_select(dim=0, index=iidx).index_select(dim=2, index=fidx)
    M_sel = (
        observed.index_select(dim=0, index=iidx)
        .index_select(dim=2, index=fidx)
    )
    return X_sel, M_sel
