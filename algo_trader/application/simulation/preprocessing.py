from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from algo_trader.domain.simulation import (
    FeatureCleaningState,
    PreprocessSpec,
    RobustScalerState,
)


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


def _nanmedian(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nanmedian(x, dim=dim).values


def _masked_variance_1d(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n = m.sum(dim=0).clamp_min(0)
    x0 = torch.where(m, x, torch.zeros_like(x))
    mean = x0.sum(dim=0) / n.clamp_min(1)
    dev2 = torch.where(m, (x - mean) ** 2, torch.zeros_like(x)).sum(dim=0)
    denom = (n - 1).clamp_min(1)
    var = dev2 / denom
    return torch.where(n >= 2, var, torch.zeros_like(var))


def _infer_scale_policy(
    feature_names: Optional[List[str]], F: int
) -> np.ndarray:
    if feature_names is None:
        return np.zeros(F, dtype=int)
    if len(feature_names) != F:
        raise ValueError("feature_names length must match F")
    policy = np.zeros(F, dtype=int)
    for i, name in enumerate(feature_names):
        if name.startswith("brk_") or name.startswith("cs_rank_"):
            policy[i] = 1
    return policy


def _compute_scale_params(
    *,
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    feature_idx: np.ndarray,
    spec: PreprocessSpec,
    total_features: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_nan = Xtr.masked_fill(~Mtr, float("nan"))
    med = _nanmedian(X_nan, dim=0)
    mad = _nanmedian(torch.abs(X_nan - med), dim=0)

    policy_all = _infer_scale_policy(spec.scaling.feature_names, F=total_features)
    policy_clean = torch.as_tensor(
        policy_all[feature_idx], dtype=torch.long, device=Xtr.device
    )

    shift = torch.where(policy_clean == 0, med, torch.zeros_like(med))
    scale = torch.where(
        policy_clean == 0, mad + spec.scaling.mad_eps, torch.ones_like(mad)
    )

    return shift, scale.clamp_min(spec.scaling.mad_eps)


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


def _observed_mask(missing_mask: torch.Tensor) -> torch.Tensor:
    return ~missing_mask


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

    observed = _observed_mask(M)
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
        return RobustScalerState(
            feature_idx=fidx_np,
            shift=torch.zeros(0, device=device),
            scale=torch.ones(0, device=device),
            mad_eps=spec.scaling.mad_eps,
        )

    observed = _observed_mask(M)
    fidx = torch.as_tensor(fidx_np, dtype=torch.long, device=device)
    tidx = torch.as_tensor(train_idx, dtype=torch.long, device=device)

    Xtr = X.index_select(dim=0, index=tidx).reshape(-1, X.shape[-1])[:, fidx]
    Mtr = (
        observed.index_select(dim=0, index=tidx)
        .reshape(-1, X.shape[-1])[:, fidx]
    )

    shift, scale = _compute_scale_params(
        Xtr=Xtr,
        Mtr=Mtr,
        feature_idx=fidx_np,
        spec=spec,
        total_features=X.shape[-1],
    )

    return RobustScalerState(
        feature_idx=fidx_np,
        shift=shift,
        scale=scale,
        mad_eps=spec.scaling.mad_eps,
    )


def transform_X(
    X: torch.Tensor,
    M: torch.Tensor,
    idx: np.ndarray,
    state: TransformState,
) -> torch.Tensor:
    device = X.device
    fidx_np = state.cleaning.feature_idx
    if fidx_np.size == 0:
        n = len(idx)
        A = X.shape[1]
        return torch.zeros((n, A, 0), device=device, dtype=X.dtype)

    observed = _observed_mask(M)
    fidx = torch.as_tensor(fidx_np, dtype=torch.long, device=device)
    iidx = torch.as_tensor(idx, dtype=torch.long, device=device)

    X_sel = X.index_select(dim=0, index=iidx).index_select(dim=2, index=fidx)
    M_sel = (
        observed.index_select(dim=0, index=iidx)
        .index_select(dim=2, index=fidx)
    )

    X_scaled = (X_sel - state.scaler.shift) / state.scaler.scale
    if state.spec.scaling.impute_missing_to_zero:
        X_scaled = torch.where(M_sel, X_scaled, torch.zeros_like(X_scaled))

    if state.spec.scaling.append_mask_as_features:
        mask_f = M_sel.to(dtype=X_scaled.dtype)
        X_scaled = torch.cat([X_scaled, mask_f], dim=-1)

    return X_scaled
