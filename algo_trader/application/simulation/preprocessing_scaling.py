from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import (
    BreakoutScaleState,
    GuardrailSpec,
    PreprocessSpec,
    WinsorState,
)
from .preprocessing_stats import _nanmedian

logger = logging.getLogger(__name__)


def _base_feature_name(name: str) -> str:
    return name.split("::", 1)[-1]


def _infer_scale_policy(
    feature_names: Optional[List[str]], F: int
) -> np.ndarray:
    if feature_names is None:
        return np.zeros(F, dtype=int)
    if len(feature_names) != F:
        raise ValueError("feature_names length must match F")
    policy = np.zeros(F, dtype=int)
    for i, name in enumerate(feature_names):
        base_name = _base_feature_name(name)
        if base_name.startswith("brk_") or base_name.startswith("cs_rank_"):
            policy[i] = 1
    return policy


def _infer_breakout_positions(
    feature_names: Optional[List[str]], feature_idx: np.ndarray
) -> np.ndarray:
    if feature_names is None:
        return np.array([], dtype=int)
    breakout_positions: list[int] = []
    for pos, raw_idx in enumerate(feature_idx):
        base_name = _base_feature_name(feature_names[int(raw_idx)])
        if base_name.startswith("brk_"):
            breakout_positions.append(pos)
    return np.asarray(breakout_positions, dtype=int)


def _infer_winsor_positions(
    feature_names: Optional[List[str]], feature_idx: np.ndarray
) -> np.ndarray:
    if feature_names is None:
        return np.array([], dtype=int)
    positions: list[int] = []
    for pos, raw_idx in enumerate(feature_idx):
        base_name = _base_feature_name(feature_names[int(raw_idx)])
        if not base_name.startswith("brk_"):
            positions.append(pos)
    return np.asarray(positions, dtype=int)


def _compute_scale_params(
    *,
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    feature_idx: np.ndarray,
    spec: PreprocessSpec,
    total_features: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_nan = Xtr.masked_fill(~Mtr, float("nan"))
    med, scale_base = _robust_scale_stats(
        X_nan, scale_floor=spec.scaling.scale_floor
    )

    policy_all = _infer_scale_policy(
        spec.scaling.inputs.feature_names, F=total_features
    )
    policy_clean = torch.as_tensor(
        policy_all[feature_idx], dtype=torch.long, device=Xtr.device
    )

    shift = torch.where(policy_clean == 0, med, torch.zeros_like(med))
    scale = torch.where(
        policy_clean == 0, scale_base, torch.ones_like(scale_base)
    )

    return shift, scale.clamp_min(spec.scaling.mad_eps)


def _robust_scale_stats(
    X_nan: torch.Tensor, scale_floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    med = _nanmedian(X_nan, dim=0)
    mad = _nanmedian(torch.abs(X_nan - med), dim=0)
    s_mad = 1.4826 * mad
    q25 = torch.nanquantile(X_nan, 0.25, dim=0)
    q75 = torch.nanquantile(X_nan, 0.75, dim=0)
    s_iqr = (q75 - q25) / 1.349
    s_mad = torch.nan_to_num(s_mad, nan=0.0, posinf=0.0, neginf=0.0)
    s_iqr = torch.nan_to_num(s_iqr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.maximum(s_mad, s_iqr)
    floor = torch.full_like(scale, scale_floor)
    return med, torch.maximum(scale, floor)


def _compute_breakout_params(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    train_idx: np.ndarray,
    feature_idx: np.ndarray,
    spec: PreprocessSpec,
) -> BreakoutScaleState:
    breakout_pos = _infer_breakout_positions(
        spec.scaling.inputs.feature_names, feature_idx
    )
    var_floor = spec.scaling.breakout_var_floor
    if breakout_pos.size == 0:
        return _empty_breakout_state(X.device, var_floor)
    Xb, Mb = _select_breakout_training_tensors(
        X=X,
        observed=observed,
        train_idx=train_idx,
        feature_idx=feature_idx,
        breakout_pos=breakout_pos,
    )
    p, denom, const = _breakout_stats(Xb=Xb, Mb=Mb, var_floor=var_floor)
    return BreakoutScaleState(
        positions=breakout_pos,
        p=p,
        denom=denom,
        const=const,
        var_floor=var_floor,
    )


def _empty_breakout_state(
    device: torch.device, var_floor: float
) -> BreakoutScaleState:
    return BreakoutScaleState(
        positions=np.array([], dtype=int),
        p=torch.zeros((0,), device=device),
        denom=torch.zeros((0,), device=device),
        const=torch.zeros((0,), device=device, dtype=torch.bool),
        var_floor=var_floor,
    )


def _select_breakout_training_tensors(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    train_idx: np.ndarray,
    feature_idx: np.ndarray,
    breakout_pos: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = X.device
    tidx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    fidx = torch.as_tensor(
        feature_idx[breakout_pos], dtype=torch.long, device=device
    )
    Xb = X.index_select(dim=0, index=tidx).index_select(dim=2, index=fidx)
    Mb = observed.index_select(dim=0, index=tidx).index_select(dim=2, index=fidx)
    return Xb, Mb


def _breakout_stats(
    *, Xb: torch.Tensor, Mb: torch.Tensor, var_floor: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_count = Mb.sum(dim=0)
    x_sum = torch.where(Mb, Xb, torch.zeros_like(Xb)).sum(dim=0)
    p = torch.where(
        obs_count > 0, x_sum / obs_count.clamp_min(1), torch.zeros_like(x_sum)
    )
    var = p * (1.0 - p)
    const = (obs_count == 0) | (var <= 0)
    denom = torch.sqrt(var + var_floor)
    return p, denom, const


def _compute_winsor_params(
    *,
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    feature_idx: np.ndarray,
    spec: PreprocessSpec,
) -> WinsorState:
    positions = _infer_winsor_positions(
        spec.scaling.inputs.feature_names, feature_idx
    )
    if positions.size == 0:
        return _empty_winsor_state(Xtr.device)
    X_nan = Xtr.masked_fill(~Mtr, float("nan"))
    lower_q = spec.scaling.winsor.lower_q
    upper_q = spec.scaling.winsor.upper_q
    q_low = torch.nanquantile(X_nan, lower_q, dim=0)
    q_high = torch.nanquantile(X_nan, upper_q, dim=0)
    q_low, q_high = _sanitize_winsor_bounds(q_low, q_high)
    pos = torch.as_tensor(positions, dtype=torch.long, device=Xtr.device)
    return WinsorState(
        positions=positions,
        lower=q_low.index_select(dim=0, index=pos),
        upper=q_high.index_select(dim=0, index=pos),
    )


def _empty_winsor_state(device: torch.device) -> WinsorState:
    return WinsorState(
        positions=np.array([], dtype=int),
        lower=torch.zeros((0,), device=device),
        upper=torch.zeros((0,), device=device),
    )


def _sanitize_winsor_bounds(
    q_low: torch.Tensor, q_high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.nan_to_num(q_low, nan=float("-inf"))
    high = torch.nan_to_num(q_high, nan=float("inf"))
    low, high = torch.minimum(low, high), torch.maximum(low, high)
    return low, high


def _compute_near_constant_mask(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    train_idx: np.ndarray,
    feature_idx: np.ndarray,
    spec: PreprocessSpec,
) -> torch.Tensor:
    if feature_idx.size == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=X.device)
    Xtr, Mtr = _select_guardrail_tensors(
        X=X, observed=observed, train_idx=train_idx, feature_idx=feature_idx
    )
    X_nan = Xtr.masked_fill(~Mtr, float("nan"))
    scale, median_abs, obs_count = _guardrail_scale_stats(X_nan, Mtr)
    return _guardrail_mask(
        scale=scale,
        median_abs=median_abs,
        obs_count=obs_count,
        guardrail=spec.scaling.guardrail,
    )


def _select_guardrail_tensors(
    *,
    X: torch.Tensor,
    observed: torch.Tensor,
    train_idx: np.ndarray,
    feature_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = X.device
    tidx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    fidx = torch.as_tensor(feature_idx, dtype=torch.long, device=device)
    Xtr = X.index_select(dim=0, index=tidx).index_select(dim=2, index=fidx)
    Mtr = observed.index_select(dim=0, index=tidx).index_select(dim=2, index=fidx)
    return Xtr, Mtr


def _guardrail_scale_stats(
    X_nan: torch.Tensor, Mtr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    med = _nanmedian(X_nan, dim=0)
    mad = _nanmedian(torch.abs(X_nan - med), dim=0)
    s_mad = 1.4826 * mad
    q25 = torch.nanquantile(X_nan, 0.25, dim=0)
    q75 = torch.nanquantile(X_nan, 0.75, dim=0)
    s_iqr = (q75 - q25) / 1.349
    scale = torch.maximum(s_mad, s_iqr)
    median_abs = _nanmedian(torch.abs(X_nan), dim=0)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    median_abs = torch.nan_to_num(median_abs, nan=0.0, posinf=0.0, neginf=0.0)
    obs_count = Mtr.sum(dim=0)
    return scale, median_abs, obs_count


def _guardrail_mask(
    *,
    scale: torch.Tensor,
    median_abs: torch.Tensor,
    obs_count: torch.Tensor,
    guardrail: GuardrailSpec,
) -> torch.Tensor:
    near_abs = scale < guardrail.abs_eps
    near_rel = scale < (guardrail.rel_eps * median_abs + guardrail.rel_offset)
    return near_abs | near_rel | (obs_count == 0)


def _apply_breakout_transform(
    *,
    X_scaled: torch.Tensor,
    X_sel: torch.Tensor,
    breakout: BreakoutScaleState,
) -> torch.Tensor:
    if breakout.positions.size == 0:
        return X_scaled
    device = X_sel.device
    bpos = torch.as_tensor(breakout.positions, dtype=torch.long, device=device)
    p = breakout.p.to(device=device, dtype=X_sel.dtype)
    denom = breakout.denom.to(device=device, dtype=X_sel.dtype)
    const = breakout.const.to(device=device)
    Xb = X_sel.index_select(dim=2, index=bpos)
    Xb_scaled = (Xb - p.unsqueeze(0)) / denom.unsqueeze(0)
    Xb_scaled = torch.where(
        const.unsqueeze(0), torch.zeros_like(Xb_scaled), Xb_scaled
    )
    updated = X_scaled.clone()
    updated.index_copy_(dim=2, index=bpos, source=Xb_scaled)
    return updated


def _apply_winsorization_2d(
    *, Xtr: torch.Tensor, Mtr: torch.Tensor, winsor: WinsorState
) -> torch.Tensor:
    if winsor.positions.size == 0:
        return Xtr
    pos = torch.as_tensor(winsor.positions, dtype=torch.long, device=Xtr.device)
    lower = winsor.lower.to(device=Xtr.device, dtype=Xtr.dtype)
    upper = winsor.upper.to(device=Xtr.device, dtype=Xtr.dtype)
    X_sel = Xtr.index_select(dim=1, index=pos)
    M_sel = Mtr.index_select(dim=1, index=pos)
    X_clip = torch.clamp(X_sel, min=lower, max=upper)
    X_clip = torch.where(M_sel, X_clip, X_sel)
    updated = Xtr.clone()
    updated.index_copy_(dim=1, index=pos, source=X_clip)
    return updated


def _apply_winsorization_3d(
    *, X_sel: torch.Tensor, M_sel: torch.Tensor, winsor: WinsorState
) -> torch.Tensor:
    if winsor.positions.size == 0:
        return X_sel
    pos = torch.as_tensor(winsor.positions, dtype=torch.long, device=X_sel.device)
    lower = winsor.lower.to(device=X_sel.device, dtype=X_sel.dtype)
    upper = winsor.upper.to(device=X_sel.device, dtype=X_sel.dtype)
    X_feat = X_sel.index_select(dim=2, index=pos)
    M_feat = M_sel.index_select(dim=2, index=pos)
    X_clip = torch.clamp(X_feat, min=lower, max=upper)
    X_clip = torch.where(M_feat, X_clip, X_feat)
    updated = X_sel.clone()
    updated.index_copy_(dim=2, index=pos, source=X_clip)
    return updated


def _apply_constant_guardrail(
    *, X_scaled: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.numel() == 0:
        return X_scaled
    guard = mask.to(device=X_scaled.device)
    return X_scaled.masked_fill(guard.unsqueeze(0), 0.0)


def _clip_scaled_features(
    X_scaled: torch.Tensor, spec: PreprocessSpec
) -> torch.Tensor:
    return X_scaled.clamp(
        min=spec.scaling.clip.min_value, max=spec.scaling.clip.max_value
    )


def _resolve_feature_names(
    feature_names: list[str] | None, feature_idx: np.ndarray
) -> list[str]:
    if feature_names is None:
        return [str(i) for i in feature_idx]
    return [feature_names[int(i)] for i in feature_idx]


def _validate_scaled_features(
    *,
    X_scaled: torch.Tensor,
    observed: torch.Tensor,
    feature_names: list[str],
    max_abs_fail: float,
) -> None:
    if observed.numel() == 0:
        return
    observed_mask = observed.to(dtype=torch.bool)
    if observed_mask.sum().item() == 0:
        return
    finite_mask = torch.isfinite(X_scaled)
    if not finite_mask[observed_mask].all():
        offenders = _collect_nonfinite_offenders(
            X_scaled=X_scaled, observed=observed_mask, feature_names=feature_names
        )
        logger.error(
            "event=preprocess.nonfinite_scaled_features context=%s",
            {"offenders": offenders},
        )
        raise SimulationError(
            "Non-finite values detected in scaled training features",
            context={"offenders": repr(offenders)},
        )
    max_abs = float(X_scaled.abs()[observed_mask].max().item())
    if max_abs <= max_abs_fail:
        return
    offenders = _collect_max_abs_offenders(
        X_scaled=X_scaled,
        observed=observed_mask,
        feature_names=feature_names,
        top_k=5,
    )
    logger.error(
        "event=preprocess.max_abs_scaled_features_exceeded context=%s",
        {"max_abs": max_abs, "max_abs_fail": max_abs_fail, "offenders": offenders},
    )
    raise SimulationError(
        "Scaled feature magnitude exceeded max_abs_fail",
        context={
            "max_abs": f"{max_abs:.6g}",
            "max_abs_fail": f"{max_abs_fail:.6g}",
        },
    )


def _collect_nonfinite_offenders(
    *, X_scaled: torch.Tensor, observed: torch.Tensor, feature_names: list[str]
) -> list[dict[str, float | int | str]]:
    bad = (~torch.isfinite(X_scaled)) & observed
    idx = bad.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return []
    idx = idx[:5]
    offenders: list[dict[str, float | int | str]] = []
    for row in idx:
        t, a, f = (int(row[0]), int(row[1]), int(row[2]))
        offenders.append(
            {
                "t": t,
                "a": a,
                "f": f,
                "name": feature_names[f],
                "value": float(X_scaled[t, a, f].item()),
            }
        )
    return offenders


def _collect_max_abs_offenders(
    *,
    X_scaled: torch.Tensor,
    observed: torch.Tensor,
    feature_names: list[str],
    top_k: int,
) -> list[dict[str, float | int | str]]:
    if observed.sum().item() == 0:
        return []
    masked = X_scaled.abs().masked_fill(~observed, float("-inf"))
    flat = masked.reshape(-1)
    k = min(top_k, int((flat > float("-inf")).sum().item()))
    if k <= 0:
        return []
    vals, idxs = torch.topk(flat, k)
    offenders: list[dict[str, float | int | str]] = []
    for rank in range(k):
        idx = int(idxs[rank])
        t, a, f = _decode_flat_index(idx, X_scaled.shape)
        offenders.append(
            {
                "t": int(t),
                "a": int(a),
                "f": int(f),
                "name": feature_names[f],
                "value": float(vals[rank].item()),
            }
        )
    return offenders


def _decode_flat_index(
    index: int, shape: Sequence[int]
) -> tuple[int, int, int]:
    a_dim = int(shape[1])
    f_dim = int(shape[2])
    t = index // (a_dim * f_dim)
    rem = index % (a_dim * f_dim)
    a = rem // f_dim
    f = rem % f_dim
    return int(t), int(a), int(f)
