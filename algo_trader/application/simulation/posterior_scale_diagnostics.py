from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from algo_trader.domain import SimulationError
from algo_trader.infrastructure import ensure_directory

from .target_space_transform import TargetSpace, TargetSpaceTransform
from .model_params import resolve_dof_shift

_TARGET_NORM_EPS = 1e-6
_SPLIT_PATTERN = re.compile(r"candidate_(\d+)_split_(\d+)\.pt")


@dataclass(frozen=True)
class _RunConfig:
    target_normalization: bool
    dof_shift: float


@dataclass(frozen=True)
class _RunInputs:
    base_dir: Path
    outer_ids: Sequence[int]
    candidate_id: int
    assets: Sequence[str]
    panel_targets: np.ndarray
    run_config: _RunConfig


@dataclass(frozen=True)
class _OuterContext:
    run: _RunInputs
    outer_k: int
    splits: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class _SplitContext:
    outer: _OuterContext
    split_id: int
    transform: TargetSpaceTransform
    z_true: np.ndarray
    z_samples: np.ndarray
    sigma_model: Mapping[str, np.ndarray]
    nu: Mapping[str, float]


def run_posterior_scale_diagnostics(
    *, base_dir: Path, outer_ids: Sequence[int], candidate_id: int
) -> None:
    output_dir = _ensure_output_dir(base_dir)
    run_inputs = _build_run_inputs(
        base_dir=base_dir,
        outer_ids=outer_ids,
        candidate_id=candidate_id,
    )
    rows: list[dict[str, Any]] = []
    for outer_k in run_inputs.outer_ids:
        outer_context = _build_outer_context(run_inputs, int(outer_k))
        rows.extend(_rows_for_outer(outer_context))
    if not rows:
        raise SimulationError(
            "No posterior scale diagnostics rows produced",
            context={"candidate_id": str(candidate_id)},
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "sigma_residual_dispersion.csv", index=False
    )


def _ensure_output_dir(base_dir: Path) -> Path:
    target_dir = (
        base_dir / "outer" / "diagnostics" / "calibration_cpcv_ensemble"
    )
    ensure_directory(
        target_dir,
        error_type=SimulationError,
        invalid_message="Posterior scale diagnostics path is not a directory",
        create_message="Failed to create posterior scale diagnostics output",
        context={"path": str(target_dir)},
    )
    return target_dir


def _build_run_inputs(
    *, base_dir: Path, outer_ids: Sequence[int], candidate_id: int
) -> _RunInputs:
    return _RunInputs(
        base_dir=base_dir,
        outer_ids=list(int(item) for item in outer_ids),
        candidate_id=int(candidate_id),
        assets=_load_asset_names(base_dir),
        panel_targets=_load_panel_targets(base_dir),
        run_config=_load_run_config(base_dir),
    )


def _load_asset_names(base_dir: Path) -> list[str]:
    path = base_dir / "inputs" / "targets.csv"
    if not path.exists():
        raise SimulationError(
            "Missing targets.csv for posterior scale diagnostics",
            context={"path": str(path)},
        )
    frame = pd.read_csv(path, nrows=0)
    assets = [str(col) for col in frame.columns if str(col) != "timestamp"]
    if not assets:
        raise SimulationError("No assets found in targets.csv")
    return assets


def _load_panel_targets(base_dir: Path) -> np.ndarray:
    panel_path = base_dir / "inputs" / "panel_tensor.pt"
    if not panel_path.exists():
        raise SimulationError(
            "Missing panel_tensor.pt for posterior scale diagnostics",
            context={"path": str(panel_path)},
        )
    payload = torch.load(panel_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise SimulationError("panel_tensor.pt payload is invalid")
    targets = payload.get("targets")
    if not isinstance(targets, torch.Tensor):
        raise SimulationError("panel_tensor.pt missing targets tensor")
    return targets.detach().cpu().numpy()


def _load_run_config(base_dir: Path) -> _RunConfig:
    config_path = base_dir / "outer" / "best_config.json"
    if not config_path.exists():
        return _RunConfig(target_normalization=False, dof_shift=2.0)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    target_normalization = _extract_target_normalization(payload)
    dof_shift = _extract_dof_shift(payload)
    return _RunConfig(
        target_normalization=target_normalization,
        dof_shift=dof_shift,
    )


def _extract_target_normalization(payload: Mapping[str, Any]) -> bool:
    training = payload.get("training")
    if not isinstance(training, Mapping):
        return False
    return bool(training.get("target_normalization", False))


def _extract_dof_shift(payload: Mapping[str, Any]) -> float:
    return resolve_dof_shift(payload, default=2.0)


def _build_outer_context(run_inputs: _RunInputs, outer_k: int) -> _OuterContext:
    return _OuterContext(
        run=run_inputs,
        outer_k=outer_k,
        splits=_load_splits(run_inputs.base_dir, outer_k),
    )


def _load_splits(base_dir: Path, outer_k: int) -> Sequence[Mapping[str, Any]]:
    path = base_dir / "inner" / f"outer_{outer_k}" / "splits.json"
    if not path.exists():
        raise SimulationError(
            "Missing splits.json for posterior scale diagnostics",
            context={"path": str(path)},
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SimulationError("splits.json payload must be a list")
    return payload


def _rows_for_outer(outer_context: _OuterContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload_path in _candidate_payload_paths(outer_context):
        split_context = _build_split_context(outer_context, payload_path)
        rows.extend(_build_asset_rows(split_context))
    return rows


def _candidate_payload_paths(outer_context: _OuterContext) -> list[Path]:
    target = (
        outer_context.run.base_dir
        / "inner"
        / f"outer_{outer_context.outer_k}"
        / "postprocessing"
        / "candidates"
    )
    pattern = f"candidate_{outer_context.run.candidate_id:04d}_split_*.pt"
    return sorted(target.glob(pattern))


def _build_split_context(
    outer_context: _OuterContext, payload_path: Path
) -> _SplitContext:
    split_id = _split_id_from_path(payload_path)
    payload = _load_payload(payload_path)
    z_true, z_samples = _extract_z_values(payload)
    transform = _resolve_transform(outer_context, split_id, payload)
    diagnostics = _load_split_diagnostics(outer_context, split_id)
    sigma_model = _extract_sigma_model(
        diagnostics=diagnostics,
        asset_count=len(outer_context.run.assets),
    )
    nu = _extract_nu_posterior(
        diagnostics=diagnostics,
        dof_shift=outer_context.run.run_config.dof_shift,
    )
    return _SplitContext(
        outer=outer_context,
        split_id=split_id,
        transform=transform,
        z_true=z_true,
        z_samples=z_samples,
        sigma_model=sigma_model,
        nu=nu,
    )


def _split_id_from_path(path: Path) -> int:
    match = _SPLIT_PATTERN.fullmatch(path.name)
    if match is None:
        raise SimulationError(
            "Unable to parse split id from payload path",
            context={"path": str(path)},
        )
    return int(match.group(2))


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise SimulationError(
            "Postprocess payload must be a mapping",
            context={"path": str(path)},
        )
    return payload


def _extract_z_values(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    z_true = payload.get("z_true")
    z_samples = payload.get("z_samples")
    if not isinstance(z_true, torch.Tensor) or not isinstance(
        z_samples, torch.Tensor
    ):
        raise SimulationError("Postprocess payload missing z_true/z_samples tensors")
    return (
        z_true.detach().cpu().numpy().astype(float),
        z_samples.detach().cpu().numpy().astype(float),
    )


def _resolve_transform(
    outer_context: _OuterContext,
    split_id: int,
    payload: Mapping[str, Any],
) -> TargetSpaceTransform:
    transform_payload = payload.get("target_space_transform")
    if isinstance(transform_payload, Mapping):
        return TargetSpaceTransform.from_payload(
            transform_payload,
            asset_count=len(outer_context.run.assets),
        )
    return _fallback_transform(outer_context, split_id, payload)


def _fallback_transform(
    outer_context: _OuterContext, split_id: int, payload: Mapping[str, Any]
) -> TargetSpaceTransform:
    scale = payload.get("scale")
    if not isinstance(scale, torch.Tensor):
        raise SimulationError("Postprocess payload missing scale tensor")
    mad_scale = scale.detach().cpu().to(dtype=torch.float64)
    if not outer_context.run.run_config.target_normalization:
        return TargetSpaceTransform(
            model_center=torch.tensor(0.0, dtype=mad_scale.dtype),
            model_scale=torch.tensor(1.0, dtype=mad_scale.dtype),
            mad_scale=mad_scale,
        )
    model_center, model_scale = _split_model_norm(outer_context, split_id)
    return TargetSpaceTransform(
        model_center=model_center,
        model_scale=model_scale,
        mad_scale=mad_scale,
    )


def _split_model_norm(
    outer_context: _OuterContext, split_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    split = _split_entry(outer_context, split_id)
    train_idx = split.get("train_idx")
    if not isinstance(train_idx, list):
        raise SimulationError("Split payload missing train_idx")
    train_values = outer_context.run.panel_targets[np.asarray(train_idx, dtype=int)]
    finite = np.isfinite(train_values)
    if not finite.any():
        return (
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
    values = train_values[finite]
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = float(mad + _TARGET_NORM_EPS)
    return (
        torch.tensor(center, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
    )


def _split_entry(
    outer_context: _OuterContext, split_id: int
) -> Mapping[str, Any]:
    if split_id < 0 or split_id >= len(outer_context.splits):
        raise SimulationError(
            "Split id out of range for splits.json",
            context={
                "split_id": str(split_id),
                "num_splits": str(len(outer_context.splits)),
            },
        )
    return outer_context.splits[split_id]


def _load_split_diagnostics(
    outer_context: _OuterContext, split_id: int
) -> Mapping[str, Any]:
    path = (
        outer_context.run.base_dir
        / "inner"
        / f"outer_{outer_context.outer_k}"
        / "postprocessing"
        / "diagnostics"
        / f"candidate_{outer_context.run.candidate_id:04d}_split_{split_id:04d}.json"
    )
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SimulationError(
            "Postprocess diagnostics JSON must be a mapping",
            context={"path": str(path)},
        )
    return payload


def _extract_sigma_model(
    *, diagnostics: Mapping[str, Any], asset_count: int
) -> Mapping[str, np.ndarray]:
    posterior = diagnostics.get("posterior")
    if not isinstance(posterior, Mapping):
        return _nan_sigma(asset_count)
    by_asset = posterior.get("sigma_by_asset")
    if isinstance(by_asset, Mapping):
        by_asset_values = _sigma_from_by_asset(by_asset, asset_count)
        if by_asset_values is not None:
            return by_asset_values
    return _sigma_from_summary(posterior, asset_count)


def _sigma_from_by_asset(
    by_asset: Mapping[str, Any], asset_count: int
) -> Mapping[str, np.ndarray] | None:
    q10 = _float_array(by_asset.get("q10"), expected=asset_count)
    median = _float_array(by_asset.get("median"), expected=asset_count)
    q90 = _float_array(by_asset.get("q90"), expected=asset_count)
    if q10 is None or median is None or q90 is None:
        return None
    return {"q10": q10, "median": median, "q90": q90}


def _sigma_from_summary(
    posterior: Mapping[str, Any], asset_count: int
) -> Mapping[str, np.ndarray]:
    sigma_summary = posterior.get("sigma")
    if not isinstance(sigma_summary, Mapping):
        return _nan_sigma(asset_count)
    median_scalar = _coerce_float(sigma_summary.get("median"))
    q10_scalar = _coerce_float(sigma_summary.get("min"))
    q90_scalar = _coerce_float(sigma_summary.get("max"))
    return {
        "q10": np.full(asset_count, q10_scalar, dtype=float),
        "median": np.full(asset_count, median_scalar, dtype=float),
        "q90": np.full(asset_count, q90_scalar, dtype=float),
    }


def _nan_sigma(asset_count: int) -> Mapping[str, np.ndarray]:
    return {
        "q10": np.full(asset_count, np.nan, dtype=float),
        "median": np.full(asset_count, np.nan, dtype=float),
        "q90": np.full(asset_count, np.nan, dtype=float),
    }


def _extract_nu_posterior(
    *, diagnostics: Mapping[str, Any], dof_shift: float
) -> Mapping[str, float]:
    posterior = diagnostics.get("posterior")
    if not isinstance(posterior, Mapping):
        return {"q10": np.nan, "median": np.nan, "q90": np.nan}
    nu = posterior.get("nu_posterior")
    if isinstance(nu, Mapping):
        return {
            "q10": _coerce_float(nu.get("q10")),
            "median": _coerce_float(nu.get("median")),
            "q90": _coerce_float(nu.get("q90")),
        }
    nu_raw = posterior.get("nu_raw_posterior")
    if isinstance(nu_raw, Mapping):
        return {
            "q10": _coerce_float(nu_raw.get("q10")) + dof_shift,
            "median": _coerce_float(nu_raw.get("median")) + dof_shift,
            "q90": _coerce_float(nu_raw.get("q90")) + dof_shift,
        }
    nu_legacy = posterior.get("nu_raw")
    if isinstance(nu_legacy, Mapping):
        median = _coerce_float(nu_legacy.get("median"))
        return {
            "q10": median + dof_shift,
            "median": median + dof_shift,
            "q90": median + dof_shift,
        }
    return {"q10": np.nan, "median": np.nan, "q90": np.nan}


def _build_asset_rows(split_context: _SplitContext) -> list[dict[str, Any]]:
    y_true, y_samples = _split_y_values(split_context)
    mu_y = np.nanmedian(y_samples, axis=0)
    residual_y = y_true - mu_y
    pred_sd_tensor = torch.as_tensor(
        np.nanstd(y_samples, axis=0), dtype=torch.float64
    )
    rows: list[dict[str, Any]] = []
    for asset_idx, asset in enumerate(split_context.outer.run.assets):
        residual_column = residual_y[:, asset_idx]
        pred_sd_column = (
            pred_sd_tensor[:, asset_idx].detach().cpu().numpy().astype(float)
        )
        rows.append(
            _build_asset_row(
                split_context=split_context,
                asset_idx=asset_idx,
                asset=asset,
                residual_column=residual_column,
                pred_sd_column=pred_sd_column,
            )
        )
    return rows


def _split_y_values(split_context: _SplitContext) -> tuple[np.ndarray, np.ndarray]:
    z_true_tensor = torch.as_tensor(split_context.z_true, dtype=torch.float64)
    z_samples_tensor = torch.as_tensor(split_context.z_samples, dtype=torch.float64)
    y_true = split_context.transform.inverse_z_to_y(z_true_tensor)
    y_samples = split_context.transform.inverse_z_to_y(z_samples_tensor)
    return (
        y_true.detach().cpu().numpy(),
        y_samples.detach().cpu().numpy(),
    )


def _build_asset_row(
    *,
    split_context: _SplitContext,
    asset_idx: int,
    asset: str,
    residual_column: np.ndarray,
    pred_sd_column: np.ndarray,
) -> dict[str, Any]:
    residual_std = _residual_std_by_space(split_context, asset_idx, residual_column)
    pred_sd = _pred_sd_med_by_space(split_context, asset_idx, pred_sd_column)
    sigma = _sigma_by_space(split_context, asset_idx)
    ratios = _ratio_fields(residual_std=residual_std, pred_sd=pred_sd, sigma=sigma)
    nu_fields = _nu_fields(split_context.nu)
    row = {
        "outer_k": int(split_context.outer.outer_k),
        "split_id": int(split_context.split_id),
        "candidate_id": int(split_context.outer.run.candidate_id),
        "asset": asset,
        "asset_idx": int(asset_idx),
        "n_test_obs": int(np.isfinite(residual_column).sum()),
        "std_e_model": residual_std["model"],
        "sigma_med_model": sigma["model"],
        "std_e_y": residual_std["y"],
        "sigma_med_y": sigma["y"],
        "std_e_z": residual_std["z"],
        "sigma_med_z": sigma["z"],
        "pred_sd_med_model": pred_sd["model"],
        "pred_sd_med_y": pred_sd["y"],
        "pred_sd_med_z": pred_sd["z"],
        "sigma_q10_model": sigma["q10_model"],
        "sigma_q90_model": sigma["q90_model"],
        "model_scale": _model_scale_for_asset(split_context.transform, asset_idx),
        "mad_scale": float(
            split_context.transform.mad_scale[asset_idx].detach().cpu().item()
        ),
        "primary_space": "y",
    }
    row.update(ratios)
    row.update(nu_fields)
    return row


def _residual_std_by_space(
    split_context: _SplitContext, asset_idx: int, residual_column: np.ndarray
) -> Mapping[str, float]:
    std_y = _finite_std(residual_column)
    model_scale = _model_scale_for_asset(split_context.transform, asset_idx)
    mad_scale = float(split_context.transform.mad_scale[asset_idx].detach().cpu().item())
    return {
        "model": std_y / model_scale,
        "y": std_y,
        "z": std_y / mad_scale,
    }


def _pred_sd_med_by_space(
    split_context: _SplitContext, asset_idx: int, pred_sd_column: np.ndarray
) -> Mapping[str, float]:
    med_y = _finite_median(pred_sd_column)
    model_scale = _model_scale_for_asset(split_context.transform, asset_idx)
    mad_scale = float(split_context.transform.mad_scale[asset_idx].detach().cpu().item())
    return {
        "model": med_y / model_scale,
        "y": med_y,
        "z": med_y / mad_scale,
    }


def _sigma_by_space(split_context: _SplitContext, asset_idx: int) -> Mapping[str, float]:
    sigma_model = float(split_context.sigma_model["median"][asset_idx])
    model_scale = _model_scale_for_asset(split_context.transform, asset_idx)
    mad_scale = float(split_context.transform.mad_scale[asset_idx].detach().cpu().item())
    sigma_y = sigma_model * model_scale
    return {
        "model": sigma_model,
        "y": sigma_y,
        "z": sigma_y / mad_scale,
        "q10_model": float(split_context.sigma_model["q10"][asset_idx]),
        "q90_model": float(split_context.sigma_model["q90"][asset_idx]),
    }


def _ratio_fields(
    *,
    residual_std: Mapping[str, float],
    pred_sd: Mapping[str, float],
    sigma: Mapping[str, float],
) -> Mapping[str, float]:
    return {
        "ratio_model": _ratio_in_space(
            numerator=sigma["model"],
            denominator=residual_std["model"],
            numerator_space="model",
            denominator_space="model",
            space="model",
        ),
        "ratio_y": _ratio_in_space(
            numerator=sigma["y"],
            denominator=residual_std["y"],
            numerator_space="y",
            denominator_space="y",
            space="y",
        ),
        "ratio_z": _ratio_in_space(
            numerator=sigma["z"],
            denominator=residual_std["z"],
            numerator_space="z",
            denominator_space="z",
            space="z",
        ),
        "ratio_pred_model": _ratio_in_space(
            numerator=pred_sd["model"],
            denominator=residual_std["model"],
            numerator_space="model",
            denominator_space="model",
            space="model",
        ),
        "ratio_pred_y": _ratio_in_space(
            numerator=pred_sd["y"],
            denominator=residual_std["y"],
            numerator_space="y",
            denominator_space="y",
            space="y",
        ),
        "ratio_pred_z": _ratio_in_space(
            numerator=pred_sd["z"],
            denominator=residual_std["z"],
            numerator_space="z",
            denominator_space="z",
            space="z",
        ),
    }


def _nu_fields(nu: Mapping[str, float]) -> Mapping[str, float]:
    nu_med = float(nu["median"])
    nu_sd_factor = (
        float(np.sqrt(nu_med / (nu_med - 2.0)))
        if np.isfinite(nu_med) and nu_med > 2.0
        else float("nan")
    )
    return {
        "nu_q10": float(nu["q10"]),
        "nu_med": nu_med,
        "nu_q90": float(nu["q90"]),
        "nu_sd_factor": nu_sd_factor,
    }


def _ratio_in_space(
    *,
    numerator: float,
    denominator: float,
    numerator_space: TargetSpace,
    denominator_space: TargetSpace,
    space: TargetSpace,
) -> float:
    if numerator_space != space or denominator_space != space:
        raise SimulationError(
            "Space mismatch in ratio computation",
            context={
                "numerator_space": numerator_space,
                "denominator_space": denominator_space,
                "space": space,
            },
        )
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _model_scale_for_asset(
    transform: TargetSpaceTransform, asset_idx: int
) -> float:
    scale = transform.model_scale.detach().cpu()
    if scale.ndim == 0:
        return float(scale.item())
    if scale.ndim == 1:
        return float(scale[asset_idx].item())
    raise SimulationError("model_scale must be scalar or 1D")


def _float_array(value: Any, *, expected: int) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != expected:
        return None
    values = np.asarray([_coerce_float(item) for item in value], dtype=float)
    return values


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=0))


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))
