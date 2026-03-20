from __future__ import annotations

from datetime import datetime
import inspect
from typing import Any, Mapping, cast

import torch

from algo_trader.domain import SimulationError
from algo_trader.pipeline.stages import modeling

from .hooks_types import (
    _DebugConfig,
    _TBPTTInputs,
    _TargetNormState,
    _TrainingParams,
)

_TARGET_NORM_EPS = 1e-6


def _prepare_training_batches(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
    params: _TrainingParams,
    debug_log_shapes: bool,
) -> tuple[list[modeling.ModelBatch], _TargetNormState | None]:
    y_train_norm, norm_state = _maybe_normalize_targets(
        y_train, params.target_normalization
    )
    if params.method == "online_filtering":
        batches = _build_online_filtering_batches(
            X_train,
            y_train_norm,
            params,
            debug_log_shapes,
            X_train_global=X_train_global,
        )
    else:
        batches = _build_tbptt_batches(
            X_train,
            y_train_norm,
            params,
            debug_log_shapes,
            X_train_global=X_train_global,
        )
    if not batches:
        raise SimulationError("No valid targets for training")
    return batches, norm_state


def _maybe_normalize_targets(
    y_train: torch.Tensor, enabled: bool
) -> tuple[torch.Tensor, _TargetNormState | None]:
    if not enabled:
        return y_train, None
    finite_mask = torch.isfinite(y_train)
    if not finite_mask.any():
        raise SimulationError("Target normalization requires finite targets")
    values = y_train[finite_mask]
    center = values.median()
    mad = (values - center).abs().median()
    scale = mad + _TARGET_NORM_EPS
    if not torch.isfinite(scale):
        raise SimulationError("Target normalization scale is not finite")
    y_norm = (y_train - center) / scale
    return y_norm, _TargetNormState(center=center, scale=scale)


def _coerce_target_norm_state(
    state: Mapping[str, Any]
) -> _TargetNormState | None:
    raw = state.get("target_norm")
    if not isinstance(raw, Mapping):
        return None
    center = raw.get("center")
    scale = raw.get("scale")
    if isinstance(center, torch.Tensor) and isinstance(scale, torch.Tensor):
        return _TargetNormState(center=center, scale=scale)
    return None


def _apply_target_denorm(
    mean: torch.Tensor,
    samples: torch.Tensor,
    state: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_state = _coerce_target_norm_state(state)
    if norm_state is None:
        return mean, samples
    mean = mean * norm_state.scale + norm_state.center
    samples = samples * norm_state.scale + norm_state.center
    return mean, samples


def _build_tbptt_batches(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    params: _TrainingParams,
    debug_log_shapes: bool = False,
    *,
    X_train_global: torch.Tensor | None = None,
) -> list[modeling.ModelBatch]:
    _validate_training_inputs(X_train, X_train_global, y_train)
    inputs = _prepare_training_observations(
        X_train, X_train_global, y_train, params
    )
    if not inputs.valid_mask.any():
        return []
    return _slice_tbptt_windows(
        inputs=inputs,
        params=params,
        debug_log_shapes=debug_log_shapes,
    )


def _build_online_filtering_batches(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    params: _TrainingParams,
    debug_log_shapes: bool = False,
    *,
    X_train_global: torch.Tensor | None = None,
) -> list[modeling.ModelBatch]:
    _validate_training_inputs(X_train, X_train_global, y_train)
    inputs = _prepare_training_observations(
        X_train, X_train_global, y_train, params
    )
    if not inputs.valid_mask.any():
        return []
    return _slice_online_filtering_batches(
        inputs=inputs,
        log_prob_scaling=params.log_prob_scaling,
        debug_log_shapes=debug_log_shapes,
    )


def _validate_training_inputs(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
) -> None:
    if X_train.ndim != 3:
        raise SimulationError("X_train must be [T, A, F]")
    if X_train_global is not None and X_train_global.ndim != 2:
        raise SimulationError("X_train_global must be [T, G]")
    if y_train.ndim != 2:
        raise SimulationError("y_train must be [T, A]")
    if X_train.shape[:2] != y_train.shape:
        raise SimulationError("X_train and y_train must align on [T, A]")
    if X_train_global is not None and X_train_global.shape[0] != y_train.shape[0]:
        raise SimulationError("X_train_global and y_train must align on T")
    if y_train.shape[1] == 0:
        raise SimulationError("y_train must have at least one asset")


def _resolve_window_len(
    y_train: torch.Tensor, params: _TrainingParams
) -> int:
    total_steps = int(y_train.shape[0])
    window_len = params.tbptt.window_len or total_steps
    if window_len <= 0:
        raise SimulationError("TBPTT window length must be positive")
    return window_len


def _prepare_training_observations(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
    params: _TrainingParams,
) -> _TBPTTInputs:
    valid_mask = cast(torch.BoolTensor, torch.isfinite(y_train))
    X_asset_obs = X_train.to(dtype=torch.float32)
    X_global_obs = (
        None
        if X_train_global is None
        else X_train_global.to(dtype=torch.float32)
    )
    y_obs = torch.nan_to_num(
        y_train, nan=0.0, posinf=0.0, neginf=0.0
    ).to(dtype=torch.float32)
    window_len = _resolve_window_len(y_train, params)
    return _TBPTTInputs(
        X_asset_obs=X_asset_obs,
        X_global_obs=X_global_obs,
        y_obs=y_obs,
        valid_mask=valid_mask,
        window_len=window_len,
    )


def _slice_tbptt_windows(
    *,
    inputs: _TBPTTInputs,
    params: _TrainingParams,
    debug_log_shapes: bool,
) -> list[modeling.ModelBatch]:
    total_steps = int(inputs.y_obs.shape[0])
    batches: list[modeling.ModelBatch] = []
    for start in range(0, total_steps, inputs.window_len):
        end = min(total_steps, start + inputs.window_len)
        mask = cast(torch.BoolTensor, inputs.valid_mask[start:end].clone())
        if params.tbptt.burn_in_len > 0:
            burn = min(params.tbptt.burn_in_len, end - start)
            mask[:burn, :] = False
        if not mask.any():
            continue
        batches.append(
            _build_training_batch(
                inputs=inputs,
                span=slice(start, end),
                mask=mask,
                log_prob_scaling=params.log_prob_scaling,
                debug_log_shapes=debug_log_shapes,
            )
        )
    return batches


def _slice_online_filtering_batches(
    *,
    inputs: _TBPTTInputs,
    log_prob_scaling: bool,
    debug_log_shapes: bool,
) -> list[modeling.ModelBatch]:
    total_steps = int(inputs.y_obs.shape[0])
    batches: list[modeling.ModelBatch] = []
    for start in range(total_steps):
        end = start + 1
        mask = cast(torch.BoolTensor, inputs.valid_mask[start:end].clone())
        if not mask.any():
            continue
        batches.append(
            _build_training_batch(
                inputs=inputs,
                span=slice(start, end),
                mask=mask,
                log_prob_scaling=log_prob_scaling,
                debug_log_shapes=debug_log_shapes,
            )
        )
    return batches


def _build_training_batch(
    *,
    inputs: _TBPTTInputs,
    span: slice,
    mask: torch.BoolTensor,
    log_prob_scaling: bool,
    debug_log_shapes: bool,
) -> modeling.ModelBatch:
    return modeling.ModelBatch(
        X=inputs.X_asset_obs[span],
        X_asset=inputs.X_asset_obs[span],
        X_global=(None if inputs.X_global_obs is None else inputs.X_global_obs[span]),
        y=inputs.y_obs[span],
        M=mask,
        obs_scale=_resolve_obs_scale(mask, log_prob_scaling),
        debug=debug_log_shapes,
    )


def _debug_config(config: Mapping[str, Any]) -> _DebugConfig:
    section = config.get("debug")
    if not isinstance(section, Mapping):
        return _DebugConfig(enabled=False, output_dir=None)
    enabled = bool(section.get("enabled", False))
    output_dir = section.get("output_dir")
    if output_dir is None:
        return _DebugConfig(enabled=enabled, output_dir=None)
    return _DebugConfig(enabled=enabled, output_dir=str(output_dir))


def _configure_debug_sink(
    *,
    debug_config: _DebugConfig,
    model_name: str,
    guide_name: str,
    model: modeling.PyroModel,
    guide: modeling.PyroGuide,
) -> None:
    if not debug_config.enabled or debug_config.output_dir is None:
        return
    run_timestamp = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    modeling.configure_debug_sink(
        output_dir=debug_config.output_dir,
        metadata=modeling.DebugMetadata(
            run_timestamp=run_timestamp,
            model_name=model_name,
            guide_name=guide_name,
            model_file=_source_path(model),
            guide_file=_source_path(guide),
        ),
    )


def _source_path(obj: object) -> str | None:
    try:
        return inspect.getsourcefile(obj.__class__)
    except TypeError:
        return None


def _resolve_obs_scale(
    mask: torch.BoolTensor, log_prob_scaling: bool
) -> float | None:
    if not log_prob_scaling:
        return None
    num_valid = int(mask.sum().item())
    if num_valid <= 0:
        return None
    return 1.0 / float(num_valid)


def _build_prediction_batch(
    X_pred: torch.Tensor, X_pred_global: torch.Tensor | None
) -> modeling.ModelBatch:
    if X_pred.ndim != 3:
        raise SimulationError("X_pred must be [T, A, F]")
    if X_pred_global is not None:
        if X_pred_global.ndim != 2:
            raise SimulationError("X_pred_global must be [T, G]")
        if X_pred_global.shape[0] != X_pred.shape[0]:
            raise SimulationError("X_pred_global and X_pred must align on T")
    return modeling.ModelBatch(
        X=X_pred,
        X_asset=X_pred,
        X_global=X_pred_global,
        y=None,
        M=None,
        obs_scale=None,
    )
