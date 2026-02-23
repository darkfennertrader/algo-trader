from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
import logging
import time
from typing import Any, Callable, Mapping, cast

import pyro
import torch
from pyro import poutine
from pyro.infer import util as infer_util

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import (
    Allocator,
    ModelFitter,
    PnLCalculator,
    Predictor,
    Scorer,
)
from algo_trader.pipeline.stages import modeling
from .metrics import build_metric_scorer
from .config_utils import require_bool

logger = logging.getLogger(__name__)

_TARGET_NORM_EPS = 1e-6

@dataclass(frozen=True)
class SimulationHooks:
    fit_model: ModelFitter
    predict: Predictor
    score: Scorer
    allocate: Allocator
    compute_pnl: PnLCalculator


def default_hooks() -> SimulationHooks:
    return SimulationHooks(
        fit_model=_fit_pyro_svi,
        predict=_predict_pyro,
        score=_score_metrics,
        allocate=_allocate_weights_stub,
        compute_pnl=_compute_weekly_pnl_stub,
    )


def stub_hooks() -> SimulationHooks:
    return SimulationHooks(
        fit_model=_fit_bayes_svi_stub,
        predict=_posterior_predict_stub,
        score=_score_predictive_stub,
        allocate=_allocate_weights_stub,
        compute_pnl=_compute_weekly_pnl_stub,
    )


@dataclass(frozen=True)
class _SVIParams:
    steps: int
    learning_rate: float
    num_elbo_particles: int
    log_every: int | None
    grad_accum_steps: int


@dataclass(frozen=True)
class _TBPTTParams:
    window_len: int | None
    burn_in_len: int


@dataclass(frozen=True)
class _TBPTTInputs:
    X_obs: torch.Tensor
    y_obs: torch.Tensor
    valid_mask: torch.BoolTensor
    window_len: int


@dataclass(frozen=True)
class _TrainingParams:
    svi: _SVIParams
    tbptt: _TBPTTParams
    log_prob_scaling: bool
    target_normalization: bool


@dataclass(frozen=True)
class _TargetNormState:
    center: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class _DebugConfig:
    enabled: bool
    output_dir: str | None


_MODEL_REGISTRY = modeling.default_model_registry()
_GUIDE_REGISTRY = modeling.default_guide_registry()


def _log_missing_targets(y_train: torch.Tensor) -> None:
    if torch.isfinite(y_train).all():
        return
    missing = int((~torch.isfinite(y_train)).sum().item())
    total = int(y_train.numel())
    logger.warning(
        "Training targets contain missing values; missing=%s total=%s",
        missing,
        total,
    )


def _resolve_model_and_guide(
    config: Mapping[str, Any],
) -> tuple[str, str, modeling.PyroModel, modeling.PyroGuide]:
    model_name, guide_name, model_params, guide_params = _resolve_model_config(
        config
    )
    model = _MODEL_REGISTRY.get(model_name, model_params)
    guide = _GUIDE_REGISTRY.get(guide_name, guide_params)
    return model_name, guide_name, model, guide


def _prepare_training_batches(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    params: _TrainingParams,
    debug_log_shapes: bool,
) -> tuple[list[modeling.ModelBatch], _TargetNormState | None]:
    y_train_norm, norm_state = _maybe_normalize_targets(
        y_train, params.target_normalization
    )
    batches = _build_tbptt_batches(
        X_train, y_train_norm, params, debug_log_shapes
    )
    if not batches:
        raise SimulationError("No valid targets for training")
    return batches, norm_state


def _build_training_state(
    model_name: str,
    guide_name: str,
    norm_state: _TargetNormState | None,
    posterior_summary: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    state: dict[str, Any] = {
        "model_name": model_name,
        "guide_name": guide_name,
    }
    if norm_state is not None:
        state["target_norm"] = {
            "center": norm_state.center,
            "scale": norm_state.scale,
        }
    if posterior_summary:
        state["posterior_summary"] = dict(posterior_summary)
    return state


def _fit_pyro_svi(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _ = init_state
    _log_missing_targets(y_train)
    model_name, guide_name, model, guide = _resolve_model_and_guide(config)
    debug_config = _debug_config(config)
    _configure_debug_sink(
        debug_config=debug_config,
        model_name=model_name,
        guide_name=guide_name,
        model=model,
        guide=guide,
    )
    params = _training_params(config)
    debug_log_shapes = debug_config.enabled
    batches, norm_state = _prepare_training_batches(
        X_train, y_train, params, debug_log_shapes
    )
    pyro.clear_param_store()
    svi = _build_svi(model=model, guide=guide, params=params)
    _run_svi_steps(
        svi=svi,
        batches=batches,
        params=params,
        context=_run_context(config),
    )
    posterior_summary = _summarize_posterior_params()
    return _build_training_state(
        model_name,
        guide_name,
        norm_state,
        posterior_summary,
    )


def _summarize_posterior_params() -> Mapping[str, Any]:
    store = pyro.get_param_store()
    summary: dict[str, Any] = {}

    nu_raw_loc = _get_param(store, "nu_raw_loc")
    nu_raw_scale = _get_param(store, "nu_raw_scale")
    nu_raw_mean = _lognormal_mean(nu_raw_loc, nu_raw_scale)
    if nu_raw_mean is not None:
        summary["nu_raw"] = _summarize_tensor(nu_raw_mean)

    sigma_loc = _get_param(store, "sigma_loc")
    sigma_scale = _get_param(store, "sigma_scale")
    sigma_mean = _lognormal_mean(sigma_loc, sigma_scale)
    if sigma_mean is not None:
        summary["sigma"] = _summarize_tensor(sigma_mean)

    w_loc = _get_param(store, "w_loc")
    w_scale = _get_param(store, "w_scale")
    if isinstance(w_loc, torch.Tensor):
        summary["w"] = {
            "abs_mean": _safe_stat(torch.mean, w_loc.abs()),
            "abs_max": _safe_stat(torch.max, w_loc.abs()),
        }
        if isinstance(w_scale, torch.Tensor):
            summary["w"]["scale_mean"] = _safe_stat(
                torch.mean, w_scale
            )
            summary["w"]["scale_max"] = _safe_stat(torch.max, w_scale)

    return summary


def _get_param(store: Any, name: str) -> torch.Tensor | None:
    if name not in store:
        return None
    value = store.get_param(name)
    if not isinstance(value, torch.Tensor):
        return None
    return value


def _lognormal_mean(
    loc: torch.Tensor | None, scale: torch.Tensor | None
) -> torch.Tensor | None:
    if not isinstance(loc, torch.Tensor) or not isinstance(
        scale, torch.Tensor
    ):
        return None
    return torch.exp(loc + 0.5 * scale.pow(2))


def _summarize_tensor(values: torch.Tensor) -> Mapping[str, float]:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return {}
    return {
        "mean": _safe_stat(torch.mean, flat),
        "median": _safe_stat(torch.median, flat),
        "min": _safe_stat(torch.min, flat),
        "max": _safe_stat(torch.max, flat),
    }


def _safe_stat(
    fn: Callable[[torch.Tensor], torch.Tensor], values: torch.Tensor
) -> float:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(fn(finite).item())


def _predict_pyro(
    X_pred: torch.Tensor,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    _, _, model, guide = _resolve_model_and_guide(config)
    num_samples = max(int(num_samples), 1)
    batch = _build_prediction_batch(X_pred)
    unconditioned = poutine.uncondition(model)
    predictive = pyro.infer.Predictive(  # type: ignore  # pylint: disable=no-member
        unconditioned,
        guide=guide,
        num_samples=num_samples,
        return_sites=("obs",),
    )
    samples = predictive(batch)["obs"]
    mean = samples.mean(dim=0)
    mean, samples = _apply_target_denorm(mean, samples, state)
    return {"samples": samples, "mean": mean}


def _score_metrics(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    scorer = build_metric_scorer(score_spec, scope="inner")
    return float(scorer(y_true, pred, score_spec))


def _resolve_model_config(
    config: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model config missing for simulation hooks")
    model_name = str(model.get("model_name", "")).strip().lower()
    guide_name = str(model.get("guide_name", "")).strip().lower()
    if not model_name:
        raise ConfigError("model.model_name must be set for simulation hooks")
    if not guide_name:
        raise ConfigError("model.guide_name must be set for simulation hooks")
    model_params = _coerce_mapping(
        model.get("params", {}),
        label="model.params",
    )
    guide_params = _coerce_mapping(
        model.get("guide_params", {}),
        label="model.guide_params",
    )
    return model_name, guide_name, model_params, guide_params


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping for simulation hooks")
    return dict(value)


def _training_params(config: Mapping[str, Any]) -> _TrainingParams:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("training config missing for simulation hooks")
    raw_svi = training.get("svi")
    if not isinstance(raw_svi, Mapping):
        raise ConfigError("training.svi config missing for simulation hooks")
    target_norm = require_bool(
        training.get("target_normalization"),
        field="training.target_normalization",
        config_path=None,
    )
    raw_log_every = raw_svi.get("log_every", None)
    log_every: int | None
    if raw_log_every is None:
        log_every = None
    else:
        log_every = int(raw_log_every)
        if log_every <= 0:
            raise ConfigError("training.svi.log_every must be >= 1 or null")
    raw_window_len = raw_svi.get("tbptt_window_len", None)
    tbptt_window_len: int | None
    if raw_window_len is None:
        tbptt_window_len = None
    else:
        tbptt_window_len = int(raw_window_len)
        if tbptt_window_len <= 0:
            raise ConfigError(
                "training.svi.tbptt_window_len must be positive or null"
            )
    tbptt_burn_in_len = int(raw_svi.get("tbptt_burn_in_len", 0))
    if tbptt_burn_in_len < 0:
        raise ConfigError(
            "training.svi.tbptt_burn_in_len must be >= 0"
        )
    if (
        tbptt_window_len is not None
        and tbptt_burn_in_len >= tbptt_window_len
    ):
        raise ConfigError(
            "training.svi.tbptt_burn_in_len must be < "
            "training.svi.tbptt_window_len"
        )
    grad_accum_steps = int(raw_svi.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ConfigError("training.svi.grad_accum_steps must be >= 1")
    log_prob_scaling = require_bool(
        training.get("log_prob_scaling"),
        field="training.log_prob_scaling",
        config_path=None,
    )
    return _TrainingParams(
        svi=_SVIParams(
            steps=int(raw_svi.get("num_steps", 0)),
            learning_rate=float(raw_svi.get("learning_rate", 1e-3)),
            num_elbo_particles=int(raw_svi.get("num_elbo_particles", 1)),
            log_every=log_every,
            grad_accum_steps=grad_accum_steps,
        ),
        tbptt=_TBPTTParams(
            window_len=tbptt_window_len,
            burn_in_len=tbptt_burn_in_len,
        ),
        log_prob_scaling=log_prob_scaling,
        target_normalization=target_norm,
    )


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


def _run_context(config: Mapping[str, Any]) -> Mapping[str, Any]:
    context = config.get("run_context")
    if not isinstance(context, Mapping):
        return {}
    return dict(context)


def _log_svi_progress(
    *, step: int, loss: float, context: Mapping[str, Any], start: float
) -> None:
    elapsed = time.perf_counter() - start
    payload = dict(context)
    payload.update(
        {
            "step": step,
            "loss": loss,
            "elapsed_s": round(elapsed, 4),
        }
    )
    logger.info("event=progress boundary=simulation.svi context=%s", payload)


def _build_svi(
    *,
    model: modeling.PyroModel,
    guide: modeling.PyroGuide,
    params: _TrainingParams,
) -> pyro.infer.SVI:  # type: ignore[name-defined]
    optimizer = pyro.optim.Adam(  # type: ignore  # pylint: disable=no-member
        {"lr": params.svi.learning_rate}
    )
    loss = pyro.infer.Trace_ELBO(  # type: ignore  # pylint: disable=no-member
        num_particles=params.svi.num_elbo_particles
    )
    return pyro.infer.SVI(  # type: ignore  # pylint: disable=no-member
        model, guide, optimizer, loss=loss
    )


def _run_svi_steps(
    *,
    svi: pyro.infer.SVI,  # type: ignore[name-defined]
    batches: list[modeling.ModelBatch],
    params: _TrainingParams,
    context: Mapping[str, Any],
) -> None:
    start = time.perf_counter()
    if not batches:
        raise SimulationError("No training batches available for SVI")
    grad_accum_steps = params.svi.grad_accum_steps
    for step in range(params.svi.steps):
        total_loss, params_in_step = _accumulate_svi_grads(
            svi=svi,
            batches=batches,
            step_index=step,
            grad_accum_steps=grad_accum_steps,
        )
        if params_in_step:
            svi.optim(params_in_step)
            infer_util.zero_grads(params_in_step)
        if params.svi.log_every and (step + 1) % params.svi.log_every == 0:
            _log_svi_progress(
                step=step + 1,
                loss=total_loss / float(grad_accum_steps),
                context=context,
                start=start,
            )


def _loss_to_float(loss: object) -> float:
    if isinstance(loss, tuple):
        if not loss:
            return 0.0
        return float(cast(Any, infer_util.torch_item(loss[0])))
    return float(cast(Any, infer_util.torch_item(loss)))


def _accumulate_svi_grads(
    *,
    svi: pyro.infer.SVI,  # type: ignore[name-defined]
    batches: list[modeling.ModelBatch],
    step_index: int,
    grad_accum_steps: int,
) -> tuple[float, set[torch.Tensor]]:
    total_batches = len(batches)
    total_loss = 0.0
    params_in_step: set[torch.Tensor] = set()
    for micro in range(grad_accum_steps):
        batch = batches[
            (step_index * grad_accum_steps + micro) % total_batches
        ]
        loss, params = _loss_and_params(svi, batch)
        total_loss += loss
        params_in_step.update(params)
    return total_loss, params_in_step


def _loss_and_params(
    svi: pyro.infer.SVI,  # type: ignore[name-defined]
    batch: modeling.ModelBatch,
) -> tuple[float, set[torch.Tensor]]:
    with poutine.trace(  # pylint: disable=not-context-manager
        param_only=True
    ) as param_capture:
        loss = svi.loss_and_grads(  # type: ignore[attr-defined]
            svi.model, svi.guide, batch
        )
    params_in_step: set[torch.Tensor] = set()
    for site in param_capture.trace.nodes.values():
        value = site.get("value")
        if value is None:
            continue
        raw_value = cast(Any, value)
        if hasattr(raw_value, "unconstrained"):
            params_in_step.add(raw_value.unconstrained())
        elif isinstance(value, torch.Tensor):
            params_in_step.add(value)
    return _loss_to_float(loss), params_in_step


def _build_tbptt_batches(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    params: _TrainingParams,
    debug_log_shapes: bool = False,
) -> list[modeling.ModelBatch]:
    _validate_training_inputs(X_train, y_train)
    inputs = _prepare_training_observations(X_train, y_train, params)
    if not inputs.valid_mask.any():
        return []
    return _slice_tbptt_windows(
        inputs=inputs,
        params=params,
        debug_log_shapes=debug_log_shapes,
    )


def _validate_training_inputs(
    X_train: torch.Tensor, y_train: torch.Tensor
) -> None:
    if X_train.ndim != 3:
        raise SimulationError("X_train must be [T, A, F]")
    if y_train.ndim != 2:
        raise SimulationError("y_train must be [T, A]")
    if X_train.shape[:2] != y_train.shape:
        raise SimulationError("X_train and y_train must align on [T, A]")
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
    X_train: torch.Tensor, y_train: torch.Tensor, params: _TrainingParams
) -> _TBPTTInputs:
    valid_mask = cast(torch.BoolTensor, torch.isfinite(y_train))
    X_obs = X_train.to(dtype=torch.float32)
    y_obs = torch.nan_to_num(
        y_train, nan=0.0, posinf=0.0, neginf=0.0
    ).to(dtype=torch.float32)
    window_len = _resolve_window_len(y_train, params)
    return _TBPTTInputs(
        X_obs=X_obs,
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
        mask = cast(
            torch.BoolTensor, inputs.valid_mask[start:end].clone()
        )
        if params.tbptt.burn_in_len > 0:
            burn = min(params.tbptt.burn_in_len, end - start)
            mask[:burn, :] = False
        if not mask.any():
            continue
        obs_scale = _resolve_obs_scale(mask, params.log_prob_scaling)
        batches.append(
            modeling.ModelBatch(
                X=inputs.X_obs[start:end],
                y=inputs.y_obs[start:end],
                M=mask,
                obs_scale=obs_scale,
                debug=debug_log_shapes,
            )
        )
    return batches


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


def _build_prediction_batch(X_pred: torch.Tensor) -> modeling.ModelBatch:
    if X_pred.ndim != 3:
        raise SimulationError("X_pred must be [T, A, F]")
    return modeling.ModelBatch(X=X_pred, y=None, M=None, obs_scale=None)


def _fit_bayes_svi_stub(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _ = (X_train, y_train, config, init_state)
    return {}


def _posterior_predict_stub(
    X_pred: torch.Tensor,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    _ = (X_pred, state, config, num_samples)
    return {}


def _score_predictive_stub(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    _ = (y_true, pred, score_spec)
    return 0.0


def _allocate_weights_stub(
    pred: Mapping[str, Any],
    alloc_spec: Mapping[str, Any],
) -> torch.Tensor:
    _ = pred
    n_assets = int(alloc_spec.get("n_assets", 1))
    return torch.full((n_assets,), 1.0 / max(n_assets, 1))


def _compute_weekly_pnl_stub(
    w: torch.Tensor,
    y_t: torch.Tensor,
    w_prev: torch.Tensor | None = None,
    cost_spec: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    gross = (w * y_t).sum()
    if w_prev is None or cost_spec is None:
        return gross
    tc_per_unit = float(cost_spec.get("tc_per_unit_turnover", 0.0))
    turnover = torch.abs(w - w_prev).sum()
    return gross - tc_per_unit * turnover
