from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Mapping, cast

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

logger = logging.getLogger(__name__)

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
class _TrainingParams:
    steps: int
    learning_rate: float
    num_elbo_particles: int
    log_every: int | None
    tbptt_window_len: int | None
    tbptt_burn_in_len: int
    grad_accum_steps: int


_MODEL_REGISTRY = modeling.default_model_registry()
_GUIDE_REGISTRY = modeling.default_guide_registry()


def _fit_pyro_svi(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _ = init_state
    model_name, guide_name = _resolve_model_guide_names(config)
    model = _MODEL_REGISTRY.get(model_name)
    guide = _GUIDE_REGISTRY.get(guide_name)
    params = _training_params(config)
    batches = _build_tbptt_batches(X_train, y_train, params)
    if not batches:
        raise SimulationError("No valid targets for training")
    pyro.clear_param_store()
    svi = _build_svi(model=model, guide=guide, params=params)
    _run_svi_steps(
        svi=svi,
        batches=batches,
        params=params,
        context=_run_context(config),
    )
    return {"model_name": model_name, "guide_name": guide_name}


def _predict_pyro(
    X_pred: torch.Tensor,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    _ = state
    model_name, guide_name = _resolve_model_guide_names(config)
    model = _MODEL_REGISTRY.get(model_name)
    guide = _GUIDE_REGISTRY.get(guide_name)
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
    return {"samples": samples, "mean": mean}


def _score_metrics(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    scorer = build_metric_scorer(score_spec, scope="inner")
    return float(scorer(y_true, pred, score_spec))


def _resolve_model_guide_names(config: Mapping[str, Any]) -> tuple[str, str]:
    model_name = str(config.get("model_name", "")).strip().lower()
    guide_name = str(config.get("guide_name", "")).strip().lower()
    if not model_name:
        raise ConfigError("model_name must be set for simulation hooks")
    if not guide_name:
        raise ConfigError("guide_name must be set for simulation hooks")
    return model_name, guide_name


def _training_params(config: Mapping[str, Any]) -> _TrainingParams:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("training config missing for simulation hooks")
    raw_log_every = training.get("log_every", None)
    log_every: int | None
    if raw_log_every is None:
        log_every = None
    else:
        log_every = int(raw_log_every)
        if log_every <= 0:
            raise ConfigError("training.log_every must be >= 1 or null")
    raw_window_len = training.get("tbptt_window_len", None)
    tbptt_window_len: int | None
    if raw_window_len is None:
        tbptt_window_len = None
    else:
        tbptt_window_len = int(raw_window_len)
        if tbptt_window_len <= 0:
            raise ConfigError(
                "training.tbptt_window_len must be positive or null"
            )
    tbptt_burn_in_len = int(training.get("tbptt_burn_in_len", 0))
    if tbptt_burn_in_len < 0:
        raise ConfigError(
            "training.tbptt_burn_in_len must be >= 0"
        )
    if (
        tbptt_window_len is not None
        and tbptt_burn_in_len >= tbptt_window_len
    ):
        raise ConfigError(
            "training.tbptt_burn_in_len must be < training.tbptt_window_len"
        )
    grad_accum_steps = int(training.get("grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ConfigError("training.grad_accum_steps must be >= 1")
    return _TrainingParams(
        steps=int(training.get("num_steps", 0)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        num_elbo_particles=int(training.get("num_elbo_particles", 1)),
        log_every=log_every,
        tbptt_window_len=tbptt_window_len,
        tbptt_burn_in_len=tbptt_burn_in_len,
        grad_accum_steps=grad_accum_steps,
    )


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
        {"lr": params.learning_rate}
    )
    loss = pyro.infer.Trace_ELBO(  # type: ignore  # pylint: disable=no-member
        num_particles=params.num_elbo_particles
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
    grad_accum_steps = params.grad_accum_steps
    for step in range(params.steps):
        total_loss, params_in_step = _accumulate_svi_grads(
            svi=svi,
            batches=batches,
            step_index=step,
            grad_accum_steps=grad_accum_steps,
        )
        if params_in_step:
            svi.optim(params_in_step)
            infer_util.zero_grads(params_in_step)
        if params.log_every and (step + 1) % params.log_every == 0:
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
) -> list[modeling.ModelBatch]:
    _validate_training_inputs(X_train, y_train)
    window_len = _resolve_window_len(y_train, params)
    X_obs, y_obs, valid_mask = _prepare_training_observations(
        X_train, y_train
    )
    if not valid_mask.any():
        return []
    return _slice_tbptt_windows(
        X_obs=X_obs,
        y_obs=y_obs,
        valid_mask=valid_mask,
        window_len=window_len,
        burn_in_len=params.tbptt_burn_in_len,
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
    window_len = params.tbptt_window_len or total_steps
    if window_len <= 0:
        raise SimulationError("TBPTT window length must be positive")
    return window_len


def _prepare_training_observations(
    X_train: torch.Tensor, y_train: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
    valid_mask = cast(torch.BoolTensor, torch.isfinite(y_train).all(dim=1))
    X_obs = X_train.to(dtype=torch.float32)
    y_obs = torch.nan_to_num(
        y_train, nan=0.0, posinf=0.0, neginf=0.0
    ).to(dtype=torch.float32)
    return X_obs, y_obs, valid_mask


def _slice_tbptt_windows(
    *,
    X_obs: torch.Tensor,
    y_obs: torch.Tensor,
    valid_mask: torch.BoolTensor,
    window_len: int,
    burn_in_len: int,
) -> list[modeling.ModelBatch]:
    total_steps = int(y_obs.shape[0])
    batches: list[modeling.ModelBatch] = []
    for start in range(0, total_steps, window_len):
        end = min(total_steps, start + window_len)
        mask = cast(torch.BoolTensor, valid_mask[start:end].clone())
        if burn_in_len > 0:
            burn = min(burn_in_len, end - start)
            mask[:burn] = False
        if not mask.any():
            continue
        batches.append(
            modeling.ModelBatch(
                X=X_obs[start:end],
                y=y_obs[start:end],
                M=mask,
            )
        )
    return batches


def _build_prediction_batch(X_pred: torch.Tensor) -> modeling.ModelBatch:
    if X_pred.ndim != 3:
        raise SimulationError("X_pred must be [T, A, F]")
    return modeling.ModelBatch(X=X_pred, y=None, M=None)


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
