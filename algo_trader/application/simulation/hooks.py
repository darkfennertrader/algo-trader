from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pyro
import torch
from pyro import poutine

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
    batch = _build_training_batch(X_train, y_train)
    if batch.y is None or batch.y.numel() == 0:
        raise SimulationError("No valid targets for training")
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam(  # type: ignore  # pylint: disable=no-member
        {"lr": params.learning_rate}
    )
    loss = pyro.infer.Trace_ELBO(  # type: ignore  # pylint: disable=no-member
        num_particles=params.num_elbo_particles
    )
    svi = pyro.infer.SVI(  # type: ignore  # pylint: disable=no-member
        model, guide, optimizer, loss=loss
    )
    for _step in range(params.steps):
        svi.step(batch)
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
    return _TrainingParams(
        steps=int(training.get("num_steps", 0)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        num_elbo_particles=int(training.get("num_elbo_particles", 1)),
    )


def _build_training_batch(
    X_train: torch.Tensor, y_train: torch.Tensor
) -> modeling.ModelBatch:
    if X_train.ndim != 3:
        raise SimulationError("X_train must be [T, A, F]")
    if y_train.ndim != 2:
        raise SimulationError("y_train must be [T, A]")
    if X_train.shape[:2] != y_train.shape:
        raise SimulationError("X_train and y_train must align on [T, A]")
    if y_train.shape[1] == 0:
        raise SimulationError("y_train must have at least one asset")
    mask = torch.isfinite(y_train).all(dim=1)
    if not mask.any():
        return modeling.ModelBatch(
            X=X_train[:0], y=y_train[:0], M=None
        )
    X_obs = X_train[mask].to(dtype=torch.float32)
    y_obs = y_train[mask].to(dtype=torch.float32)
    return modeling.ModelBatch(X=X_obs, y=y_obs, M=None)


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
