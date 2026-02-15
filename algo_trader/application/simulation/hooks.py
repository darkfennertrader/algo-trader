from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain.simulation import (
    Allocator,
    ModelFitter,
    PnLCalculator,
    Predictor,
    Scorer,
)


@dataclass(frozen=True)
class SimulationHooks:
    fit_model: ModelFitter
    predict: Predictor
    score: Scorer
    allocate: Allocator
    compute_pnl: PnLCalculator


def default_hooks() -> SimulationHooks:
    return SimulationHooks(
        fit_model=_fit_bayes_svi_stub,
        predict=_posterior_predict_stub,
        score=_score_predictive_stub,
        allocate=_allocate_weights_stub,
        compute_pnl=_compute_weekly_pnl_stub,
    )


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
