from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from algo_trader.domain.simulation import CPCVSplit, PreprocessSpec
from .hooks import SimulationHooks
from .preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)


@dataclass(frozen=True)
class InnerObjectiveContext:
    X: torch.Tensor
    M: torch.Tensor
    y: torch.Tensor
    inner_splits: list[CPCVSplit]
    preprocess_spec: PreprocessSpec
    score_spec: Mapping[str, Any]
    num_pp_samples: int


def make_inner_objective(
    *,
    context: InnerObjectiveContext,
    hooks: SimulationHooks,
) -> Callable[[Mapping[str, Any]], float]:
    def objective(config: Mapping[str, Any]) -> float:
        fold_scores: list[float] = []
        for split in context.inner_splits:
            cleaning = fit_feature_cleaning(
                X=context.X,
                M=context.M,
                train_idx=split.train_idx,
                spec=context.preprocess_spec,
                frozen_feature_idx=None,
            )
            if cleaning.feature_idx.size == 0:
                continue

            scaler = fit_robust_scaler(
                X=context.X,
                M=context.M,
                train_idx=split.train_idx,
                cleaning=cleaning,
                spec=context.preprocess_spec,
            )

            state = TransformState(
                cleaning=cleaning,
                scaler=scaler,
                spec=context.preprocess_spec,
            )

            X_train = transform_X(
                context.X,
                context.M,
                split.train_idx,
                state,
            )
            y_train = context.y[
                torch.as_tensor(
                    split.train_idx, dtype=torch.long, device=context.y.device
                )
            ]

            X_test = transform_X(
                context.X,
                context.M,
                split.test_idx,
                state,
            )
            y_test = context.y[
                torch.as_tensor(
                    split.test_idx, dtype=torch.long, device=context.y.device
                )
            ]

            model_state = hooks.fit_model(
                X_train=X_train,
                y_train=y_train,
                config=config,
                init_state=None,
            )

            pred = hooks.predict(
                X_pred=X_test,
                state=model_state,
                config=config,
                num_samples=context.num_pp_samples,
            )

            fold_scores.append(
                float(
                    hooks.score(
                        y_true=y_test,
                        pred=pred,
                        score_spec=context.score_spec,
                    )
                )
            )

        return float(np.mean(fold_scores)) if fold_scores else float("-inf")

    return objective
