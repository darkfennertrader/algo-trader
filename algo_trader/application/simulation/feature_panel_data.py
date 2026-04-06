from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algo_trader.domain.simulation import PreprocessSpec

from .preprocessing import (
    GlobalBlockInputs,
    TransformState,
    fit_global_feature_cleaning,
    fit_global_robust_scaler,
    transform_global_X,
)


@dataclass(frozen=True)
class FeaturePanelData:
    X: torch.Tensor
    M: torch.Tensor
    X_global: torch.Tensor | None
    M_global: torch.Tensor | None
    global_feature_names: Sequence[str]


def prepare_global_feature_batches(
    *,
    panel: FeaturePanelData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    preprocess_spec: PreprocessSpec,
    validate_train: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if panel.X_global is None or panel.M_global is None:
        return None, None
    global_inputs = GlobalBlockInputs(
        X=panel.X_global,
        M=panel.M_global,
        feature_names=panel.global_feature_names,
    )
    cleaning = fit_global_feature_cleaning(
        inputs=global_inputs,
        train_idx=train_idx,
        spec=preprocess_spec,
        frozen_feature_idx=None,
    )
    if cleaning.feature_idx.size == 0:
        return None, None
    scaler = fit_global_robust_scaler(
        inputs=global_inputs,
        train_idx=train_idx,
        cleaning=cleaning,
        spec=preprocess_spec,
    )
    state = TransformState(
        cleaning=cleaning,
        scaler=scaler,
        spec=preprocess_spec,
    )
    return (
        transform_global_X(
            global_inputs,
            train_idx,
            state,
            validate=validate_train,
        ),
        transform_global_X(global_inputs, test_idx, state),
    )


def with_run_context_updates(
    config: Mapping[str, Any],
    **updates: Any,
) -> Mapping[str, Any]:
    run_context = config.get("run_context")
    merged_context = dict(run_context) if isinstance(run_context, Mapping) else {}
    merged_context.update(updates)
    return {
        **config,
        "run_context": merged_context,
    }
