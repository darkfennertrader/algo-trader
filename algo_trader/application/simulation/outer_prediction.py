from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from algo_trader.domain.simulation import (
    FeatureCleaningState,
    OuterFold,
    PredictionPacket,
    PreprocessSpec,
)

from .feature_panel_data import (
    FeaturePanelData,
    prepare_global_feature_batches,
    with_run_context_updates,
)
from .hooks import SimulationHooks
from .prediction_handoff import build_prediction_packet
from .preprocessing import (
    TransformState,
    fit_feature_cleaning,
    fit_robust_scaler,
    transform_X,
)


@dataclass(frozen=True)
class OuterPredictionData:
    panel: FeaturePanelData
    y: torch.Tensor
    timestamps: Sequence[Any]


@dataclass(frozen=True)
class OuterPredictionRuntime:
    outer_fold: OuterFold
    preprocess_spec: PreprocessSpec
    num_pp_samples: int
    assets: Sequence[str]
    execution_mode: str
    week_progress: Callable[[int, Any], None] | None = None


@dataclass(frozen=True)
class OuterPredictionContext:
    data: OuterPredictionData
    runtime: OuterPredictionRuntime


@dataclass(frozen=True)
class OuterPredictionInputs:
    panel: FeaturePanelData
    y: torch.Tensor
    timestamps: Sequence[Any]
    preprocess_spec: PreprocessSpec
    num_pp_samples: int
    assets: Sequence[str]
    execution_mode: str


class SupportsOuterPredictionInputs(Protocol):
    @property
    def panel(self) -> FeaturePanelData: ...

    @property
    def y(self) -> torch.Tensor: ...

    @property
    def timestamps(self) -> Sequence[Any]: ...

    @property
    def preprocess_spec(self) -> PreprocessSpec: ...

    @property
    def num_pp_samples(self) -> int: ...

    @property
    def assets(self) -> Sequence[str]: ...

    @property
    def execution_mode(self) -> str: ...


@dataclass(frozen=True)
class PreparedOuterBatches:
    X_train: torch.Tensor
    X_train_global: torch.Tensor | None
    y_train: torch.Tensor
    X_pred: torch.Tensor
    X_pred_global: torch.Tensor | None


@dataclass(frozen=True)
class WeeklyPredictionResult:
    outer_k: int
    timestamp: Any
    prediction: PredictionPacket
    realized_returns: torch.Tensor


@dataclass(frozen=True)
class PredictionLoopContext:
    context: OuterPredictionContext
    best_config: Mapping[str, Any]
    hooks: SimulationHooks
    base_train: np.ndarray
    test_weeks: np.ndarray
    cleaning_outer: FeatureCleaningState


def build_outer_prediction_context(
    *,
    owner: SupportsOuterPredictionInputs,
    outer_fold: OuterFold,
    week_progress: Callable[[int, Any], None] | None,
) -> OuterPredictionContext:
    return OuterPredictionContext(
        data=OuterPredictionData(
            panel=owner.panel,
            y=owner.y,
            timestamps=owner.timestamps,
        ),
        runtime=OuterPredictionRuntime(
            outer_fold=outer_fold,
            preprocess_spec=owner.preprocess_spec,
            num_pp_samples=owner.num_pp_samples,
            assets=owner.assets,
            execution_mode=owner.execution_mode,
            week_progress=week_progress,
        ),
    )


def evaluate_outer_predictions(
    *,
    context: OuterPredictionContext,
    best_config: Mapping[str, Any],
    hooks: SimulationHooks,
) -> tuple[list[WeeklyPredictionResult], FeatureCleaningState | None]:
    test_weeks = _sorted_indices(context.runtime.outer_fold.test_idx)
    base_train = _sorted_indices(context.runtime.outer_fold.train_idx)
    cleaning_outer = fit_feature_cleaning(
        X=context.data.panel.X,
        M=context.data.panel.M,
        train_idx=base_train,
        spec=context.runtime.preprocess_spec,
        frozen_feature_idx=None,
    )
    if cleaning_outer.feature_idx.size == 0:
        return [], cleaning_outer
    loop_context = PredictionLoopContext(
        context=context,
        best_config=best_config,
        hooks=hooks,
        base_train=base_train,
        test_weeks=test_weeks,
        cleaning_outer=cleaning_outer,
    )
    results: list[WeeklyPredictionResult] = []
    state: Mapping[str, Any] | None = None
    for current_t in test_weeks.tolist():
        state, weekly = _evaluate_week(
            loop_context=loop_context,
            current_t=int(current_t),
            state=state,
        )
        results.append(weekly)
        _report_week_progress(context, int(current_t))
    return results, cleaning_outer


def _sorted_indices(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(indices, dtype=int))


def _evaluate_week(
    *,
    loop_context: PredictionLoopContext,
    current_t: int,
    state: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, WeeklyPredictionResult]:
    config = _with_asset_names(
        loop_context.best_config,
        loop_context.context.runtime.assets,
        loop_context.context.runtime.execution_mode,
    )
    train_idx_t = _expanding_train(
        loop_context.base_train,
        loop_context.test_weeks,
        current_t,
    )
    batches = _prepare_batches(
        context=loop_context.context,
        train_idx=train_idx_t,
        pred_t=current_t,
        cleaning_outer=loop_context.cleaning_outer,
    )
    state = loop_context.hooks.fit_model(
        X_train=batches.X_train,
        X_train_global=batches.X_train_global,
        y_train=batches.y_train,
        config=config,
        init_state=state,
    )
    prediction = _build_weekly_prediction(
        loop_context=loop_context,
        batches=batches,
        state=state,
        current_t=current_t,
        config=config,
    )
    return state, WeeklyPredictionResult(
        outer_k=loop_context.context.runtime.outer_fold.k_test,
        timestamp=prediction.rebalance_timestamp,
        prediction=prediction,
        realized_returns=loop_context.context.data.y[current_t],
    )


def _with_asset_names(
    config: Mapping[str, Any],
    asset_names: Sequence[str],
    execution_mode: str,
) -> Mapping[str, Any]:
    return with_run_context_updates(
        config,
        asset_names=list(asset_names),
        execution_mode=execution_mode,
    )


def _prepare_batches(
    *,
    context: OuterPredictionContext,
    train_idx: np.ndarray,
    pred_t: int,
    cleaning_outer: FeatureCleaningState,
) -> PreparedOuterBatches:
    scaler_t = fit_robust_scaler(
        X=context.data.panel.X,
        M=context.data.panel.M,
        train_idx=train_idx,
        cleaning=cleaning_outer,
        spec=context.runtime.preprocess_spec,
    )
    state_t = TransformState(
        cleaning=cleaning_outer,
        scaler=scaler_t,
        spec=context.runtime.preprocess_spec,
    )
    X_train_t = transform_X(
        context.data.panel.X,
        context.data.panel.M,
        train_idx,
        state_t,
        validate=True,
    )
    y_train_t = context.data.y[
        torch.tensor(
            np.array(train_idx, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=context.data.y.device,
        )
    ]
    X_pred_t = transform_X(
        context.data.panel.X,
        context.data.panel.M,
        np.array([pred_t], dtype=int),
        state_t,
    )
    X_train_global_t, X_pred_global_t = prepare_global_feature_batches(
        panel=context.data.panel,
        train_idx=train_idx,
        test_idx=np.array([pred_t], dtype=int),
        preprocess_spec=context.runtime.preprocess_spec,
        validate_train=True,
    )
    return PreparedOuterBatches(
        X_train=X_train_t,
        X_train_global=X_train_global_t,
        y_train=y_train_t,
        X_pred=X_pred_t,
        X_pred_global=X_pred_global_t,
    )


def _build_weekly_prediction(
    *,
    loop_context: PredictionLoopContext,
    batches: PreparedOuterBatches,
    state: Mapping[str, Any] | None,
    current_t: int,
    config: Mapping[str, Any],
) -> PredictionPacket:
    pred = loop_context.hooks.predict(
        X_pred=batches.X_pred,
        X_pred_global=batches.X_pred_global,
        state=state or {},
        config=config,
        num_samples=loop_context.context.runtime.num_pp_samples,
    )
    return build_prediction_packet(
        pred=pred,
        asset_names=loop_context.context.runtime.assets,
        rebalance_index=current_t,
        rebalance_timestamp=loop_context.context.data.timestamps[current_t],
    )


def _expanding_train(
    base_train: np.ndarray,
    test_weeks: np.ndarray,
    current_t: int,
) -> np.ndarray:
    realized_test = test_weeks[test_weeks < current_t]
    return np.unique(np.concatenate([base_train, realized_test]))


def _report_week_progress(
    context: OuterPredictionContext,
    current_t: int,
) -> None:
    if context.runtime.week_progress is None:
        return
    context.runtime.week_progress(
        context.runtime.outer_fold.k_test,
        context.data.timestamps[current_t],
    )
