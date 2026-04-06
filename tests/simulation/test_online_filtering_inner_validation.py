from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from algo_trader.application.simulation import inner_objective
from algo_trader.application.simulation.hooks import SimulationHooks
from algo_trader.domain.simulation import AllocationRequest, AllocationResult


def test_predict_validation_horizon_uses_single_block_for_tbptt() -> None:
    recorder = _HookRecorder()
    batches = _build_split_batches()

    pred = inner_objective._predict_validation_horizon(  # pylint: disable=protected-access
        hooks=recorder.hooks,
        batches=batches,
        config=_config("tbptt"),
        num_samples=4,
        model_state={"state_id": 1},
    )

    assert recorder.predict_lengths == [2]
    assert not recorder.fit_lengths
    assert pred["samples"].shape == (4, 2, 2)


def test_predict_validation_horizon_steps_for_online_filtering() -> None:
    recorder = _HookRecorder()
    batches = _build_split_batches()

    pred = inner_objective._predict_validation_horizon(  # pylint: disable=protected-access
        hooks=recorder.hooks,
        batches=batches,
        config=_config("online_filtering"),
        num_samples=4,
        model_state={"state_id": 1},
    )

    assert recorder.predict_lengths == [1, 1]
    assert recorder.fit_lengths == [1, 1]
    assert pred["samples"].shape == (4, 2, 2)
    assert pred["mean"].shape == (2, 2)
    assert pred["covariance"].shape == (2, 2, 2)


class _HookRecorder:
    def __init__(self) -> None:
        self.predict_lengths: list[int] = []
        self.fit_lengths: list[int] = []
        self._next_state_id = 1
        self.hooks = SimulationHooks(
            fit_model=cast(Any, self._fit_model),
            predict=cast(Any, self._predict),
            score=self._score,
            allocate=self._allocate,
            compute_pnl=cast(Any, self._compute_pnl),
        )

    def _fit_model(self, **kwargs: Any) -> Mapping[str, Any]:
        X_train = cast(torch.Tensor, kwargs["X_train"])
        X_train_global = kwargs.get("X_train_global")
        y_train = cast(torch.Tensor, kwargs["y_train"])
        config = kwargs.get("config")
        init_state = kwargs.get("init_state")
        del X_train_global, config, init_state
        self.fit_lengths.append(int(X_train.shape[0]))
        self._next_state_id += 1
        return {"state_id": self._next_state_id, "y_shape": tuple(y_train.shape)}

    def _predict(self, **kwargs: Any) -> Mapping[str, Any]:
        X_pred = cast(torch.Tensor, kwargs["X_pred"])
        X_pred_global = kwargs.get("X_pred_global")
        state = cast(Mapping[str, Any], kwargs["state"])
        config = kwargs.get("config")
        num_samples = int(kwargs["num_samples"])
        del X_pred_global, config
        length = int(X_pred.shape[0])
        self.predict_lengths.append(length)
        asset_count = int(X_pred.shape[1])
        state_scale = float(state["state_id"])
        samples = torch.full(
            (num_samples, length, asset_count),
            fill_value=state_scale,
            dtype=torch.float32,
        )
        mean = samples.mean(dim=0)
        covariance = torch.stack(
            [torch.eye(asset_count, dtype=torch.float32) for _ in range(length)],
            dim=0,
        )
        return {
            "samples": samples,
            "mean": mean,
            "covariance": covariance,
        }

    def _score(
        self,
        y_true: torch.Tensor,
        pred: Mapping[str, Any],
        score_spec: Mapping[str, Any],
    ) -> float:
        del y_true, pred, score_spec
        return 0.0

    def _allocate(self, request: AllocationRequest) -> AllocationResult:
        del request
        return AllocationResult(
            rebalance_index=0,
            rebalance_timestamp=None,
            asset_names=(),
            weights=torch.zeros(0),
        )

    def _compute_pnl(
        self,
        w: torch.Tensor,
        y_t: torch.Tensor,
        w_prev: torch.Tensor | None = None,
        cost_spec: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        del w, y_t, w_prev, cost_spec
        return torch.tensor(0.0)


def _build_split_batches() -> inner_objective.SplitBatches:
    return inner_objective.SplitBatches(
        cleaning=inner_objective.FeatureCleaningState(
            feature_idx=torch.tensor([], dtype=torch.int64).cpu().numpy(),
            usable_ratio=torch.tensor([], dtype=torch.float32).cpu().numpy(),
            variance=torch.tensor([], dtype=torch.float32).cpu().numpy(),
            dropped_low_usable=torch.tensor([], dtype=torch.int64).cpu().numpy(),
            dropped_low_var=torch.tensor([], dtype=torch.int64).cpu().numpy(),
            dropped_duplicates=torch.tensor([], dtype=torch.int64).cpu().numpy(),
            duplicate_pairs=[],
        ),
        X_train=torch.zeros((3, 2, 1)),
        X_train_global=torch.zeros((3, 1)),
        y_train=torch.zeros((3, 2)),
        X_test=torch.arange(4, dtype=torch.float32).reshape(2, 2, 1),
        X_test_global=torch.zeros((2, 1)),
        y_test=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )


def _config(method: str) -> dict[str, Any]:
    return {"training": {"method": method}}
