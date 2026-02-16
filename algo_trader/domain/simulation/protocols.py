from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import torch

class PanelDataset(Protocol):
    @property
    def data(self) -> torch.Tensor: ...

    @property
    def targets(self) -> torch.Tensor: ...

    @property
    def missing_mask(self) -> torch.Tensor: ...

    @property
    def dates(self) -> Sequence[Any]: ...

    @property
    def assets(self) -> Sequence[str]: ...

    @property
    def features(self) -> Sequence[str]: ...

    @property
    def device(self) -> str: ...


class ModelFitter(Protocol):
    def __call__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        config: Mapping[str, Any],
        init_state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class Predictor(Protocol):
    def __call__(
        self,
        X_pred: torch.Tensor,
        state: Mapping[str, Any],
        config: Mapping[str, Any],
        num_samples: int,
    ) -> Mapping[str, Any]: ...


class Scorer(Protocol):
    def __call__(
        self,
        y_true: torch.Tensor,
        pred: Mapping[str, Any],
        score_spec: Mapping[str, Any],
    ) -> float: ...


class Allocator(Protocol):
    def __call__(
        self,
        pred: Mapping[str, Any],
        alloc_spec: Mapping[str, Any],
    ) -> torch.Tensor: ...


class PnLCalculator(Protocol):
    def __call__(
        self,
        w: torch.Tensor,
        y_t: torch.Tensor,
        w_prev: torch.Tensor | None = None,
        cost_spec: Mapping[str, Any] | None = None,
    ) -> torch.Tensor: ...
