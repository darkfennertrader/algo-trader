from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from .types import CVConfig, DataConfig, TrainingConfig

Batch = Mapping[str, torch.Tensor]
MetricFn = Callable[["BaseTrainer", Batch], float]


class PanelDataset(Protocol):
    @property
    def data(self) -> torch.Tensor: ...

    @property
    def targets(self) -> torch.Tensor: ...

    @property
    def dates(self) -> Sequence[Any]: ...

    @property
    def assets(self) -> Sequence[str]: ...

    @property
    def features(self) -> Sequence[str]: ...

    @property
    def device(self) -> str: ...

    def select_period_and_subsets(self, config: DataConfig) -> "PanelDataset":
        ...

    def slice_by_indices(self, indices: Sequence[int]) -> Batch:
        ...


class BaseCVSplitter(Protocol):
    def split(
        self, dates: Sequence[Any]
    ) -> Sequence[tuple[list[int], list[int]]]:
        ...


class BaseTrainer(Protocol):
    def fit(
        self,
        train_batch: Batch,
        training_config: TrainingConfig | None = None,
    ) -> object:
        ...

    def evaluate(
        self,
        val_batch: Batch,
        metrics: Mapping[str, MetricFn],
    ) -> Mapping[str, float]:
        ...
