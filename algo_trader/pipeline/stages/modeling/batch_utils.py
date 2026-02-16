from __future__ import annotations

from typing import NamedTuple

import torch

from algo_trader.domain import ConfigError
from .protocols import ModelBatch


class BatchShape(NamedTuple):
    T: int
    A: int
    device: torch.device
    dtype: torch.dtype
    y_obs: torch.Tensor | None


def resolve_batch_shape(batch: ModelBatch) -> BatchShape:
    if batch.y is not None:
        if batch.y.ndim != 2:
            raise ConfigError("batch.y must have shape [T, A]")
        T, A = int(batch.y.shape[0]), int(batch.y.shape[1])
        return BatchShape(T, A, batch.y.device, batch.y.dtype, batch.y)
    if batch.X is not None:
        if batch.X.ndim != 3:
            raise ConfigError("batch.X must have shape [T, A, F]")
        T, A = int(batch.X.shape[0]), int(batch.X.shape[1])
        return BatchShape(T, A, batch.X.device, batch.X.dtype, None)
    raise ConfigError("ModelBatch must provide X or y")
