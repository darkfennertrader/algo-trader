from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ModelBatch:
    X: torch.Tensor | None
    y: torch.Tensor | None
    M: torch.Tensor | None = None


class PyroModel(Protocol):
    def __call__(self, batch: ModelBatch) -> None:
        """Define a probabilistic model over the provided data."""
        ...


class PyroGuide(Protocol):
    def __call__(self, batch: ModelBatch) -> None:
        """Define a variational guide for the model."""
        ...
