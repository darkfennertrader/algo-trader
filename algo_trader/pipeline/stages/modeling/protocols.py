from __future__ import annotations

from typing import Protocol

import torch


class PyroModel(Protocol):
    def __call__(self, data: torch.Tensor) -> None:
        """Define a probabilistic model over the provided data."""
        ...


class PyroGuide(Protocol):
    def __call__(self, data: torch.Tensor) -> None:
        """Define a variational guide for the model."""
        ...
