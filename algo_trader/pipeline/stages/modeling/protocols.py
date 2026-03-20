from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, kw_only=True)
class ModelBatch:
    X: torch.Tensor | None = None
    X_asset: torch.Tensor | None = None
    X_global: torch.Tensor | None = None
    y: torch.Tensor | None
    M: torch.BoolTensor | None = None
    obs_scale: float | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        asset = self.X_asset if self.X_asset is not None else self.X
        legacy = self.X if self.X is not None else self.X_asset
        object.__setattr__(self, "X_asset", asset)
        object.__setattr__(self, "X", legacy)


class PyroModel(Protocol):
    def __call__(self, batch: ModelBatch) -> None:
        """Define a probabilistic model over the provided data."""
        ...


class PyroGuide(Protocol):
    def __call__(self, batch: ModelBatch) -> None:
        """Define a variational guide for the model."""
        ...
