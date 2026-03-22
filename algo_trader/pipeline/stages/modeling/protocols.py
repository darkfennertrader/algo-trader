from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True, kw_only=True)
# pylint: disable=too-many-instance-attributes
class ModelBatch:
    X: torch.Tensor | None = None
    X_asset: torch.Tensor | None = None
    X_global: torch.Tensor | None = None
    y: torch.Tensor | None
    M: torch.BoolTensor | None = None
    obs_scale: float | None = None
    filtering_state: object | None = None
    asset_names: Sequence[str] | None = None
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

    def supported_training_methods(self) -> tuple[str, ...]:
        """Return the training methods this runtime model supports."""
        ...


class PyroGuide(Protocol):
    def __call__(self, batch: ModelBatch) -> None:
        """Define a variational guide for the model."""
        ...


@dataclass(frozen=True)
class PredictiveRequest:
    model: PyroModel
    guide: PyroGuide
    batch: ModelBatch
    num_samples: int
    state: Mapping[str, Any] | None = None


class PyroPredictor(Protocol):
    def __call__(
        self,
        request: PredictiveRequest,
    ) -> Mapping[str, Any] | None:
        """Generate predictive outputs from a model/guide pair."""
        ...
