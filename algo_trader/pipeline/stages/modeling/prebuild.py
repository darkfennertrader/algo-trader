from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class PrebuildContext:
    X_train: torch.Tensor
    y_train: torch.Tensor
    M_train: torch.Tensor
    feature_names: Sequence[str]
    assets: Sequence[str]
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrebuildResult:
    model_params: Mapping[str, Any] = field(default_factory=dict)
    guide_params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PrebuildHook(Protocol):
    def __call__(self, context: PrebuildContext) -> PrebuildResult:
        """Compute data-driven defaults for model/guide parameters."""
        ...
