from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class PredictionPacket:
    rebalance_index: int
    rebalance_timestamp: Any | None
    asset_names: tuple[str, ...]
    tradable_mask: torch.BoolTensor
    mu: torch.Tensor
    covariance: torch.Tensor
    samples: torch.Tensor | None = None


@dataclass(frozen=True)
class AllocationRequest:
    prediction: PredictionPacket
    allocation_spec: Mapping[str, Any]
    previous_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class AllocationResult:
    rebalance_index: int
    rebalance_timestamp: Any | None
    asset_names: tuple[str, ...]
    weights: torch.Tensor
    expected_return: torch.Tensor | None = None
    expected_risk: torch.Tensor | None = None
    turnover: torch.Tensor | None = None
