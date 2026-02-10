from __future__ import annotations

from typing import Callable, Mapping

import torch

from algo_trader.domain.model_selection import BaseTrainer, MetricFn as DomainMetricFn

Batch = Mapping[str, torch.Tensor]
ModelFn = Callable[[Batch], None]
GuideFn = Callable[[Batch], None]
MetricFn = DomainMetricFn
