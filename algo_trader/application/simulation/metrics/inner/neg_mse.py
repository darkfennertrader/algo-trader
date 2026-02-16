from __future__ import annotations

from typing import Any, Mapping

import torch

from algo_trader.domain import SimulationError
from algo_trader.application.simulation.metrics.registry import MetricFn, register_metric


@register_metric("neg_mse", scope="inner")
def build_neg_mse(*, spec: Mapping[str, Any]) -> MetricFn:
    _ = spec

    def score(
        y_true: torch.Tensor,
        pred: Mapping[str, Any],
        score_spec: Mapping[str, Any],
    ) -> float:
        _ = score_spec
        mean = pred.get("mean")
        if not isinstance(mean, torch.Tensor):
            raise SimulationError("Prediction missing mean tensor")
        mask = torch.isfinite(y_true)
        if not mask.any():
            return float("-inf")
        err = (mean - y_true)[mask]
        mse = torch.mean(err ** 2)
        return float(-mse.detach().cpu())

    return score
