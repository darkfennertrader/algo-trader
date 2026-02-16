from __future__ import annotations

from typing import Any, Mapping

import torch

from algo_trader.application.simulation.metrics.registry import MetricFn, register_metric


@register_metric("stub", scope="outer")
def build_stub(*, spec: Mapping[str, Any]) -> MetricFn:
    _ = spec

    def score(
        y_true: torch.Tensor,
        pred: Mapping[str, Any],
        score_spec: Mapping[str, Any],
    ) -> float:
        _ = (y_true, pred, score_spec)
        return 0.0

    return score
