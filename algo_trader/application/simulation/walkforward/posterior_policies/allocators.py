from __future__ import annotations

import torch

from algo_trader.domain.simulation import PredictionPacket

from .scores import scoped_confidence_scores
from .types import PosteriorConfidencePolicyConfig


def allocate_posterior_confidence(
    *,
    prediction: PredictionPacket,
    config: PosteriorConfidencePolicyConfig,
) -> torch.Tensor:
    scoped_scores = scoped_confidence_scores(
        prediction=prediction,
        score_name=config.score_name,
        block_scope=config.block_scope,
    )
    active_scores = torch.relu(scoped_scores - config.score_threshold)
    score_sum = active_scores.sum()
    if float(score_sum.item()) <= 1e-12:
        return torch.zeros_like(prediction.mu)
    return active_scores / score_sum


__all__ = ["allocate_posterior_confidence"]
