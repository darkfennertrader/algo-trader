from __future__ import annotations

from typing import Sequence, cast

import torch

from algo_trader.domain import (
    COMMODITIES_BLOCK,
    FULL_BLOCK,
    FX_BLOCK,
    INDICES_BLOCK,
    ConfigError,
    build_asset_block_index_map,
)
from algo_trader.domain.simulation import PredictionPacket

VALID_POSTERIOR_POLICY_SCORE_NAMES = (
    "posterior_mean_over_std",
    "positive_probability_edge",
)
VALID_POSTERIOR_POLICY_BLOCK_SCOPES = (
    FULL_BLOCK,
    FX_BLOCK,
    INDICES_BLOCK,
    COMMODITIES_BLOCK,
)


def scoped_confidence_scores(
    *,
    prediction: PredictionPacket,
    score_name: str,
    block_scope: str,
) -> torch.Tensor:
    scores = posterior_scores(prediction=prediction, score_name=score_name)
    active_mask = scope_mask(
        asset_names=prediction.asset_names,
        block_scope=block_scope,
        device=prediction.mu.device,
    )
    return torch.where(active_mask, scores, torch.zeros_like(scores))


def posterior_scores(
    *,
    prediction: PredictionPacket,
    score_name: str,
) -> torch.Tensor:
    if score_name == "posterior_mean_over_std":
        return _posterior_mean_over_std(prediction)
    if score_name == "positive_probability_edge":
        return _positive_probability_edge(prediction)
    raise ConfigError(
        "allocation.spec.score_name must be posterior_mean_over_std or "
        "positive_probability_edge"
    )


def scope_mask(
    *,
    asset_names: Sequence[str],
    block_scope: str,
    device: torch.device,
) -> torch.BoolTensor:
    if block_scope not in VALID_POSTERIOR_POLICY_BLOCK_SCOPES:
        raise ConfigError(
            "allocation.spec.block_scope must be full, fx, indices, or "
            "commodities"
        )
    block_map = build_asset_block_index_map(asset_names)
    indices = block_map[block_scope]
    if not indices:
        raise ConfigError(
            "allocation.spec.block_scope resolved to an empty asset block"
        )
    mask = torch.zeros((len(asset_names),), device=device, dtype=torch.bool)
    mask[list(indices)] = True
    return cast(torch.BoolTensor, mask)


def _posterior_mean_over_std(prediction: PredictionPacket) -> torch.Tensor:
    if prediction.samples is not None:
        std = prediction.samples.std(dim=0, unbiased=False)
    else:
        std = prediction.covariance.diag().clamp_min(1e-12).sqrt()
    return prediction.mu / std.clamp_min(1e-12)


def _positive_probability_edge(prediction: PredictionPacket) -> torch.Tensor:
    if prediction.samples is None:
        raise ConfigError(
            "allocation.spec.score_name=positive_probability_edge requires "
            "posterior predictive samples"
        )
    p_positive = (prediction.samples > 0.0).to(prediction.mu.dtype).mean(dim=0)
    return p_positive - 0.5


__all__ = [
    "VALID_POSTERIOR_POLICY_BLOCK_SCOPES",
    "VALID_POSTERIOR_POLICY_SCORE_NAMES",
    "posterior_scores",
    "scope_mask",
    "scoped_confidence_scores",
]
