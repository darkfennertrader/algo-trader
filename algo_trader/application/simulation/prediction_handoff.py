from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import torch

from algo_trader.domain import SimulationError
from algo_trader.domain.simulation import PredictionPacket


def build_prediction_packet(
    *,
    pred: Mapping[str, Any],
    asset_names: Sequence[str],
    rebalance_index: int,
    rebalance_timestamp: Any | None,
) -> PredictionPacket:
    samples = _extract_samples(pred)
    mu = _extract_mu(pred, samples)
    covariance = _extract_covariance(pred, samples, mu)
    names = _resolve_asset_names(asset_names, mu.shape[0])
    tradable_mask = cast(
        torch.BoolTensor,
        torch.ones(
            (mu.shape[0],),
            dtype=torch.bool,
            device=mu.device,
        ),
    )
    return PredictionPacket(
        rebalance_index=rebalance_index,
        rebalance_timestamp=rebalance_timestamp,
        asset_names=names,
        tradable_mask=tradable_mask,
        mu=mu,
        covariance=covariance,
        samples=samples,
    )


def _extract_samples(pred: Mapping[str, Any]) -> torch.Tensor | None:
    raw = pred.get("samples")
    if raw is None:
        return None
    if not isinstance(raw, torch.Tensor):
        raise SimulationError("pred['samples'] must be a tensor when provided")
    if raw.ndim == 2:
        return raw.detach()
    if raw.ndim == 3:
        return raw[:, 0, :].detach()
    raise SimulationError(
        "pred['samples'] must have shape [S, A] or [S, T, A]"
    )


def _extract_mu(
    pred: Mapping[str, Any], samples: torch.Tensor | None
) -> torch.Tensor:
    raw = pred.get("mean")
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 1:
            return raw.detach()
        if raw.ndim == 2:
            return raw[0].detach()
        raise SimulationError("pred['mean'] must have shape [A] or [T, A]")
    if samples is None:
        raise SimulationError(
            "prediction handoff requires pred['mean'] or pred['samples']"
        )
    return samples.mean(dim=0)


def _extract_covariance(
    pred: Mapping[str, Any],
    samples: torch.Tensor | None,
    mu: torch.Tensor,
) -> torch.Tensor:
    raw = pred.get("covariance")
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 2:
            return raw.detach()
        if raw.ndim == 3:
            return raw[0].detach()
        raise SimulationError(
            "pred['covariance'] must have shape [A, A] or [T, A, A]"
        )
    if samples is None:
        return torch.zeros(
            (mu.shape[0], mu.shape[0]),
            dtype=mu.dtype,
            device=mu.device,
        )
    return _sample_covariance(samples)


def _sample_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] < 2:
        return torch.zeros(
            (samples.shape[1], samples.shape[1]),
            dtype=samples.dtype,
            device=samples.device,
        )
    centered = samples - samples.mean(dim=0, keepdim=True)
    scale = 1.0 / float(samples.shape[0] - 1)
    return scale * centered.transpose(0, 1).matmul(centered)


def _resolve_asset_names(
    asset_names: Sequence[str], asset_count: int
) -> tuple[str, ...]:
    if not asset_names:
        return tuple(f"asset_{idx}" for idx in range(asset_count))
    names = tuple(asset_names)
    if len(names) != asset_count:
        raise SimulationError(
            "asset names must match predictive asset dimension",
            context={
                "asset_name_count": str(len(names)),
                "asset_count": str(asset_count),
            },
        )
    return names
