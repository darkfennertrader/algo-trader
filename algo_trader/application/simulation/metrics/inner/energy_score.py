from __future__ import annotations

from typing import Any, Mapping

import torch

from algo_trader.domain import SimulationError
from algo_trader.application.simulation.metrics.registry import MetricFn, register_metric


@register_metric("energy_score", scope="inner")
def build_energy_score(*, spec: Mapping[str, Any]) -> MetricFn:
    _ = spec

    def score(
        y_true: torch.Tensor,
        pred: Mapping[str, Any],
        score_spec: Mapping[str, Any],
    ) -> float:
        _ = score_spec
        samples = _require_samples(pred)
        scale, whitener = _require_transform(pred)
        y_true_eval = _normalize_y_true(y_true)
        samples_eval = _normalize_samples(samples, y_true_eval)
        _ensure_finite(y_true_eval, samples_eval)
        u_true, u_samples = _whiten(
            y_true=y_true_eval,
            samples=samples_eval,
            scale=scale,
            whitener=whitener,
        )
        score_values = _energy_score_terms(u_true, u_samples)
        return float(-score_values.mean().detach().cpu())

    return score


def _require_samples(pred: Mapping[str, Any]) -> torch.Tensor:
    samples = pred.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise SimulationError("Energy score requires pred['samples'] tensor")
    return samples


def _require_transform(
    pred: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = pred.get("energy_score")
    if not isinstance(transform, Mapping):
        raise SimulationError(
            "Energy score requires pred['energy_score'] transform"
        )
    scale = transform.get("scale")
    whitener = transform.get("whitener")
    if not isinstance(scale, torch.Tensor) or not isinstance(
        whitener, torch.Tensor
    ):
        raise SimulationError(
            "Energy score transform must include scale and whitener tensors"
        )
    return scale, whitener


def _normalize_y_true(y_true: torch.Tensor) -> torch.Tensor:
    if y_true.ndim == 1:
        y_true_eval = y_true.unsqueeze(0)
    else:
        y_true_eval = y_true
    if y_true_eval.ndim != 2:
        raise SimulationError("Energy score y_true must be [T, A]")
    return y_true_eval


def _normalize_samples(
    samples: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    if samples.ndim == 2:
        samples_eval = samples.unsqueeze(1)
    else:
        samples_eval = samples
    if samples_eval.ndim != 3:
        raise SimulationError("Energy score samples must be [S, T, A]")
    if samples_eval.shape[1] != y_true.shape[0]:
        raise SimulationError(
            "Energy score samples and y_true must align on T"
        )
    if samples_eval.shape[2] != y_true.shape[1]:
        raise SimulationError(
            "Energy score samples and y_true must align on A"
        )
    return samples_eval


def _ensure_finite(y_true: torch.Tensor, samples: torch.Tensor) -> None:
    if not torch.isfinite(y_true).all():
        raise SimulationError("Energy score y_true contains NaNs/Infs")
    if not torch.isfinite(samples).all():
        raise SimulationError("Energy score samples contain NaNs/Infs")


def _whiten(
    *,
    y_true: torch.Tensor,
    samples: torch.Tensor,
    scale: torch.Tensor,
    whitener: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_true = y_true / scale
    z_samples = samples / scale
    u_true = torch.matmul(z_true, whitener.T)
    u_samples = torch.matmul(z_samples, whitener.T)
    return u_true, u_samples


def _energy_score_terms(
    u_true: torch.Tensor, u_samples: torch.Tensor
) -> torch.Tensor:
    diff = u_samples - u_true.unsqueeze(0)
    term1 = torch.norm(diff, dim=-1).mean(dim=0)
    u_samples_t = u_samples.transpose(0, 1)
    pairwise = torch.cdist(u_samples_t, u_samples_t)
    term2 = 0.5 * pairwise.mean(dim=(1, 2))
    return term1 - term2


def energy_score_terms(
    u_true: torch.Tensor, u_samples: torch.Tensor
) -> torch.Tensor:
    return _energy_score_terms(u_true, u_samples)
