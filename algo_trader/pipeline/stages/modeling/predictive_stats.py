from __future__ import annotations

import torch


def predictive_covariance(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 2:
        return _sample_covariance(samples)
    if samples.ndim != 3:
        raise ValueError("Predictive samples must be [S, A] or [S, T, A]")
    covariances = []
    for t in range(int(samples.shape[1])):
        covariances.append(_sample_covariance(samples[:, t, :]))
    return torch.stack(covariances, dim=0)


def _sample_covariance(samples_t: torch.Tensor) -> torch.Tensor:
    if int(samples_t.shape[0]) <= 1:
        return torch.zeros(
            (samples_t.shape[1], samples_t.shape[1]),
            device=samples_t.device,
            dtype=samples_t.dtype,
        )
    return torch.cov(samples_t.transpose(0, 1))
