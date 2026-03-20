from __future__ import annotations

from typing import NamedTuple

import pyro
import torch
from pyro import poutine
from pyro.distributions.torch_distribution import TorchDistributionMixin

from algo_trader.domain import ConfigError
from .protocols import ModelBatch


class BatchShape(NamedTuple):
    T: int
    A: int
    device: torch.device
    dtype: torch.dtype
    y_obs: torch.Tensor | None


def resolve_batch_shape(batch: ModelBatch) -> BatchShape:
    if batch.y is not None:
        if batch.y.ndim != 2:
            raise ConfigError("batch.y must have shape [T, A]")
        T, A = int(batch.y.shape[0]), int(batch.y.shape[1])
        return BatchShape(T, A, batch.y.device, batch.y.dtype, batch.y)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is not None:
        if X_asset.ndim != 3:
            raise ConfigError("batch.X_asset must have shape [T, A, F]")
        T, A = int(X_asset.shape[0]), int(X_asset.shape[1])
        return BatchShape(T, A, X_asset.device, X_asset.dtype, None)
    raise ConfigError("ModelBatch must provide X_asset or y")


def sample_observation(
    *,
    obs_dist: TorchDistributionMixin,
    y_obs: torch.Tensor | None,
    mask: torch.BoolTensor | None,
    scale: float | None = None,
) -> None:
    if mask is None:
        if scale is None:
            pyro.sample("obs", obs_dist, obs=y_obs)
            return
        with poutine.scale(  # pylint: disable=not-context-manager
            scale=scale
        ):
            pyro.sample("obs", obs_dist, obs=y_obs)
        return
    with poutine.mask(  # pylint: disable=not-context-manager
        mask=mask
    ):
        if scale is None:
            pyro.sample("obs", obs_dist, obs=y_obs)
            return
        with poutine.scale(  # pylint: disable=not-context-manager
            scale=scale
        ):
            pyro.sample("obs", obs_dist, obs=y_obs)
