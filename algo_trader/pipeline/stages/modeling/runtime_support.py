from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState


@dataclass(frozen=True)
class RuntimeObservations:
    y_input: torch.Tensor
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None


def build_runtime_observations(
    *,
    y_input: torch.Tensor,
    y_obs: torch.Tensor | None,
    time_mask: torch.BoolTensor | None,
    obs_scale: float | None,
) -> RuntimeObservations:
    return RuntimeObservations(
        y_input=y_input,
        y_obs=y_obs,
        time_mask=time_mask,
        obs_scale=obs_scale,
    )


def require_tensor_entry(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ConfigError(
            "structural_posterior_means must include tensor entries",
            context={"field": key},
        )
    return value


def sample_time_observations(
    *,
    time_count: int,
    obs_dist: dist.TorchDistribution,
    y_obs: torch.Tensor | None,
    time_mask: torch.BoolTensor | None,
    obs_scale: float | None,
) -> None:
    with pyro.plate("time", time_count, dim=-1):
        if time_mask is None:
            _sample_with_optional_scale(
                obs_dist=obs_dist,
                y_obs=y_obs,
                obs_scale=obs_scale,
            )
            return
        with _managed_context(poutine.mask(mask=time_mask)):
            _sample_with_optional_scale(
                obs_dist=obs_dist,
                y_obs=y_obs,
                obs_scale=obs_scale,
            )


def move_filtering_state(
    *,
    filtering_state: FilteringState,
    device: torch.device,
    dtype: torch.dtype,
    coerce_tensor: Callable[[torch.Tensor], torch.Tensor],
) -> FilteringState:
    return FilteringState(
        h_loc=coerce_tensor(filtering_state.h_loc).to(device=device, dtype=dtype),
        h_scale=coerce_tensor(filtering_state.h_scale).to(
            device=device, dtype=dtype
        ),
        steps_seen=int(filtering_state.steps_seen),
    )


def _sample_with_optional_scale(
    *,
    obs_dist: dist.TorchDistribution,
    y_obs: torch.Tensor | None,
    obs_scale: float | None,
) -> None:
    if obs_scale is None:
        pyro.sample("obs", obs_dist, obs=y_obs)
        return
    with _managed_context(poutine.scale(scale=float(obs_scale))):
        pyro.sample("obs", obs_dist, obs=y_obs)


@contextmanager
def _managed_context(handler_obj: object) -> Iterator[None]:
    handler = handler_obj
    enter = getattr(handler, "__enter__", None)
    exit_handler = getattr(handler, "__exit__", None)
    if not callable(enter) or not callable(exit_handler):
        raise ConfigError("Invalid Pyro context manager")
    typed_enter = enter
    typed_exit = exit_handler
    typed_enter()
    try:
        yield
    finally:
        typed_exit(None, None, None)
