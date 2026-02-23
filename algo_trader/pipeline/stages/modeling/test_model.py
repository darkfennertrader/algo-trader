from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch

from .batch_utils import resolve_batch_shape, sample_observation
from .protocols import ModelBatch, PyroModel
from .registry_core import register_model


@dataclass(frozen=True)
class TestModel(PyroModel):
    def __call__(self, batch: ModelBatch) -> None:
        shape = resolve_batch_shape(batch)
        zeros = torch.zeros(shape.A, device=shape.device, dtype=shape.dtype)
        ones = torch.ones(shape.A, device=shape.device, dtype=shape.dtype)
        loc = pyro.sample(
            "loc",
            dist.Normal(zeros, ones).to_event(1),
        )
        scale = pyro.sample(
            "scale",
            dist.LogNormal(zeros, ones).to_event(1),
        )
        with pyro.plate("data", shape.T):
            obs_dist = dist.Normal(loc, scale).to_event(1)
            sample_observation(
                obs_dist=obs_dist,
                y_obs=shape.y_obs,
                mask=batch.M,
                scale=batch.obs_scale,
            )


@register_model("test_model")
def build_test_model(params: Mapping[str, Any]) -> PyroModel:
    _ = params
    return TestModel()
