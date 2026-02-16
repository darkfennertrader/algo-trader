from __future__ import annotations

from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch

from .batch_utils import resolve_batch_shape
from .protocols import ModelBatch, PyroModel
from .registry_core import register_model


@dataclass(frozen=True)
class TestModel(PyroModel):
    def __call__(self, batch: ModelBatch) -> None:
        shape = resolve_batch_shape(batch)
        T, A, device, dtype, y_obs = (
            shape.T,
            shape.A,
            shape.device,
            shape.dtype,
            shape.y_obs,
        )
        zeros = torch.zeros(A, device=device, dtype=dtype)
        ones = torch.ones(A, device=device, dtype=dtype)
        loc = pyro.sample(
            "loc",
            dist.Normal(zeros, ones).to_event(1),
        )
        scale = pyro.sample(
            "scale",
            dist.LogNormal(zeros, ones).to_event(1),
        )
        with pyro.plate("data", T):
            pyro.sample(
                "obs",
                dist.Normal(loc, scale).to_event(1),
                obs=y_obs,
            )


@register_model("test_model")
def build_test_model() -> PyroModel:
    return TestModel()
