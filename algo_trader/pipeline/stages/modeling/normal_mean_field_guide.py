from __future__ import annotations

from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from .batch_utils import resolve_batch_shape
from .protocols import ModelBatch, PyroGuide
from .registry import register_guide


@dataclass(frozen=True)
class NormalMeanFieldGuide(PyroGuide):
    def __call__(self, batch: ModelBatch) -> None:
        shape = resolve_batch_shape(batch)
        A, device, dtype = shape.A, shape.device, shape.dtype
        zeros = torch.zeros(A, device=device, dtype=dtype)
        ones = torch.ones(A, device=device, dtype=dtype)
        loc_loc = pyro.param(
            "loc_loc",
            zeros.clone(),
        )
        loc_scale = pyro.param(
            "loc_scale",
            ones.clone(),
            constraint=constraints.positive,
        )
        pyro.sample(
            "loc",
            dist.Normal(loc_loc, loc_scale).to_event(1),
        )
        scale_loc = pyro.param(
            "scale_loc",
            zeros.clone(),
        )
        scale_scale = pyro.param(
            "scale_scale",
            ones.clone(),
            constraint=constraints.positive,
        )
        pyro.sample(
            "scale",
            dist.LogNormal(scale_loc, scale_scale).to_event(1),
        )


@register_guide("normal_mean_field")
def build_normal_mean_field_guide() -> PyroGuide:
    return NormalMeanFieldGuide()
