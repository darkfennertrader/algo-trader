from __future__ import annotations

from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from .protocols import PyroGuide, PyroModel


@dataclass(frozen=True)
class NormalModel(PyroModel):
    def __call__(self, data: torch.Tensor) -> None:
        feature_count = data.shape[1]
        zeros = data.new_zeros(feature_count)
        ones = data.new_ones(feature_count)
        loc = pyro.sample(
            "loc",
            dist.Normal(zeros, ones).to_event(1),
        )
        scale = pyro.sample(
            "scale",
            dist.LogNormal(zeros, ones).to_event(1),
        )
        with pyro.plate("data", data.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(loc, scale).to_event(1),
                obs=data,
            )


@dataclass(frozen=True)
class NormalMeanFieldGuide(PyroGuide):
    def __call__(self, data: torch.Tensor) -> None:
        feature_count = data.shape[1]
        zeros = data.new_zeros(feature_count)
        ones = data.new_ones(feature_count)
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
