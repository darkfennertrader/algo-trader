from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from algo_trader.domain import ConfigError
from ..batch_utils import resolve_batch_shape
from ..protocols import ModelBatch, PyroGuide
from ..registry_core import register_guide
from .model_v1 import FactorModelPriors, _build_factor_priors


@dataclass(frozen=True)
class FactorGuideV1(PyroGuide):
    priors: FactorModelPriors = field(default_factory=FactorModelPriors)

    def __call__(self, batch: ModelBatch) -> None:
        """Define the variational guide for FactorModelV1.

        ModelBatch:
        - X: torch.Tensor | None (features [T, A, F], scaled; may include mask
          channels if preprocessing.append_mask_as_features is true)
        - y: torch.Tensor | None (targets [T, A], realized returns)
        - M: torch.BoolTensor | None (target mask [T, A] for missing y)

        Frequency and horizon:
        - The simulation alignment is typically weekly.
        - That weekly timestep applies to X, y, and the T axis.
        - y[t] is a 1-step-ahead target: return realized over week t+1.

        Requirements:
        - Match all latent sample site names used in the model.
        - Use pyro.param for variational parameters with correct constraints.
        - Keep the call signature identical to the model (batch: ModelBatch).
        """
        context = _build_context(batch)
        _sample_global_params(self.priors, context)
        _sample_lambda(context)
        _sample_asset_params(context)


@register_guide("factor_guide_v1")
def build_factor_guide_v1(params: Mapping[str, Any]) -> PyroGuide:
    priors = _build_factor_priors(params)
    return FactorGuideV1(priors=priors)


@dataclass(frozen=True)
class _GuideContext:
    A: int
    F: int
    device: torch.device
    dtype: torch.dtype


def _build_context(batch: ModelBatch) -> _GuideContext:
    shape = resolve_batch_shape(batch)
    if batch.X is None:
        raise ConfigError("FactorGuideV1 requires batch.X with shape [T, A, F]")
    return _GuideContext(
        A=shape.A,
        F=int(batch.X.shape[-1]),
        device=shape.device,
        dtype=shape.dtype,
    )


def _sample_global_params(
    priors: FactorModelPriors, context: _GuideContext
) -> None:
    device, dtype = context.device, context.dtype
    nu_raw_loc = pyro.param(
        "nu_raw_loc",
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    nu_raw_scale = pyro.param(
        "nu_raw_scale",
        torch.tensor(1.0, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample("nu_raw", dist.LogNormal(nu_raw_loc, nu_raw_scale))

    tau0_loc = pyro.param(
        "tau0_loc",
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    tau0_scale = pyro.param(
        "tau0_scale",
        torch.tensor(1.0, device=device, dtype=dtype),
        constraint=constraints.positive,
    )
    pyro.sample("tau0", dist.LogNormal(tau0_loc, tau0_scale))

    if priors.horseshoe.use_regularized:
        c_loc = pyro.param(
            "c_loc",
            torch.tensor(0.0, device=device, dtype=dtype),
        )
        c_scale = pyro.param(
            "c_scale",
            torch.tensor(1.0, device=device, dtype=dtype),
            constraint=constraints.positive,
        )
        pyro.sample("c", dist.LogNormal(c_loc, c_scale))


def _sample_lambda(context: _GuideContext) -> None:
    device, dtype = context.device, context.dtype
    F = context.F
    with pyro.plate("feature", F, dim=-1):
        lambda_loc = pyro.param(
            "lambda_loc",
            torch.zeros(F, device=device, dtype=dtype),
        )
        lambda_scale = pyro.param(
            "lambda_scale",
            torch.ones(F, device=device, dtype=dtype),
            constraint=constraints.positive,
        )
        pyro.sample("lambda", dist.LogNormal(lambda_loc, lambda_scale))


def _sample_asset_params(context: _GuideContext) -> None:
    device, dtype = context.device, context.dtype
    A, F = context.A, context.F
    alpha_shape = (A, 1)
    sigma_shape = (A, 1)
    with pyro.plate("asset", A, dim=-2):
        alpha_loc = pyro.param(
            "alpha_loc",
            torch.zeros(alpha_shape, device=device, dtype=dtype),
        )
        alpha_scale = pyro.param(
            "alpha_scale",
            torch.ones(alpha_shape, device=device, dtype=dtype),
            constraint=constraints.positive,
        )
        pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale))

        sigma_loc = pyro.param(
            "sigma_loc",
            torch.zeros(sigma_shape, device=device, dtype=dtype),
        )
        sigma_scale = pyro.param(
            "sigma_scale",
            torch.ones(sigma_shape, device=device, dtype=dtype),
            constraint=constraints.positive,
        )
        pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))

        with pyro.plate("feature_w", F, dim=-1):
            kappa_loc = pyro.param(
                "kappa_loc",
                torch.zeros((A, F), device=device, dtype=dtype),
            )
            kappa_scale = pyro.param(
                "kappa_scale",
                torch.ones((A, F), device=device, dtype=dtype),
                constraint=constraints.positive,
            )
            pyro.sample("kappa", dist.LogNormal(kappa_loc, kappa_scale))

            w_loc = pyro.param(
                "w_loc",
                torch.zeros((A, F), device=device, dtype=dtype),
            )
            w_scale = pyro.param(
                "w_scale",
                torch.ones((A, F), device=device, dtype=dtype),
                constraint=constraints.positive,
            )
            pyro.sample("w", dist.Normal(w_loc, w_scale))
