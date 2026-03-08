"""Level 1 learning model: intercept-only Bayesian regression.

Goal:
- Start with the smallest useful probabilistic model.
- No features yet; each asset has only:
  - alpha[a]: mean return
  - sigma[a]: residual volatility

Math:
    alpha_a ~ Normal(0, alpha_scale)
    sigma_a ~ HalfCauchy(sigma_scale)
    y_t,a   ~ Normal(alpha_a, sigma_a)

Shapes:
- y: [T, A]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyro
import pyro.distributions as dist
import torch


@dataclass(frozen=True)
class Batch:
    """Minimal batch for learning examples."""

    y: torch.Tensor  # [T, A]


@dataclass(frozen=True)
class InterceptOnlyModel:
    """Intercept-only model with per-asset volatility."""

    alpha_scale: float = 0.02
    sigma_scale: float = 0.05

    def __call__(self, batch: Batch) -> None:
        # We expect a 2D target matrix:
        # T = number of time steps, A = number of assets.
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")

        T, A = batch.y.shape
        device = batch.y.device
        dtype = batch.y.dtype

        # One plate over assets means:
        # "sample one alpha and one sigma for each asset".
        with pyro.plate("asset", A, dim=-1):
            alpha = pyro.sample(
                "alpha",
                dist.Normal(
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(self.alpha_scale, device=device, dtype=dtype),
                ),
            )  # [A]
            sigma = pyro.sample(
                "sigma",
                dist.HalfCauchy(
                    torch.tensor(self.sigma_scale, device=device, dtype=dtype)
                ),
            )  # [A]

        # alpha and sigma are [A], but observations are [T, A].
        # So we broadcast asset parameters across time:
        # - unsqueeze(0): [1, A]
        # - expand(T, A): [T, A]
        mu = alpha.unsqueeze(0).expand(T, A)  # [T, A]
        scale = sigma.unsqueeze(0).expand(T, A)  # [T, A]

        # Observation model:
        # y[t, a] is Normal with mean alpha[a] and std sigma[a].
        # Plate layout here matches y shape [T, A].
        with pyro.plate("time", T, dim=-2):
            with pyro.plate("asset_obs", A, dim=-1):
                pyro.sample("obs", dist.Normal(mu, scale), obs=batch.y)


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A = 3, 2
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(y=y)
    model = InterceptOnlyModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l1_intercept.png"

    pyro.render_model(
        model,
        model_args=(batch,),
        filename=str(output_path),
        render_params=True,
        render_distributions=True,
    )
    return output_path


if __name__ == "__main__":
    output = _render_model_graph()
    print(f"Saved model graph to {output}")
