"""Level 2 learning model: linear features + Gaussian residuals.

What is new vs Level 1:
- Add feature weights w[a, f].
- Predict mean using linear combination:
      mu[t, a] = alpha[a] + sum_f X[t, a, f] * w[a, f]

Math:
    alpha_a ~ Normal(0, alpha_scale)
    sigma_a ~ HalfCauchy(sigma_scale)
    w_a,f   ~ Normal(0, w_scale)
    y_t,a   ~ Normal(mu_t,a, sigma_a)

Shapes:
- X: [T, A, F]
- y: [T, A]
- w: [A, F]
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

    X: torch.Tensor  # [T, A, F]
    y: torch.Tensor  # [T, A]


@dataclass(frozen=True)
class LinearGaussianModel:
    """Per-asset linear regression with Gaussian noise."""

    alpha_scale: float = 0.02
    sigma_scale: float = 0.05
    w_scale: float = 0.05

    def __call__(self, batch: Batch) -> None:
        # Expected inputs:
        # X[t, a, f] -> feature f for asset a at time t
        # y[t, a]    -> realized return for asset a at time t
        if batch.X.ndim != 3:
            raise ValueError("Expected X with shape [T, A, F]")
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")
        if batch.X.shape[:2] != batch.y.shape:
            raise ValueError("X and y must agree on [T, A]")

        T, A, F = batch.X.shape
        device = batch.X.device
        dtype = batch.X.dtype

        # Plate over assets first, then features:
        # - alpha[a]: baseline return per asset
        # - sigma[a]: noise std per asset
        # - w[a,f]: feature weights per asset
        with pyro.plate("asset", A, dim=-2):
            alpha = pyro.sample(
                "alpha",
                dist.Normal(
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(self.alpha_scale, device=device, dtype=dtype),
                ),
            )  # [A, 1]
            sigma = pyro.sample(
                "sigma",
                dist.HalfCauchy(
                    torch.tensor(self.sigma_scale, device=device, dtype=dtype)
                ),
            )  # [A, 1]
            with pyro.plate("feature_w", F, dim=-1):
                w = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(self.w_scale, device=device, dtype=dtype),
                    ),
                )  # [A, F]

        # Remove singleton dims from alpha/sigma then compute linear predictor:
        # mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_vec = sigma.squeeze(-1)  # [A]
        mu = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T, A]
        # sigma is per asset, so broadcast across all times.
        scale = sigma_vec.unsqueeze(0).expand(T, A)  # [T, A]

        # Observation model for every (t, a) pair.
        with pyro.plate("time", T, dim=-2):
            with pyro.plate("asset_obs", A, dim=-1):
                pyro.sample("obs", dist.Normal(mu, scale), obs=batch.y)


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A, F = 3, 2, 4
    X = torch.zeros((T, A, F), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X=X, y=y)
    model = LinearGaussianModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l2_linear_gaussian.png"

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
