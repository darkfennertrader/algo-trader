"""Level 3 learning model: linear features + Student-t residuals.

What is new vs Level 2:
- Keep linear predictor the same.
- Replace Gaussian observation noise with Student-t to handle outliers.
- Learn degrees of freedom nu from:
      nu_raw ~ Gamma(shape, rate), nu = nu_raw + shift

Math:
    nu_raw  ~ Gamma(shape, rate)
    nu      = nu_raw + shift
    alpha_a ~ Normal(0, alpha_scale)
    sigma_a ~ HalfCauchy(sigma_scale)
    w_a,f   ~ Normal(0, w_scale)
    y_t,a   ~ StudentT(df=nu, loc=mu_t,a, scale=sigma_a)
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
class LinearStudentTModel:
    """Per-asset linear regression with heavy-tailed residuals."""

    alpha_scale: float = 0.02
    sigma_scale: float = 0.05
    w_scale: float = 0.05
    nu_shape: float = 2.0
    nu_rate: float = 0.2
    nu_shift: float = 2.0

    def __call__(self, batch: Batch) -> None:
        # Expected inputs:
        # X[t, a, f] -> feature tensor
        # y[t, a]    -> observed returns
        if batch.X.ndim != 3:
            raise ValueError("Expected X with shape [T, A, F]")
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")
        if batch.X.shape[:2] != batch.y.shape:
            raise ValueError("X and y must agree on [T, A]")

        T, A, F = batch.X.shape
        device = batch.X.device
        dtype = batch.X.dtype

        # Heavy-tail parameter:
        # low nu => heavier tails; high nu => closer to Gaussian.
        # We sample nu_raw from Gamma and shift it upward to keep nu > 2.
        nu_raw = pyro.sample(
            "nu_raw",
            dist.Gamma(
                torch.tensor(self.nu_shape, device=device, dtype=dtype),
                torch.tensor(self.nu_rate, device=device, dtype=dtype),
            ),
        )
        nu = nu_raw + torch.tensor(self.nu_shift, device=device, dtype=dtype)

        # Per asset:
        # alpha[a] baseline mean, sigma[a] noise scale, w[a,f] feature weights.
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

        # Linear predictor:
        # mu[t,a] = alpha[a] + dot(X[t,a,:], w[a,:]).
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_vec = sigma.squeeze(-1)  # [A]
        mu = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T, A]
        # Broadcast per-asset sigma across time.
        scale = sigma_vec.unsqueeze(0).expand(T, A)  # [T, A]

        # Observation model for all (t, a):
        # StudentT handles outliers better than Normal.
        with pyro.plate("time", T, dim=-2):
            with pyro.plate("asset_obs", A, dim=-1):
                pyro.sample(
                    "obs",
                    dist.StudentT(df=nu, loc=mu, scale=scale),
                    obs=batch.y,
                )


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A, F = 3, 2, 4
    X = torch.zeros((T, A, F), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X=X, y=y)
    model = LinearStudentTModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l3_linear_student_t.png"

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
