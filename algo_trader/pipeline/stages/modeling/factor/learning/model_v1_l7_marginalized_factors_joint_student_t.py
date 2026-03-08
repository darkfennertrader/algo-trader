"""Level 7 learning model: marginalized factors + joint heavy tails.

What is new vs Level 6:
- Keep the same mean model as Level 4/6:
  mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]
- Keep the same priors for B, sigma_idio, and weekly u[t].
- Integrate out weekly latent factor shocks f[t] from the likelihood.

Collapsed likelihood:
    Sigma = B B' + diag(sigma_idio^2)
    y[t] | u[t] ~ MVN(mean=mu[t], covariance=Sigma / u[t])

Practical implementation:
- Use LowRankMultivariateNormal with:
  cov_factor[t] = B / sqrt(u[t])      # [A,K]
  cov_diag[t]   = sigma_idio^2 / u[t] # [A]
- One time plate only.
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
class MarginalizedFactorJointStudentTModel:
    """Low-rank multivariate residual model with shared weekly heavy-tail scale."""

    # Mean model priors (same spirit as Level 4)
    alpha_scale: float = 0.02
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12

    # Residual factor priors
    factor_count: int = 3
    b_scale: float = 0.20
    b_col_shrink_scale: float = 0.50
    sigma_idio_scale: float = 0.05

    # Fixed df for weekly scale mixture.
    nu: float = 10.0

    def __call__(self, batch: Batch) -> None:
        # Expected inputs:
        # X[t,a,f] and y[t,a], aligned on [T, A].
        if batch.X.ndim != 3:
            raise ValueError("Expected X with shape [T, A, F]")
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")
        if batch.X.shape[:2] != batch.y.shape:
            raise ValueError("X and y must agree on [T, A]")

        T, A, F = batch.X.shape
        K = int(self.factor_count)
        if K <= 0:
            raise ValueError("factor_count must be positive")
        if self.nu <= 2.0:
            raise ValueError("nu should be > 2 for finite variance")
        device = batch.X.device
        dtype = batch.X.dtype

        # Horseshoe globals for feature weights.
        tau0 = pyro.sample(
            "tau0",
            dist.HalfCauchy(
                torch.tensor(self.tau0_scale, device=device, dtype=dtype)
            ),
        )
        with pyro.plate("feature", F, dim=-1):
            lam = pyro.sample(
                "lambda",
                dist.HalfCauchy(
                    torch.full((F,), self.lambda_scale, device=device, dtype=dtype)
                ),
            )  # [F]
        c = pyro.sample(
            "c",
            dist.HalfCauchy(
                torch.tensor(self.c_scale, device=device, dtype=dtype)
            ),
        )

        # Column-wise shrinkage for factor loadings B[:,k].
        with pyro.plate("factor_loading_col", K, dim=-1):
            b_col = pyro.sample(
                "b_col",
                dist.HalfNormal(
                    torch.full(
                        (K,), self.b_col_shrink_scale, device=device, dtype=dtype
                    )
                ),
            )  # [K]

        # Asset-level params: alpha, w, sigma_idio, B.
        with pyro.plate("asset", A, dim=-2):
            alpha = pyro.sample(
                "alpha",
                dist.Normal(
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(self.alpha_scale, device=device, dtype=dtype),
                ),
            )  # [A,1]
            sigma_idio = pyro.sample(
                "sigma_idio",
                dist.HalfNormal(
                    torch.tensor(self.sigma_idio_scale, device=device, dtype=dtype)
                ),
            )  # [A,1]

            with pyro.plate("feature_w", F, dim=-1):
                kappa = pyro.sample(
                    "kappa",
                    dist.HalfCauchy(
                        torch.tensor(self.kappa_scale, device=device, dtype=dtype)
                    ),
                )  # [A,F]

                # Regularized horseshoe for w scale.
                base = lam * kappa
                base2 = base.pow(2)
                c2 = c * c
                tau2 = tau0 * tau0
                lam_tilde = torch.sqrt((c2 * base2) / (c2 + tau2 * base2 + self.eps))
                w_scale = tau0 * lam_tilde

                w = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        w_scale,
                    ),
                )  # [A,F]

            with pyro.plate("factor_loading_k", K, dim=-1):
                B = pyro.sample(
                    "B",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(self.b_scale, device=device, dtype=dtype) * b_col,
                    ),
                )  # [A,K]

        # Shared weekly stress scale.
        nu_half = torch.tensor(self.nu / 2.0, device=device, dtype=dtype)
        with pyro.plate("time_scale", T, dim=-1):
            u = pyro.sample("u", dist.Gamma(nu_half, nu_half))  # [T]

        # Mean block (unchanged from Level 4 style).
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_idio_vec = sigma_idio.squeeze(-1)  # [A]
        mu = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T,A]

        # Collapsed low-rank covariance for each week t:
        # cov = (B B' + diag(sigma_idio^2)) / u[t].
        inv_sqrt_u = torch.rsqrt(u).unsqueeze(-1)  # [T,1]
        cov_factor = inv_sqrt_u.unsqueeze(-1) * B.unsqueeze(0)  # [T,A,K]
        cov_diag = sigma_idio_vec.pow(2).unsqueeze(0) / u.unsqueeze(-1)  # [T,A]

        obs_dist = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
        )

        # One plate over time; each obs[t] is an A-dim event.
        with pyro.plate("time", T, dim=-1):
            pyro.sample("obs", obs_dist, obs=batch.y)


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A, F = 3, 2, 4
    X = torch.zeros((T, A, F), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X=X, y=y)
    model = MarginalizedFactorJointStudentTModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l7_marginalized_factors_joint_student_t.png"

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
