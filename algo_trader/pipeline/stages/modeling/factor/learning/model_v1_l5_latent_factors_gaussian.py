"""Level 5 learning model: horseshoe mean + latent cross-asset factors.

What is new vs Level 4:
- Keep the same mean model as Level 4:
  mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]
- Replace independent residuals with shared latent factor shocks:
  residual[t,a] = sum_k B[a,k] * f[t,k] + eps[t,a]
- This introduces cross-asset dependence through B and f.

This level uses Gaussian residuals (no heavy-tail scale mixture yet).
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
class LatentFactorGaussianModel:
    """Cross-asset latent factor residual model with horseshoe mean."""

    # Mean model priors (same spirit as Level 4)
    alpha_scale: float = 0.02
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12

    # New residual-factor priors
    factor_count: int = 3
    b_scale: float = 0.20
    b_col_shrink_scale: float = 0.50
    sigma_idio_scale: float = 0.05

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
        device = batch.X.device
        dtype = batch.X.dtype

        # Horseshoe global terms for feature weights w[a,f].
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

        # Asset-level parameters: alpha, horseshoe weights w, idiosyncratic std.
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
                # Each factor-column k has its own shrinkage b_col[k].
                B = pyro.sample(
                    "B",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(self.b_scale, device=device, dtype=dtype) * b_col,
                    ),
                )  # [A,K]

        # Weekly latent factor shocks f[t,k].
        with pyro.plate("time_latent", T, dim=-2):
            with pyro.plate("factor_shock", K, dim=-1):
                f = pyro.sample(
                    "f",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(1.0, device=device, dtype=dtype),
                    ),
                )  # [T,K]

        # Mean from Level 4 structure.
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_idio_vec = sigma_idio.squeeze(-1)  # [A]
        mu_base = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T,A]

        # Cross-asset latent factor contribution: B[a,:] dot f[t,:].
        factor_term = torch.einsum("ak,tk->ta", B, f)  # [T,A]
        loc = mu_base + factor_term
        scale = sigma_idio_vec.unsqueeze(0).expand(T, A)  # [T,A]

        with pyro.plate("time", T, dim=-2):
            with pyro.plate("asset_obs", A, dim=-1):
                pyro.sample("obs", dist.Normal(loc, scale), obs=batch.y)


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A, F = 3, 2, 4
    X = torch.zeros((T, A, F), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X=X, y=y)
    model = LatentFactorGaussianModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l5_latent_factors_gaussian.png"

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
