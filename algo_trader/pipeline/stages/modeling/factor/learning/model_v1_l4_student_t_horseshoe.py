"""Level 4 learning model: full structure close to FactorModelV1.

What is new vs Level 3:
- Replace fixed Normal weight scale with hierarchical horseshoe shrinkage.
- Add feature-level and local scales to promote sparsity in w[a, f].
- Keep Student-t likelihood for heavy tails.

Math:
    nu_raw     ~ Gamma(shape, rate)
    nu         = nu_raw + shift

    tau0       ~ HalfCauchy(tau0_scale)            # global shrinkage
    lambda_f   ~ HalfCauchy(lambda_scale)          # per-feature shrinkage
    kappa_a,f  ~ HalfCauchy(kappa_scale)           # local shrinkage
    c          ~ HalfCauchy(c_scale)               # slab scale (regularized HS)

    base_a,f   = lambda_f * kappa_a,f
    lam_tilde^2= (c^2 * base^2) / (c^2 + tau0^2 * base^2 + eps)
    w_scale_a,f= tau0 * sqrt(lam_tilde^2)

    alpha_a    ~ Normal(0, alpha_scale)
    sigma_a    ~ HalfCauchy(sigma_scale)
    w_a,f      ~ Normal(0, w_scale_a,f)
    mu_t,a     = alpha_a + sum_f X[t,a,f] * w_a,f
    y_t,a      ~ StudentT(df=nu, loc=mu_t,a, scale=sigma_a)
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
class StudentTHorseshoeModel:
    """Hierarchical sparse linear model with Student-t residuals."""

    alpha_scale: float = 0.02
    sigma_scale: float = 0.05
    nu_shape: float = 2.0
    nu_rate: float = 0.2
    nu_shift: float = 2.0
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12

    def __call__(self, batch: Batch) -> None:
        # Expected inputs:
        # X[t, a, f] and y[t, a] with matching first two dims [T, A].
        if batch.X.ndim != 3:
            raise ValueError("Expected X with shape [T, A, F]")
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")
        if batch.X.shape[:2] != batch.y.shape:
            raise ValueError("X and y must agree on [T, A]")

        T, A, F = batch.X.shape
        device = batch.X.device
        dtype = batch.X.dtype

        # Student-t degrees of freedom:
        # nu > 2 keeps variance finite and usually stabilizes training.
        nu_raw = pyro.sample(
            "nu_raw",
            dist.Gamma(
                torch.tensor(self.nu_shape, device=device, dtype=dtype),
                torch.tensor(self.nu_rate, device=device, dtype=dtype),
            ),
        )
        nu = nu_raw + torch.tensor(self.nu_shift, device=device, dtype=dtype)

        # Horseshoe global terms:
        # tau0 = global shrinkage, lambda[f] = feature-level shrinkage.
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
                    torch.full(
                        (F,),
                        self.lambda_scale,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )  # [F]
        # c controls slab size in the regularized horseshoe.
        c = pyro.sample(
            "c",
            dist.HalfCauchy(torch.tensor(self.c_scale, device=device, dtype=dtype)),
        )

        # Per asset parameters and local shrinkage.
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
                kappa = pyro.sample(
                    "kappa",
                    dist.HalfCauchy(
                        torch.tensor(self.kappa_scale, device=device, dtype=dtype)
                    ),
                )  # [A, F]

                # Regularized horseshoe scale construction:
                # base[a,f] combines feature and local shrinkage.
                base = lam * kappa  # [A, F], lam broadcasts across assets
                base2 = base.pow(2)
                c2 = c * c
                tau2 = tau0 * tau0
                lam_tilde = torch.sqrt((c2 * base2) / (c2 + tau2 * base2 + self.eps))
                # Final std used to sample weight w[a,f].
                w_scale = tau0 * lam_tilde

                w = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        w_scale,
                    ),
                )  # [A, F]

        # Linear predictor:
        # mu[t,a] = alpha[a] + dot(X[t,a,:], w[a,:]).
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_vec = sigma.squeeze(-1)  # [A]
        mu = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T, A]
        # sigma is per asset, so broadcast over time.
        scale = sigma_vec.unsqueeze(0).expand(T, A)  # [T, A]

        # Student-t observation over all (time, asset) cells.
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
    model = StudentTHorseshoeModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l4_student_t_horseshoe.png"

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
