"""Level 9 learning model: Level 8 + global macro features with pooling.

What is new vs Level 8:
- Keep Level 8 collapsed covariance and persistent regime scale:
  y[t] | u[t] ~ LowRankMVN(mu[t], B, sigma_idio)
  u[t] = s_regime[t] * v[t]
- Split mean features into asset-local and global blocks:
  mu[t,a] = alpha[a]
            + sum_f X_asset[t,a,f] * w[a,f]
            + sum_g X_global[t,g] * beta[a,g]
- Add hierarchical pooling for global loadings:
  beta[a,g] ~ Normal(beta0[g], tau_beta[g])
  beta0[g]  ~ Normal(0, s_beta0)
  tau_beta[g] ~ HalfNormal(s_tau)

X_global is expected in fixed column order from the pipeline.
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

    X_asset: torch.Tensor  # [T, A, F]
    X_global: torch.Tensor  # [T, G]
    y: torch.Tensor  # [T, A]


def _half_student_t(
    *, df: float, scale: float, device: torch.device, dtype: torch.dtype
) -> dist.FoldedDistribution:
    """Build a half-StudentT via FoldedDistribution(StudentT)."""
    base = dist.StudentT(
        torch.tensor(df, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(scale, device=device, dtype=dtype),
    )
    return dist.FoldedDistribution(base)


@dataclass(frozen=True)
class MarginalizedFactorSVScaleMixtureGlobalModel:
    """Collapsed low-rank MVN with AR(1) regime + pooled global loadings."""

    # Mean model priors for asset-local features (same spirit as Level 8)
    alpha_scale: float = 0.02
    tau0_scale: float = 0.10
    lambda_scale: float = 1.0
    kappa_scale: float = 1.0
    c_scale: float = 0.5
    eps: float = 1e-12

    # Hierarchical priors for global-feature loadings beta[a,g].
    beta0_scale: float = 0.05
    tau_beta_scale: float = 0.05

    # Residual factor priors
    factor_count: int = 3
    b_scale: float = 0.20
    b_col_shrink_scale: float = 0.50
    sigma_idio_scale: float = 0.05

    # Weekly heavy-tail shock (fixed df).
    nu: float = 10.0

    # Persistent latent regime settings.
    phi: float = 0.97  # fixed high persistence
    s_u_df: float = 4.0
    s_u_scale: float = 0.20

    def __call__(self, batch: Batch) -> None:
        # Expected inputs:
        # X_asset[t,a,f], X_global[t,g], y[t,a], aligned on time T.
        if batch.X_asset.ndim != 3:
            raise ValueError("Expected X_asset with shape [T, A, F]")
        if batch.X_global.ndim != 2:
            raise ValueError("Expected X_global with shape [T, G]")
        if batch.y.ndim != 2:
            raise ValueError("Expected y with shape [T, A]")

        T, A, F = batch.X_asset.shape
        T_global, G = batch.X_global.shape
        if batch.y.shape != (T, A):
            raise ValueError("X_asset and y must agree on [T, A]")
        if T_global != T:
            raise ValueError("X_global and X_asset must agree on time dimension T")

        K = int(self.factor_count)
        if K <= 0:
            raise ValueError("factor_count must be positive")
        if G <= 0:
            raise ValueError("X_global must have at least one feature")
        if self.nu <= 2.0:
            raise ValueError("nu should be > 2 for finite variance")
        if not 0.0 < self.phi < 1.0:
            raise ValueError("phi must be in (0, 1)")

        device = batch.X_asset.device
        dtype = batch.X_asset.dtype

        # Horseshoe globals for asset-local feature weights.
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

        # Hierarchical global-feature loading priors.
        with pyro.plate("global_feature", G, dim=-1):
            beta0 = pyro.sample(
                "beta0",
                dist.Normal(
                    torch.zeros((G,), device=device, dtype=dtype),
                    torch.full((G,), self.beta0_scale, device=device, dtype=dtype),
                ),
            )  # [G]
            tau_beta = pyro.sample(
                "tau_beta",
                dist.HalfNormal(
                    torch.full((G,), self.tau_beta_scale, device=device, dtype=dtype)
                ),
            )  # [G]

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

        # Asset-level params: alpha, w, beta, sigma_idio, B.
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

            with pyro.plate("global_loading", G, dim=-1):
                beta = pyro.sample(
                    "beta",
                    dist.Normal(beta0, tau_beta),
                )  # [A,G]

            with pyro.plate("factor_loading_k", K, dim=-1):
                B = pyro.sample(
                    "B",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(self.b_scale, device=device, dtype=dtype) * b_col,
                    ),
                )  # [A,K]

        # Learn regime innovation scale s_u with robust half-StudentT prior.
        s_u = pyro.sample(
            "s_u",
            _half_student_t(
                df=self.s_u_df,
                scale=self.s_u_scale,
                device=device,
                dtype=dtype,
            ),
        )

        # Persistent regime h[t] (AR(1)).
        phi_t = torch.tensor(self.phi, device=device, dtype=dtype)
        denom = (
            torch.tensor(1.0, device=device, dtype=dtype)
            - phi_t.pow(2)
            + self.eps
        )
        h0_scale = s_u / torch.sqrt(denom)
        h1 = pyro.sample(
            "h_1",
            dist.Normal(
                torch.tensor(0.0, device=device, dtype=dtype),
                h0_scale,
            ),
        )
        if T > 1:
            with pyro.plate("time_h_eps", T - 1, dim=-1):
                h_eps = pyro.sample(
                    "h_eps",
                    dist.Normal(
                        torch.tensor(0.0, device=device, dtype=dtype),
                        torch.tensor(1.0, device=device, dtype=dtype),
                    ),
                )  # [T-1]
            h_values = [h1]
            for t in range(T - 1):
                h_next = phi_t * h_values[-1] + s_u * h_eps[t]
                h_values.append(h_next)
            h = torch.stack(h_values, dim=0)  # [T]
        else:
            h = h1.unsqueeze(0)  # [1]

        # One-week iid heavy-tail shock.
        nu_half = torch.tensor(self.nu / 2.0, device=device, dtype=dtype)
        with pyro.plate("time_v", T, dim=-1):
            v = pyro.sample("v", dist.Gamma(nu_half, nu_half))  # [T]

        # Scale anchoring and total weekly scale.
        var_h = s_u.pow(2) / denom
        s_regime = torch.exp(h - 0.5 * var_h)  # [T]
        u = s_regime * v  # [T]

        # Mean decomposition:
        # mu[t,a] = alpha[a] + X_asset[t,a,:]·w[a,:] + X_global[t,:]·beta[a,:]
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_idio_vec = sigma_idio.squeeze(-1)  # [A]
        mu_asset = (batch.X_asset * w.unsqueeze(0)).sum(dim=-1)  # [T,A]
        mu_global = batch.X_global @ beta.transpose(0, 1)  # [T,A]
        mu = alpha_vec.unsqueeze(0) + mu_asset + mu_global  # [T,A]

        # Collapsed low-rank covariance each week t:
        # cov = (B B' + diag(sigma_idio^2)) / u[t].
        inv_sqrt_u = torch.rsqrt(u).unsqueeze(-1)  # [T,1]
        cov_factor = inv_sqrt_u.unsqueeze(-1) * B.unsqueeze(0)  # [T,A,K]
        cov_diag = sigma_idio_vec.pow(2).unsqueeze(0) / u.unsqueeze(-1)  # [T,A]

        obs_dist = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=cov_factor,
            cov_diag=cov_diag,
        )

        # One time plate only; y[t] is an A-dimensional event.
        with pyro.plate("time", T, dim=-1):
            pyro.sample("obs", obs_dist, obs=batch.y)


def _render_model_graph() -> Path:
    """Render the plate graph for learning/debugging."""
    T, A, F, G = 3, 2, 4, 5
    X_asset = torch.zeros((T, A, F), dtype=torch.float32)
    X_global = torch.zeros((T, G), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X_asset=X_asset, X_global=X_global, y=y)
    model = MarginalizedFactorSVScaleMixtureGlobalModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / "model_v1_l9_marginalized_factors_sv_scale_mixture_global.png"
    )

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
