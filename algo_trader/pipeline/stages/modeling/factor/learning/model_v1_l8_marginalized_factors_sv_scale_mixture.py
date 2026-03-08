"""Level 8 learning model: collapsed factors + persistent volatility regime.

What is new vs Level 7:
- Keep the same collapsed cross-asset structure:
  y[t] | u[t] ~ LowRankMVN(mu[t], B, sigma_idio)
- Replace iid weekly scale u[t] with:
  u[t] = s_regime[t] * v[t]
  s_regime[t] = exp(h[t] - 0.5 * var_h)
  var_h = s_u^2 / (1 - phi^2)
  where:
    h[t] is a persistent AR(1) latent log-volatility regime
    v[t] is an iid Gamma heavy-tail shock

This adds volatility clustering while preserving joint heavy tails.
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
class MarginalizedFactorSVScaleMixtureModel:
    """Collapsed low-rank MVN with AR(1) latent volatility regime."""

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

    # Weekly heavy-tail shock (fixed df).
    nu: float = 10.0

    # Persistent latent regime settings.
    phi: float = 0.97  # fixed high persistence
    s_u_df: float = 4.0
    s_u_scale: float = 0.20

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
        if not 0.0 < self.phi < 1.0:
            raise ValueError("phi must be in (0, 1)")

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

        # Persistent regime h[t] (AR(1)):
        # h[1] ~ Normal(0, s0), s0 = s_u / sqrt(1 - phi^2 + eps)
        # h[t] = phi*h[t-1] + s_u*eps[t], eps[t]~N(0,1)
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

        # Scale anchoring:
        # var_h is stationary variance of AR(1) h[t].
        # Centering by -0.5*var_h keeps E[s_regime] ~ 1 under stationarity.
        var_h = s_u.pow(2) / denom
        s_regime = torch.exp(h - 0.5 * var_h)  # [T]
        u = s_regime * v  # [T]

        # Mean block (unchanged from Level 4 style).
        alpha_vec = alpha.squeeze(-1)  # [A]
        sigma_idio_vec = sigma_idio.squeeze(-1)  # [A]
        mu = alpha_vec.unsqueeze(0) + (batch.X * w.unsqueeze(0)).sum(dim=-1)  # [T,A]

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
    T, A, F = 3, 2, 4
    X = torch.zeros((T, A, F), dtype=torch.float32)
    y = torch.zeros((T, A), dtype=torch.float32)
    batch = Batch(X=X, y=y)
    model = MarginalizedFactorSVScaleMixtureModel()

    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1_l8_marginalized_factors_sv_scale_mixture.png"

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
