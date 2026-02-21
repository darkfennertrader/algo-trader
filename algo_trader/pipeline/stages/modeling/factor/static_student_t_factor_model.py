"""
static_student_t_factor_model.py

Static Student-t factor model for *standardized* (z-scored) asset returns.

Model (per time t and asset i, in standardized space):

    z_t  (N-dimensional vector of standardized returns)
    f_t  (K-dimensional vector of latent factors)

    mu_i           ~ Normal(0, mu_prior_scale^2)
    L_{i,k}        ~ Normal(0, loading_prior_scale^2)
    factor_scale_k ~ HalfNormal(factor_scale_prior)
    idio_scale_i   ~ HalfNormal(idio_scale_prior)

    df_factor_raw  ~ Gamma(df_factor_alpha, df_factor_beta)
    df_idio_raw    ~ Gamma(df_idio_alpha, df_idio_beta)
    df_factor      = df_factor_raw + df_shift   # df_factor > df_shift
    df_idio        = df_idio_raw + df_shift     # df_idio  > df_shift

    For each time t:
        f_t        ~ StudentT(df_factor, 0, factor_scale)      # K-dimensional
        z_t       | f_t ~ StudentT(df_idio, mu + L f_t, idio_scale)

Notes:
    - Inputs `returns` are assumed *already standardized* per asset:
          z_{t,i} = (r_{t,i} - m_i) / s_i
      using some fixed means m_i and stds s_i (e.g. long-run estimates).
    - Hyperparameters are chosen to be reasonable on the z-score scale
      (typical variance per asset ~ 1).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling import ModelBatch, PyroModel


@dataclass(frozen=True)
class StaticStudentTFactorReturnsModel(PyroModel):
    """
    Static Student-t factor model for standardized (z-scored) asset returns.

    Usage:
        # K latent factors, e.g. 4
        model = StaticStudentTFactorReturnsModel(num_factors=4)

        # In your pipeline, make sure `returns` are z-scored per asset
        # using fixed means/stds (not recomputed on the same 60-day window).
        loss = svi.step(ModelBatch(X=None, y=standardized_returns, M=None))

    Hyperparameters (defaults are tuned for z-scored daily returns):

        - mu_prior_scale:
            Prior std for per-asset intercept mu_i in z-space.
            Small but not microscopic; centered strongly at 0.

        - loading_prior_scale:
            Prior std for factor loadings L_{i,k}. With K ~ 4, this gives
            order-1 factor contributions to variance without being too tight.

        - factor_scale_prior, idio_scale_prior:
            HalfNormal priors for factor and idiosyncratic volatilities
            in standardized units. Together with L, they imply that
            total prior variance per asset is O(1).

        - df_*:
            Gamma + shift priors for Student-t degrees of freedom, independent
            of scaling. Defaults give moderately heavy tails.
    """

    num_factors: int

    # Priors on intercepts and loadings (z-score scale)
    mu_prior_scale: float = 0.05  # prior std for mu_i (z-space)
    loading_prior_scale: float = 0.4  # prior std for L_{i,k}

    # Priors on factor and idiosyncratic scales (z-score scale)
    factor_scale_prior: float = 0.7  # HalfNormal scale for factor_scale_k
    idio_scale_prior: float = 0.7  # HalfNormal scale for idio_scale_i

    # Priors for degrees of freedom (moderately heavy tails)
    df_factor_alpha: float = 2.0
    df_factor_beta: float = 0.2  # prior mean ~ 10
    df_idio_alpha: float = 2.0
    df_idio_beta: float = 0.2  # prior mean ~ 10
    df_shift: float = 2.0  # ensures df > 2 (finite variance)

    def __call__(self, batch: ModelBatch):
        """
        Pyro model definition.

        Args:
            batch: data batch containing y and/or X. Returns are inferred from y
                and expected to be standardized (z-scored).
                - T: number of time steps in the current window (e.g., ~60 days)
                - N: number of assets

            The model defines:

                p(mu, L, factor_scale, idio_scale,
                  df_factor, df_idio, {f_t}, {z_t})

            and conditions on `returns` (z_t) via obs=returns.

        Shapes:
            - mu:          [N]
            - L:           [N, K]
            - factor_scale:[K]
            - idio_scale:  [N]
            - f_t:         [T, K]
            - mean_t:      [T, N]
            - returns:     [T, N]  (standardized)
        """
        returns, y_obs = _resolve_returns(batch)
        T, N = returns.shape  # T: time, N: assets
        K = self.num_factors
        device = returns.device

        # ------------------------------------------------------------------
        # 1. Priors on intercepts (mu) and factor loadings (L) in z-space
        # ------------------------------------------------------------------
        mu = pyro.sample(
            "mu",
            dist.Normal(
                loc=torch.zeros(N, device=device),
                scale=self.mu_prior_scale * torch.ones(N, device=device),
            ).to_event(1),
        )  # [N]

        L = pyro.sample(
            "L",
            dist.Normal(
                loc=torch.zeros(N, K, device=device),
                scale=self.loading_prior_scale
                * torch.ones(N, K, device=device),
            ).to_event(2),
        )  # [N, K]

        # ------------------------------------------------------------------
        # 2. Priors on factor and idiosyncratic scales (z-space)
        # ------------------------------------------------------------------
        factor_scale = pyro.sample(
            "factor_scale",
            dist.HalfNormal(
                self.factor_scale_prior * torch.ones(K, device=device)
            ).to_event(1),
        )  # [K]

        idio_scale = pyro.sample(
            "idio_scale",
            dist.HalfNormal(
                self.idio_scale_prior * torch.ones(N, device=device)
            ).to_event(1),
        )  # [N]

        # ------------------------------------------------------------------
        # 3. Priors on degrees of freedom (scale-free)
        # ------------------------------------------------------------------
        df_factor_raw = pyro.sample(
            "df_factor_raw",
            dist.Gamma(self.df_factor_alpha, self.df_factor_beta),
        )
        df_idio_raw = pyro.sample(
            "df_idio_raw",
            dist.Gamma(self.df_idio_alpha, self.df_idio_beta),
        )

        df_factor = df_factor_raw + self.df_shift
        df_idio = df_idio_raw + self.df_shift

        # ------------------------------------------------------------------
        # 4. Time plate and latent factors
        # ------------------------------------------------------------------
        with pyro.plate("time", T, dim=-2):

            # f_t: [T, K]
            f_t = pyro.sample(
                "f",
                dist.StudentT(
                    df=df_factor,
                    loc=torch.zeros(K, device=device),
                    scale=factor_scale,
                ).to_event(1),
            )

            # mean_t: [T, N] = mu + f_t @ L^T
            mean_t = mu + torch.matmul(f_t, L.transpose(0, 1))

            # ------------------------------------------------------------------
            # 5. Observation model in z-space
            # ------------------------------------------------------------------
            obs_dist = dist.StudentT(
                df=df_idio,
                loc=mean_t,  # [T, N]
                scale=idio_scale,  # [N], broadcast across T
            ).to_event(
                1
            )  # event dim is asset dimension
            if batch.M is None:
                pyro.sample("obs", obs_dist, obs=y_obs)
            else:
                with poutine.mask(  # pylint: disable=not-context-manager
                    mask=batch.M
                ):
                    pyro.sample("obs", obs_dist, obs=y_obs)


def _resolve_returns(
    batch: ModelBatch,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if batch.y is not None:
        if batch.y.ndim != 2:
            raise ConfigError("batch.y must have shape [T, A]")
        return batch.y, batch.y
    if batch.X is not None:
        if batch.X.ndim != 3:
            raise ConfigError("batch.X must have shape [T, A, F]")
        T, A = int(batch.X.shape[0]), int(batch.X.shape[1])
        zeros = torch.zeros((T, A), device=batch.X.device, dtype=batch.X.dtype)
        return zeros, None
    raise ConfigError("ModelBatch must provide X or y")


# SIDE NOTES:
# Things to watch out for (cons / caveats)
# How you estimate the z‑score (mean and std)

# Do not recompute the z‑score on the same short window (e.g. the 60‑day model window). That mixes the scaling with the latent volatility you are trying to model.
# Instead, use means and stds from a long, fixed history (e.g. all past data up to your backtest start, or an expanding window that changes slowly). Keep those (m_i, s_i) fixed when fitting the model and when doing out‑of‑sample predictions.
# You still need to map back to raw space

# For portfolio construction, you usually want covariances and expected returns in raw units:

# If you standardize as z_{t,i} = (r_{t,i} - m_i) / s_i, collect s_i into S = diag(s_1, ..., s_N).
# If Σ_z is the covariance in z‑space, then Σ_raw = S · Σ_z · S.
# If μ_z is the mean in z‑space, then μ_raw = m + S · μ_z.
# So keep track of m_i and s_i for each asset.

# Interaction with future conditional volatility modeling

# You mentioned you eventually want conditional volatility and discount factors.
# Z‑scoring with a fixed long‑run std is compatible with that (you still have conditional dynamics around that baseline scale).
# But if you were to z‑score using a rolling realized volatility, you’d effectively remove a big part of the very volatility you want to model.
# Robustness to outliers

# Daily returns are heavy‑tailed; a few big moves can distort sample std and thus your z‑score.
# If you see this in practice, consider robust scale estimates (e.g. shrink sample std or use some winsorized/robust estimator) for s_i.
# Bottom line
# For your SVI + GPU Bayesian factor model and cross‑asset universe, yes, I would standardize returns per asset, using a long‑run mean and volatility.
# The rewritten model above assumes exactly that and uses priors calibrated to z‑space.
# Just make sure you:
# compute (m_i, s_i) from a stable historical sample,
# keep them fixed when fitting/predicting, and
# correctly transform posterior means/covariances back to raw space before feeding them to HERC.
