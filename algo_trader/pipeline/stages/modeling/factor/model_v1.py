"""
FactorModelV1 (Pyro) implements a Bayesian heavy-tailed regression with
hierarchical horseshoe shrinkage for feature selection. It is a linear model
in the features, with per-asset intercepts and residual scales, where each
asset gets its own coefficient vector but coefficients are strongly shrunk
toward zero unless supported by the data.

Basis and hypotheses:
    - Linear predictability: returns can be explained by a linear combination
      of features plus an asset-specific intercept.
    - Sparsity: only a small subset of features matter for each asset, so we
      place a horseshoe prior to shrink irrelevant coefficients toward zero
      while allowing a few large signals to remain.
    - Heavy tails: realized returns have outliers, so we model residuals with
      a Student-t likelihood rather than Gaussian noise.
    - Conditional independence: given (alpha, w, sigma, nu), observations are
      independent across time and assets.
    - Coupling: assets are coupled a priori / in the posterior through shared
      shrinkage hyperparameters (tau0, lambda) and shared nu (and c if used).

Data:
    X[t,a,f]  features for time t, asset a, feature f
    y[t,a]    realized return (1-step-ahead, typically weekly)
    Assumes data are pre-aligned so that X[t] is information available at
    decision time t and y[t] is the realized return over week t+1.

Model:
    nu_raw        ~ Gamma(shape, rate)
    nu            = nu_raw + shift

    tau0          ~ HalfCauchy(tau0_scale)                       (global shrinkage)
    lambda_f      ~ HalfCauchy(lambda_scale)                     (feature shrinkage)
    kappa_a,f     ~ HalfCauchy(kappa_scale)                      (local shrinkage)
    c             ~ HalfCauchy(c_scale)                          (slab scale, optional)

    alpha_a       ~ Normal(0, alpha_scale)                       (asset intercept)
    sigma_a       ~ HalfCauchy(sigma_scale)                      (asset residual scale)

    base_a,f      = lambda_f * kappa_a,f
    if regularized:
        lambda_tilde_a,f^2 = (c^2 * base_a,f^2) / (c^2 + tau0^2 * base_a,f^2)
        w_scale_a,f        = tau0 * lambda_tilde_a,f
    else:
        w_scale_a,f        = tau0 * base_a,f

    w_a,f         ~ Normal(0, w_scale_a,f)                       (feature weights)

    mu_t,a        = alpha_a + sum_f X[t,a,f] * w_a,f
    y[t,a]        ~ StudentT(df=nu, loc=mu_t,a, scale=sigma_a)

Masking:
    If batch.M is provided, it masks missing y entries in the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pyro
import pyro.distributions as dist
import torch
from algo_trader.domain import ConfigError
from ..batch_utils import resolve_batch_shape, sample_observation
from ..protocols import ModelBatch, PyroModel
from ..registry_core import register_model


@dataclass(frozen=True)
class LikelihoodPriors:
    # Priors for per-asset intercepts and residual scale.
    alpha_scale: float = 0.02
    sigma_scale: float = 0.05


@dataclass(frozen=True)
class DofPriors:
    # Student-t degrees of freedom: nu = Gamma(shape, rate) + shift.
    shape: float = 2.0
    rate: float = 0.2
    shift: float = 2.0


@dataclass(frozen=True)
class HorseshoePriors:
    # Hierarchical shrinkage priors for feature weights.
    tau0_scale: float = 0.10  # global shrinkage scale (weekly)
    lambda_scale: float = 1.0  # feature shrinkage scale (usually 1)
    kappa_scale: float = 1.0  # local shrinkage scale (usually 1)
    use_regularized: bool = True
    c_scale: float = 0.5  # slab scale (weekly); tune
    eps: float = 1e-12


@dataclass(frozen=True)
class FactorModelPriors:
    # Grouped priors so FactorModelV1 remains minimal.
    likelihood: LikelihoodPriors = field(default_factory=LikelihoodPriors)
    dof: DofPriors = field(default_factory=DofPriors)
    horseshoe: HorseshoePriors = field(default_factory=HorseshoePriors)


@dataclass(frozen=True)
class _ModelContext:
    # Resolved batch metadata used across helper functions.
    T: int
    A: int
    F: int
    device: torch.device
    dtype: torch.dtype
    X: torch.Tensor
    y_obs: torch.Tensor | None


@dataclass(frozen=True)
class _HorseshoeScales:
    # Sampled shrinkage scales shared across asset-level weights.
    tau0: torch.Tensor
    lam: torch.Tensor
    c: torch.Tensor | None


@dataclass(frozen=True)
class FactorModelV1(PyroModel):
    # Hierarchical regression with Student-t likelihood and horseshoe shrinkage.
    priors: FactorModelPriors = field(default_factory=FactorModelPriors)

    def __call__(self, batch: ModelBatch) -> None:
        """Define the Pyro model over a batch.

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
        - Keep an observation site named "obs" so simulation prediction works.
        - If batch.M is provided, mask missing targets in the likelihood.

        Shape helper:
        - resolve_batch_shape returns:
          - T: int, number of time steps
          - A: int, number of assets
          - device: torch.device for tensors
          - dtype: torch.dtype for tensors
          - y_obs: torch.Tensor | None, observed returns [T, A]
        """
        context = _build_context(batch)
        # 1) Tail thickness for Student-t likelihood.
        nu = _sample_nu(self.priors.dof, context)
        # 2) Global and feature-level shrinkage scales.
        scales = _sample_horseshoe(self.priors.horseshoe, context)
        # 3) Asset-level intercepts, residual scales, and feature weights.
        alpha, sigma, w = _sample_asset_params(
            self.priors.likelihood,
            self.priors.horseshoe,
            context,
            scales,
        )
        # 4) Build likelihood and condition on observed targets.
        obs_dist = _build_likelihood(context, alpha, sigma, w, nu)
        with pyro.plate("time", context.T, dim=-2):
            with pyro.plate("asset_obs", context.A, dim=-1):
                sample_observation(
                    obs_dist=obs_dist,
                    y_obs=context.y_obs,
                    mask=batch.M,
                    scale=batch.obs_scale,
                )


def _build_factor_priors(params: Mapping[str, Any]) -> FactorModelPriors:
    if not params:
        return FactorModelPriors()
    extra = set(params) - {"likelihood", "dof", "horseshoe"}
    if extra:
        raise ConfigError(
            "Unknown factor_model_v1 params",
            context={"params": ", ".join(sorted(extra))},
        )
    return FactorModelPriors(
        likelihood=_build_likelihood_priors(params.get("likelihood")),
        dof=_build_dof_priors(params.get("dof")),
        horseshoe=_build_horseshoe_priors(params.get("horseshoe")),
    )


def _build_likelihood_priors(
    raw: object,
) -> LikelihoodPriors:
    values = _coerce_mapping(raw, label="model.params.likelihood")
    if not values:
        return LikelihoodPriors()
    extra = set(values) - {"alpha_scale", "sigma_scale"}
    if extra:
        raise ConfigError(
            "Unknown factor_model_v1 likelihood params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = LikelihoodPriors()
    try:
        return replace(
            base,
            alpha_scale=float(values.get("alpha_scale", base.alpha_scale)),
            sigma_scale=float(values.get("sigma_scale", base.sigma_scale)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid factor_model_v1 likelihood params",
            context={"params": str(dict(values))},
        ) from exc


def _build_dof_priors(raw: object) -> DofPriors:
    values = _coerce_mapping(raw, label="model.params.dof")
    if not values:
        return DofPriors()
    extra = set(values) - {"shape", "rate", "shift"}
    if extra:
        raise ConfigError(
            "Unknown factor_model_v1 dof params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = DofPriors()
    try:
        return replace(
            base,
            shape=float(values.get("shape", base.shape)),
            rate=float(values.get("rate", base.rate)),
            shift=float(values.get("shift", base.shift)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid factor_model_v1 dof params",
            context={"params": str(dict(values))},
        ) from exc


def _build_horseshoe_priors(raw: object) -> HorseshoePriors:
    values = _coerce_mapping(raw, label="model.params.horseshoe")
    if not values:
        return HorseshoePriors()
    extra = set(values) - {
        "tau0_scale",
        "lambda_scale",
        "kappa_scale",
        "use_regularized",
        "c_scale",
        "eps",
    }
    if extra:
        raise ConfigError(
            "Unknown factor_model_v1 horseshoe params",
            context={"params": ", ".join(sorted(extra))},
        )
    base = HorseshoePriors()
    use_regularized = values.get("use_regularized", base.use_regularized)
    if not isinstance(use_regularized, bool):
        raise ConfigError(
            "factor_model_v1 horseshoe.use_regularized must be a boolean",
            context={"value": str(use_regularized)},
        )
    try:
        return replace(
            base,
            tau0_scale=float(values.get("tau0_scale", base.tau0_scale)),
            lambda_scale=float(values.get("lambda_scale", base.lambda_scale)),
            kappa_scale=float(values.get("kappa_scale", base.kappa_scale)),
            use_regularized=use_regularized,
            c_scale=float(values.get("c_scale", base.c_scale)),
            eps=float(values.get("eps", base.eps)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Invalid factor_model_v1 horseshoe params",
            context={"params": str(dict(values))},
        ) from exc


def _coerce_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return dict(value)


def _build_context(batch: ModelBatch) -> _ModelContext:
    # Resolve shape metadata and coerce features onto the batch dtype/device.
    shape = resolve_batch_shape(batch)
    if batch.X is None:
        raise ConfigError("FactorModelV1 requires batch.X with shape [T, A, F]")
    X = batch.X.to(device=shape.device, dtype=shape.dtype)
    return _ModelContext(
        T=shape.T,
        A=shape.A,
        F=int(X.shape[-1]),
        device=shape.device,
        dtype=shape.dtype,
        X=X,
        y_obs=shape.y_obs,
    )


def _sample_nu(priors: DofPriors, context: _ModelContext) -> torch.Tensor:
    # Sample nu from a Gamma prior and shift to keep nu > 2.
    nu_raw = pyro.sample(
        "nu_raw",
        dist.Gamma(
            torch.tensor(
                priors.shape, device=context.device, dtype=context.dtype
            ),
            torch.tensor(
                priors.rate, device=context.device, dtype=context.dtype
            ),
        ),
    )
    return nu_raw + torch.tensor(
        priors.shift, device=context.device, dtype=context.dtype
    )


def _sample_horseshoe(
    priors: HorseshoePriors, context: _ModelContext
) -> _HorseshoeScales:
    # Global shrinkage for all weights.
    tau0 = pyro.sample(
        "tau0",
        dist.HalfCauchy(
            torch.tensor(
                priors.tau0_scale, device=context.device, dtype=context.dtype
            )
        ),
    )
    # Feature-level shrinkage (shared across assets).
    with pyro.plate("feature", context.F, dim=-1):
        lam = pyro.sample(
            "lambda",
            dist.HalfCauchy(
                torch.full(
                    (context.F,),
                    priors.lambda_scale,
                    device=context.device,
                    dtype=context.dtype,
                )
            ),
        )
    c = None
    if priors.use_regularized:
        # Slab scale for the regularized horseshoe.
        c = pyro.sample(
            "c",
            dist.HalfCauchy(
                torch.tensor(
                    priors.c_scale, device=context.device, dtype=context.dtype
                )
            ),
        )
    return _HorseshoeScales(tau0=tau0, lam=lam, c=c)


def _sample_asset_params(
    like: LikelihoodPriors,
    priors: HorseshoePriors,
    context: _ModelContext,
    scales: _HorseshoeScales,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Asset-level parameters and weights.
    with pyro.plate("asset", context.A, dim=-2):
        alpha = pyro.sample(
            "alpha",
            dist.Normal(
                torch.tensor(0.0, device=context.device, dtype=context.dtype),
                torch.tensor(
                    like.alpha_scale, device=context.device, dtype=context.dtype
                ),
            ),
        )
        sigma = pyro.sample(
            "sigma",
            dist.HalfCauchy(
                torch.tensor(
                    like.sigma_scale, device=context.device, dtype=context.dtype
                )
            ),
        )
        with pyro.plate("feature_w", context.F, dim=-1):
            kappa = pyro.sample(
                "kappa",
                dist.HalfCauchy(
                    torch.tensor(
                        priors.kappa_scale,
                        device=context.device,
                        dtype=context.dtype,
                    )
                ),
            )
            # Combine global, feature, and local scales.
            w_scale = _weight_scale(priors, scales, kappa)
            w = pyro.sample(
                "w",
                dist.Normal(
                    torch.tensor(
                        0.0, device=context.device, dtype=context.dtype
                    ),
                    w_scale,
                ),
            )
    return alpha, sigma, w


def _weight_scale(
    priors: HorseshoePriors,
    scales: _HorseshoeScales,
    kappa: torch.Tensor,
) -> torch.Tensor:
    # Base local scale per asset-feature pair (lam broadcast over assets).
    base = scales.lam * kappa
    if not priors.use_regularized:
        return scales.tau0 * base
    if scales.c is None:
        raise ConfigError("Regularized horseshoe requires slab scale c")
    # Regularized horseshoe shrinkage.
    base2 = base.pow(2)
    c2 = scales.c * scales.c
    tau2 = scales.tau0 * scales.tau0
    lam_tilde = torch.sqrt((c2 * base2) / (c2 + tau2 * base2 + priors.eps))
    return scales.tau0 * lam_tilde


def _build_likelihood(
    context: _ModelContext,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    w: torch.Tensor,
    nu: torch.Tensor,
) -> dist.StudentT:
    # Pyro plates may introduce singleton dims; normalize to [A].
    alpha_vec = alpha.squeeze(-1)
    sigma_vec = sigma.squeeze(-1)
    # Linear predictor: alpha[a] + sum_f X[t,a,f] * w[a,f].
    mu = alpha_vec.unsqueeze(0) + (context.X * w.unsqueeze(0)).sum(dim=-1)
    # Broadcast residual scale across time.
    scale = sigma_vec.unsqueeze(0).expand(context.T, context.A)
    return dist.StudentT(df=nu, loc=mu, scale=scale)


def _render_model_graph() -> Path:
    T, A, F = 2, 3, 4
    device = torch.device("cpu")
    dtype = torch.float32
    X = torch.zeros((T, A, F), device=device, dtype=dtype)
    y = torch.zeros((T, A), device=device, dtype=dtype)
    batch = ModelBatch(X=X, y=y, M=None)
    model = FactorModelV1()
    output_dir = Path(__file__).with_name("render")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v1.png"
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


def render_model_graph() -> Path:
    """Public wrapper for rendering the model graph."""
    return _render_model_graph()


@register_model("factor_model_v1")
def build_factor_model_v1(params: Mapping[str, Any]) -> PyroModel:
    priors = _build_factor_priors(params)
    return FactorModelV1(priors=priors)
