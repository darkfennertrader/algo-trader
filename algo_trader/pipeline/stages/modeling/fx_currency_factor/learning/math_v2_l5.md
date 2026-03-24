# FX Currency Factor V2 L5

This note describes the `v2_l5` FX-only Bayesian model family implemented in:

- `model_v2_l5.py`
- `guide_v2_l5.py`
- `predict_v2_l5.py`

The goal of `v2_l5` is to keep the stronger `v2_l2` covariance block, but soften the `v2_l4` mean hierarchy.

So `v2_l5` is:

- `v2_l2` covariance
- `v2_l2` static hierarchical nugget
- a currency-centered prior for pair intercepts and pair feature weights
- no hard decomposition of pair means into `currency part + delta part`

## Universe and Geometry

The model remains FX-only.

Each observed asset is a pair return:

- `a = BASE.QUOTE`

We build a pair-to-currency exposure matrix `P[a, c]`:

- `+1` if currency `c` is the base of pair `a`
- `-1` if currency `c` is the quote of pair `a`
- one anchor currency is removed for identifiability, usually `USD`

Only relative currency moves are modeled.

## Mean

The predictive mean still uses pair-level coefficients:

```text
mu[t,a] =
  alpha[a]
  + sum_f X_asset[t,a,f] * w[a,f]
  + sum_g X_global[t,g] * (P[a] Gamma[:,g])
```

The change is in the prior for `alpha[a]` and `w[a,f]`.

### Currency-Centered Pair Intercepts

We introduce latent currency intercepts:

```text
alpha_currency[c] ~ Normal(0, alpha_currency_scale)
tau_alpha_pair ~ HalfNormal(tau_alpha_pair_scale)
```

Then each pair intercept is centered on the currency difference implied by `P`:

```text
alpha[a] ~ Normal(P[a] alpha_currency, tau_alpha_pair)
```

Interpretation:

- the model prefers triangle-consistent pair intercepts
- but pair intercepts are still free to move away from that center if the data supports it

This is softer than `v2_l4`, where the mean used `P[a] alpha_currency + delta_alpha[a]` directly.

### Currency-Centered Pair Feature Weights

We also add a currency hierarchy for pair feature weights.

First draw currency-level feature centers:

```text
theta0[f] ~ Normal(0, theta0_scale)
tau_theta[f] ~ HalfNormal(tau_theta_scale)
theta_currency[c,f] ~ Normal(theta0[f], tau_theta[f])
```

Then center pair feature weights on the currency difference:

```text
w[a,f] ~ Normal(P[a] theta_currency[:,f], rhs_scale[a,f])
```

where `rhs_scale[a,f]` is the same regularized horseshoe scale used in `v2_l2`.

Interpretation:

- feature effects are encouraged to be currency-native
- but the actual fitted pair weights remain pair-specific
- shrinkage still decides how far each pair-feature coefficient can move from the center

This is the key difference from `v2_l4`:

- `v2_l4` parameterized pair weights as `P theta_currency + delta_w`
- `v2_l5` keeps `w[a,f]` as the main predictive parameter and only changes its prior center

## Covariance

The covariance block is unchanged from `v2_l2`.

### Dynamic Common FX Covariance

Currency factor loadings:

```text
B_currency[c,k]
```

project to pair space:

```text
B_pair[a,k] = P[a] B_currency[:,k]
```

The shared FX regime is still a scalar AR(1):

```text
h[t] = phi * h[t-1] + s_u * eps[t]
eps[t] ~ Normal(0, 1)
```

with heavy tails:

```text
v[t] ~ Gamma(nu/2, nu/2)
u[t] = exp(h[t] - 0.5 Var[h]) * v[t]
```

The common covariance block is scaled by `u[t]`:

```text
z_common[t] ~ Normal(0, B_pair B_pair' / u[t])
```

### Static Hierarchical Nugget

The pair-specific residual nugget also stays as in `v2_l2`:

```text
log_sigma_idio[a] =
  sigma0
  + sum_c |P[a,c]| * sigma_currency[c]
  + delta_pair[a]
```

with:

```text
sigma0 ~ Normal(log(sigma_idio_scale), 0.5)
sigma_currency[c] ~ Normal(0, sigma_currency_scale)
tau_sigma_pair ~ HalfNormal(tau_sigma_pair_scale)
delta_pair[a] ~ Normal(0, tau_sigma_pair)
```

and the nugget is not scaled by the shared volatility shock.

## Full Observation Distribution

Conditioned on structural sites and latent regime:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor = B_pair / sqrt(u[t]),
  cov_diag = sigma_idio^2
)
```

So `v2_l5` only changes the prior geometry of the mean. The covariance geometry is intentionally unchanged.

## Guide

The guide remains an online-filtering variational family with the same gain-style local filtering path.

Structural variational sites now include:

- `alpha_currency`
- `theta0`
- `tau_theta`
- `theta_currency`
- `tau_alpha_pair`
- `alpha`
- `w`
- `sigma0`
- `sigma_currency`
- `tau_sigma_pair`
- `delta_pair`
- `gamma0`
- `tau_gamma`
- `Gamma`
- `b_col`
- `B_currency`
- `s_u`

The predictive summaries still expose pair-level `alpha` and `w`, so prediction stays aligned with the `v2_l2` runtime path.

## Predict

Prediction still uses:

1. stored structural posterior summaries
2. the carried online filtering state
3. AR(1) rollout of the shared FX regime
4. the same `LowRankMVN` covariance as `v2_l2`

The predictive mean is still:

```text
alpha[a] + X_asset[t,a] dot w[a] + X_global[t] dot (P[a] Gamma)
```

The difference is that `alpha[a]` and `w[a]` were learned under currency-centered priors.

## Intended Diagnostic Effect

Relative to `v2_l2`, the intended effect of `v2_l5` is:

- keep the better `v2_l2` covariance calibration
- improve triangle consistency in the mean without forcing a hard decomposition
- reduce the risk of the `v2_l4` regression, where the mean hierarchy was too restrictive

If `v2_l5` helps, it suggests the remaining FX problem is partly in mean geometry, but that the hierarchy needs to stay soft rather than deterministic.
