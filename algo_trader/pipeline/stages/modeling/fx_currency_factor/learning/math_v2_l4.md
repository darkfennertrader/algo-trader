# FX Currency Factor V2 L4

This note describes the `v2_l4` FX-only Bayesian model family implemented in:

- `model_v2_l4.py`
- `guide_v2_l4.py`
- `predict_v2_l4.py`

`v2_l4` is built from the `v2_l2` covariance base.

The empirical result from `v2_l3` was that simplifying the common covariance to diagonal currency shocks was too aggressive. So `v2_l4` keeps the stronger `v2_l2` covariance structure and instead changes the mean:

- keep the `v2_l2` low-rank currency covariance block
- keep the `v2_l2` static hierarchical nugget
- replace the free pair-native mean with a currency-centered hierarchy plus pair deltas

## Universe and Geometry

The model remains FX-only.

Each observed asset is a pair return:

- `a = BASE.QUOTE`

We build a pair-to-currency exposure matrix `P[a, c]`:

- `+1` if currency `c` is the base of pair `a`
- `-1` if currency `c` is the quote of pair `a`
- one anchor currency is removed for identifiability, usually `USD`

Only relative currency moves are modeled.

## Currency-Centered Mean

This is the main `v2_l4` change.

Instead of free pair intercepts `alpha[a]` and free pair feature weights `w[a,f]`, the mean is centered on currency-level components.

### Intercepts

We introduce latent currency intercepts:

```text
alpha_currency[c] ~ Normal(0, alpha_currency_scale)
```

and pair residual intercepts:

```text
tau_alpha_pair ~ HalfNormal(tau_alpha_pair_scale)
delta_alpha[a] ~ Normal(0, tau_alpha_pair)
```

The pair intercept is then:

```text
alpha_pair[a] = P[a] alpha_currency + delta_alpha[a]
```

So pair intercepts are mostly explained by base-minus-quote currency structure, with a shrunk residual correction.

### Feature Weights

We introduce currency-centered feature effects:

```text
theta0[f] ~ Normal(0, theta0_scale)
tau_theta[f] ~ HalfNormal(tau_theta_scale)
theta_currency[c,f] ~ Normal(theta0[f], tau_theta[f])
```

and pair-specific feature deltas:

```text
delta_w[a,f] ~ shrinkage prior centered at 0
```

The pair feature loading becomes:

```text
w_pair[a,f] = P[a] theta_currency[:,f] + delta_w[a,f]
```

So the mean for pair `a` at time `t` is:

```text
mu[t,a] =
  alpha_pair[a]
  + sum_f X_asset[t,a,f] * w_pair[a,f]
  + sum_g X_global[t,g] * (P[a] Gamma[:,g])
```

Expanded:

```text
mu[t,a] =
  P[a] alpha_currency
  + delta_alpha[a]
  + sum_f X_asset[t,a,f] *
      (P[a] theta_currency[:,f] + delta_w[a,f])
  + sum_g X_global[t,g] * (P[a] Gamma[:,g])
```

Interpretation:

- triangle-consistent mean structure is now the default
- pair-specific flexibility still exists
- but that flexibility is pushed into shrunk residual corrections rather than unconstrained pair means

## Dynamic Common Covariance

The covariance block is unchanged from `v2_l2`.

We keep low-rank currency loadings:

```text
B_currency[c,k]
```

and project them to pair space:

```text
B_pair[a,k] = P[a] B_currency[:,k]
```

The shared FX regime remains a scalar AR(1):

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

## Hierarchical Static Idiosyncratic Nugget

The nugget is also unchanged from `v2_l2`.

Residual pair scales use:

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

This nugget remains static:

```text
eps_idio[t,a] ~ Normal(0, sigma_idio[a]^2)
```

It is not scaled by the shared regime.

## Full Observation Distribution

Conditioned on structural sites and latent regime:

```text
y[t] = mu[t] + z_common[t] + eps_idio[t]
```

implemented as:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor = B_pair / sqrt(u[t]),
  cov_diag = sigma_idio^2
)
```

Compared with `v2_l2`:

- covariance is unchanged
- the nugget is unchanged
- only the mean hierarchy changes

## Guide

The guide remains an online-filtering variational family.

Structural sites now include:

- `alpha_currency`
- `tau_alpha_pair`
- `delta_alpha`
- `theta0`
- `tau_theta`
- `theta_currency`
- `delta_w`
- `gamma_currency`
- `B_currency`
- sigma hierarchy sites
- `s_u`

The latent regime path still uses the same gain-style encoder inherited from the earlier online-filtering family.

## Predict

Prediction uses:

1. stored structural posterior summaries
2. the carried filtering state from online training
3. AR(1) rollout of the shared FX regime
4. simulation from the same `v2_l2` covariance block

At prediction time, pair means are reconstructed from:

- `alpha_currency`
- `delta_alpha`
- `theta_currency`
- `delta_w`
- `gamma_currency`

using the same exposure matrix `P`.

## Intended Diagnostic Effect

The expected effect of `v2_l4` relative to `v2_l2` is:

- better triangle-consistent mean behavior across related FX pairs
- less need for arbitrary pair-specific intercepts and feature coefficients
- tighter central calibration on remaining quiet crosses such as `EUR.GBP`, `EUR.CAD`, `AUD.CAD`, and `NZD.CAD`

If `v2_l4` still remains too wide, the next likely move is not another covariance simplification but a more structured volatility split or a tighter hierarchy on the pair residual feature deltas.
