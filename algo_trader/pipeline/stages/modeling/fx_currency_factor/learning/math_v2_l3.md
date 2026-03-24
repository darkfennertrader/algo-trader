# FX Currency Factor V2 L3

This note describes the `v2_l3` FX-only Bayesian model family implemented in:

- `model_v2_l3.py`
- `guide_v2_l3.py`
- `predict_v2_l3.py`

`v2_l3` is the next structural step after `v2_l2`.

The `v2_l2` result suggested that the static hierarchical nugget was a good change, but the dynamic common block was still too flexible. `v2_l3` therefore simplifies the common FX covariance:

- keep the `v2_l2` static hierarchical idiosyncratic nugget
- keep the scalar online-filtered FX regime and heavy tails
- replace the low-rank currency loading block with diagonal currency shocks

This is a covariance-only step. The mean remains pair-native in `v2_l3a`.

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

The mean is unchanged from `v2_l2`.

For pair `a` at time `t`:

```text
mu[t,a] =
  alpha[a]
  + sum_f X_asset[t,a,f] * w[a,f]
  + sum_g X_global[t,g] * (P[a] Gamma[:,g])
```

where:

- `alpha[a]` is a pair intercept
- `w[a,f]` are sparse pair feature weights
- `Gamma[c,g]` are currency-level macro loadings

So macro effects are still currency-native even though the local pair feature block is not yet hierarchical.

## Dynamic Common Covariance

This is the main `v2_l3` change.

Instead of a low-rank currency loading matrix

```text
B_currency[c,k]
```

we use one positive shock scale per non-anchor currency:

```text
omega_currency[c] > 0
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

At time `t`, latent currency shocks are independent conditional on `u[t]`:

```text
z_currency[t,c] ~ Normal(0, omega_currency[c]^2 / u[t])
```

The pair-space common covariance is induced by the exposure matrix:

```text
z_common[t] = P z_currency[t]
Cov[z_common[t]] = P diag(omega_currency^2 / u[t]) P'
```

This keeps triangle-consistent FX dependence, but removes the extra flexibility of an arbitrary low-rank currency factor geometry.

## Hierarchical Static Idiosyncratic Nugget

The idiosyncratic block is unchanged from `v2_l2`.

Residual pair scales use a hierarchical log-scale model:

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

Interpretation:

- `sigma0` is the global FX residual level
- `sigma_currency[c]` adjusts residual scale by currency participation
- `delta_pair[a]` captures pair-specific deviations, with shrinkage

The nugget remains static:

```text
eps_idio[t,a] ~ Normal(0, sigma_idio[a]^2)
```

It is not scaled by the shared regime.

## Full Observation Distribution

Conditioned on structural sites and latent regime:

```text
y[t] = mu[t] + z_common[t] + eps_idio[t]
```

which is implemented as:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor[a,c] = P[a,c] * omega_currency[c] / sqrt(u[t]),
  cov_diag[a] = sigma_idio[a]^2
)
```

Compared with `v2_l2`:

- `cov_diag` is unchanged
- `u[t]` is unchanged
- only the common covariance geometry changes

## Guide

The guide remains an online-filtering variational family.

Structural sites use independent variational distributions for:

- `alpha`
- `w`
- `gamma0`
- `tau_gamma`
- `Gamma`
- `sigma0`
- `sigma_currency`
- `tau_sigma_pair`
- `delta_pair`
- `omega_currency`
- `s_u`

The latent regime path still uses the same gain-style encoder inherited from the earlier online-filtering family.

So `v2_l3` changes common covariance structure while keeping the filtering mechanics fixed.

## Predict

Prediction uses:

1. stored structural posterior summaries
2. the carried filtering state from online training
3. AR(1) rollout of the shared FX regime
4. simulation from the same diagonal-currency-shock `LowRankMVN`

The predictive step mirrors the model:

- the common currency shock block is scaled by forecast `u[t+1]`
- the hierarchical nugget remains static

## Intended Diagnostic Effect

The expected effect of `v2_l3` relative to `v2_l2` is:

- narrower central intervals on quiet G10 and cross-style pairs
- less arbitrary spread from the common FX block
- stronger pressure for pair dependence to come directly from shared currencies

If `v2_l3` still remains too wide, the next likely extension is `v2_l3b`: move pair intercepts and feature weights to a currency-centered hierarchy instead of leaving them pair-native.
