# FX Currency Factor V2 L2

This note describes the `v2_l2` FX-only Bayesian model family implemented in:

- `model_v2_l2.py`
- `guide_v2_l2.py`
- `predict_v2_l2.py`

The goal of `v2_l2` is narrower and more FX-native posterior uncertainty than `v2_l1`.

The key structural change is:

- the common currency covariance block remains dynamic and heavy-tailed
- the idiosyncratic pair nugget becomes hierarchical but static

That is, `v2_l2` removes the `v2_l1` assumption that the same shared volatility shock should inflate both the common FX block and the pair-specific residual block.

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

So macro effects remain currency-native:

- global macro signals first hit currencies
- pair means are then formed as base-minus-quote differences

## Dynamic Common Covariance

The common FX covariance remains low-rank in currency space.

We draw currency factor loadings:

```text
B_currency[c,k]
```

and project them to pair space:

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

## Hierarchical Static Idiosyncratic Nugget

This is the main `v2_l2` change.

Instead of free pair residual scales:

```text
sigma_idio[a] ~ HalfNormal(...)
```

we use a hierarchical log-scale model:

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
- `sigma_currency[c]` adjusts that residual level by currency participation
- `delta_pair[a]` allows pair-specific deviations, but with shrinkage

Most importantly, this nugget is not multiplied by the shared time-varying shock.

So the observation model becomes:

```text
y[t] = mu[t] + z_common[t] + eps_idio[t]
eps_idio[t] ~ Normal(0, D_pair)
```

with:

```text
D_pair = diag(sigma_idio[a]^2)
```

This is the intended calibration fix:

- shared FX stress widens the common block
- but quiet pair-specific residual noise does not automatically balloon at the same time

## Full Observation Distribution

Conditioned on structural sites and latent regime:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor = B_pair / sqrt(u[t]),
  cov_diag = sigma_idio^2
)
```

Compared with `v2_l1`, only `cov_factor` is dynamic.

## Guide

The guide remains an online-filtering variational family.

Structural sites use independent variational distributions:

- `alpha`
- `w`
- `gamma0`
- `tau_gamma`
- `Gamma`
- `b_col`
- `B_currency`
- `sigma0`
- `sigma_currency`
- `tau_sigma_pair`
- `delta_pair`
- `s_u`

The latent regime path still uses the gain-style encoder inherited from the earlier online-filtering family.

So `v2_l2` changes structural variance geometry while keeping the same filtering mechanics.

## Predict

Prediction uses:

1. stored structural posterior summaries
2. the carried filtering state from online training
3. AR(1) rollout of the shared FX regime
4. simulation from the same `LowRankMVN` structure

The predictive step mirrors the model:

- common currency covariance is scaled by forecast `u[t+1]`
- idiosyncratic nugget remains static

## Intended Diagnostic Effect

The expected effect of `v2_l2` relative to `v2_l1` is:

- lower over-coverage at central levels like 50% and 80%
- less contamination of quiet pairs by broad shared-volatility shocks
- cleaner separation between common FX movement and pair-specific residual noise

If `v2_l2` still remains too wide, the next likely extension is not more pair pruning but a `v2_l3` mean/feature hierarchy or a multi-state FX volatility model.
