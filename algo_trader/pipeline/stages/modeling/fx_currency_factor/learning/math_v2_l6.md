# FX Currency Factor V2 L6

This note describes the proposed `v2_l6` FX-only Bayesian model family.

`v2_l6` is not a mean redesign. It is a volatility and covariance redesign built
on the stronger `v2_l2` base.

The main conclusion from the calibrated `v2_l2` rerun is:

- the new selector fixed part of the model-choice problem
- `v2_l2` remained the best base family
- coverage improved materially
- but the posterior is still too wide, especially at `p90` and `p95`

So `v2_l6` should attack the remaining problem directly:

- keep the `v2_l2` mean
- keep the `v2_l2` static hierarchical nugget
- keep the FX-native currency loadings
- replace the single shared common-block volatility state with a two-state FX
  volatility structure

## Core Idea

`v2_l2` uses one scalar latent regime:

```text
u[t]
```

to scale the whole dynamic common FX covariance block.

That is still too blunt. When broad FX stress rises, every part of the common
block widens together. Quiet G10 and cross-style pairs remain too conservative
because they are still being carried by one shared volatility path.

`v2_l6` splits that common block into two pieces:

- a broad FX stress block
- a quieter cross-volatility block

Each block keeps currency-native geometry, but gets its own latent volatility
state.

## Universe and Geometry

The model remains FX-only.

Each asset is a pair:

- `a = BASE.QUOTE`

with pair-to-currency exposure matrix:

```text
P[a,c] in {-1, 0, +1}
```

where one anchor currency is removed for identifiability, usually `USD`.

Only relative currency moves are modeled.

## Mean

The mean stays the `v2_l2` mean:

```text
mu[t,a] =
  alpha[a]
  + sum_f X_asset[t,a,f] * w[a,f]
  + sum_g X_global[t,g] * (P[a] Gamma[:,g])
```

This is deliberate.

The recent experiments suggest the remaining misspecification is not primarily in
the mean geometry. So `v2_l6` isolates the next structural change on the
volatility side.

## Two Common Covariance Blocks

We split the currency factor loadings into two groups:

```text
B_broad[c,k]
B_cross[c,j]
```

and project each group to pair space:

```text
B_pair_broad[a,k] = P[a] B_broad[:,k]
B_pair_cross[a,j] = P[a] B_cross[:,j]
```

Recommended first pass:

- `K_broad = 1 or 2`
- `K_cross = 1`

Interpretation:

- `B_broad` captures global FX co-movement and crisis-style stress
- `B_cross` captures quieter, more persistent cross-style covariance

The important point is that both remain currency-native. We are not returning to
free pair-level covariance geometry.

## Two Latent Volatility States

Instead of one AR(1) regime, use two:

```text
h_broad[t] =
  phi_broad * h_broad[t-1] + s_broad * eps_broad[t]

h_cross[t] =
  phi_cross * h_cross[t-1] + s_cross * eps_cross[t]
```

with:

```text
eps_broad[t], eps_cross[t] ~ Normal(0, 1)
```

and heavy tails:

```text
v_broad[t] ~ Gamma(nu_broad / 2, nu_broad / 2)
v_cross[t] ~ Gamma(nu_cross / 2, nu_cross / 2)
```

Then define:

```text
u_broad[t] =
  exp(h_broad[t] - 0.5 Var[h_broad]) * v_broad[t]

u_cross[t] =
  exp(h_cross[t] - 0.5 Var[h_cross]) * v_cross[t]
```

### Prior Intention

`v2_l6` should encourage the two states to play different roles.

Suggested prior direction:

- `s_cross_scale < s_broad_scale`
- `phi_cross >= phi_broad`
- optionally `b_cross_scale < b_broad_scale`

Interpretation:

- the broad block is allowed to move more sharply
- the cross block is intended to be calmer and more persistent

That is exactly the behavior missing from the current single-state family.

## Static Hierarchical Nugget

Keep the `v2_l2` hierarchical static nugget unchanged:

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

and:

```text
eps_idio[t] ~ Normal(0, D_pair)
D_pair = diag(sigma_idio[a]^2)
```

The nugget remains static on purpose. The `v2_l2` experiments already showed
that letting the shared volatility shock inflate the pair nugget was part of the
over-coverage problem.

## Full Observation Distribution

Conditioned on structural sites and latent regimes:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor =
    [
      B_pair_broad / sqrt(u_broad[t]),
      B_pair_cross / sqrt(u_cross[t])
    ],
  cov_diag = sigma_idio^2
)
```

Equivalently:

```text
Cov[y[t]] =
  B_pair_broad B_pair_broad' / u_broad[t]
  + B_pair_cross B_pair_cross' / u_cross[t]
  + D_pair
```

This is the key `v2_l6` change.

Compared with `v2_l2`:

- common covariance is still low-rank and currency-native
- but it no longer has to widen uniformly through one shared state

## Guide

The guide should remain online-filtering, but the local latent state becomes
two-dimensional:

```text
(h_broad[t], h_cross[t])
```

The simplest first-pass guide is:

- keep the same gain-style chronological update idea
- emit diagonal Normal variational updates for the two regime states
- keep structural sites independent as in `v2_l2`

New structural variational sites would include:

- `b_col_broad`
- `B_broad`
- `b_col_cross`
- `B_cross`
- `s_broad`
- `s_cross`
- optionally separate `nu_broad`, `nu_cross`

All mean sites can remain as in `v2_l2`.

## Predict

Prediction should carry a two-state filtering summary:

```text
state_t = (
  mean_h_broad,
  scale_h_broad,
  mean_h_cross,
  scale_h_cross,
  steps_seen
)
```

At prediction time:

1. roll forward `h_broad`
2. roll forward `h_cross`
3. simulate `u_broad`, `u_cross`
4. sample from the same `LowRankMVN`

So prediction changes only in the common-block regime rollout. The mean and
static nugget stay on the proven `v2_l2` path.

## Intended Diagnostic Effect

Relative to calibrated `v2_l2`, the intended effect of `v2_l6` is:

- reduce persistent over-coverage at `p90` and `p95`
- let quiet crosses avoid inheriting the full width of broad FX shocks
- preserve the better selector-aware calibration gains already achieved
- avoid reopening the mean-geometry branch that failed to produce clear gains

The expected signature of success is:

- `p50` stays near or below the calibrated `v2_l2` level
- `p90` drops materially from the current `~0.986`
- weekly `p90` coverage is no longer pinned near `1.0`
- ES does not collapse the way it did in `v2_l4`

## First Recommended Version

The first implementation should stay conservative:

1. keep the `v2_l2` mean unchanged
2. keep the `v2_l2` nugget unchanged
3. split the common block into `broad` and `cross`
4. give each block its own scalar AR(1) heavy-tailed regime
5. keep the guide diagonal in the two local states at first

That version is the cleanest test of the current hypothesis:

the remaining FX misspecification is mainly in the fact that one shared common
volatility state is still too blunt.
