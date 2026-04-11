# Math: `v13_l2`

`v13_l2` is the narrow follow-up to `v13_l1` inside the
`basket_consistency` family.

It is motivated by the posterior-signal slice diagnostics recorded after the
11-year `v13_l1` work:

- commodities retained some useful signal
- FX was roughly flat
- indices were the main drag

So `v13_l2` keeps the `v13_l1` raw-space backbone and the same four decision
baskets, but changes the auxiliary basket regularizer from a single weighted
term into two weighted pieces.

## Core Thesis

The `v13_l1` auxiliary basket likelihood may still be too level-dominated.

The level baskets:

- `us_index`
- `europe_index`
- `index_equal_weight`

help keep broad basket geometry sane, but they can also pull the auxiliary
term toward the same index-level structure that is already handled by the
trusted raw-space likelihood.

The spread basket:

- `us_minus_europe`

is more targeted at the index slice where posterior-signal diagnostics suggest
the remaining misspecification still lives.

So `v13_l2` tests:

**keep the same raw-space model and the same whitened basket coordinates, but
downweight the level baskets and upweight the spread basket inside the
auxiliary basket likelihood.**

## What Stays Fixed

`v13_l2` keeps unchanged:

- the full `v13_l1` raw-space `v4_l1`-based backbone
- the basket map `B`
- the basket whitening transform
- the auxiliary diagonal Student-t family
- the learned positive basket-scale multiplier

So if `c_t in R^4` is the whitened basket vector from `v13_l1`, then the
auxiliary moments still come from:

`mu_c,t`

`Sigma_c,t`

and the diagonal auxiliary scale still uses:

`d_t = a * sqrt(diag(Sigma_c,t) + eps)`

with the same learned positive `a`.

## Split Auxiliary Likelihood

Partition the whitened basket vector into:

- level block `c_level,t in R^3`
- spread block `c_spread,t in R`

where:

`c_level,t = (c_us_index,t, c_europe_index,t, c_index_equal_weight,t)`

`c_spread,t = c_us_minus_europe,t`

and partition the model-implied mean and scale the same way:

`mu_level,t, d_level,t`

`mu_spread,t, d_spread,t`

Then `v13_l2` uses two auxiliary likelihood pieces:

`c_level,obs,t ~ StudentT_nu(mu_level,t, diag(d_level,t))`

`c_spread,obs,t ~ StudentT_nu(mu_spread,t, d_spread,t)`

with the same auxiliary degrees of freedom `nu` as in `v13_l1`.

## Composite Objective

Instead of one common basket weight `w_basket`, `v13_l2` uses:

- `w_level`
- `w_spread`

So the composite objective becomes:

`L(theta) = L_raw(theta) + w_level * L_level(theta) + w_spread * L_spread(theta)`

where:

- `L_raw` is the ordinary `v13_l1` raw-space log-likelihood
- `L_level` is the auxiliary level-basket log-likelihood
- `L_spread` is the auxiliary spread-basket log-likelihood

The intended regime is:

`w_spread > w_level`

not because the level baskets are unimportant, but because the spread basket
is the more targeted diagnostic lever for the remaining index weakness.

## What `v13_l2` Tests

`v13_l2` tests whether the Family 13 auxiliary regularizer should be
rebalanced rather than abandoned.

So the question is:

- can the same whitened basket-space idea be made more useful for posterior
  signal
- by reducing the emphasis on broad level baskets
- while increasing emphasis on the `us_minus_europe` spread basket

without sacrificing the raw-space calibration and dependence quality that made
`v13_l1` a credible Family 13 opener.
