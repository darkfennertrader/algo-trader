# V3 L3 Unified

`v3_l3_unified` keeps `v3_l1_unified` as the base mixed-asset model and changes
only one structural piece: the index block now has a small **static**
covariance component in addition to the dynamic index regime block.

## Why L3 Exists

`v3_l1_unified` was the best first unified model, but its hardest names were
still mostly indices. `v3_l2_unified` tried to fix that by making the global and
index side richer and looser by default. That moved the posterior in the wrong
direction:

- worse score
- worse tail calibration
- no clear robustness gain

So `v3_l3_unified` is a narrower intervention:

- keep FX unchanged
- keep commodities simple
- give indices a static low-rank carrier so persistent index co-movement does
  not have to be explained only through the dynamic index state

## Structural Change

The mean is unchanged:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

The latent AR(1) states are also unchanged:

```text
h_fx_broad[t]
h_fx_cross[t]
h_index[t]
h_commodity[t]
```

with the same interpretation as `v3_l1_unified`.

## Covariance

`v3_l1_unified` used:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

`v3_l3_unified` adds one static index block:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    M_index * B_index_static,
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

So the index covariance is now split into:

- a **static** block for persistent index co-movement
- a **dynamic** block for time-varying index stress

This is the core `v3_l3` hypothesis.

## Priors

The default priors stay close to `v3_l1`:

- one global factor
- one dynamic index factor
- one static index factor
- slightly smaller default dynamic index scale than `v3_l1`

So the model is not globally looser; it is more structured on the index side.

## Guide

The guide is the same mean-field online-filtering guide as `v3_l1`, plus one
extra structural site:

```text
B_index_static
```

Everything else is unchanged:

- same local 4-state latent path
- same filtering state interface
- same predictive-state handoff

## Predict

Prediction is also the same as `v3_l1`, except the covariance builder now
includes:

```text
M_index * B_index_static
```

at every forecast step before adding the dynamic index block.

## What L3 Is Testing

`v3_l3_unified` is testing a specific claim:

- indices in the mixed universe need a persistent covariance carrier
- they should not have to borrow all their structure from either the static
  global block or the dynamic index regime state

If that claim is right, `v3_l3_unified` should improve the index-heavy failure
pattern without disturbing the FX block.
