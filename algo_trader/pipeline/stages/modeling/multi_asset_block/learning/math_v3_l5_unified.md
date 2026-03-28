# V3 L5 Unified

`v3_l5_unified` keeps `v3_l1_unified` as the mixed-universe baseline and makes
one targeted index-block change: the deterministic index-group carrier becomes
a **dynamic group-aware index channel**.

## Why L5 Exists

`v3_l3_unified` and `v3_l4_unified` taught two useful but incomplete lessons:

- indices likely need structure beyond one generic dynamic index block
- static structure alone is not enough

`v3_l3_unified` added a learned static index factor and improved on the looser
`v3_l2_unified` branch, but it did not replace `v3_l1_unified`.
`v3_l4_unified` then used deterministic country or region group structure, but
its static group carrier became too narrow in the tails.

So `v3_l5_unified` tests a more specific claim:

- some of the missing index structure is still group-shaped
- but that structure should be allowed to move over time
- the right fix is not a wider generic index block
- the right fix is a small **dynamic regional spread channel**

## Structural Change

The mean remains unchanged:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

The broad latent states remain:

```text
h_fx_broad[t]
h_fx_cross[t]
h_index[t]
h_commodity[t]
```

`v3_l5_unified` adds one more latent vector for the index groups:

```text
g_index[t,g]
```

where each component is an AR(1) state for one deterministic index group.

## Dynamic Index-Group Channel

Indices are still assigned to deterministic groups from their symbols, for
example:

- `IBUS500`, `IBUS30`, `IBUST100` -> `US`
- `IBDE40`, `IBEU50`, `IBFR40`, `IBGB100` -> Europe-style country or region groups
- `IBCH20` -> `CH`

Let `M_group` be the binary index-group exposure matrix:

```text
M_group[a,g] = 1 if index asset a belongs to group g
             = 0 otherwise
```

`v3_l4_unified` used a static positive group scale:

```text
lambda_group[g] > 0
```

`v3_l5_unified` keeps that base scale, but now modulates it with a dynamic
group state:

```text
lambda_group[g] * exp(0.5 * g_index[t,g])
```

So each index group gets its own time-varying shared covariance carrier.

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

`v3_l5_unified` becomes:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    M_group * diag(lambda_group * exp(0.5 g_index[t])),
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

So the index covariance is split into:

- a broad dynamic index market channel
- a dynamic deterministic group channel

The point is to let the model express both:

- market-wide index stress
- time-varying regional or country divergence

without loosening the full mixed universe.

## Guide

The guide stays in the same mean-field online-filtering family as `v3_l1`,
but the local latent state now includes:

```text
[ h_fx_broad, h_fx_cross, h_index, g_index[1], ..., g_index[G], h_commodity ]
```

The structural sites remain simple:

- `alpha`
- `sigma_idio`
- `w`
- `beta`
- loading matrices
- `lambda_group`
- regime innovation scales

with Normal or LogNormal variational families as appropriate.

## Predict

Prediction is the same online filtering rollout style as `v3_l1_unified`, but
the forecasting state now carries the extra index-group latent vector. At each
forecast step:

1. roll the broad latent states forward
2. roll the group-state vector forward
3. rebuild the full covariance with the dynamic group channel
4. sample from the mixed-universe `LowRankMVN`

## What L5 Is Testing

`v3_l5_unified` is testing a narrower and more dynamic claim than both
`v3_l3_unified` and `v3_l4_unified`:

- the remaining index problem is not purely static
- the missing structure is likely group-aware and time-varying
- one broad index state is not enough to capture both market mode and
  regional spread mode

If that claim is right, `v3_l5_unified` should:

- keep `v3_l1_unified`-level mixed-universe calibration
- improve index-only dependence quality
- reduce residual dependence in the hard index pairs
- improve fixed-basket diagnostics such as `us_index`, `europe_index`, and
  `us_minus_europe`
