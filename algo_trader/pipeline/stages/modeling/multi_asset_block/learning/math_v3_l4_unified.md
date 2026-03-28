# V3 L4 Unified

`v3_l4_unified` keeps `v3_l1_unified` as the mixed-universe baseline and
changes only one part of the unified covariance structure: the index block gets
an explicit **static index-group carrier**.

## Why L4 Exists

`v3_l1_unified` established that the block-structured unified family is viable.
`v3_l2_unified` and `v3_l3_unified` then tested broader and looser index-side
fixes, but neither branch became the new baseline:

- `v3_l2_unified` made the global/index side too loose
- `v3_l3_unified` added a learned static index block, but it still did not beat
  `v3_l1_unified`

The residual problem remained index-heavy. The hard names were not random; they
were mostly country or region index families such as:

- `IBUS500`, `IBUS30`
- `IBDE40`
- `IBFR40`
- `IBEU50`
- `IBGB100`

So `v3_l4_unified` tests a more specific claim:

- part of the persistent index covariance should be tied to **index groups**
- that structure should be deterministic from the universe itself
- it should not have to be rediscovered only through generic static loadings

## Structural Change

The mean remains unchanged:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

The dynamic latent states also remain unchanged:

```text
h_fx_broad[t]
h_fx_cross[t]
h_index[t]
h_commodity[t]
```

with the same interpretation as `v3_l1_unified`.

## Index Group Carrier

Indices are assigned to deterministic groups from their symbols. For example:

- `IBUS500`, `IBUS30` -> `US`
- `IBDE40` -> `DE`
- `IBFR40` -> `FR`
- `IBEU50` -> `EU`

Let `M_group` be the index-group exposure matrix:

```text
M_group[a,g] = 1 if index asset a belongs to group g
             = 0 otherwise
```

Then `v3_l4_unified` adds a positive scale for each group:

```text
lambda_group[g] > 0
```

and the new static index-group block is:

```text
M_group * diag(lambda_group)
```

This is deterministic given the asset names and the learned group scales.

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

`v3_l4_unified` becomes:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    M_group * diag(lambda_group),
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

So the index covariance is split into:

- a deterministic static group carrier
- a dynamic index regime block

The purpose is to keep persistent country/region index co-movement out of the
generic dynamic regime channel.

## Guide

The guide is the same mean-field online-filtering guide as `v3_l1_unified`,
plus one optional positive structural site:

```text
lambda_group
```

This site is only active when index-group support is enabled in the wrapper.

## Predict

Prediction is the same as `v3_l1_unified`, except the covariance builder now
adds the static index-group block at every forecast step before the dynamic
index block.

## What L4 Is Testing

`v3_l4_unified` is testing a narrower claim than `v3_l3_unified`:

- persistent index structure should be tied to observable index families
- that structure should be captured by a deterministic group carrier
- this should reduce index-block brittleness without loosening the whole mixed
  model

If that claim is right, `v3_l4_unified` should improve mixed-universe
calibration and stability relative to `v3_l1_unified`, while keeping the FX
block behavior intact.
