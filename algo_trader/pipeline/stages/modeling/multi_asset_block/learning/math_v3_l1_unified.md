# V3 L1 Unified

`v3_l1_unified` is the first mixed-asset model in the new `multi_asset_block`
family. The goal is one posterior over FX, indices, and commodities without
forcing one homogeneous geometry on every asset.

## Design Goal

The model keeps the lesson from `fx_currency_factor`:

- FX needs dedicated structure.
- Indices and commodities should not be forced into FX geometry.
- The unified model should still produce one posterior predictive
  distribution over the whole universe.

## Asset Blocks

Assets are partitioned into three blocks from their names:

- FX: names in `BASE.QUOTE` form like `EUR.USD`
- Indices: non-FX names such as `IBUS500`
- Commodities: symbols such as `XAUUSD`, `XAGUSD`

`v3_l1_unified` then uses:

- one static global low-rank block across all assets
- one dynamic FX broad block
- one dynamic FX cross block
- one dynamic index block
- one dynamic commodity block

The first version is intentionally simple:

- FX structure is block-aware but not yet currency-latent
- cross-block dependence is introduced through the shared global factor and the
  shared macro mean layer
- the idiosyncratic nugget stays static

## Generative Structure

For asset `a` and time `t`:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

Static idiosyncratic scale:

```text
sigma_idio[a] > 0
```

Dynamic latent log-vol states:

```text
h_fx_broad[t]     = phi_fx_broad     * h_fx_broad[t-1]     + s_u_fx_broad * eps1[t]
h_fx_cross[t]     = phi_fx_cross     * h_fx_cross[t-1]     + s_u_fx_cross * eps2[t]
h_index[t]        = phi_index        * h_index[t-1]        + s_u_index * eps3[t]
h_commodity[t]    = phi_commodity    * h_commodity[t-1]    + s_u_commodity * eps4[t]
```

with `epsk[t] ~ Normal(0, 1)`.

Low-rank blocks:

```text
B_global      : asset x global_factor
B_fx_broad    : asset x fx_broad_factor
B_fx_cross    : asset x fx_cross_factor
B_index       : asset x index_factor
B_commodity   : asset x commodity_factor
```

At each time:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

where each `M_*` is a diagonal asset-class mask.

Observations:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor = cov_factor[t],
  cov_diag = sigma_idio^2
)
```

## Guide

The guide is simpler than `v2_l6` on purpose.

Global sites:

- `alpha`
- `sigma_idio`
- `w`
- `beta`
- all loading matrices
- all `s_u_*` scales

Each has a standard mean-field variational family:

- Normal for signed parameters
- LogNormal for positive scales

Local states:

```text
h_t ~ Normal(h_loc[t], h_scale[t])
```

with one 4-dimensional latent vector per time.

This guide is less expressive than the FX gain-style guide, but it gives a
stable first production mixed-asset model while keeping the online-filtering
state interface intact.

## Predict

Prediction reuses:

- structural posterior means from the guide/state
- the saved 4-dimensional filtering state

For each forecast step:

1. roll the four latent states forward with AR(1)
2. rebuild the block covariance
3. sample from the full-universe `LowRankMVN`

Outputs:

- predictive samples
- predictive mean
- predictive covariance

## Why This Is Only L1

`v3_l1_unified` is a production-wireable unified baseline, not the final
multi-asset architecture.

Expected future upgrades:

- make the FX block currency-latent inside the unified model
- improve cross-block dependence
- replace the mean-field local state guide with a stronger online guide
- add richer class-specific residual hierarchies
