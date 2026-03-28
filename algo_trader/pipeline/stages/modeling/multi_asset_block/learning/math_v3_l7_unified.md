# V3 L7 Unified

`v3_l7_unified` keeps `v3_l1_unified` as the mixed-universe baseline and
replaces the one-spread `v3_l6_unified` index intervention with a slightly more
explicit **two-spread index submodel**.

The point of `v3_l7_unified` is not to widen the whole model. The point is to
admit that the remaining unified misspecification is now concentrated enough in
the index block that the next branch should model that structure directly.

## Why L7 Exists

The unified sequence now gives a fairly coherent picture:

- `v3_l2_unified` showed that generic extra freedom is too loose
- `v3_l3_unified` showed that indices likely need extra structure, but a broad
  static addition is not enough
- `v3_l4_unified` showed that static deterministic group structure is too rigid
- `v3_l5_unified` showed that a dynamic group-channel still uses the wrong
  geometry
- `v3_l6_unified` showed that a named spread factor is more sensible, but one
  spread factor is still not enough

`v3_l6_unified` failed in a very specific way:

- `us_index` and `us_minus_europe` became too wide
- `index_equal_weight` became too narrow
- residual dependence in the hard index set still did not improve after
  whitening

That combination suggests the remaining problem is no longer "add a bit more
index flexibility." The problem is that the index block still needs one more
named structural mode.

So `v3_l7_unified` tests a stricter claim:

- indices need one broad market mode
- indices need one large cross-region spread mode
- indices also need one narrower Europe-internal mode
- residual covariance should stay deliberately small

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

`v3_l7_unified` adds two named index-only spread states:

```text
s_us_eu[t]
s_eu_vs_ukch[t]
```

Interpretation:

- `s_us_eu[t]`: broad `US minus Europe`
- `s_eu_vs_ukch[t]`: continental Europe versus the names that did not fit the
  first spread cleanly in `v3_l6_unified`, namely `IBGB100` and `IBCH20`

## Fixed Spread Carriers

Let the asset names be the canonical names derived from
[tickers.yml](/home/ray/projects/algo-trader/config/tickers.yml).

### US-Europe Carrier

Let `d_us_eu[a]` be:

- positive for `IBUS30`, `IBUS500`, `IBUST100`
- negative for `IBDE40`, `IBES35`, `IBEU50`, `IBFR40`, `IBGB100`, `IBNL25`
- zero for everything else, including FX, commodities, and `IBCH20`

This carrier should be:

- centered on the active index subset
- normalized to unit norm
- orthogonalized to the broad index carrier

### Europe-versus-UK/CH Carrier

Let `d_eu_vs_ukch[a]` be:

- positive for continental European indices:
  `IBDE40`, `IBES35`, `IBEU50`, `IBFR40`, `IBNL25`
- negative for `IBGB100` and `IBCH20`
- zero for US indices, FX, and commodities

This second carrier should also be:

- centered on the active index subset
- normalized to unit norm
- orthogonalized to both the broad index carrier and `d_us_eu`

So the two spread modes are:

- one global cross-region spread
- one narrower Europe-internal spread

## State Dynamics

The two spread states evolve as persistent AR(1) processes:

```text
s_us_eu[t]     = phi_us_eu     * s_us_eu[t-1]     + tau_us_eu     * z_us_eu[t]
s_eu_vs_ukch[t] = phi_eu_vs_ukch * s_eu_vs_ukch[t-1] + tau_eu_vs_ukch * z_eu_vs_ukch[t]
```

where:

- `|phi_*| < 1`
- `tau_* > 0`
- the innovation shocks are heavy-tailed rather than Gaussian

The heavy-tailed shocks should be kept, because `v3_l6_unified` suggests that
named spread structure is directionally right, but the tails in the targeted
baskets still need protection.

The first-pass `v3_l7_unified` branch should still avoid separate
spread-specific volatility states. That would add too many new degrees of
freedom at once.

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

`v3_l7_unified` becomes:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])   * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])   * M_fx        * B_fx_cross,
    exp(0.5 h_index[t])      * M_index     * B_index,
    s_us_eu[t]               * d_us_eu,
    s_eu_vs_ukch[t]          * d_eu_vs_ukch,
    exp(0.5 h_commodity[t])  * M_commodity * B_commodity ]
```

Conceptually, the index block now contains:

- one broad market mode
- one `US minus Europe` mode
- one continental-Europe versus `UK/CH` mode

This is the minimum explicit index submodel that matches what the diagnostics
have been saying.

## Residual Covariance

The residual index covariance should remain intentionally small:

- diagonal idiosyncratic noise is preferred
- at most a very small static nugget is tolerable
- no generic learned residual factor block should be added in this branch

If `v3_l7_unified` cannot work under that restriction, the correct conclusion
is not "add more generic covariance." The correct conclusion is that the index
submodel still has the wrong named structure.

## Guide

The guide remains in the same mean-field online-filtering family as
`v3_l1_unified`, but the local latent state now includes:

```text
[ h_fx_broad, h_fx_cross, h_index, s_us_eu, s_eu_vs_ukch, h_commodity ]
```

The structural sites remain simple:

- `alpha`
- `sigma_idio`
- `w`
- `beta`
- loading matrices
- `phi_*`
- innovation scales

Both new spread states should be strongly shrunk:

- persistent but not near-unit `phi`
- small innovation scales by default
- heavy-tailed shocks that protect against rare spread moves without
  encouraging generic over-dispersion

## Predict

Prediction stays in the same online-filtering rollout style:

1. roll the broad latent states forward
2. roll both named spread states forward
3. rebuild the covariance with the two fixed spread carriers
4. sample from the mixed-universe `LowRankMVN`

## What L7 Is Testing

`v3_l7_unified` is testing the next explicit claim:

- one named spread was not enough
- two named spread modes may be enough
- the second mode should be narrow and interpretable, not generic

`v3_l7_unified` should only be considered promising if it:

- matches or improves on `v3_l1_unified` aggregate calibration
- improves the `indices` and `full` block scores
- improves dependence and residual-dependence diagnostics on the hard index set
- improves the basket diagnostics for `us_index`, `europe_index`,
  `us_minus_europe`, and `index_equal_weight`
- avoids the `v3_l6_unified` pattern of making some spread baskets too wide
  while making the equal-weight basket too narrow

If that still fails, the next conclusion should be that the index block needs a
more explicit standalone redesign, not another small unified patch.
