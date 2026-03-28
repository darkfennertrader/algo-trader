# V3 L6 Unified

`v3_l6_unified` keeps `v3_l1_unified` as the mixed-universe baseline and makes
one more targeted index-block change than `v3_l5_unified`: the next branch
uses an explicit **dynamic US-minus-Europe spread factor** rather than another
generic or group-wide index extension.

## Why L6 Exists

The unified research history now gives a fairly specific picture of what has
and has not worked in the index block:

- `v3_l2_unified` showed that simply giving the global or index side more
  generic freedom is too loose
- `v3_l3_unified` showed that extra index structure may help, but a broad
  static covariance addition is not enough
- `v3_l4_unified` showed that deterministic group structure is informative, but
  static group structure alone becomes too rigid
- `v3_l5_unified` showed that a small dynamic group-aware channel still did not
  solve the remaining index problem and regressed aggregate calibration

So `v3_l6_unified` tests a narrower and more interpretable claim:

- one broad index state is still too coarse
- the missing structure is likely a named regional spread, not another generic
  learned factor
- the basket that matters most is `US minus Europe`
- the next branch should make that structure explicit and strongly shrunk

## Structural Change

The mean remains unchanged:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

The broad latent states also remain:

```text
h_fx_broad[t]
h_fx_cross[t]
h_index[t]
h_commodity[t]
```

`v3_l6_unified` adds one more index-only latent state:

```text
s_us_eu[t]
```

This is a scalar AR(1) spread state that is meant to move US indices against
continental European indices while leaving the rest of the mixed universe
unchanged.

## US-Europe Spread Carrier

Let `d_us_eu[a]` be a fixed deterministic exposure vector over the full asset
universe.

For non-index assets:

```text
d_us_eu[a] = 0
```

For the index block, the first-pass carrier is:

- positive for US indices such as `IBUS30`, `IBUS500`, `IBUST100`
- negative for continental Europe indices such as `IBDE40`, `IBEU50`,
  `IBFR40`, `IBES35`, `IBNL25`
- zero for names that do not fit the first named spread cleanly, such as
  `IBGB100` and `IBCH20`

The vector should be:

- centered on the active index subset
- normalized to unit norm
- orthogonalized to the broad index market carrier

so that the spread factor does not simply relearn the same movement as the
global or broad index state.

## State Dynamics

The new spread state evolves as:

```text
s_us_eu[t] = phi_us_eu * s_us_eu[t-1] + tau_us_eu * z[t]
```

where:

- `|phi_us_eu| < 1`
- `tau_us_eu > 0`
- `z[t]` is heavy-tailed rather than Gaussian

The first-pass `v3_l6_unified` hypothesis uses heavy-tailed innovations for
the spread shock itself, but does **not** add a separate volatility state for
`s_us_eu[t]`. That keeps the branch identifiable:

- one new named spread state
- one new heavy-tailed shock source
- no second new degree of freedom for spread volatility

If the branch partially works, a later follow-up could add a spread-specific
volatility state. It should not be added in the first `v3_l6_unified` test.

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

`v3_l6_unified` becomes:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    exp(0.5 h_index[t])     * M_index     * B_index,
    s_us_eu[t] * d_us_eu,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

Conceptually, the index block is now split into:

- one broad index market mode
- one explicit US-minus-Europe spread mode

without adding a wide extra learned factor family.

The residual index covariance should remain small:

- diagonal idiosyncratic noise is preferred
- at most a very small static nugget should be tolerated
- no additional wide residual covariance channel should be added in this
  branch

## Guide

The guide remains in the same mean-field online-filtering family as
`v3_l1_unified`, but the local latent state now includes:

```text
[ h_fx_broad, h_fx_cross, h_index, s_us_eu, h_commodity ]
```

The structural sites remain simple:

- `alpha`
- `sigma_idio`
- `w`
- `beta`
- loading matrices
- `phi_*` and innovation scales

The new spread state should be strongly regularized:

- shrink `tau_us_eu`
- keep `phi_us_eu` persistent but not near-unit by default
- use prior structure that prefers a weak spread factor unless the data
  repeatedly demands it

The goal is not to let the index block explore generic extra flexibility. The
goal is to test one interpretable structural missing mode.

## Predict

Prediction stays in the same online-filtering rollout style as `v3_l1_unified`.
At each forecast step:

1. roll the broad latent states forward
2. roll the US-Europe spread state forward with heavy-tailed shock
3. rebuild the mixed-universe covariance with the named spread carrier
4. sample from the mixed-universe `LowRankMVN`

## What L6 Is Testing

`v3_l6_unified` is testing a stricter and more interpretable claim than
`v3_l5_unified`:

- the missing index structure is not well described by another general group
  channel
- the dominant remaining misspecification may be one persistent regional spread
  mode
- the relevant spread to test first is `US minus Europe`

`v3_l6_unified` should only be considered promising if it:

- matches or improves on `v3_l1_unified` aggregate calibration
- improves the `indices` and `full` block scores
- improves dependence and residual-dependence diagnostics on the hard index set
- improves fixed basket diagnostics for `us_index`, `europe_index`, and
  `us_minus_europe`
- does not become narrower in the tails than `v3_l1_unified`

If that fails, the next conclusion should not be "add one more generic factor."
It should be that the index block needs a more explicit standalone submodel
with:

- one broad market factor
- one or two named regional spread factors
- a deliberately small residual covariance layer
