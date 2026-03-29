# V3 L8 Unified

`v3_l8_unified` keeps `v3_l1_unified` as the mixed-universe baseline and
replaces the incremental named-spread path with a smaller explicit
**index-block redesign**.

The point of `v3_l8_unified` is not to make the whole model looser. The point
is to test whether the corrected evidence now points to a tighter geometry:

- one broad equity mode
- one regional `US minus Europe` mode
- one small residual covariance term inside indices only

## Why L8 Exists

The corrected unified sequence now says:

- `v3_l2_unified` made the model too loose
- `v3_l3_unified` showed that extra index covariance structure can matter
- `v3_l5_unified` showed that a dynamic deterministic group channel is still
  the wrong fix
- `v3_l6_bug_fixed` showed that one explicit named regional factor is more
  sensible than the `v3_l5` idea, but still over-widens the targeted baskets
- `v3_l7_unified` showed that adding another named spread on top of that path
  still does not displace the baseline

So the next step should not be "one more spread patch." It should be a more
direct index submodel:

- one broad dynamic index factor
- one dynamic regional factor
- one very small static residual index factor

## Structural Change

The mean remains unchanged:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

The latent state remains deliberately small:

```text
h_fx_broad[t]
h_fx_cross[t]
h_index[t]
s_region[t]
h_commodity[t]
```

Interpretation:

- `h_index[t]`: broad equity market mode
- `s_region[t]`: persistent `US minus Europe` regional mode with heavy-tailed
  shocks

The remaining index covariance remainder is not given its own dynamic state.
Instead it is handled through one very small static residual index factor.

## Regional Carrier

Let `d_region[a]` be:

- positive for `IBUS30`, `IBUS500`, `IBUST100`
- negative for `IBDE40`, `IBES35`, `IBEU50`, `IBFR40`, `IBNL25`
- zero for everything else, including FX, commodities, `IBGB100`, and `IBCH20`

This carrier should be:

- centered on the active index subset
- normalized to unit norm

The residual index covariance term is not another named spread. It is a small
learned static factor carried only by the index block and strongly shrunk by
prior scale.

## State Dynamics

The regional factor evolves as:

```text
s_region[t] = phi_region * s_region[t-1] + tau_region * z_region[t]
```

where:

- `|phi_region| < 1`
- `tau_region > 0`
- `z_region[t]` is heavy-tailed

The broad index factor remains Gaussian AR(1), as in the earlier unified
branches.

## Covariance

`v3_l8_unified` uses:

```text
cov_factor[t] =
  [ B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    M_index                 * B_index_static_resid,
    exp(0.5 h_index[t])     * M_index     * B_index,
    s_region[t]             * d_region,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity ]
```

Interpretation of the index block:

- `B_index`: broad dynamic index mode
- `d_region * s_region[t]`: named regional spread mode
- `B_index_static_resid`: small residual covariance term for whatever narrow
  structure remains after the first two modes

This is intentionally different from `v3_l3_unified`. The static residual
factor here is not the whole story and should not be allowed to dominate. It is
only there to mop up a small covariance remainder.

## Prior Discipline

The branch should stay conservative:

- `index_factor_count = 1`
- `index_static_factor_count = 1`
- `index_b_scale` stays moderate
- `index_static_b_scale` is much smaller than the broad factor scale
- `s_region` innovations remain strongly shrunk

If this branch only works when the static residual term becomes large, then the
design has failed. The intended result is:

- broad structure handled by the broad factor
- regional structure handled by the named factor
- only a small leftover absorbed by the residual term

## Guide

The guide stays in the same online-filtering mean-field family as
`v3_l1_unified` and `v3_l6_unified`.

The local latent state is:

```text
[ h_fx_broad, h_fx_cross, h_index, s_region, h_commodity ]
```

Structural sites remain simple:

- `alpha`
- `sigma_idio`
- `w`
- `beta`
- loading matrices
- regime innovation scales

## Predict

Prediction stays in the standard rollout pattern:

1. roll the five latent states forward
2. rebuild the covariance with the broad index block, regional carrier, and
   small static residual index term
3. sample from the mixed-universe `LowRankMVN`

## What L8 Is Testing

`v3_l8_unified` is testing a sharper claim than `v3_l6_unified`:

- one named regional spread factor was directionally right
- but it was not enough on its own
- the index block likely needs a small explicit residual covariance term in
  addition to the broad and regional modes

The branch is successful only if it:

- matches or beats `v3_l1_bug_fixed` aggregate calibration
- improves `indices` and `full` diagnostics
- reduces index residual dependence after whitening
- improves the fixed baskets without pushing them toward `1.0` overcoverage

If `v3_l8_unified` still fails, the next step should be a deeper explicit index
submodel rather than another small additive tweak.
