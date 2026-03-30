# Math v3_l10c Clean Unified

`v3_l10c_clean_unified` is the last narrow interpolation-style refinement in
the current dependence-only unified research line.

It keeps the corrected `v3_l1_bug_fixed` Gaussian unified backbone for:
- the FX block
- the commodity block
- the mean layer
- the shared macro/global linkage
- the baseline Gaussian index marginals and loadings

and changes only the **index dependence layer** by adding one **index-only
regional scale-mixture overlay**.

The branch is called `v3_l10c_clean` because it deliberately avoids inheriting
the `v3_l6` spread-base geometry that already over-widened the targeted index
baskets.

## Research Position

The corrected research sequence before `v3_l10c_clean` is:
- `v3_l1_bug_fixed`: best calibrated unified baseline
- `v3_l6_bug_fixed`: informative named-spread branch, but still too wide in the
  key index baskets
- `v3_l7` through `v3_l9`: progressively stronger Gaussian index-geometry
  redesigns that still failed
- `v3_l10`: first shallow flow branch, but still anchored to the `v3_l6`
  Gaussian base rather than the best baseline
- `v3_l10a_clean`: strongest clean challenger, but still too wide on the
  US-side baskets
- `v3_l10b_clean`: stronger aggregate calibration and better US-side control,
  but too much give-back on Europe and equal-weight baskets

So the working hypothesis for `v3_l10c_clean` is:

**the best remaining move is not a new architecture, but a softer
interpolation between `v3_l10a_clean` and `v3_l10b_clean`: keep the regional
dependence-only overlay, but weaken the US-differential component enough to
preserve more of the Europe-side and equal-weight gains.**

That motivates a dependence-only overlay on top of `v3_l1_bug_fixed`.

## Why This Is A t-Copula-Style Overlay

The ideal object would be a true post-hoc index-only t-copula fitted on top of
the corrected unified marginals.

In the current Pyro stack, the most faithful practical next step is a
**minimal asymmetric Gamma scale-mixture overlay** on the index block:
- it preserves the `v3_l1` Gaussian backbone
- it introduces one broad index tail factor
- it introduces one even more strongly shrunk US-differential tail factor than
  `v3_l10b_clean`
- it remains cheaper and more interpretable than a deeper flow
- it isolates the dependence correction from the rest of the unified model

So `v3_l10c_clean` should be read as:

**the smallest practical asymmetric t-copula-style overlay on top of
`v3_l1_bug_fixed`**

rather than as a full copula estimation program.

## Base Gaussian Backbone

As in corrected `v3_l1`, let:

- `r_t in R^N` be the full asset return vector
- `mu_t in R^N` be the mean
- `Sigma_t` be the Gaussian covariance implied by the unified block structure

Then the baseline observation model is:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

where `Sigma_t` is built from:
- shared global loadings
- FX broad and FX cross regimes
- one broad index regime
- one static index factor
- one commodity regime

## Index-Only Broad Plus US-Differential Scale Mixture

Let `I` denote the index coordinates in the full asset vector, and let
`I_US` denote the US index subset.

For each time step `t`, introduce two positive latent scales:

```text
g_t^(broad) ~ Gamma(nu_b / 2, nu_b / 2)
g_t^(us)    ~ Gamma(nu_u / 2, nu_u / 2)
```

Both have mean `1`.

Then apply these scales only to the **factor part** of the Gaussian covariance
construction:
- `g_t^(broad)` applies to all index rows
- `g_t^(us)` applies only to US index rows
- the diagonal residual term is left unchanged

Operationally, if:

```text
Sigma_t = L_t L_t^T + D_t
```

then `v3_l10c_clean` uses a factor-row scaling matrix `S_t` and keeps `D_t`
anchored:

```text
Sigma_t^(overlay) = (S_t L_t)(S_t L_t)^T + D_t
```

where:

```text
S_t[ii] = (g_t^(broad))^(-1/2)                         for i in I \ I_US
S_t[ii] = (g_t^(broad))^(-1/2) * (g_t^(us))^(-a/2)    for i in I_US
S_t[ii] = 1                                            otherwise
```

with `a in (0, 1)` a fixed shrinkage coefficient.

Relative to `v3_l10b_clean`, `v3_l10c_clean` uses a weaker US-differential
component in practice:
- `us_diff_df = 14.0` rather than `10.0`
- `us_diff_strength = 0.20` rather than `0.35`

This is a scale-mixture representation of a multivariate-t-style dependence
overlay focused only on the index block, but now with a minimal asymmetric
component aimed specifically at the remaining US-side over-width.

## Why This Is Cleaner Than v3_l10

`v3_l10` added a shallow flow on top of corrected `v3_l6`.

That still inherited:
- the one-spread `v3_l6` geometry
- the over-wide `us_index`
- the over-wide `us_minus_europe`

`v3_l10c_clean` instead:
- starts from `v3_l1_bug_fixed`
- keeps the best-known Gaussian marginals
- changes only the index dependence tail behavior
- does not scale the diagonal residual variance
- introduces only one additional asymmetric component beyond the broad tail
  factor

So if this branch wins, the interpretation is much cleaner:

**the baseline Gaussian unified model was already the right marginal anchor,
and only the index dependence tail structure needed a small asymmetric
regional correction.**

## Practical Regularization

`v3_l10c_clean` is intentionally conservative.

The overlay is constrained by:
- one broad scalar latent per time step
- one more strongly shrunk US-differential scalar latent per time step
- index block only
- mean-one Gamma prior
- no extra flow depth
- no separate Europe-specific latent scale
- no diagonal residual rescaling
- no change to FX or commodity dependence

So this branch is explicitly **not**:
- a deep normalizing flow
- a full-universe flow
- a Bayesian neural network
- another spread-layout redesign

## Acceptance Criteria

`v3_l10c_clean` only survives if it beats the corrected baseline
`v3_l1_bug_fixed`.

The branch should be judged by:
- aggregate calibration summary
- block scoring for `indices` and `full`
- dependence scoring for `indices` and `full`
- residual dependence diagnostics on the hard index set
- fixed basket diagnostics:
- `us_index`
- `europe_index`
- `us_minus_europe`
- `index_equal_weight`

It is not enough for `v3_l10c_clean` to add width. It must improve dependence
structure while preserving the corrected `v3_l1` marginal advantage.

More specifically, it should:
- preserve more of the Europe-side and `index_equal_weight` gains from
  `v3_l10a_clean_unified`
- preserve more of the aggregate calibration and US-side control from
  `v3_l10b_clean_unified`
- keep residual whitening on `indices` better than the raw residual
  dependence summaries

If this still fails, the current narrow-refinement line should be treated as
exhausted.
