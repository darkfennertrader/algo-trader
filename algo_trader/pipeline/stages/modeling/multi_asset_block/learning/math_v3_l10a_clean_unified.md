# Math v3_l10a Clean Unified

`v3_l10a_clean_unified` is the cleanest remaining dependence-only test in the
unified multi-asset block family.

It keeps the corrected `v3_l1_bug_fixed` Gaussian unified backbone for:
- the FX block
- the commodity block
- the mean layer
- the shared macro/global linkage
- the baseline Gaussian index marginals and loadings

and changes only the **index dependence layer** by adding one **index-only
shared scale mixture**.

The branch is called `v3_l10a_clean` because it deliberately avoids inheriting
the `v3_l6` spread-base geometry that already over-widened the targeted index
baskets.

## Research Position

The corrected research sequence before `v3_l10a_clean` is:
- `v3_l1_bug_fixed`: best calibrated unified baseline
- `v3_l6_bug_fixed`: informative named-spread branch, but still too wide in the
  key index baskets
- `v3_l7` through `v3_l9`: progressively stronger Gaussian index-geometry
  redesigns that still failed
- `v3_l10`: first shallow flow branch, but still anchored to the `v3_l6`
  Gaussian base rather than the best baseline

So the working hypothesis for `v3_l10a_clean` is:

**the remaining index failure may be mostly a tail-dependence / common-shock
problem, while the corrected `v3_l1` marginals are already the best anchor.**

That motivates a dependence-only overlay on top of `v3_l1_bug_fixed`.

## Why This Is A t-Copula-Style Overlay

The ideal object would be a true post-hoc index-only t-copula fitted on top of
the corrected unified marginals.

In the current Pyro stack, the most faithful practical first step is a
**shared Gamma scale-mixture overlay** on the index block:
- it preserves the `v3_l1` Gaussian backbone
- it introduces common index tail dependence
- it is cheaper and more interpretable than a deeper flow
- it cleanly isolates the dependence correction from the rest of the unified
  model

So `v3_l10a_clean` should be read as:

**the smallest practical t-copula-style overlay on top of `v3_l1_bug_fixed`**

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

## Index-Only Shared Scale Mixture

Let `I` denote the index coordinates in the full asset vector.

For each time step `t`, introduce one positive latent scale:

```text
g_t ~ Gamma(nu / 2, nu / 2)
```

with mean `1`.

Then apply this scale only to the index rows of the Gaussian covariance
construction.

Operationally, if:

```text
Sigma_t = L_t L_t^T + D_t
```

then `v3_l10a_clean` replaces the index rows and columns by the common factor:

```text
Sigma_t^(overlay) = S_t^(1/2) Sigma_t S_t^(1/2)
```

where:

```text
S_t[ii] = g_t^(-1)    for index assets
S_t[ii] = 1           otherwise
```

This is a scale-mixture representation of a multivariate-t-style dependence
overlay focused only on the index block.

## Why This Is Cleaner Than v3_l10

`v3_l10` added a shallow flow on top of corrected `v3_l6`.

That still inherited:
- the one-spread `v3_l6` geometry
- the over-wide `us_index`
- the over-wide `us_minus_europe`

`v3_l10a_clean` instead:
- starts from `v3_l1_bug_fixed`
- keeps the best-known Gaussian marginals
- changes only the index dependence tail behavior

So if this branch wins, the interpretation is much cleaner:

**the baseline Gaussian unified model was already the right marginal anchor,
and only the index dependence tail structure needed correction.**

## Practical Regularization

`v3_l10a_clean` is intentionally conservative.

The overlay is constrained by:
- one scalar latent per time step
- index block only
- mean-one Gamma prior
- no extra flow depth
- no change to FX or commodity dependence

So this branch is explicitly **not**:
- a deep normalizing flow
- a full-universe flow
- a Bayesian neural network
- another spread-layout redesign

## Acceptance Criteria

`v3_l10a_clean` only survives if it beats the corrected baseline
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

It is not enough for `v3_l10a_clean` to add width. It must improve dependence
structure while preserving the corrected `v3_l1` marginal advantage.
