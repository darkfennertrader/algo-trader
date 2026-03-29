# Math v3_l10 Unified

`v3_l10_unified` is the first **index-only expressive dependence** branch in
the unified multi-asset block family.

It keeps the corrected `v3_l6` Gaussian backbone for:
- the FX block
- the commodity block
- the mean layer
- the shared macro/global linkage
- the broad plus one-spread Gaussian index base

and changes only the **index dependence layer** by applying one shallow
affine-coupling flow to the index subvector.

The goal is not to replace the whole unified model with a flexible density. The
goal is to test the narrowest possible non-Gaussian extension after the
Gaussian index covariance redesigns from `v3_l7` through `v3_l9` failed to beat
the corrected `v3_l1` baseline.

## Research Position

The corrected research sequence before `v3_l10` is:
- `v3_l5`: wrong dynamic group-channel direction
- `v3_l6_bug_fixed`: explicit `US minus Europe` spread is more sensible, but
  still non-winning
- `v3_l7`: more named spread structure still over-widens baskets
- `v3_l8`: one regional factor plus one small residual factor still loses
- `v3_l9`: dedicated structured Gaussian index submodel still over-widens the
  index block

So the working hypothesis for `v3_l10` is:

**the remaining index failure may no longer be a Gaussian covariance-layout
problem, but a joint-dependence-shape problem inside the index block.**

That motivates an index-only expressive dependence layer, while leaving the
rest of the unified model unchanged.

## Why This Is A Flow Branch, Not Yet A True Copula Branch

The ideal next mathematical object after `v3_l9` would be a small
copula-oriented dependence model for the index block.

In the current stack, the practical first step is instead a shallow
affine-coupling flow:
- it is available in installed Pyro
- it integrates cleanly with the existing online-filtering pipeline
- it can be initialized at identity
- it provides a minimal expressive extension without replacing the model
  backbone

So `v3_l10` should be read as:

**the smallest practical expressive dependence test on the index block**

rather than as the final copula design.

## Base Gaussian Backbone

As in corrected `v3_l6`, let:

- `r_t in R^N` be the full asset return vector
- `mu_t in R^N` be the mean
- `Sigma_t` be the Gaussian covariance implied by the unified block structure

Then the base observation model is:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

where the Gaussian covariance is built from:
- shared global loadings
- FX broad and FX cross regimes
- one broad index regime
- one named `US minus Europe` index spread regime
- one commodity regime

This is exactly the Gaussian base we already judged informative but still
insufficient in corrected `v3_l6`.

## Index-Only Flow Construction

Let `I` denote the index coordinates within the full asset vector.

Write the Gaussian base draw as:

```text
r_t = [r_fx,t, r_idx,t, r_cmd,t]
```

Only `r_idx,t` is transformed.

Define one shallow affine-coupling transform:

```text
f_idx : R^(N_idx) -> R^(N_idx)
```

with identity initialization.

Then the final observation vector is:

```text
r_t^* =
[
  r_fx,t,
  f_idx(r_idx,t),
  r_cmd,t
]
```

Equivalently, the full observation distribution is:

```text
r_t^* ~ T_f # N(mu_t, Sigma_t)
```

where `T_f` is the full-space transform that:
- applies `f_idx` on the index coordinates
- leaves FX and commodities unchanged

This means `v3_l10` keeps the Gaussian marginals and Gaussian joint structure
as the starting point, but allows one small nonlinear reshaping of the index
dependence geometry.

## Affine-Coupling Parameterization

The implemented branch uses one affine-coupling layer on the index subvector.

For a partition of the index coordinates:

```text
x = (x_a, x_b)
```

the layer acts as:

```text
y_a = x_a
y_b = x_b * exp(s(x_a)) + t(x_a)
```

where:
- `s(.)` is a small scale network
- `t(.)` is a small shift network
- the final layer is initialized to zero so that initially:

```text
s(x_a) ~= 0
t(x_a) ~= 0
```

and therefore:

```text
f_idx(x) ~= x
```

This is important because the branch should begin close to corrected `v3_l6`,
not as a large uncontrolled departure from it.

## Practical Regularization

`v3_l10` is intentionally conservative.

The expressive layer is constrained by:
- one shallow coupling layer only
- small hidden dimension
- identity initialization
- clipped log-scale output
- no flow applied to FX or commodities

So this branch is explicitly **not**:
- a full-universe normalizing flow
- a deep flow stack
- a Bayesian neural network
- a replacement for the block-structured unified model

It is only a narrow dependence correction on the index block.

## Interpretation

If `v3_l10` wins, the correct lesson is:

**the Gaussian block structure was close, but the remaining index misspecification
required a small nonlinear dependence correction.**

If `v3_l10` fails, the lesson is stronger:

**the project has probably exhausted the useful small extensions of the current
index block, and any future expressive dependence work should be treated as a
more explicit index copula / deeper flow program rather than another local
patch.**

## Acceptance Criteria

`v3_l10` only survives if it beats the corrected baseline
`v3_l1_bug_fixed`, not merely the contaminated historical archives.

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

It is not enough for `v3_l10` to sharpen scores. It must also reduce the
structural basket and residual-dependence failures that survived the Gaussian
redesign sequence.
