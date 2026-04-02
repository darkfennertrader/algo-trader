# Math: `v6_l1`

`v6_l1` opens the `mixture_copula` family.

It keeps the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone unchanged:
- same marginals
- same FX block
- same commodity block
- same mean/global structure
- same online-filtering protocol

The only change is the **index dependence layer**.

## Hypothesis

The fitted residual-copula study exposed a mismatch:

- descriptive diagnostics said dependence is stronger in stress than calm
- descriptive diagnostics also said US/Europe structure matters
- but the fitted offline model comparison still preferred one global static
  t-copula over hard regime splits and hard regional blocks

This suggests the issue is not that state dependence or regional structure are
absent. The issue is that the current parameterization loses too much signal by:

- splitting the sample too hard into calm vs stress
- or forcing regional blocks that remove useful cross-region dependence

So `v6_l1` tests:

**Can the remaining index misspecification be reduced by a soft state-mixture
copula that preserves full cross-region dependence while allowing stress
intensity to rise continuously with the latent index state?**

## Construction

Start from the `v4_l1` backbone and apply an index-only mixture-copula layer.

The overlay has:
- one calm broad scale
- one sampled mixture weight
- one stress broad scale
- one US regional stress tilt
- one Europe regional stress tilt

The effective stress weight is:
- sampled from a Beta prior
- then modulated by a smooth gate based on `|index_state|`

So the model does not use a hard calm/stress split. It uses a **continuous
stress mixture**.

## Intent

This branch is dependence-only in spirit:
- diagonal residual scale is unchanged
- only factor-row scaling is modified

It is therefore a new family because the dependence layer is parameterized in a
new way:
- no hard regime partition
- no regional block independence
- full cross-region broad dependence retained

## Success criteria

`v6_l1` only survives if it improves on `v4_l1` by:
- preserving or improving aggregate calibration
- reducing the remaining `indices` over-width
- improving the main index baskets together
- preserving or improving residual dependence diagnostics
