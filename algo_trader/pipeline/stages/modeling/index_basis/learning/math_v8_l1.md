# Math: `v8_l1`

`v8_l1` opens the `index_basis` family.

It is derived directly from the transformed-coordinate research summarized in
[index_representation_study.md](/home/ray/projects/algo-trader/docs/simulation/index_representation_study.md).

The study conclusion was:
- the remaining index misspecification is concentrated in a small set of spread
  coordinates
- diagonal transformed-space Student-t marginals already improve the key basket
  diagnostics materially
- meaningful transformed-space dependence still remains after that marginal
  correction

So `v8_l1` changes the **representation of the index block itself**, rather
than adding another dependence overlay in the original index-asset basis.

## Construction

Start from the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone and keep
unchanged:
- the mean structure
- the global macro feature path
- the FX block
- the commodity block
- online latent-state propagation

Replace only the index block with a transformed representation.

The transformed index space is split into:
- `z_global,t`: one global index level coordinate
- `z_spread,t`: four spread coordinates
  - `us_minus_europe`
  - `us_internal_style`
  - `euro_core_vs_uk_ch`
  - `spain_vs_euro_core`

`v8_l1` then models:
- `z_global,t` as a standalone Student-t factor
- `z_spread,t` as a small 4D joint block with:
  - Student-t scale-mixture behavior
  - a strongly shrunk correlation structure
  - strong shrinkage on spread scales

The transformed-space factor block is mapped back into the raw index asset
space and combined with the unchanged FX and commodity blocks inside the
unified observation model.

## Hypothesis

`v8_l1` tests:

**Is the remaining index misspecification primarily a representation problem,
so that modeling global level separately from a small joint spread subspace
beats additional copula or observable-state dependence tweaks in the original
asset basis?**

## Success Criteria

`v8_l1` only survives if it improves on `v4_l1` by:
- preserving or improving aggregate calibration
- improving the key decision baskets together
- reducing the remaining transformed-space spread misspecification
- preserving or improving residual dependence diagnostics
