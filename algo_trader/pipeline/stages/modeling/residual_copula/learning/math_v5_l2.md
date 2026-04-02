# Math: `v5_l2`

`v5_l2` is the second branch in the `residual_copula` family.

It keeps the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone unchanged:
- same marginals
- same FX block
- same commodity block
- same mean/global structure
- same online-filtering protocol

The only change is the **index residual dependence layer**.

## Hypothesis

The offline `v4_l1` residual study suggested that the remaining index
misspecification is not explained only by:
- one common heavy-tail correction
- or one calm-versus-stress split

Instead, the data suggested:
- Europe and the US have different tail behavior
- upper and lower tail dependence are not symmetric

So `v5_l2` tests:

**Can the remaining index misspecification be reduced by an asymmetric regional
residual-copula overlay that treats positive and negative index states
differently for US and Europe?**

## Construction

Start from the `v4_l1` backbone and apply an index-only residual-copula layer.

The overlay has:
- one calm broad scale
- one stress weight
- one broad stress scale
- one US upper-tail scale
- one US lower-tail scale
- one Europe upper-tail scale
- one Europe lower-tail scale

The upper/lower contribution is gated by the sign of the latent index state via
a smooth logistic weight, so the overlay remains continuous.

## Intent

This branch is still dependence-only in spirit:
- diagonal residual scale is unchanged
- only factor-row scaling is modified

So `v5_l2` is not a new marginal model. It is a sharper residual-dependence
hypothesis than `v5_l1`.

## Success criteria

`v5_l2` only survives if it improves on `v4_l1` by:
- reducing the remaining `indices` over-width
- improving `us_index`, `europe_index`, and `us_minus_europe`
- preserving or improving aggregate calibration
- preserving or improving residual dependence diagnostics
