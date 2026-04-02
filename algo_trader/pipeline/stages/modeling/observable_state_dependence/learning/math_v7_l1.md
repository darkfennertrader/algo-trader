# Math: `v7_l1`

`v7_l1` opens the `observable_state_dependence` family.

It keeps the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone
unchanged:
- same marginals
- same FX block
- same commodity block
- same mean/global structure
- same online-filtering protocol

The only change is the **index dependence layer**.

## Hypothesis

The post-`v6_l1` research conclusion is that richer unconditional copula
families are no longer the best next bet. The remaining index misspecification
may instead be driven by dependence that changes with **observable market
state**, not by more latent copula machinery.

So `v7_l1` tests:

**Can the remaining index misspecification be reduced by conditioning the
index dependence layer on observed feature-state strength, while keeping the
trusted Gaussian backbone untouched?**

## Construction

Start from the `v4_l1` backbone and apply an index-only observed-state
dependence adapter.

The adapter uses:
- one observed global-feature magnitude summary
- one observed index-feature magnitude summary
- one learned scalar bias
- one learned global sensitivity
- one learned index sensitivity
- one learned broad index dependence strength
- one learned US regional strength
- one learned Europe regional strength

These combine into a smooth gate:

- low observed-state magnitude -> dependence stays close to the `v4_l1`
  backbone
- high observed-state magnitude -> index factor rows are widened in a
  structured way

The broad strength applies to all indices. The regional strengths apply only
to US and Europe names respectively.

## Intent

This branch is dependence-only in spirit:
- diagonal residual scale is left unchanged
- only factor-row scaling changes

So `v7_l1` is a genuinely new family because it changes the source of
dependence adaptation:
- not more Gaussian structure
- not more latent copula structure
- but **observable-state-conditioned dependence**

## Success criteria

`v7_l1` only survives if it improves on `v4_l1` by:
- preserving or improving aggregate calibration
- reducing the remaining `indices` misspecification
- improving the key index baskets together
- preserving or improving residual dependence diagnostics
