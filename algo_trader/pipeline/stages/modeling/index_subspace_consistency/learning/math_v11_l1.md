# Index Subspace Consistency Family

`v11_l1` tests a new dual-view hypothesis for the index block.

The key claim is:

- the raw index-space model from `v4_l1` is broadly right
- the remaining misspecification is concentrated in a small spread subspace
- the fix is therefore not another raw-space replacement, but a soft auxiliary
  transformed-space consistency layer

So `v11_l1` keeps:
- the trusted `v4_l1` backbone
- the FX block
- the commodity block
- the macro/global structure
- the online filtering state propagation

And it changes only the index block treatment:
- keep raw index returns in the main likelihood
- define a deterministic transformed coordinate system with:
  - `global_level`
  - `us_minus_europe`
  - `us_internal_style`
  - `euro_core_vs_uk_ch`
  - `spain_vs_euro_core`
- add an auxiliary transformed-space likelihood:
  - `global_level` uses a Student-t consistency term
  - the four spread coordinates use a small shrunk multivariate Student-t block

This family should be read as:
- not another copula family
- not a hard basis replacement
- not another measurement model
- but a dual-view consistency family that keeps raw-space realism while
  regularizing the low-dimensional spread subspace where the basket
  misspecification lives
