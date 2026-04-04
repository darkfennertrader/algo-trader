# `v11_l2`: Spread-Only Index Subspace Consistency

`v11_l1` showed that the dual-view idea is directionally useful:

- keep the trusted raw-space `v4_l1` backbone
- add a soft transformed-space consistency block
- improve some key decision baskets without rebuilding the index block

But `v11_l1` was still too broad:

- it supervised both `global_level` and the spread coordinates
- the auxiliary block was too strong overall
- aggregate calibration and residual-dependence quality still lagged `v4_l1`

So `v11_l2` is a narrow follow-up, not a new architecture.

## Core hypothesis

The useful part of the dual-view idea is the **spread subspace**, not the
global transformed coordinate.

So `v11_l2` keeps:

- the raw-space `v4_l1` index block unchanged
- the same unified FX / commodity / macro backbone
- the same dependence-layer raw observation path

And changes only the auxiliary consistency block:

- drop the transformed `global_level` term entirely
- keep only the 4 spread coordinates:
  - `us_minus_europe`
  - `us_internal_style`
  - `euro_core_vs_uk_ch`
  - `spain_vs_euro_core`
- reduce the auxiliary observation weight materially
- increase shrinkage on spread scales and spread correlation

## Intended interpretation

`v11_l2` should be read as:

- not another dependence family
- not another measurement family
- not a hard transformed-basis replacement
- but a weaker, spread-only dual-view regularizer aimed directly at the basket
  subspace where `v11_l1` was helpful

## What would count as success

Relative to `v11_l1`, `v11_l2` should:

- preserve part of the basket gains on:
  - `us_index`
  - `us_minus_europe`
  - `index_equal_weight`
- avoid worsening `europe_index`
- improve aggregate calibration and PIT
- improve or at least not worsen residual-dependence quality

If it fails, the `index_subspace_consistency` family should be treated as
exhausted and `v4_l1` should remain the best current promotion candidate.
