# Math v3_l9 Unified

`v3_l9_unified` is the next unified research branch after `v3_l8_unified`.

The corrected evidence from `v3_l1_bug_fixed`, `v3_l6_bug_fixed`, `v3_l7`, and
`v3_l8` points to one persistent conclusion: the main unresolved bottleneck is
the covariance geometry of the index block.

`v3_l9_unified` therefore keeps the corrected `v3_l1` backbone intact and
replaces the incremental-patch mentality with a small dedicated structured
index covariance submodel.

## Design Goal

Preserve what is already stable:
- FX block
- commodity block
- mean layer
- global linkage

Change only the index covariance structure.

## Index Submodel

The index block is modeled as:

- one broad dynamic market factor
- one dynamic `US minus Europe` regional factor
- one dynamic `core Europe minus UK/CH` regional factor
- one very small static residual index factor

This gives:

```text
Sigma_idx,t ≈ B_idx B_idx^T exp(h_index,t)
           + d_us_eu d_us_eu^T exp(h_region_us_eu,t)
           + d_eu_core d_eu_core^T exp(h_region_eu_core,t)
           + B_idx_static B_idx_static^T
           + diag(sigma_idio_idx^2)
```

The two regional carriers are fixed and interpretable.

The static residual term is deliberately tiny and strongly shrunk. It is meant
to absorb what the named dynamic factors miss, not to turn the index block into
another loose latent-factor model.

## Why This Is Different From v3_l7 And v3_l8

`v3_l7` added more named spread structure, but without a residual learned
remainder.

`v3_l8` added a tiny residual factor, but collapsed the regional dynamics back
to one regional mode.

`v3_l9` keeps both:
- two explicit regional dynamic modes
- one tiny static residual factor

So `v3_l9` is the first branch in this line that actually matches the proposed
structured index submodel:
- broad market mode
- regional structure
- constrained residual remainder

## State Layout

The online-filtering state vector is:

```text
[ h_fx_broad,
  h_fx_cross,
  h_index,
  h_index_region_us_eu,
  h_index_region_eu_core_vs_uk_ch,
  h_commodity ]
```

The two regional states use heavy-tailed innovations.

## Acceptance Criteria

`v3_l9_unified` is only interesting if it improves on `v3_l1_bug_fixed`.

Minimum review set:
- aggregate calibration summary
- `indices` and `full` block scoring
- `indices` and `full` dependence scoring
- residual dependence on the hard index set
- baskets:
  - `us_index`
  - `europe_index`
  - `us_minus_europe`
  - `index_equal_weight`

Failure signs:
- `us_index` and `us_minus_europe` still near `1.0`
- `index_equal_weight` still too narrow
- whitened residual correlation worse than raw residual correlation

If `v3_l9` still fails, that would be stronger evidence that the project needs
an even more explicit dedicated index model rather than another unified-branch
variation.
