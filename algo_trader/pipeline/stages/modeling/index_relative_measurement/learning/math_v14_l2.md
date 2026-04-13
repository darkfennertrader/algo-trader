# Family 14, Version 2: Per-Index Relative Measurement

`v14_l2` is a narrow follow-up to `v14_l1`.

The main lesson from `v14_l1` was:

- index calibration improved,
- index rank ordering improved slightly,
- but monetization proxies such as top-k spread and hit rate did not improve
  enough.

So `v14_l2` keeps the Family 14 measurement-geometry idea, but makes the
relative layer more directly cross-sectional.

## High-level idea

Keep:

- the trusted unified raw-return backbone for FX and commodities,
- the same latent factor and regime structure,
- the same non-index raw observation path.

Change only the index measurement basis.

Instead of using a small set of coarse regional relative coordinates,
`v14_l2` uses:

- one global index level coordinate,
- one relative coordinate for each index against the equal-weight index level,
- and only uses residual coordinates if they are still needed to complete the
  basis.

## Coordinate system

Let `y_index_t` be the vector of raw index returns at time `t`.

Let `1_eq` be the equal-weight index vector.

The level coordinate is:

`index_level_t = mean(y_index_t)`

For each index asset `i`, define a relative coordinate:

`index_relative_i_t = y_index_i_t - mean(y_index_t)`

These relative coordinates directly encode whether each index is outperforming
 or underperforming the broad index block.

The full basis matrix `B` is built from:

- the level vector,
- the per-index relative vectors,
- then orthonormalized and completed only if residual coordinates are still
  needed.

The observed coordinate vector is:

`z_t = transpose(y_index_t) * B`

## Observation model

Given model-implied index moments:

`mu_index_t` and `Sigma_index_t`,

the implied coordinate moments are:

`mu_z_t = transpose(mu_index_t) * B`

`Sigma_z_t = transpose(B) * Sigma_index_t * B`

Each coordinate is then standardized with train-split median and MAD:

`z_tilde_t_j = (z_t_j - center_j) / mad_j`

The coordinate groups are:

- level coordinate,
- per-index relative coordinates,
- residual coordinates only if present.

The observation families remain diagonal Student-t, with group weights:

- `w_level`
- `w_relative`
- `w_residual`

## Why `v14_l2` exists

`v14_l1` asked the model to fit a better relative measurement geometry, but
its leading coordinates were still somewhat coarse.

`v14_l2` tightens the hypothesis:

> if the real weakness is cross-sectional monetization within the index block,
> the auxiliary coordinates should supervise the actual per-index relative
> ranking problem more directly.

So `v14_l2` is not a new family. It is the first narrow Family 14 follow-up
designed to improve:

- top-k spread,
- top-k hit rate,
- and downstream usefulness of the index slice,

while preserving the same probabilistic backbone.
