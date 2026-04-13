# Family 14, Version 1: Index Relative Measurement

`v14_l1` opens a genuinely new family after the nearby Family 13 repair line
was exhausted.

The stable lesson from the `v13_l1` through `v13_l3` line was:

- the pooled posterior can be made slightly cleaner,
- but the **index slice** remains the main drag on the full 11-year horizon,
- and weak auxiliary basket regularizers are not strong enough to repair it.

So `v14_l1` changes the index measurement object itself.

## High-level idea

Keep the trusted unified raw-return backbone for:

- FX
- commodities
- the latent factor / regime structure

But replace the **index raw observation path** with a structured relative
measurement layer.

That means:

- non-index assets are still observed directly in raw return space;
- the index block is observed in a full-rank **relative coordinate system**
  rather than asset-by-asset raw coordinates.

## Index coordinate system

Let the vector of index returns at time `t` be called `y_index_t`.

It is an `n_index`-dimensional vector containing the raw returns of the index
block at that time.

We construct a full-rank basis matrix:

`B`, an `n_index x n_index` matrix,

whose leading columns are aligned to economically meaningful relative
directions:

- global index level,
- US relative to the broad index level,
- Europe relative to the broad index level,
- US minus Europe spread,

and whose remaining columns span the residual index subspace.

The observed index coordinate vector is then:

`z_t = transpose(y_index_t) * B`

Because \(B\) is full rank, this is not a lossy basket summary. It is a
re-expression of the full index block in a measurement basis that isolates the
coordinates we actually care about.

## Relative measurement layer

Given the model-implied index mean and covariance:

`mu_index_t` and `Sigma_index_t`,

the implied coordinate moments are:

`mu_z_t = transpose(mu_index_t) * B`

`Sigma_z_t = transpose(B) * Sigma_index_t * B`

We then robustly standardize each coordinate using the training-split median
and MAD:

`z_tilde_t_j = (z_t_j - center_j) / mad_j`

and likewise for the model-implied moments.

The coordinate observations are grouped into:

- level coordinate,
- relative coordinates,
- residual coordinates.

Each group is observed with its own scale weight:

`w_level`, `w_relative`, and `w_residual`

The observation family is diagonal Student-\(t\):

`z_tilde_t_group` is modeled with a diagonal Student-t distribution having:

- degrees of freedom `nu`
- location `mu_tilde_z_t_group`
- scale `sigma_tilde_z_t_group`

with the log-density contribution scaled by the group weight.

## Why this is a new family

Family 13 kept the raw index observation model untouched and added weak
auxiliary basket regularizers.

`v14_l1` is different:

- it does **not** merely add another basket penalty,
- it changes the actual index measurement coordinates,
- and it applies that change in a full-rank way so the whole index block is
  supervised through the relative basis.

So this is the first family whose local hypothesis is:

> the persistent index problem is not just a missing weak regularizer; it is a
> misaligned **measurement geometry**.

## Success criterion

`v14_l1` is only justified if it improves:

- posterior-signal diagnostics on the index slice,
- especially rank IC / linear IC / top-k behavior,
- without materially damaging the rest of the unified posterior.
