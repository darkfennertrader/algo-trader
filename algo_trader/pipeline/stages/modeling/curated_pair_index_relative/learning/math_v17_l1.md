# Family 17: Curated Pair Index Relative, version 17 level 1

## Purpose

Family 17 keeps the shared unified raw-return posterior and replaces the broad
or all-pairs index auxiliary object with a very small curated pair set.

The goal is not to explain the entire reduced index block equally well.

The goal is to supervise only the pair relationships that survived:

- pair 1: `IBCH20 - IBDE40`
- pair 2: `IBUS30 - IBUST100`

These are the cleanest weekly pair objects found by:

- the reduced-universe pairwise index study
- the curated pair stability study

## Model idea

The model still has two heads:

1. raw return head
   - predicts next-week raw returns for all assets
2. curated pair head
   - predicts a small set of reduced-universe index spreads

The latent state, factor structure, dependence layer, and online filtering
machinery are still shared.

## Curated pair coordinates

Let `y_index_t` be the reduced-universe index return vector at week `t`.

Define two curated spread coordinates:

- `z_pair_1_t = r_IBCH20_t - r_IBDE40_t`
- `z_pair_2_t = r_IBUS30_t - r_IBUST100_t`

These form the supervised auxiliary coordinates.

The remaining orthogonal directions are still completed into a basis and are
treated as residual auxiliary coordinates.

## Auxiliary observation groups

The auxiliary index head is split into:

- `curated_pair_index_relative_obs`
  - the curated pair coordinates only
- `curated_pair_index_residual_obs`
  - the remaining orthogonal residual coordinates

Their weights are:

- `curated_pair_obs_weight`
- `residual_obs_weight`

So the model can learn to emphasize the curated pair spreads without forcing
the residual directions to carry equal importance.

## Expected gain

The hypothesis is:

- the weekly reduced-universe index problem is too diffuse when modeled as a
  whole block
- but a very small set of stable pair relationships is still learnable

So Family 17 should be judged by whether it improves:

- pair-aware posterior ordering
- reduced-universe index-slice signal
- top-k spread and hit-rate behavior

without using portfolio metrics in model promotion.
