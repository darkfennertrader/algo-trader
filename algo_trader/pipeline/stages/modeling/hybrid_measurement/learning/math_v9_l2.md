# Math: `v9_l2`

`v9_l2` is the narrow post-mortem follow-up to
[math_v9_l1.md](/home/ray/projects/algo-trader/algo_trader/pipeline/stages/modeling/hybrid_measurement/learning/math_v9_l1.md)
inside the `hybrid_measurement` family.

It is derived from the post-mortem research written in:

- [unified_model_research.md](/home/ray/projects/algo-trader/docs/simulation/unified_model_research.md)

## Core Thesis

`v9_l1` taught that the first hybrid-measurement parameterization was too
measurement-state dominated by construction.

The main failure was not:

- large drift of `H` away from `H0`
- loose composite residual variances
- exploding latent-state scales

The failure was structural:

- the measurement layer carried too much broad index variance
- the local latent states competed with the shared global index channel
- the composite rows inherited that over-dominance and became far too wide

So `v9_l2` asks a narrower question:

**can the hybrid composite-measurement family work if the local measurement
states are restricted to contrast structure only, with broad index level left
to the existing shared global backbone?**

## What Stays Fixed

`v9_l2` keeps the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone:

- mean structure
- FX block
- commodity block
- macro/global structure
- online filtering/state propagation
- posterior predictive workflow

It also keeps the soft composite-measurement idea from `v9_l1`:

- observed index returns remain in the likelihood
- composite rows remain tightly anchored measurements rather than deterministic
  replacements

## New Index Block

`v9_l2` removes the broad local level states from `v9_l1` and keeps only
orthogonalized contrast states:

- `q_us_style`
- `q_euro_periphery`
- `q_uk_ch_vs_euro`

Observed index returns are still modeled as:

`r_idx,t = mu_idx,t + B_global * g_t + H * q_t + eps_idx,t`

but the local state vector `q_t` is now interpreted much more narrowly:

- `B_global * g_t` carries the broad shared index level
- `H * q_t` carries only regional/style contrasts
- `eps_idx,t` remains diagonal index-specific residual noise

## Orthogonalized Measurement Structure

The key structural change is that the local measurement states are projected so
they do not carry broad level structure inside the index block.

In practice:

- start from an anchored measurement matrix `H0`
- allow a small learned deviation `Delta`
- form `H = H0 + Delta`
- explicitly orthogonalize the columns of `H` against the broad index level on
  the observed index rows

So the measurement states are prevented from reintroducing another broad US or
Europe level channel on top of the shared global factor.

## Initial Contrast Anchors

The starting anchor rows are:

- `IBUS30`: negative `us_style`
- `IBUST100`: positive `us_style`
- `IBUS500`: near-zero `us_style`
- `IBDE40`, `IBFR40`, `IBNL25`: negative `euro_periphery` and slightly
  negative `uk_ch_vs_euro`
- `IBES35`: positive `euro_periphery`
- `IBEU50`: near-zero `euro_periphery`
- `IBGB100`, `IBCH20`: positive `uk_ch_vs_euro`

This is intentionally a contrast-state geometry, not a second broad regional
level model.

## Residual And Tail Design

`v9_l2` simplifies the tail treatment relative to `v9_l1`:

- local state innovations are only mildly heavy-tailed
- residual scales remain diagonal
- the model avoids putting strong heavy tails in both the local state layer and
  the residual layer at the same time

The point is to keep the hybrid measurement thesis while removing unnecessary
sources of width.

## Key Priors

The most important priors in `v9_l2` are:

- very strong shrinkage on composite-row loading deviations
- tighter residual priors on composite rows than on primitive/style-sensitive
  rows
- smaller local state scale than in `v9_l1`
- stronger correlation shrinkage on the local contrast-state block

## What `v9_l2` Tests

`v9_l2` tests whether the `hybrid_measurement` family was structurally right
but parameterized at the wrong variance allocation.

In other words, it asks whether the family can improve once:

- broad level is carried by the shared global index channel
- local latent states are restricted to contrasts only
- the composite rows stay soft measurements
- the local block is prevented from dominating the whole index variance budget
