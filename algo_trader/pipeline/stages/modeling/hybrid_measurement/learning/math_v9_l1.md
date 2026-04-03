# Math: `v9_l1`

`v9_l1` opens the `hybrid_measurement` family.

This family is derived from the composite-measurement research documented in:

- [index_measurement_study.md](/home/ray/projects/algo-trader/docs/simulation/index_measurement_study.md)

## Core Thesis

The index block is neither:

- 10 primitive peer assets, nor
- something that should be fully replaced by a hard transformed basis.

Instead, it is a **hybrid composite-measurement block**:

- some indices are closer to primitive regional/style measurements
- some indices are clearly composite measurements of latent underlying equity
  states

The offline study supports this strongly for:

- `IBUS500`
- `IBEU50`

## What Stays Fixed

`v9_l1` keeps the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone:

- mean structure
- FX block
- commodity block
- macro/global structure
- online filtering/state propagation
- posterior predictive workflow

## New Index Block

Introduce a small latent primitive index state vector:

- `q_us_broad`
- `q_us_style`
- `q_euro_core`
- `q_iberia`
- `q_uk_ch`

Then model observed index returns as:

`r_idx,t = mu_idx,t + B_global * g_t + H * q_t + eps_idx,t`

where:

- `mu_idx,t` is the existing conditional mean
- `g_t` is the existing shared/global backbone state
- `H * q_t` is the new structured index measurement layer
- `eps_idx,t` is diagonal index-specific residual noise

## Soft Composite Structure

Do not deterministically replace `IBUS500` or `IBEU50`.

Instead:

- keep them as observed returns in the likelihood
- force them to be tight, noisy measurements of latent primitive states

So:

- `IBUS500` is strongly anchored to `q_us_broad` with only a tiny style loading
  and small residual noise
- `IBEU50` is strongly anchored to `q_euro_core` with a small Iberia loading
  and small residual noise

## First Branch

`v9_l1` uses:

- a fixed anchor matrix `H0`
- a small learned deviation `Delta`
- `H = H0 + Delta`
- strong shrinkage on `Delta`, especially on composite rows
- heavy-tailed latent-state innovations on `q_t`
- a small full covariance structure on the latent primitive states
- diagonal residual noise on observed indices

## Initial Measurement Anchors

Start with:

- `IBUS30`: `us_broad` and negative `us_style`
- `IBUST100`: `us_broad` and positive `us_style`
- `IBUS500`: mostly `us_broad`, slight `us_style`
- `IBDE40`, `IBFR40`, `IBNL25`: `euro_core`
- `IBES35`: `euro_core` plus `iberia`
- `IBEU50`: mostly `euro_core`, small `iberia`
- `IBGB100`, `IBCH20`: `uk_ch`

## Key Priors

- very tight residual priors for `IBUS500` and `IBEU50`
- looser residual priors for the more primitive/style-sensitive rows
- strong shrinkage on loading deviations
- Student-t / scale-mixture latent shocks on the 5D primitive state block

## What `v9_l1` Tests

`v9_l1` tests whether the remaining index misspecification is best handled by:

- keeping the unified backbone fixed
- replacing raw peer-asset index geometry with a soft composite-measurement
  layer

If it works, that means the unresolved problem was representational, but the
correct fix was **soft measurement structure**, not hard basis replacement and
not another raw-space copula overlay.
