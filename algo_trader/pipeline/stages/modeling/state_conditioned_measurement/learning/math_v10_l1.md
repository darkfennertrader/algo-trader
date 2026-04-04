# State-Conditioned Measurement Family, `v10_l1`

`v10_l1` opens the `state_conditioned_measurement` family.

This family combines two lessons that remained separately useful but
non-winning:

- Family 7 showed that **observable market state** can narrow the index block,
  but `v7_l1` did not improve the overall balance enough to beat `v4_l1`.
- Family 9 showed that **soft composite measurement structure** is a more
  plausible representation of some indices than treating all 10 names as
  primitive peers, but `v9_l1` was too measurement-state dominated and
  `v9_l2` still over-corrected.

So the new Family 10 thesis is:

**keep the trusted `v4_l1` backbone and the Family 9 contrast-state
measurement structure, but let the measurement block adapt with observed market
state rather than treating that measurement structure as fixed through time.**

## What stays fixed

`v10_l1` keeps:

- the trusted `v4_l1` / corrected `v3_l1_bug_fixed` backbone
- the FX block
- the commodity block
- the macro/global structure
- the online filtering protocol
- the posterior predictive workflow

## What changes

Only the index block changes.

`v10_l1` starts from the narrow `v9_l2` contrast-state geometry:

- broad index level remains with the shared global channel
- local measurement states carry only contrast structure:
  - `us_style`
  - `euro_periphery`
  - `uk_ch_vs_euro`

Then it adds a small observed-state gate derived from:

- global feature magnitude
- index feature magnitude

That gate modulates two things only:

1. the effective strength of the contrast-state block
2. the tightness of the composite-measurement rows

The gate is centered, so `v10_l1` can both tighten and loosen the local
measurement block relative to a neutral state, instead of only widening it.

## Why this is a genuinely new family

This is not:

- another residual-copula family
- another observable-state dependence family in raw index space
- another hard transformed-basis model
- another generic hybrid-measurement tweak

It is a new family because the research object is now:

**state-conditioned measurement reliability inside the index block**

rather than:

- richer raw-space dependence
- richer copula structure
- or another fixed latent measurement geometry

## First branch hypothesis

`v10_l1` tests whether the remaining index misspecification is better explained
by:

- a stable contrast-state measurement geometry
- whose effective contribution varies with observed market state

rather than by:

- more copula structure
- or another fixed hybrid-measurement parameterization

## Decision standard

`v10_l1` only promotes if it improves:

- aggregate calibration relative to `v4_l1`
- index block balance
- the key baskets:
  - `us_index`
  - `europe_index`
  - `us_minus_europe`
  - `index_equal_weight`
- residual dependence on `indices`
