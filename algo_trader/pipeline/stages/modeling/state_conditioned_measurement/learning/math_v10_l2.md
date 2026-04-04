# State-Conditioned Measurement `v10_l2`

`v10_l2` is the narrow post-mortem follow-up to
[math_v10_l1.md](/home/ray/projects/algo-trader/algo_trader/pipeline/stages/modeling/state_conditioned_measurement/learning/math_v10_l1.md)
and should be read together with
[state_conditioned_measurement_postmortem.md](/home/ray/projects/algo-trader/docs/simulation/state_conditioned_measurement_postmortem.md).

## Motivation

The `v10_l1` post-mortem showed that the branch did not fail because:

- composite rows became too tight
- measurement-state scales exploded
- the latent measurement block dominated total variance

Instead, the mechanical failure was narrower:

- the observed-state gate stayed high most of the time
- the contrast-state modulation was the main over-tightening lever
- composite residual modulation was not the primary problem

So `v10_l2` is intentionally a **repair branch**, not a new Family 10 design.

## Thesis

Keep the useful Family 10 idea:

- trusted `v4_l1` backbone
- `v9_l2`-style contrast-state measurement geometry
- observed-state-conditioned measurement block

But correct the specific `v10_l1` pathology by:

- recentering the gate upward so it is not effectively always on
- weakening the gate through a larger scale
- heavily shrinking contrast-state gating
- retaining only mild composite residual modulation

## Structural Interpretation

The branch still treats the index block as:

- broad level carried by the shared global/index channel
- contrast structure carried by local measurement states
- composite rows (`IBUS500`, `IBEU50`) kept as soft measurements

But the state conditioning is now much more conservative:

- contrast-state strength should remain near the `v9_l2` baseline most of the
  time
- composite residual modulation can still move a little with state
- the gate should no longer collapse the spread baskets by acting as a
  near-always-on narrowing device

## Parameter Changes Relative to `v10_l1`

`v10_l2` changes only the state-conditioned measurement defaults:

- gate center: `0.50 -> 1.75`
- gate scale: `0.75 -> 1.25`
- bias prior scale: `1.0 -> 0.50`
- global/index gate-weight prior scales: `0.50 -> 0.35`
- contrast-strength prior scale: `0.12 -> 0.02`
- composite residual modulation prior scale: `0.10 -> 0.05`

Everything else stays aligned with `v10_l1`.

## Decision Rule

`v10_l2` is justified only because the post-mortem identified a clear
mechanical fix. It should be judged as:

- a narrow gate repair
- not a new architecture
- not another broad state-conditioned measurement exploration

If `v10_l2` still fails to beat `v4_l1`, the Family 10 line should likely
stop rather than continue with nearby gate tweaks.
