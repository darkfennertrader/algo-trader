# Math v4_l2

`v4_l2` is the first narrow follow-up to `v4_l1`.

The design goal is:

- keep the corrected `v3_l1_bug_fixed` Gaussian backbone unchanged
- keep the `v4` dependence-layer family interpretation unchanged
- keep the correction dependence-only
- replace the single shared index tail scale with a lightly regionalized overlay

## Why v4_l2 Exists

`v4_l1` was the first branch that clearly beat the corrected `v3_l1_bug_fixed`
baseline on the main aggregate calibration summary while also improving the
index basket balance.

But it still left the `indices` block somewhat too wide overall. That leaves
one precise next question:

**is the remaining index misspecification caused by the shared index-only
overlay still being too blunt across US and Europe?**

`v4_l2` tests exactly that and nothing broader.

## v4_l2 Definition

Let the corrected Gaussian backbone remain:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

with `mu_t` and `Sigma_t` still supplied by the corrected
`v3_l1_bug_fixed` unified block model.

`v4_l2` then adds three positive scale-mixture variables per time step:

```text
g_t^(broad)  ~ Gamma(nu_b / 2, nu_b / 2)
g_t^(us)     ~ Gamma(nu_u / 2, nu_u / 2)
g_t^(eu)     ~ Gamma(nu_e / 2, nu_e / 2)
```

All have mean `1`.

The broad scale acts on all index assets. The regional scales act only on
their respective index subsets and are heavily shrunk through small exponents.

If the Gaussian covariance factorization is:

```text
Sigma_t = L_t L_t^T + D_t
```

then `v4_l2` changes only the factor-loading part through row-wise scaling:

```text
L_t^(overlay) = S_t L_t
```

where:

- all index rows receive the broad scaling
- US index rows receive an additional softly shrunk US scaling
- Europe index rows receive an additional softly shrunk Europe scaling
- non-index rows remain unchanged

The diagonal residual term `D_t` is left unchanged, so the overlay remains a
dependence correction rather than a marginal-residual rewrite.

## Research Meaning

The hypothesis is:

**`v4_l1` was directionally right, but a single shared index tail scale is
still too coarse. A lightly regionalized dependence-only overlay may tighten
the remaining `indices` over-width without giving back the broader calibration
gain.**

So `v4_l2` should be judged by whether it:

- preserves or improves aggregate calibration relative to `v4_l1`
- reduces the remaining `indices` over-width
- improves the hard index baskets together, not one at the expense of all the
  others
- keeps residual dependence diagnostics moving in the right direction
