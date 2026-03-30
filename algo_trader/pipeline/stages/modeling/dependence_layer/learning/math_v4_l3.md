# Math v4_l3

`v4_l3` is the first sharply asymmetric follow-up to `v4_l2`.

The design goal is:

- keep the corrected `v3_l1_bug_fixed` Gaussian backbone unchanged
- keep the correction dependence-only
- keep the `v4_l2` broad and symmetric regional t-copula scales
- add one explicitly shrunk `US-vs-Europe` differential tail component

## Why v4_l3 Exists

`v4_l2` showed that a regionalized t-copula overlay can improve the overall
index block while still beating the corrected Gaussian baseline on the global
calibration objective.

But the result was not a clean promotion over `v4_l1`:

- the US side narrowed in a useful direction
- the Europe side and `index_equal_weight` gave back too much
- residual dependence remained acceptable but weaker than `v4_l1`

That leaves one specific next question:

**is the remaining index misspecification driven by regional tail asymmetry
rather than by missing one more purely regional scale?**

`v4_l3` tests exactly that.

## v4_l3 Definition

Keep the corrected Gaussian backbone:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

with `mu_t` and `Sigma_t` still supplied by the corrected
`v3_l1_bug_fixed` unified block model.

`v4_l3` adds four positive scale-mixture variables per time step:

```text
g_t^(broad)   ~ Gamma(nu_b / 2, nu_b / 2)
g_t^(us)      ~ Gamma(nu_u / 2, nu_u / 2)
g_t^(eu)      ~ Gamma(nu_e / 2, nu_e / 2)
g_t^(spread)  ~ Gamma(nu_s / 2, nu_s / 2)
```

All have mean `1`.

The first three scales are inherited from `v4_l2`:

- broad shared index tail scale
- one symmetric regional tail scale sampled separately for US and Europe rows

The new `spread` scale is applied in opposite directions:

- US rows receive a softly shrunk positive spread exponent
- Europe rows receive the same exponent with the opposite sign

So if the Gaussian factorization is:

```text
Sigma_t = L_t L_t^T + D_t
```

then `v4_l3` again changes only the factor-loading part through row-wise
scaling:

```text
L_t^(overlay) = S_t L_t
```

where:

- all index rows receive the broad scaling
- US rows receive additional regional and spread scaling
- Europe rows receive additional regional scaling and the inverse spread scaling
- non-index rows remain unchanged

The diagonal residual term `D_t` is still left unchanged, so the overlay stays
dependence-only rather than rewriting marginal residual scale.

## Research Meaning

The hypothesis is:

**the remaining `v4_l2` tradeoff is not just "US versus Europe need different
tail thickness", but "US versus Europe need a softly asymmetric differential
tail channel on top of the shared regional structure".**

`v4_l3` should win only if it:

- preserves or improves aggregate calibration relative to `v4_l1`
- narrows the remaining `us_index` / `us_minus_europe` over-width
- does so without giving back the Europe-side and `index_equal_weight` profile
- keeps residual dependence diagnostics at least as strong as `v4_l1`
