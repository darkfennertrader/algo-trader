# Math v4_l1

`v4_l1` starts a new modeling family:

```text
v4 = dependence-layer unified posterior family
```

The purpose of `v4` is to stop overloading the `v3` multi-asset block family
with ever more local index patches.

`v4_l1` therefore keeps the corrected `v3_l1_bug_fixed` Gaussian backbone as
the trusted base and makes the remaining research problem explicit:

- keep the FX block unchanged
- keep the commodity block unchanged
- keep the mean layer unchanged
- keep the shared macro/global linkage unchanged
- correct only the index dependence layer

## Why A New Family Is Justified

The end of the `v3` sequence established three things:

- `v3_l1_bug_fixed` is still the best calibrated Gaussian unified baseline
- the remaining problem is concentrated in index dependence geometry
- cheap Gaussian redesigns and cheap `v3`-backbone flow patches did not produce
  a clean winner

The strongest `v3` challenger was `v3_l10a_clean_unified`:

- start from the corrected `v3_l1_bug_fixed` Gaussian base
- preserve the best-known marginals
- add only an index-only t-copula-style shared scale-mixture overlay

That idea is strong enough to deserve a new family identity rather than another
`v3` suffix.

## v4_l1 Definition

Let:

- `r_t in R^N` be the full asset return vector
- `mu_t` be the corrected `v3_l1_bug_fixed` conditional mean
- `Sigma_t` be the corrected `v3_l1_bug_fixed` Gaussian covariance

Then the base observation model remains:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

where `Sigma_t` is still built from:

- one global factor block
- FX broad and FX cross regimes
- one broad index factor plus one static index factor
- one commodity factor block

## Index-Only Dependence Overlay

`v4_l1` adds one positive latent scale per time step:

```text
g_t ~ Gamma(nu / 2, nu / 2)
```

with mean `1`.

This scale acts only on the index coordinates of the covariance construction.
If the Gaussian covariance factorization is:

```text
Sigma_t = L_t L_t^T + D_t
```

then `v4_l1` applies:

```text
Sigma_t^(overlay) = S_t^(1/2) Sigma_t S_t^(1/2)
```

where:

```text
S_t[ii] = g_t^(-1)    for index assets
S_t[ii] = 1           otherwise
```

This is the smallest practical index-only t-copula-style dependence adapter on
top of the trusted Gaussian unified base.

## Research Meaning

The key hypothesis is:

**the corrected `v3_l1_bug_fixed` marginals are already the right anchor, and
the remaining error is mostly joint index dependence rather than marginal
scale.**

So `v4_l1` should win only if it improves:

- aggregate calibration
- `indices` and `full` block behavior
- `us_index`, `europe_index`, `us_minus_europe`, `index_equal_weight`
- residual dependence after whitening

If `v4_l1` fails, the project should treat that as evidence about the whole
dependence-layer program, not just another `v3` patch failure.
