# Math v5_l1

`v5_l1` opens a new modeling family:

```text
v5 = residual-copula family
```

The point of `v5` is to act on the evidence produced by the offline
`v4_l1` residual study rather than keep extending the `v4` dependence-layer
line with small static tweaks.

## Why A New Family Is Justified

The offline residual-copula study on `v4_l1` showed:

- one static dependence correction is probably not enough
- dependence becomes much stronger in stress than in calm periods
- Europe and the US have different tail-dependence patterns
- sparse conditional links exist, but they are not the primary signal

So the next useful hypothesis is not “one more static overlay.” The next useful
hypothesis is:

**the trusted Gaussian backbone is still the right marginal anchor, but the
index residual dependence layer should be conditional on stress and allowed to
vary by region.**

## v5_l1 Definition

`v5_l1` keeps unchanged:

- the corrected `v3_l1_bug_fixed` / `v4_l1` mean layer
- the FX block
- the commodity block
- the shared macro/global linkage
- the online-filtering training protocol

It changes only the index residual dependence adapter.

### Base Layer

Let the trusted Gaussian backbone remain:

```text
r_t | z_t ~ N(mu_t, Sigma_t)
```

with `Sigma_t = L_t L_t^T + D_t` built exactly as in the trusted base.

### Conditional Residual-Copula Layer

`v5_l1` adds:

- one calm broad scale mixture
- one stress weight in `[0, 1]`
- one broad stress scale
- one US stress scale
- one Europe stress scale

The latent variables are:

```text
g_calm,t         ~ Gamma(nu_calm / 2, nu_calm / 2)
omega_t          ~ Beta(a_stress, b_stress)
g_stress,t       ~ Gamma(nu_stress / 2, nu_stress / 2)
g_us_stress,t    ~ Gamma(nu_us / 2, nu_us / 2)
g_eu_stress,t    ~ Gamma(nu_eu / 2, nu_eu / 2)
```

with mean-one Gamma scales and a stress weight `omega_t` that is shrunk toward
low values.

The index factor rows are then rescaled as:

```text
S_idx,t =
  g_calm,t^(-1/2)
  * g_stress,t^(-0.5 * lambda_stress * omega_t)
```

and regionally:

```text
S_us,t = S_idx,t * g_us_stress,t^(-0.5 * lambda_regional * omega_t)
S_eu,t = S_idx,t * g_eu_stress,t^(-0.5 * lambda_regional * omega_t)
```

Only the low-rank factor rows are rescaled. The diagonal residual variance is
left unchanged.

So `v5_l1` remains a dependence-only model family in the same spirit as `v4`,
but it is explicitly:

- conditional on stress
- regional across US and Europe
- still anchored to the trusted Gaussian backbone

## Research Meaning

`v5_l1` is the first model that directly tests the main conclusion of the
offline residual study:

**the remaining index problem is not just joint fat tails, but conditional
regional tail dependence.**

It should only be promoted if it improves on `v4_l1` for:

- aggregate calibration
- `indices` block width and scoring
- `us_index`
- `europe_index`
- `us_minus_europe`
- `index_equal_weight`
- residual dependence after whitening

If it does not, the correct decision is not another small patch. The correct
decision is to stop the line and take `v4_l1` to long-horizon validation.
