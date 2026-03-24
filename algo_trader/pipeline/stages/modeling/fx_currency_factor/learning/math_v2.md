# FX Currency Factor V2 Learning Notes

This package starts a new learning track for FX-native models.

Files:
- `model_v2_l1.py`
- `guide_v2_l1.py`
- `predict_v2_l1.py`

## V2 Level 1: FX-native currency-factor model with online filtering

### What changes conceptually

The old FX track still treated each pair as an independent asset with its own
pair-level covariance loading matrix. This version changes that geometry.

V2 L1 keeps:
- the online-filtering boundary state,
- the gain-style guide update,
- heavy tails,
- pair-level feature shrinkage for `X_asset`,
- a collapsed LowRank MVN likelihood over pair returns.

But it replaces the old pair-level covariance block with a currency-native one.

### Pair-to-currency exposure matrix

For each FX pair, build a signed exposure row:

- `+1` for the base currency,
- `-1` for the quote currency,
- drop one anchor currency for identifiability.

If the anchor is `USD`, then:

- `EUR.USD` loads on `EUR` with `+1`,
- `USD.JPY` loads on `JPY` with `-1`,
- `EUR.JPY` loads on `EUR` with `+1` and `JPY` with `-1`.

Call this matrix `P[a,c]`.

### Mean block

The conditional mean is:

- `mu[t,a] = alpha[a]`
  `         + sum_f X_asset[t,a,f] * w[a,f]`
  `         + sum_c P[a,c] * m_currency[t,c]`

where currency macro effects are:

- `m_currency[t,c] = sum_g X_global[t,g] * Gamma[c,g]`

So global features are now learned at the currency level, then projected back to
pairs through `P`.

### Covariance block

Currency covariance loadings live in:

- `B_currency[c,k]`

and pair covariance is induced by projection:

- `B_pair[a,k] = sum_c P[a,c] * B_currency[c,k]`

The collapsed covariance for pairs is then:

- `Sigma_pair = B_pair B_pair' + diag(sigma_idio[a]^2)`

### Online volatility state

V2 L1 keeps one shared FX regime:

- `h[t] = phi * h[t-1] + s_u * eps[t]`
- `eps[t] ~ Normal(0, 1)`

and one shared heavy-tail shock:

- `v[t] ~ Gamma(nu/2, nu/2)`

The total weekly scale is:

- `u[t] = exp(h[t] - 0.5 * Var_t[h]) * v[t]`

That scalar scale is then applied to the full pair covariance:

- `y[t] | u[t] ~ MVN(mu[t], Sigma_pair / u[t])`

### LowRank MVN implementation

The collapsed likelihood is implemented with:

- `cov_factor[t,a,k] = B_pair[a,k] / sqrt(u[t])`
- `cov_diag[t,a] = sigma_idio[a]^2 / u[t]`

### Guide

The guide keeps the same high-level split as the current online-filtering
family:

1. Structural latents use global variational parameters.
2. Local state uses an amortized gain-style update.

Local filtering update:

- `h_loc_t = prior_loc_t + gain_t * innovation_t`

with:

- `gain_t = max_gain * prior_scale_t * sigmoid(raw_gain_t)`
- `innovation_t = tanh(raw_innovation_t)`

So the regime update remains uncertainty-aware, but the structure it is
supporting is now FX-native.

### Predictor

The predictor keeps the production-equivalent online rollout:

1. Start from the filtered boundary message.
2. Propagate the scalar AR(1) regime variance conditionally.
3. Sample the next heavy-tail shock.
4. Build the pair mean from pair-local features plus projected currency macro
   effects.
5. Build pair covariance from projected currency loadings.
6. Sample from the LowRank MVN and return `samples`, `mean`, and `covariance`.

### Why this version exists

The mixed-universe experiments showed that the old family was structurally too
conservative for FX, especially for crosses and quiet / managed pairs.

This version changes the part that was still wrong:

- not the online filter,
- not the heavy-tail mechanism,
- not the pair feature shrinkage,
- but the latent geometry used to explain FX dependence.
