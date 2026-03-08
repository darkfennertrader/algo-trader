# Factor Model V1 Learning Notes (Plain English)

This file explains the 9 learning models from simplest to more advanced.

## Level 1: Intercept-only
File: `model_v1_l1_intercept.py`

What it means:
- Each asset `a` has a baseline return `alpha[a]`.
- Each asset `a` also has a noise level `sigma[a]` (always positive).
- Return at time `t` for asset `a` is centered around `alpha[a]` with spread `sigma[a]`.

Simple formula:
- `mu[t,a] = alpha[a]`
- `y[t,a] ~ Normal(mean=mu[t,a], std=sigma[a])`

Shapes:
- `y`: `[T, A]`
- `alpha`: `[A]`
- `sigma`: `[A]`
- `mu`: `[T, A]`

Read order:
1. Check `y` shape.
2. Sample `alpha` and `sigma`.
3. Expand `alpha` across all `T`.
4. Use Normal likelihood for `y`.

## Level 2: Linear + Gaussian
File: `model_v1_l2_linear_gaussian.py`

What is added:
- Feature weights `w[a,f]`.
- Mean is now linear in features.

Simple formula:
- `mu[t,a] = alpha[a] + sum over f of X[t,a,f] * w[a,f]`
- `y[t,a] ~ Normal(mean=mu[t,a], std=sigma[a])`

Shapes:
- `X`: `[T, A, F]`
- `w`: `[A, F]`
- `mu`: `[T, A]`

Read order:
1. Check `X` and `y` shapes.
2. Sample `alpha`, `sigma`, and `w`.
3. Build linear `mu`.
4. Use Normal likelihood for `y`.

## Level 3: Linear + Student-t
File: `model_v1_l3_linear_student_t.py`

What is added:
- Same linear mean as Level 2.
- Observation noise changes from Normal to Student-t.
- Student-t has a parameter `nu` (degrees of freedom), learned from data.

Simple formula:
- `nu_raw ~ Gamma(shape, rate)`
- `nu = nu_raw + shift`
- `mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]`
- `y[t,a] ~ StudentT(df=nu, mean=mu[t,a], scale=sigma[a])`

Why this matters:
- Student-t handles outliers better than Normal.

Read order:
1. Sample `nu_raw`, then compute `nu`.
2. Sample `alpha`, `sigma`, `w`.
3. Build `mu`.
4. Use Student-t likelihood for `y`.

## Level 4: Student-t + Regularized Horseshoe
File: `model_v1_l4_student_t_horseshoe.py`

What is added:
- Weight scale is no longer fixed.
- It is built from hierarchical shrinkage terms:
  - `tau0`: global shrinkage
  - `lambda[f]`: feature-level shrinkage
  - `kappa[a,f]`: local shrinkage
  - `c`: slab scale
- This pushes many weights near zero and allows a few important ones to stay large.

Simple formula:
- `base[a,f] = lambda[f] * kappa[a,f]`
- `lam_tilde_sq[a,f] = (c^2 * base[a,f]^2) / (c^2 + tau0^2 * base[a,f]^2 + eps)`
- `w_scale[a,f] = tau0 * sqrt(lam_tilde_sq[a,f])`
- `w[a,f] ~ Normal(0, w_scale[a,f])`
- `mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]`
- `y[t,a] ~ StudentT(df=nu, mean=mu[t,a], scale=sigma[a])`

Read order:
1. Sample `nu`.
2. Sample horseshoe globals (`tau0`, `lambda`, `c`).
3. Sample asset params (`alpha`, `sigma`) and locals (`kappa`).
4. Build `w_scale`, then sample `w`.
5. Build `mu` and observe `y` with Student-t.

## Level 5: Cross-Asset Latent Factors (Gaussian)
File: `model_v1_l5_latent_factors_gaussian.py`

What is added:
- Keep Level 4 mean:
  `mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]`
- Replace independent residuals with a shared latent-factor residual:
  - `f[t,k]` = week-level factor shocks
  - `B[a,k]` = asset loading on factor `k`
  - `eps[t,a]` = idiosyncratic noise
- Add column-wise shrinkage on `B` (each factor column has its own shrink control).

Simple formula:
- `b_col[k] ~ HalfNormal(b_col_shrink_scale)`  (column shrinkage)
- `B[a,k] ~ Normal(0, b_scale * b_col[k])`
- `sigma_idio[a] ~ HalfNormal(sigma_idio_scale)`
- `f[t,k] ~ Normal(0, 1)`
- `y[t,a] ~ Normal(mean = mu[t,a] + sum_k B[a,k] * f[t,k], std = sigma_idio[a])`

Why this matters:
- Assets become correlated through shared `f[t,:]`.
- You can model common market moves and sector-style co-movement.

Read order:
1. Build Level 4-style `mu`.
2. Sample factor-loading shrinkage and `B`.
3. Sample weekly factor shocks `f`.
4. Add `Bf` term to mean.
5. Observe with Gaussian idiosyncratic noise.

## Level 6: Cross-Asset Latent Factors + Joint Heavy Tails
File: `model_v1_l6_latent_factors_joint_student_t.py`

What is added:
- Keep Level 5 structure.
- Add a shared weekly stress scale `u[t]`:
  `u[t] ~ Gamma(nu/2, nu/2)` with fixed `nu` (for stability).
- Scale both common factor term and idiosyncratic noise by `1/sqrt(u[t])`.

Simple formula:
- `u[t] ~ Gamma(nu/2, nu/2)`
- `f[t,k] ~ Normal(0,1)`
- `y[t,a] ~ Normal(`  
  `mean = mu[t,a] + (sum_k B[a,k] * f[t,k]) / sqrt(u[t]),`  
  `std  = sigma_idio[a] / sqrt(u[t]) )`

Key effect:
- Same-week tails become joint across assets:
  when `u[t]` is small, many assets can move more at the same time.
- This gives crisis-week behavior that independent Student-t per asset misses.

Equivalent intuition (marginal view):
- `y[t]` behaves like a multivariate heavy-tailed distribution.
- Dependence comes from `B f[t]`.
- Heavy tails come from shared scale `u[t]`.

Read order:
1. Build Level 4-style `mu`.
2. Build factor residual structure (`B`, `f`) as in Level 5.
3. Sample weekly stress scale `u[t]`.
4. Apply `1/sqrt(u[t])` scaling to common and idiosyncratic parts.
5. Observe with Normal conditional likelihood (heavy tails appear marginally).

## Level 7: Collapsed (Marginalized) Latent Factors + Joint Heavy Tails
File: `model_v1_l7_marginalized_factors_joint_student_t.py`

What is added:
- Keep the same mean block as Level 4/6:
  `mu[t,a] = alpha[a] + sum_f X[t,a,f] * w[a,f]`
- Keep the same priors for `B`, `sigma_idio`, and shared weekly `u[t]`.
- Remove per-week latent shocks `f[t,k]` from sampling by integrating them out.

Collapsed covariance:
- `D = diag(sigma_idio^2)`
- `Sigma = B B' + D`
- Conditional likelihood per week:
  `y[t] | u[t] ~ MVN(mean=mu[t], covariance=Sigma / u[t])`

Practical implementation:
- `cov_factor[t] = B / sqrt(u[t])`
- `cov_diag[t] = sigma_idio^2 / u[t]`
- `y[t] ~ LowRankMultivariateNormal(loc=mu[t], cov_factor[t], cov_diag[t])`
- One `time` plate only, no `asset_obs` plate.

Why this helps:
- Fewer latent variables (no `f[t,k]` block).
- Better identifiability and stability in SVI.
- More reliable posterior for covariance structure used in portfolio/risk steps.

Read order:
1. Build Level 4-style `mu`.
2. Sample `B`, `sigma_idio`, and shared `u[t]`.
3. Build low-rank covariance pieces (`cov_factor`, `cov_diag`).
4. Observe each `y[t]` as one A-dimensional event from LowRank MVN.

## Level 8: Collapsed Factors + Persistent Volatility Regime
File: `model_v1_l8_marginalized_factors_sv_scale_mixture.py`

What is added:
- Keep Level 7 collapsed covariance structure.
- Replace iid weekly scale with a product:
  - persistent regime part (anchored): `s_regime[t] = exp(h[t] - 0.5 * var_h)`
  - iid heavy-tail shock: `v[t]`
- Total scale:
  - `u[t] = s_regime[t] * v[t]`

Persistent regime state:
- `phi` fixed high (default 0.97).
- Learn `s_u` (regime vol-of-vol).
- AR(1) log-volatility:
  - `h[1] ~ Normal(0, s0)`
  - `h[t] = phi * h[t-1] + s_u * eps[t]`
  - `eps[t] ~ Normal(0,1)`
  - `s0 = s_u / sqrt(1 - phi^2 + eps)`
- Stationary variance and anchoring:
  - `var_h = s_u^2 / (1 - phi^2 + eps)`
  - `s_regime[t] = exp(h[t] - 0.5 * var_h)` so `E[s_regime] ~ 1`

Heavy-tail shock:
- `v[t] ~ Gamma(nu/2, nu/2)` with fixed `nu` (default 10).

Collapsed likelihood remains:
- `Sigma = B B' + diag(sigma_idio^2)`
- `y[t] | u[t] ~ MVN(mu[t], Sigma / u[t])`

Practical LowRank MVN form:
- `cov_factor[t] = B / sqrt(u[t])`
- `cov_diag[t] = sigma_idio^2 / u[t]`
- `y[t] ~ LowRankMultivariateNormal(loc=mu[t], cov_factor[t], cov_diag[t])`

Why this helps:
- Captures volatility clustering over weeks (persistent high/low risk regimes).
- Still allows week-specific tail shocks.
- Keeps the stable collapsed-factor inference from Level 7.

Read order:
1. Build Level 7 mean and collapsed covariance pieces.
2. Sample/construct AR(1) regime `h[t]`, compute `var_h`, then build anchored `s_regime[t]`.
3. Sample iid heavy-tail `v[t]`.
4. Combine into total scale `u[t] = s_regime[t] * v[t]`.
5. Apply `u[t]` in LowRank MVN covariance scaling.

## Level 9: Level 8 + Global Macro Features with Hierarchical Pooling
File: `model_v1_l9_marginalized_factors_sv_scale_mixture_global.py`

What is added:
- Keep Level 8 collapsed covariance and persistent volatility regime.
- Split the mean into asset-local and global feature blocks:
  - `X_asset[t,a,f]` stays asset-specific.
  - `X_global[t,g]` is shared across assets at each time.
- Add hierarchical partial pooling for global loadings:
  - `beta[a,g] ~ Normal(beta0[g], tau_beta[g])`
  - `beta0[g] ~ Normal(0, 0.05)`
  - `tau_beta[g] ~ HalfNormal(0.05)`

Mean decomposition:
- `mu[t,a] = alpha[a]`
  `         + sum_f X_asset[t,a,f] * w[a,f]`
  `         + sum_g X_global[t,g] * beta[a,g]`

Covariance and scale (same as Level 8):
- `Sigma = B B' + diag(sigma_idio^2)`
- `u[t] = s_regime[t] * v[t]`
- `y[t] | u[t] ~ MVN(mu[t], Sigma / u[t])`
- LowRank implementation:
  - `cov_factor[t] = B / sqrt(u[t])`
  - `cov_diag[t] = sigma_idio^2 / u[t]`

Why this helps:
- Macro regime signals (VIX, credit spreads, DXY, real yields, liquidity stress)
  affect all assets through `X_global`.
- Hierarchical pooling shares information across assets for each macro loading
  and stabilizes estimates when per-asset history is limited.
- Keeps the robust Level 8 covariance/tail structure unchanged.

Read order:
1. Build the Level 8 regime-scale and collapsed covariance pieces.
2. Build asset-local mean term from `X_asset` and horseshoe `w`.
3. Build global mean term from `X_global` and pooled `beta`.
4. Sum both mean terms plus intercept `alpha`.
5. Observe with the same LowRank MVN likelihood scaled by `u[t]`.

## Map to production `model_v1.py` and beyond
- Level 1: baseline + noise.
- Level 2: add linear features.
- Level 3: add heavy tails.
- Level 4: add sparse shrinkage and reach near-production structure.
- Level 5: add cross-asset dependence via latent factors.
- Level 6: add joint heavy tails via shared weekly scale mixture.
- Level 7: collapse weekly factors for a cleaner, more stable covariance posterior.
- Level 8: add persistent volatility clustering on top of collapsed heavy-tail covariance.
- Level 9: add global macro features in the mean with hierarchical pooling over
  per-asset macro loadings.
