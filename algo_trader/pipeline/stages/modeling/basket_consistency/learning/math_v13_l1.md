# Math: `v13_l1`

`v13_l1` is the first branch in the `basket_consistency` family.

It is derived from the research logic summarized in:

- [unified_model_research.md](/home/ray/projects/algo-trader/docs/simulation/unified_model_research.md)

## Core Thesis

The strongest stable 5-year model is still `v4_l1`, which means the main
raw-space unified likelihood is already broadly correct.

The remaining failure is narrower:

- it keeps appearing in a small set of decision baskets
- it is not fixed by rebuilding the raw index block
- it is not fixed by replacing the raw index block with a hard transformed
  basis

So `v13_l1` tests a different claim:

**keep the raw-space `v4_l1` model unchanged, and add only a weak auxiliary
likelihood in a whitened basket space built from the key decision baskets.**

This is therefore:

- not another copula family
- not another measurement family
- not another transformed-basis replacement
- but a composite-likelihood-style regularizer on the exact low-dimensional
  basket space where the residual misspecification keeps showing up

## What Stays Fixed

`v13_l1` keeps the trusted `v4_l1` backbone:

- mean structure
- FX block
- commodity block
- macro/global structure
- raw index likelihood
- online filtering/state propagation
- posterior predictive workflow

So if `y_t` is the full observed mixed-universe return vector, the main model
stays:

`y_t ~ p_v4_l1(y_t | x_t, theta)`

with the same raw-space factor/regime/dependence structure as `v4_l1`.

## Index Subvector And Basket Map

Let:

- `y_idx,t in R^m` be the raw index return subvector at time `t`
- `m = 10` for the current index universe

Define a fixed basket map:

`b_t = B y_idx,t`

where `b_t in R^4` contains:

- `b_1,t = us_index`
- `b_2,t = europe_index`
- `b_3,t = us_minus_europe`
- `b_4,t = index_equal_weight`

and `B in R^(4 x 10)` is the deterministic basket exposure matrix implied by
the decision-basket definitions.

The same map is applied to:

- observed training targets
- the model-implied raw index mean
- the model-implied raw index covariance

So if the raw-space index block has predictive mean `mu_idx,t` and covariance
`Sigma_idx,t`, then the basket-space projection is:

`mu_b,t = B mu_idx,t`

`Sigma_b,t = B Sigma_idx,t B^T`

## Training-Split Basket Standardization

For each training split only, construct the basket time series:

`b_train,t = B y_idx,t`

Then define a robust basket center and scale:

`m_b = median_t(b_train,t)`

`s_b,j = max(MAD_t(b_train,j,t), s_min)`

where:

- `MAD` is the median absolute deviation
- `s_min > 0` is a small floor

Form the scaled basket observations:

`u_train,t = D_b^(-1) (b_train,t - m_b)`

with:

`D_b = diag(s_b,1, ..., s_b,4)`

This step is fixed per training split and is **not** learned in the Pyro
model.

## Basket Whitening

Compute the empirical scaled basket covariance on the training split:

`S_u = Cov(u_train,t)`

Then shrink it toward diagonal:

`S_shrunk = (1 - lambda) S_u + lambda diag(S_u) + delta I`

where:

- `lambda in [0, 1]` is the shrinkage coefficient
- `delta > 0` is a small covariance floor

Define a whitening matrix `W` such that:

`W S_shrunk W^T ~= I`

In implementation, `W` is built from the Cholesky factor of `S_shrunk`.

The whitened basket coordinates are then:

`c_t = W D_b^(-1) (b_t - m_b)`

This same fixed split-specific transform is applied to:

- observed basket targets
- model-implied basket means
- model-implied basket covariances

So the model-implied whitened basket moments are:

`mu_c,t = W D_b^(-1) (mu_b,t - m_b)`

`Sigma_c,t = W D_b^(-1) Sigma_b,t D_b^(-1) W^T`

## Auxiliary Basket Likelihood

The auxiliary basket term is deliberately weak.

Let `d_t` be the diagonal auxiliary basket scale vector. In `v13_l1`, this is
parameterized as:

`d_t = a * sqrt(diag(Sigma_c,t) + eps)`

where:

- `a` is a learned positive basket-scale multiplier
- `eps > 0` is a small stabilizer

The whitened observed basket coordinates `c_obs,t` are then given a diagonal
Student-t likelihood:

`c_obs,t ~ StudentT_nu(mu_c,t, diag(d_t))`

with:

- one common auxiliary degrees-of-freedom parameter `nu`
- diagonal noise in the whitened basket space

No extra full 4D auxiliary covariance is learned in this first branch.

That is intentional:

- the whitening transform already removes most linear basket correlation
- the goal is targeted regularization, not another dependence family

## Composite Objective

`v13_l1` should be read as a composite objective:

`L(theta) = L_raw(theta) + w_basket * L_basket(theta)`

where:

- `L_raw` is the ordinary `v4_l1` raw-space log-likelihood
- `L_basket` is the auxiliary whitened basket-space Student-t log-likelihood
- `w_basket` is a small fixed observation weight

Equivalently, the basket term is implemented as a scaled auxiliary likelihood:

`p(y_t, c_t | theta) propto p_v4_l1(y_t | theta) * p_basket(c_t | theta)^w_basket`

The purpose of `w_basket` is to keep the auxiliary term weak enough that:

- the raw-space model still dominates
- the branch does not recreate the Family 11 problem in a more targeted form
- the basket term nudges the raw model toward better local consistency instead
  of becoming a second full supervisor

## What Is Fixed Versus Learned

Fixed per training split:

- basket map `B`
- basket median center `m_b`
- basket MAD scales `D_b`
- shrunk basket covariance `S_shrunk`
- whitening transform `W`

Learned in the model:

- all ordinary `v4_l1` latent variables and parameters
- the auxiliary basket scale multiplier `a`

Configured but not learned:

- basket auxiliary degrees of freedom `nu`
- basket observation weight `w_basket`
- covariance shrinkage and floor hyperparameters

## What `v13_l1` Tests

`v13_l1` tests whether the remaining index problem is best understood as a
**decision-subspace consistency problem** rather than as another raw-space
structural problem.

So the branch asks:

- can the trusted `v4_l1` model remain the main raw-space posterior
- while a weak whitened basket-space consistency term improves:
  - `us_index`
  - `europe_index`
  - `us_minus_europe`
  - `index_equal_weight`
- without sacrificing the aggregate calibration and residual-dependence quality
  that make `v4_l1` the current best candidate

That is the entire point of the family:

**keep the raw-space model, and regularize only the exact basket space where
the persistent misspecification keeps appearing.**
