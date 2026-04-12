# Math: `v13_l3`

`v13_l3` is the next narrow follow-up to `v13_l1` inside the
`basket_consistency` family.

It is motivated by the posterior-signal slice diagnostics, not by downstream
portfolio metrics:

- aggregate posterior signal remained weak
- the weakness was concentrated in the index slice
- the remaining index misspecification looked more relative than purely
  level-based

So `v13_l3` keeps the trusted `v13_l1` raw-space backbone and the same
decision-basket spirit, but changes the auxiliary target from broad index
levels to index-relative basket coordinates.

## Core Thesis

The remaining index problem may live in **relative regional structure** rather
than in absolute regional basket levels.

So instead of regularizing:

- `us_index`
- `europe_index`
- `index_equal_weight`
- `us_minus_europe`

directly, `v13_l3` regularizes:

- `us_relative_index = us_index - index_equal_weight`
- `europe_relative_index = europe_index - index_equal_weight`
- `us_minus_europe`

The raw-space model still carries the main return likelihood. The new relative
targets only act as a weak auxiliary pressure on the posterior.

## What Stays Fixed

`v13_l3` keeps unchanged:

- the full `v13_l1` raw return likelihood
- the `v4_l1`-derived factor and regime backbone
- the index `t`-copula overlay
- the auxiliary diagonal Student-t family
- the learned positive basket-scale multiplier

So if `mu_t` and `Sigma_t` are the raw-space posterior moments, the main
likelihood and predictive object are unchanged.

## Relative Basket Coordinates

Let:

- `b_us`
- `b_europe`
- `b_equal`
- `b_spread`

be the original `v13_l1` basket vectors for:

- US index basket
- Europe index basket
- equal-weight index basket
- US minus Europe spread

Then `v13_l3` defines the relative auxiliary basis:

`b_us_rel = b_us - b_equal`

`b_europe_rel = b_europe - b_equal`

`b_spread = b_spread`

and stacks them into a reduced basis `B_rel`.

The auxiliary coordinates are then:

`c_rel,t = y_t B_rel`

and their model-implied moments are:

`mu_rel,t = mu_t B_rel`

`Sigma_rel,t = B_rel' Sigma_t B_rel`

## Whitening and Auxiliary Scale

As in `v13_l1`, the relative coordinates are standardized and whitened on the
training split using:

- componentwise median centering
- componentwise MAD scaling
- shrunk covariance whitening

If `tilde(c_rel,t)` denotes the whitened coordinate vector, then the auxiliary
scale still uses the model-implied whitened covariance:

`d_rel,t = a * sqrt(diag(tilde(Sigma_rel,t)) + eps)`

with the same learned positive multiplier `a`.

## Split Auxiliary Likelihood

Partition the whitened relative vector into:

- relative block
  - `us_relative_index`
  - `europe_relative_index`
- spread block
  - `us_minus_europe`

Then `v13_l3` uses:

`c_relative,obs,t ~ StudentT_nu(mu_relative,t, diag(d_relative,t))`

`c_spread,obs,t ~ StudentT_nu(mu_spread,t, d_spread,t)`

with two weights:

- `w_relative`
- `w_spread`

and composite objective:

`L(theta) = L_raw(theta) + w_relative * L_relative(theta) + w_spread * L_spread(theta)`

with the intended regime still:

`w_spread > w_relative`

## What `v13_l3` Tests

`v13_l3` tests whether Family 13 should regularize the **index-relative
subspace** rather than the broad level baskets themselves.

So the question is:

- can the same probabilistic raw-space model become more useful on the index
  slice
- by keeping the auxiliary term focused on relative regional structure
- without using any trading metric inside model selection

This keeps the Family 13 thesis intact:

- do not replace the main return model
- do not optimize for portfolio performance inside model research
- only add a weak auxiliary probabilistic measurement where the posterior-signal
  diagnostics suggest the remaining index misspecification still lives
