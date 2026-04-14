Family 16, version v16_l1, keeps the trusted dependence-layer raw-return
backbone and adds one auxiliary index-spread head built from curated pairwise
index coordinates.

High-level structure:

- head 1: raw next-week returns for the full universe
- head 2: pairwise next-week index spreads on the cleaned reduced index set

The model still learns one shared latent posterior for the full universe.
The difference from Family 15 is the definition of the auxiliary object.
Instead of supervising coarse relative baskets, v16_l1 supervises pairwise
index spreads that are closer to the weekly discrimination problem we want the
posterior to solve.

Let y_t be the raw asset-return vector at week t, and let y_index_t be the
subvector restricted to index assets.

The raw-return head is unchanged:

- y_t is modeled by the dependence-layer backbone
- posterior prediction still yields samples, mean, and covariance for raw
  returns

For the auxiliary head, define a reduced-universe pairwise basis matrix P.
Each column of P is one selected spread vector over the index block, for
example:

- IBUS30 minus IBUST100
- IBUST100 minus IBDE40
- IBUST100 minus IBFR40
- IBUST100 minus IBGB100
- IBUST100 minus IBES35
- IBUST100 minus IBNL25
- IBUST100 minus IBCH20

The pairwise auxiliary coordinates are:

- z_t = transpose(y_index_t) * P

Given the raw-head posterior moments for the index block:

- mu_index_t
- Sigma_index_t

the implied auxiliary moments are:

- mu_z_t = transpose(mu_index_t) * P
- Sigma_z_t = transpose(P) * Sigma_index_t * P

As in Family 14 and Family 15, the auxiliary coordinates are standardized
robustly with median and MAD so that the auxiliary likelihood does not become
dominated by scale differences across coordinates.

The auxiliary observation likelihood is then applied to:

- the curated pairwise coordinates
- the remaining residual coordinates that complete the basis

with separate weights:

- pairwise_obs_weight
- residual_obs_weight

Interpretation:

- the raw head keeps the model anchored to the original unified forecasting
  problem
- the pairwise head asks the same latent backbone to also explain direct
  weekly index-against-index spreads

The hypothesis is:

- on the cleaned reduced index universe, pairwise spread supervision is closer
  to the true weekly cross-sectional ordering problem than the coarse relative
  basket objects used in Families 14 and 15

Success criterion:

- improve index-slice posterior signal, especially rank IC and monetizable
  spread diagnostics, without using trading metrics inside model selection
