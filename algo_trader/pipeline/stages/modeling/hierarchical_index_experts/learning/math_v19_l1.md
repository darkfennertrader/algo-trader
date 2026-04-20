Family 19 keeps the trusted v4_l1 raw-return backbone and adds a broader
reduced-index auxiliary head built from three soft experts.

The model still predicts the full raw return vector y_t for all assets through
the same latent probabilistic backbone used by the dependence-layer follow-up
families.

On top of that, it defines an index-only auxiliary coordinate system:

- broad expert:
  z_broad_t = average of the active reduced-universe index returns
- anchor-pair expert:
  z_anchor_t = r_IBCH20_t - r_IBDE40_t
- residual expert:
  the remaining orthonormal residual index coordinates

The basis is completed to a full orthonormal system over the active index
block, so the auxiliary head still spans the reduced-universe index subspace,
but the first coordinates are deliberately ordered by research priority:

1. broad reduced-index view
2. durable anchor-pair view
3. residual cleanup view

The auxiliary likelihood is grouped into:

- hierarchical_index_broad_obs
- hierarchical_index_anchor_obs
- hierarchical_index_residual_obs

The family-specific configuration exposes three raw expert weights:

- broad_obs_weight
- anchor_pair_obs_weight
- residual_obs_weight

These are converted into a static soft mixture by normalizing them and scaling
them by the common auxiliary observation weight obs_weight:

- effective_broad_weight = obs_weight * broad_obs_weight / total
- effective_anchor_weight = obs_weight * anchor_pair_obs_weight / total
- effective_residual_weight = obs_weight * residual_obs_weight / total

where:

- total = broad_obs_weight + anchor_pair_obs_weight + residual_obs_weight

So v19_l1 is a broad hierarchical opener, but it is still conservative:

- no latent expert switch
- no explicit regime gate
- no second anchor pair yet

The thesis is:

- broad reduced-index structure still matters
- the durable anchor pair should receive extra supervision weight
- the remaining index variation should be regularized, not ignored

This keeps the model broad while reusing the strongest lessons from the
reduced-universe pair research.
