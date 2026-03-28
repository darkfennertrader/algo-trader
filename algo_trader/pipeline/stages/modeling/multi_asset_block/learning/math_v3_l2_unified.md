# V3 L2 Unified

`v3_l2_unified` keeps the same unified block architecture as `v3_l1_unified`
but makes the mixed-equity side more expressive.

The first unified run showed two things:

- the overall block design is viable;
- the hardest names were often indices, and the tails were a bit too tight.

So `v3_l2_unified` is a conservative follow-up, not a full redesign.

## Change From V3 L1

The structure is unchanged:

- one static global low-rank block;
- dynamic FX broad block;
- dynamic FX cross block;
- dynamic index block;
- dynamic commodity block;
- static idiosyncratic nugget.

The changes are in the defaults:

- `global_factor_count` increases from `1` to `2`;
- `index_factor_count` increases from `1` to `2`;
- `sigma_index_scale` is widened;
- `global_b_scale` and `index_b_scale` are widened slightly;
- `index` regime `s_u_scale` is loosened slightly.

In one sentence:

`v3_l2_unified` gives the global/index side more room before changing the
overall unified geometry.

## Generative Structure

The generative equations remain the same as `v3_l1_unified`.

Mean:

```text
mu[t,a] =
  alpha[a]
  + X_asset[t,a] dot w[a]
  + X_global[t] dot beta[a]
```

Observation model:

```text
y[t] ~ LowRankMVN(
  loc = mu[t],
  cov_factor = [
    B_global,
    exp(0.5 h_fx_broad[t])  * M_fx        * B_fx_broad,
    exp(0.5 h_fx_cross[t])  * M_fx        * B_fx_cross,
    exp(0.5 h_index[t])     * M_index     * B_index,
    exp(0.5 h_commodity[t]) * M_commodity * B_commodity
  ],
  cov_diag = sigma_idio^2
)
```

The practical difference is that the global block and the index block now carry
more latent columns by default.

## Why This Is L2

`v3_l2_unified` is meant to answer a narrow question:

- was the first unified model mainly underfitting the index/global side?

If this version improves robustness on the mixed 5-year universe, then it is a
better base for the later 11-year unified validation.

If it does not, the next unified version should be more structural rather than
just widening defaults.
