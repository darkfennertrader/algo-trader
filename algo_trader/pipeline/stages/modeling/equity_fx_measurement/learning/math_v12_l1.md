# Family 12: Equity-FX Measurement Family

`v12_l1` tests a new hypothesis derived from the full post-`v4` research line:
the remaining index error is partly an **equity-versus-currency measurement**
problem, not only an index-dependence problem.

The key idea is:

- keep the trusted `v4_l1` backbone intact
- keep the FX block intact
- keep the commodity block intact
- keep raw index returns in the main likelihood
- replace only the latest index-local patching mindset with a soft
  equity-plus-FX measurement layer for non-US indices

In `v12_l1`, the index block is treated as a hybrid of:

- primitive equity-style states
- composite measurement rows
- explicit FX-translation states tied to the FX-cross regime

The latent index-state vector is:

- `q_us_style`
- `q_euro_local`
- `q_euro_periphery`
- `q_uk_ch_local`
- `q_eur_translation`
- `q_gbp_chf_translation`

The intended economic interpretation is:

- US indices are mostly local equity measurements, plus a small style channel
- euro-area indices mix local equity structure with EUR translation into the
  USD return space
- UK / Swiss indices mix local equity structure with GBP / CHF translation
  into the USD return space

The observation block is therefore:

`r_idx,t = mu_idx,t + base_v4_l1_idx,t + H q_t + eps_idx,t`

where:

- `base_v4_l1_idx,t` is the trusted raw-space `v4_l1` index component
- `H q_t` is the new equity-FX measurement block
- `eps_idx,t` is index-specific residual noise

The crucial design choice is that the new translation states are not generic
index-local factors:

- local equity states scale with the existing index regime channel
- translation states scale with the existing FX-cross regime channel

So this family should be read as a genuine new hypothesis:

- not another copula extension
- not another hard transformed-basis family
- not another pure measurement family
- but an index measurement family that explicitly uses the unified FX lesson

Priors are strongly regularized:

- composite rows (`IBUS500`, `IBEU50`) get tight residual priors
- loading deviations are strongly shrunk, especially on composite rows
- state correlation is small and regularized
- both the base `v4_l1` t-copula overlay and the new measurement block are
  kept modest

The branch only wins if it improves on `v4_l1` jointly in:

- aggregate calibration
- `indices`
- `us_index`
- `europe_index`
- `us_minus_europe`
- `index_equal_weight`
- residual dependence after whitening
