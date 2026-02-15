# Algo Trader

Basic commands using uv:

- Run the CLI: `uv run algotrader [command]`.
- Install deps: `uv sync --all-groups` (installs runtime + dev deps like pytest).
- Run tests: `uv run pytest`.
- Add runtime deps: `uv add <package>`.
- Add dev deps (lint/type/etc.): `uv add <package> --group dev`.

## CLI
- Wizard (interactive command builder): `uv run algotrader wizard`
- Default pipeline (placeholder): `uv run algotrader`
- Historical download: `uv run algotrader historical`
- Data cleaning: `uv run algotrader data_cleaning --start YYYY-MM --end YYYY-MM --return-type simple --assets EUR.USD,IBUS30`
- Data processing: `uv run algotrader data_processing` (defaults to `identity`; add `--preprocessor <name>` to override)
- Feature engineering: `uv run algotrader feature_engineering`
- Modeling (Pyro inference): `uv run algotrader modeling --model normal --guide normal_mean_field`
- Simulation (nested CV): `uv run algotrader simulation --simulation-config config/model_selection.yml`
- Backtest (placeholder): `uv run algotrader backtest`

### Nested CV simulation

The `simulation` command runs a **nested time‑series CV**:

- **Outer loop:** walk‑forward (train on past, test on future group)
- **Inner loop:** CPCV over past groups with purge + embargo for leakage control

Details and flow diagrams:
- `docs/nestedcv.md`
- `docs/nestedcv_sequence.md`

Data cleaning return options:
- `--return-type simple` (default): `(P_t / P_{t-1}) - 1`
- `--return-type log`: `log(P_t) - log(P_{t-1})`
- `--start` is required; `--end` is optional (defaults to the latest available data).
- Data cleaning runs per-asset loading in parallel using `max(1, os.cpu_count() - 1)` workers, capped by asset count.
- Return frequency is weekly: weekly returns from hourly closes grouped by week starting Monday; uses the first available price in the week (prefers Monday) and last available price (prefers Friday), labeled with the latest timestamp in that week across assets

### Feature engineering

Compute feature groups from data-cleaning outputs (weekly momentum/mean-reversion/breakout plus daily-input volatility and regime features, all output weekly), plus cross-sectional transforms.

```bash
uv run algotrader feature_engineering \
  --group momentum
```

Args and defaults:
- Horizons are fixed to each group's defaults (see the group sections below). The CLI does not accept horizon overrides.
- `group`: feature group to compute (repeatable; default = all registered groups).
   Valid values: `momentum`, `mean_reversion`, `breakout`, `cross_sectional`, `volatility`, `seasonal`, `regime`, `all`.
   `all` runs the non-cross-sectional groups in parallel using (logical CPUs - 1) workers, then computes `cross_sectional` last.
- `feature`: feature key within a group (repeatable; default = group default set).
  - momentum keys: `vol_scaled_momentum`, `ema_spread`
  - mean_reversion keys: `z_price_ema`, `z_price_med`, `donch_pos`, `rsi_centered`, `rev`, `shock`, `range_pos`, `range_z`
  - breakout keys: `brk_up`, `brk_dn`
  - cross_sectional keys: `cs_rank`
  - volatility keys: `vol_cc_d`, `atrp_d`, `vol_regime_cc`, `vov_norm`, `vol_ts_cc_1w_4w`, `vol_ts_cc_4w_26w`, `down_up_vol_ratio_4w`, `realized_skew_d_12w`, `tail_5p_sigma_ratio_12w`, `jump_freq_4w`
  - seasonal keys: `dow_alpha`, `dow_spread`
  - regime keys: `glob_vol_cc_d`, `glob_vol_regime_cc`, `glob_corr_mean`, `glob_pc1_share`, `glob_disp_ret`, `glob_disp_mom`, `glob_disp_vol`

Outputs (per group):
- `FEATURE_STORE_SOURCE/features/YYYY-WW/<group>/features.csv` (MultiIndex columns: asset, feature)
- `FEATURE_STORE_SOURCE/features/YYYY-WW/<group>/features_tensor.pt` (TxAxF values, timestamps, missing_mask)
- `FEATURE_STORE_SOURCE/features/YYYY-WW/<group>/metadata.json`

Notes:
- Weekly inputs are loaded from `DATA_LAKE_SOURCE/YYYY-WW/weekly_ohlc.csv`.
- Volatility features use daily inputs from `DATA_LAKE_SOURCE/YYYY-WW/daily_ohlc.csv` and are sampled to weekly output.
- Seasonal features use daily inputs from `DATA_LAKE_SOURCE/YYYY-WW/daily_ohlc.csv` and are sampled to weekly output.
- Cross-sectional features use weekly inputs because they derive from momentum and mean-reversion base features.
- Regime features use daily inputs for volatility, correlation, and one-factor-ness plus weekly inputs for return/momentum dispersion; outputs are broadcast to each asset.
- Missing weekly OHLC values for any asset will raise an error.
- Metadata includes `feature_name_units` and `days_to_weeks` for horizon mapping.

#### Breakout group features

Computed on weekly OHLC data per asset (horizons shown in weeks):

Outputs:
- `brk_up_4w`, `brk_up_26w`: 1 if close_t > max(High_{t-h..t-1}), else 0.
- `brk_dn_4w`, `brk_dn_26w`: 1 if close_t < min(Low_{t-h..t-1}), else 0.

#### Cross-sectional group features

Computed on weekly output by applying cross-sectional transforms to a fixed base feature set. For each week t, U_t is the assets with valid base features and N_eff(t) is the count.

`cs_rank_*` is the cross-sectional percentile rank at week t using average rank for ties: (rank - 0.5) / N_eff(t).

Outputs:
- `cs_rank_z_mom_4w`, `cs_rank_z_mom_12w`
- `cs_rank_z_price_ema_26w`
- `cs_rank_z_price_med_26w`
- `cs_rank_rev_1w`
- `cs_rank_shock_4w`

Base features (examples: `cs_rank_z_mom_4w`, `cs_rank_rev_1w`):

Trend:
- `z_mom_4w`, `z_mom_12w`

Mean-reversion:
- `z_price_ema_26w`
- `z_price_med_26w`
- `rev_1w`
- `shock_4w`

#### Mean-reversion group features

Computed on weekly OHLC data per asset (horizons shown in weeks).

Outputs:
- `z_price_ema_12w`, `z_price_ema_26w`: (p_t − EMA_h(p)) / (EWM_std_h(p) + eps), where p_t = log(C_t^w) and both EWM mean/std use halflife = h.
- `z_price_med_26w`: (p_t − median(p_{t−25..t})) / (1.4826 * MAD_{26}(p) + eps), with MAD computed on log closes in the 26w window.
- `donch_pos_4w`: (C_t^w − low_4w(t)) / (high_4w(t) − low_4w(t) + eps), where high_4w/low_4w are rolling 4‑week High/Low extrema.
- `rsi_centered_4w`: RSI_4w(C^w) − 50, computed on weekly closes.
- `rev_1w`: −log(C_{t−1}^w / C_{t−2}^w), i.e., negative 1‑week log return lagged one week.
- `shock_4w`: r_{t−1}^w / (std(r_{t−4..t−1}^w) + eps), where r^w is weekly log returns.
- `range_pos_1w`: (C_t^w − mid_t) / (0.5 * range_t + eps), mid_t = (H_t^w + L_t^w)/2, range_t = H_t^w − L_t^w.
- `range_z_12w`: (range_t − mean(range_{t−11..t})) / (std(range_{t−11..t}) + eps), where range_t = H_t^w − L_t^w.

#### Momentum group features

Computed on weekly OHLC data per asset (horizons shown in weeks; returns are log for momentum features):

Outputs:
- `z_mom_4w`, `z_mom_12w`, `z_mom_26w`: log cumulative return divided by (sigma_ref * sqrt(H) + eps), where sigma_ref is the 26w rolling std of weekly log returns.
- `ema_spread_4w_26w`, `ema_spread_12w_26w`: (EMA_s − EMA_l) on weekly closes normalized by ATR_l on weekly bars (longer horizon).

#### Regime group features

Computed on daily inputs (volatility, correlation, one-factor-ness) and weekly inputs (dispersion), then broadcast to all assets at week t (horizons shown in weeks; defaults = 1, 4, 12, 26).

Outputs:
- `glob_vol_cc_d_4w`: mean close-to-close realized vol across assets over daily returns, sampled weekly.
- `glob_vol_regime_cc_4w_26w`: log ratio of current 4w global vol vs the prior 26w median of 4w global vol (eps=1e-6).
- `glob_corr_mean_12w`: average pairwise correlation across assets over the last 12w of daily returns.
- `glob_pc1_share_12w`: leading eigenvalue share of the 12w daily-return covariance matrix.
- `glob_disp_ret_1w`: cross-sectional dispersion of 1w log returns using MAD scaling (1.4826 * MAD).
- `glob_disp_mom_12w`: cross-sectional dispersion of 12w momentum using MAD scaling (1.4826 * MAD).
- `glob_disp_vol_4w`: cross-sectional dispersion of 4w close-to-close vol using MAD scaling (1.4826 * MAD).

#### Seasonal group features

Computed on daily OHLC data per asset, then sampled to weekly output (horizons shown in weeks; defaults = 26).
Daily returns are log returns on daily closes: r_d = log(C_d) − log(C_{d−1}).
For each weekly output t and horizon h, use the last h completed weeks (Mon–Fri) and compute the weekday means:

Outputs:
- `dow_alpha_Mon_26w`, `dow_alpha_Fri_26w`
- `dow_spread_26w`: max weekday mean − min weekday mean over the last 26 weeks

#### Volatility group features

Computed on daily OHLC data per asset, then sampled to weekly output (horizons shown in weeks; defaults = 1, 4, 12, 26).
Missing daily OHLC rows are dropped before indicator computation; data-quality ratios are written to `goodness.json` as valid / horizon for volatility and regime and missing / horizon for weekly groups. Timestamp keys in `goodness.json` use `YYYY-MM-DD_HH:MM:SS+00:00`.

Outputs:
- `vol_cc_d_4w`, `vol_cc_d_26w`: close-to-close realized vol over the stated horizon of daily returns.
- `atrp_d_4w`: ATR over the stated horizon of daily bars, divided by close.
- `vol_regime_cc_4w_26w`: current 4w vol vs median of prior 26 weeks of 4w vol history.
- `vov_norm_12w`: log((a_12w + eps) / (B + eps)), where a_12w is std(|r_t|) over last 12w daily returns and B is the median of the prior 26w of a_12w.
- `vol_ts_cc_1w_4w`: log ratio of 1w vs 4w close-to-close vol.
- `vol_ts_cc_4w_26w`: log ratio of 4w vs 26w close-to-close vol.
- `down_up_vol_ratio_4w`: log ratio of downside vs upside vol (eps=1e-6).
- `realized_skew_d_12w`: clipped realized skewness over 12 weeks.
- `tail_5p_sigma_ratio_12w`: log ratio of empirical 5% tail vs 1.645*sigma_12w (eps=1e-6).
- `jump_freq_4w`: fraction of last 4w days with |r| > 2*sigma_12w.

#### Feature reference

This section provides a quick explanation of each feature key (independent of horizon
suffixes).

Breakout:
- `brk_up`: 1 if close exceeds the prior rolling high over the horizon.
- `brk_dn`: 1 if close breaks below the prior rolling low over the horizon.

Cross-sectional:
- `cs_rank`: cross-sectional percentile rank at week t using average rank for ties.

Mean-reversion:
- `z_price_ema`: z-score of log price versus EWM mean with same halflife.
- `z_price_med`: z-score of log price versus rolling median using MAD-based scale.
- `donch_pos`: position of close within rolling high/low range (0 to 1).
- `rsi_centered`: RSI centered at zero (RSI minus 50).
- `rev`: negative of the prior horizon log return (lagged one week).
- `shock`: prior 1W log return scaled by 4W rolling std of 1W returns.
- `range_pos`: close position within current bar range using mid/half-range.
- `range_z`: z-score of weekly range versus rolling mean/std.

Momentum:
- `vol_scaled_momentum`: momentum divided by rolling std of 1W returns over the horizon.
- `ema_spread`: EMA differences normalized by ATR for the longer horizon.

Regime:
- `glob_vol_cc_d`: average close-to-close realized vol across assets (daily log returns), sampled weekly.
- `glob_vol_regime_cc`: log ratio of current 4w global vol vs prior 26w median of 4w global vol (eps=1e-6).
- `glob_corr_mean`: average pairwise correlation across assets over the horizon.
- `glob_pc1_share`: leading eigenvalue share of the return covariance matrix over the horizon.
- `glob_disp_ret`: cross-sectional dispersion of 1w log returns using MAD scaling (1.4826 * MAD).
- `glob_disp_mom`: cross-sectional dispersion of 12w momentum using MAD scaling (1.4826 * MAD).
- `glob_disp_vol`: cross-sectional dispersion of 4w close-to-close vol using MAD scaling (1.4826 * MAD).

Seasonal:
- `dow_alpha`: average daily log return for a weekday over the last h weeks.
- `dow_spread`: max weekday mean minus min weekday mean over the last h weeks.

Volatility:

  Volatility Level (log returns only):
- `vol_cc_d`: close-to-close realized vol over daily **log** returns.
- `atrp_d`: ATR over daily bars as a percent of price.
- `vol_regime_cc`: log((sigma_4w+eps)/(median_26w+eps)) using sigma_4w from log-return vol, eps=1e-6.
- `vov_norm`: normalized volatility-of-vol using |r_t| over 12w (log returns), normalized by the median of the prior 26w of 12w vol-of-vol and reported as a log-ratio with eps.

  Volatility term structure (ratios/slope):
- `vol_ts_cc_1w_4w`: log((v_1w+eps)/(v_4w+eps)) using v=max(vol, eps), eps=1e-6.
- `vol_ts_cc_4w_26w`: log((v_4w+eps)/(v_26w+eps)) using v=max(vol, eps), eps=1e-6.

  Volatility asymmetry and tails:
- `down_up_vol_ratio_4w`: log((d+eps)/(u+eps)) with v=max(vol, eps), eps=1e-6.
- `realized_skew_d_12w`: clipped to [-5, 5], computed from 12w returns.
- `tail_5p_sigma_ratio_12w`: log(((-q05)+eps)/(1.645*sigma_12w+eps)), eps=1e-6.
- `jump_freq_4w`: count(|r| > 2*sigma_12w) / 20 over the last 4w.


### Preprocessors

**Identity** (no-op, optional copy):

```bash
uv run algotrader data_processing --preprocessor identity
uv run algotrader data_processing --preprocessor identity --preprocessor-arg copy=true
```

Args and defaults:
- `copy`: `true` or `false` (optional, default = `false`)

**Z-score** (per-column normalization over a date range, defaults missing=zero):

```bash
uv run algotrader data_processing --preprocessor zscore \
  --preprocessor-arg start_date=YYYY-MM-DD \
  --preprocessor-arg end_date=YYYY-MM-DD \
  --preprocessor-arg missing=drop \
  --preprocessor-arg pipeline=my_pipeline
```

Args and defaults:
- `start_date`: `YYYY-MM-DD` (optional, default = full range)
- `end_date`: `YYYY-MM-DD` (optional, default = full range)
- `missing`: `zero` or `drop` (optional, default = `zero`)
- `pipeline`: `A-Za-z0-9._-` (optional, default = `debug`)

Outputs:
- `processed.csv` (main output)
- `processed_tensor.pt` (values + timestamps + missing_mask)
- `mean_ref.pt`, `std_ref.pt`
- `metadata.json`

**PCA** (z-score + PCA factors; choose k or variance, not both):

```bash
uv run algotrader data_processing --preprocessor pca \
  --preprocessor-arg k=5 \
  --preprocessor-arg missing=zero \
  --preprocessor-arg pipeline=my_pipeline
```

```bash
uv run algotrader data_processing --preprocessor pca \
  --preprocessor-arg variance=0.9 \
  --preprocessor-arg missing=drop
```

Args and defaults:
- `k`: positive integer (required if `variance` is not set)
- `variance`: float in `(0, 1]` (required if `k` is not set)
- `start_date`: `YYYY-MM-DD` (optional, default = full range)
- `end_date`: `YYYY-MM-DD` (optional, default = full range)
- `missing`: `zero` or `drop` (optional, default = `zero`)
- `pipeline`: `A-Za-z0-9._-` (optional, default = `debug`)

Outputs:
- `factors.csv` (main output)
- `factors.pt`
- `loadings.csv`, `loadings.pt`
- `eigenvalues.csv`
- `metadata.json`


## Directory structure

```
algo_trader/
  cli/
  application/
    data_cleaning/
    data_processing/
    feature_engineering/
    historical/
    modeling/
    simulation/
  domain/
    market_data/
    simulation/
  infrastructure/
    data/
    exporters/
  preprocessing/
  providers/
    historical/
    live/
  pipeline/
    stages/
```

- `cli/` parses command-line args and routes to use-cases; entrypoints only.
- `application/` orchestrates workflows and wires protocols to concrete providers.
- `application/data_cleaning/` builds weekly returns and OHLC from raw data.
- `application/data_processing/` runs preprocessors and writes processed feature sets.
- `application/feature_engineering/` computes feature groups from cleaned inputs.
- `application/historical/` handles historical data downloads and requests.
- `application/modeling/` runs inference workflows and model I/O.
- `application/simulation/` runs nested CV simulation and evaluation wiring.
- `domain/` holds core models and protocols; stays stable and vendor-agnostic.
- `domain/market_data/` domain models and interfaces for market data.
- `domain/simulation/` simulation interfaces and configuration types.
- `infrastructure/` shared plumbing (env, logging, storage, event bus).
- `infrastructure/data/` shared data utilities, schemas, and storage helpers.
- `infrastructure/exporters/` exporters for saving or emitting artifacts.
- `preprocessing/` feature preprocessing implementations (z-score, PCA, etc.).
- `providers/` vendor adapters like IB or Alpaca.
- `providers/historical/` historical data provider adapters.
- `providers/live/` live data/execution provider adapters.
- `pipeline/` composable stages (preprocess, features, model, metrics).
- `pipeline/stages/` concrete pipeline stages and registries.

## Configuration

- Copy `config/tickers.example.yml` to `config/tickers.yml` and edit `tickers`, `duration`
  `bar_size`, and `what_to_show` for your run.
- Set `MAX_PARALLEL_REQUESTS`, `IB_HOST`, `IB_PORT`, and `IB_CLIENT_ID` in your `.env`
  (see `.env.example`).
- For data cleaning, set `DATA_SOURCE` and `DATA_LAKE_SOURCE` in `.env`. Output is
  located at `DATA_LAKE_SOURCE` under `.env` and is written to
  `DATA_LAKE_SOURCE/YYYY-WW/` (includes `returns.csv`, `weekly_ohlc.csv`,
  `weekly_ohlc_meta.json`, `daily_ohlc.csv`, `daily_ohlc_meta.json`).
  `daily_ohlc_meta.json` includes per-asset missing days (Mon–Fri only), recorded at
  the last available hour across all assets for that day, plus monthly missing counts.
- For data processing, set `DATA_LAKE_SOURCE` and `FEATURE_STORE_SOURCE` in `.env`.
  Output is located at `FEATURE_STORE_SOURCE` under `.env`. The command selects the
  latest `YYYY-WW` directory, reads `returns.csv`, and writes `processed.csv` to the
  feature store.
- For modeling/inference, set `FEATURE_STORE_SOURCE` (input) and `MODEL_STORE_SOURCE`
  (outputs) in `.env`. Output is located at `MODEL_STORE_SOURCE` under `.env`. The
  command reads the latest prepared data from the feature store (or `--input`), then
  writes parameter outputs to the model store.
- For simulation, edit `config/model_selection.yml` or pass
  `--simulation-config` to the CLI. The simulation expects `data.paths.tensor_path`
  to point at a tensor bundle (optionally with `missing_mask`) and uses
  `data.selection` for slicing. Nested CV parameters live under `cv` and
  `preprocessing`, with outer fold selection under `outer`.

## Adding a new preprocessor

1) Implement `Preprocessor` in `algo_trader/preprocessing` and return a pandas `DataFrame`
   with a UTC datetime index.
2) Register the implementation in `algo_trader/preprocessing/registry.py` by adding it to `default_registry()`.
3) Invoke it via CLI: `uv run algotrader data_processing --preprocessor <name> --preprocessor-arg key=value`.

## Adding a new model/guide (Pyro)

1) Implement a Pyro model and guide in `algo_trader/pipeline/stages/modeling/`.
   - Model must implement `PyroModel` and define `__call__(self, data: torch.Tensor) -> None`.
   - Guide must implement `PyroGuide` and define `__call__(self, data: torch.Tensor) -> None`.
   - See `algo_trader/pipeline/stages/modeling/dummy.py` for the current example.
2) Register both in `algo_trader/pipeline/stages/modeling/registry.py`:
   - Add `registry.register("<model_name>", YourModel())` in `default_model_registry()`.
   - Add `registry.register("<guide_name>", YourGuide())` in `default_guide_registry()`.
3) Expose the new classes in `algo_trader/pipeline/stages/modeling/__init__.py`.
4) Run it via CLI:
   - `uv run algotrader modeling --model <model_name> --guide <guide_name>`

## Simulation extensions

To extend simulation, define concrete dataset loaders in `algo_trader/infrastructure/data/` (e.g., a `PanelTensorDataset` loader), then register them in `algo_trader/application/simulation/registry.py` under the keys you want to reference; finally, wire those keys in `config/model_selection.yml` (`data.dataset_name`) so the CLI can build the simulation from config.

## Scripts

- Show the latest return tensor: `scripts/show_return_tensor.sh [--head N] [DATA_CLEANING_DIR]`
- Show the latest feature tensor for a group: `scripts/show_feature_tensor.sh --group GROUP [--head N]`
