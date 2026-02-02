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
- Modeling (Pyro inference): `uv run algotrader modeling --model normal --guide normal_mean_field`
- Backtest (placeholder): `uv run algotrader backtest`

Data cleaning return options:
- `--return-type simple` (default): `(P_t / P_{t-1}) - 1`
- `--return-type log`: `log(P_t) - log(P_{t-1})`
- Return frequency is weekly: weekly returns from hourly closes grouped by week starting Monday; uses the first available price in the week (prefers Monday) and last available price (prefers Friday), labeled with the latest timestamp in that week across assets


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
    historical/
  domain/
  infrastructure/
  preprocessing/
  providers/
  pipeline/
```

- `cli/` parses command-line args and routes to use-cases. It contains entrypoints only.
- `application/` orchestrates workflows like historical downloads or backtests. It wires protocols to concrete providers.
- `domain/` holds core models and Protocols. It stays stable and avoids vendor or CLI dependencies.
- `infrastructure/` contains shared plumbing (env, logging, storage, event bus). It supports the app but isn’t domain logic.
- `providers/` houses vendor adapters like IB or Alpaca. These implementations satisfy domain protocols.
- `pipeline/` groups swappable stages and composition (preprocess, features, model, metrics). It lets you reorder or replace components easily.

## Configuration

- Copy `config/tickers.example.yml` to `config/tickers.yml` and edit `tickers`, `duration`
  `bar_size`, and `what_to_show` for your run.
- Set `MAX_PARALLEL_REQUESTS`, `IB_HOST`, `IB_PORT`, and `IB_CLIENT_ID` in your `.env`
  (see `.env.example`).
- For data cleaning, set `DATA_SOURCE` and `DATA_LAKE_SOURCE` in `.env`. Output is
  located at `DATA_LAKE_SOURCE` under `.env` and is written to
  `DATA_LAKE_SOURCE/YYYY-WW/returns.csv`.
- For data processing, set `DATA_LAKE_SOURCE` and `FEATURE_STORE_SOURCE` in `.env`.
  Output is located at `FEATURE_STORE_SOURCE` under `.env`. The command selects the
  latest `YYYY-WW` directory, reads `returns.csv`, and writes `processed.csv` to the
  feature store.
- For modeling/inference, set `FEATURE_STORE_SOURCE` (input) and `MODEL_STORE_SOURCE`
  (outputs) in `.env`. Output is located at `MODEL_STORE_SOURCE` under `.env`. The
  command reads the latest prepared data from the feature store (or `--input`), then
  writes parameter outputs to the model store.

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
