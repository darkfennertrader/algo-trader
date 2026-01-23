# Algo Trader

Basic commands using uv:

- Run the CLI: `uv run algotrader [command]`.
- Install deps: `uv sync --all-groups` (installs runtime + dev deps like pytest).
- Run tests: `uv run pytest`.
- Add runtime deps: `uv add <package>`.
- Add dev deps (lint/type/etc.): `uv add <package> --group dev`.

## CLI

- Default pipeline (placeholder): `uv run algotrader`
- Historical download: `uv run algotrader historical`
- Backtest (placeholder): `uv run algotrader backtest`

## Directory structure

```
algo_trader/
  cli/
  application/
  domain/
  infrastructure/
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
