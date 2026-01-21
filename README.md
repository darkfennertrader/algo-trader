# Algo Trader

Basic commands using uv:

- Run the app: `uv run algo-trader`.
- Install deps: `uv sync --all-groups` (installs runtime + dev deps like pytest).
- Run tests: `uv run pytest`.
- Add runtime deps: `uv add <package>`.
- Add dev deps (lint/type/etc.): `uv add <package> --group dev`.

## Configuration

- Copy `config/tickers.example.yml` to `config/tickers.yml` and edit `tickers`, `duration`, `bar_size`, and `what_to_show` for your run.
- Override the path with `TICKER_CONFIG_PATH` if you keep multiple ticker files.
- Set `MAX_PARALLEL_REQUESTS` in your `.env` (required for the IB client throttle).
