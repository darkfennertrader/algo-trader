# Algo Trader

Quickstart:
- Run the CLI: `uv run algotrader [command]`.
- Install deps: `uv sync --all-groups` (installs runtime + dev deps like pytest).
- Run tests: `uv run pytest`.
- Add runtime deps: `uv add <package>`.
- Add dev deps (lint/type/etc.): `uv add <package> --group dev`.

Ray Tune note: Ray workers build a fresh environment from default
dependencies. Local path dependencies that live outside the repo (like
`ibapi` from the IBKR SDK) are not available to those workers and can
cause failures. Keep `ibapi` vendored under `vendor/ibapi` so Ray can
package it with the working directory.

## Documentation
- Start here: `docs/README.md`.
- Coding rules and workflow: `AGENTS.md`.
- Third-party type stubs live under `typings/` (used by Pyright).

Note: Simulation models and guides implement `PyroModel`/`PyroGuide` using the
`ModelBatch` dataclass (features/targets/mask). See `docs/workflows.md` for
the current model/guide contract.

Note: Hyperparameter tuning uses `tuning.space` in
`config/simulation.yml`. Candidate configs are persisted to
`cv/candidates.json` under the simulation output directory
(`SIMULATION_SOURCE/<simulation_output_path or latest YYYY-WW>`).

Note: Post‑tune model selection computes CRPS/QL only for ES‑survivor
candidates; non‑survivors will show `NaN` for those metrics in
`outer/metrics.json`.

LLM reading order:
- `AGENTS.md` - coding rules and workflow
- `docs/repo_structure.md` - architecture and module map
- `docs/configuration.md` - config and env setup
- `docs/workflows.md` - common tasks and CLI
- `docs/feature_engineering.md` - feature groups and definitions
- `docs/simulation/nestedcv.md` - nested CV details
- `docs/simulation/nestedcv_sequence.md` - nested CV sequence flow
