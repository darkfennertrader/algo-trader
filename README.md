# Algo Trader

Quickstart:
- Run the CLI: `uv run algotrader [command]`.
- Install deps: `uv sync --all-groups` (installs runtime + dev deps like pytest).
- Run tests: `uv run pytest`.
- Add runtime deps: `uv add <package>`.
- Add dev deps (lint/type/etc.): `uv add <package> --group dev`.

## Documentation
- Start here: `docs/README.md`.
- Coding rules and workflow: `AGENTS.md`.
- Third-party type stubs live under `typings/` (used by Pyright).

Note: Simulation models and guides implement `PyroModel`/`PyroGuide` using the
`ModelBatch` dataclass (features/targets/mask). See `docs/workflows.md` for
the current model/guide contract.

Note: Hyperparameter tuning uses `tuning.space` in
`config/model_selection.yml`. Candidate configs are persisted to
`hyperparams_space.json` under the dataset version `inner/` directory derived
from `data.paths.tensor_path` and reused if present.

LLM reading order:
- `AGENTS.md` - coding rules and workflow
- `docs/repo_structure.md` - architecture and module map
- `docs/configuration.md` - config and env setup
- `docs/workflows.md` - common tasks and CLI
- `docs/feature_engineering.md` - feature groups and definitions
- `docs/simulation/nestedcv.md` - nested CV details
- `docs/simulation/nestedcv_sequence.md` - nested CV sequence flow
