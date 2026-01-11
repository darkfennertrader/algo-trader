# Repository Agent Instructions

- Package management and tooling: always use `uv` for Python tasks (installs, running scripts, dependency updates). Do not use `pip`, `pipenv`, `poetry`, or `venv` directly.
- Use `uv run` for executing project commands and `uv add` / `uv remove` for dependency changes to keep `pyproject.toml` in sync.
