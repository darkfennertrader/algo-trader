from __future__ import annotations

import os

from dotenv import load_dotenv

from algo_trader.domain import EnvVarError

load_dotenv()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise EnvVarError(
            f"{name} must be set in .env", context={"env_var": name}
        )
    return value


def require_int(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise EnvVarError(
            f"{name} must be an integer in .env",
            context={"env_var": name, "value": raw},
        ) from exc


def optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def optional_int(name: str) -> int | None:
    value = optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise EnvVarError(
            f"{name} must be an integer in .env",
            context={"env_var": name, "value": value},
        ) from exc
