from __future__ import annotations

import functools
import logging
from typing import Callable, Mapping, ParamSpec, TypeVar

from algo_trader.domain import EnvVarError
from .env import optional_env, require_env

LogContext = Mapping[str, str]
ContextFactory = Callable[..., LogContext]

P = ParamSpec("P")
R = TypeVar("R")

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _build_console_handler() -> logging.Handler:
    return logging.StreamHandler()


def _build_file_handler() -> logging.Handler:
    path = require_env("LOG_FILE_PATH")
    return logging.FileHandler(path)


_HANDLER_BUILDERS: dict[str, Callable[[], logging.Handler]] = {
    "console": _build_console_handler,
    "file": _build_file_handler,
}


def configure_logging() -> None:
    dest_value = optional_env("LOG_DEST") or "console"
    dest_names = [name.strip().lower() for name in dest_value.split(",") if name]
    if not dest_names:
        raise EnvVarError(
            "LOG_DEST must include at least one destination",
            context={"env_var": "LOG_DEST", "value": dest_value},
        )

    handlers: list[logging.Handler] = []
    for name in dest_names:
        builder = _HANDLER_BUILDERS.get(name)
        if builder is None:
            raise EnvVarError(
                "LOG_DEST must be 'console' or 'file'",
                context={"env_var": "LOG_DEST", "value": name},
            )
        handlers.append(builder())

    level = (optional_env("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=handlers,
    )
    logging.getLogger("ibapi").setLevel(logging.WARNING)


def log_boundary(
    name: str,
    *,
    logger: logging.Logger | None = None,
    context: LogContext | ContextFactory | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        log = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            resolved = _resolve_context(context, *args, **kwargs)
            _log_event(log, logging.INFO, "start", name, resolved)
            result = func(*args, **kwargs)
            _log_event(log, logging.INFO, "complete", name, resolved)
            return result

        return wrapper

    return decorator


def _resolve_context(
    context: LogContext | ContextFactory | None,
    *args: object,
    **kwargs: object,
) -> LogContext | None:
    if context is None:
        return None
    if callable(context):
        return context(*args, **kwargs)
    return context


def _log_event(
    log: logging.Logger,
    level: int,
    event: str,
    name: str,
    context: LogContext | None,
) -> None:
    if context:
        log.log(level, "event=%s boundary=%s context=%s", event, name, context)
    else:
        log.log(level, "event=%s boundary=%s", event, name)
