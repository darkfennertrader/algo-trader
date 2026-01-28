from __future__ import annotations

import re

from algo_trader.domain import ConfigError

_PIPELINE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_pipeline_name(raw: str) -> str:
    normalized = raw.strip()
    if not normalized:
        return "debug"
    if not _PIPELINE_PATTERN.match(normalized):
        raise ConfigError(
            "pipeline contains invalid characters",
            context={"pipeline": raw},
        )
    return normalized
