from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from algo_trader.domain import ConfigError
from .validation import validate_no_unknown_params


@dataclass(frozen=True)
class IdentityPreprocessorConfig:
    copy: bool = False


class IdentityPreprocessor:
    def process(
        self, data: pd.DataFrame, *, params: Mapping[str, str]
    ) -> pd.DataFrame:
        config = _parse_config(params)
        if config.copy:
            return data.copy()
        return data


def _parse_config(params: Mapping[str, str]) -> IdentityPreprocessorConfig:
    copy_value = params.get("copy")
    if copy_value is None:
        validate_no_unknown_params(params, allowed={"copy"})
        return IdentityPreprocessorConfig()
    validate_no_unknown_params(params, allowed={"copy"})
    return IdentityPreprocessorConfig(copy=_parse_bool(copy_value))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ConfigError(
        "copy must be a boolean",
        context={"value": value},
    )
