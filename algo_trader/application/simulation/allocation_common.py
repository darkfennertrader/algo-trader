from __future__ import annotations

from typing import Any

from algo_trader.domain import ConfigError

VALID_ALLOCATION_FAMILIES = (
    "long_only",
    "equal_weight",
    "random",
    "de_risked",
    "skfolio_risk_budgeting",
)

VALID_PORTFOLIO_STYLES = (
    "long_only",
    "long_short_bounded_net",
    "factor_neutral_long_short",
)


def optional_float_value(raw: Any, *, location: str) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{location} must be numeric") from exc


__all__ = [
    "VALID_ALLOCATION_FAMILIES",
    "VALID_PORTFOLIO_STYLES",
    "optional_float_value",
]
