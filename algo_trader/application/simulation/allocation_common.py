from __future__ import annotations

from typing import Any

from algo_trader.domain import ConfigError

VALID_ALLOCATION_FAMILIES = (
    "long_only",
    "equal_weight",
    "random",
    "de_risked",
    "herc",
    "schur",
    "skfolio_risk_budgeting",
)

VALID_PORTFOLIO_STYLES = (
    "long_only",
    "long_short_bounded_net",
    "factor_neutral_long_short",
)

VALID_SKFOLIO_DISTANCE_ESTIMATORS = (
    "pearson",
    "mutual_information",
)

VALID_SKFOLIO_RISK_MEASURES = (
    "mean_absolute_deviation",
    "first_lower_partial_moment",
    "variance",
    "semi_variance",
    "cvar",
    "evar",
    "worst_realization",
    "cdar",
    "max_drawdown",
    "average_drawdown",
    "edar",
    "ulcer_index",
    "gini_mean_difference",
    "value_at_risk",
    "drawdown_at_risk",
    "entropic_risk_measure",
    "fourth_central_moment",
    "fourth_lower_partial_moment",
    "kurtosis",
    "skew",
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
    "VALID_SKFOLIO_DISTANCE_ESTIMATORS",
    "VALID_SKFOLIO_RISK_MEASURES",
    "optional_float_value",
]
