from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from algo_trader.domain import ConfigError, EnvVarError
from algo_trader.domain.market_data import HistoricalDataProvider
from algo_trader.infrastructure import optional_env, require_env, require_int
from algo_trader.providers.historical import IbConnectionSettings, build_ib_provider
from .config import HistoricalRequestConfig

_EUROPE_TZ = ZoneInfo("Europe/Paris")
_ET_TZ = ZoneInfo("America/New_York")
_DAILY_RESET_START = time(6, 25)
_DAILY_RESET_END = time(7, 45)
_WEEKEND_RESET_START = time(23, 0)
_WEEKEND_RESET_END = time(3, 0)
_WEEKEND_RESET_DURATION = timedelta(hours=4)

logger = logging.getLogger(__name__)


def _resolve_provider_name(request_config: HistoricalRequestConfig) -> str:
    env_value = optional_env("HISTORICAL_DATA_PROVIDER")
    if env_value:
        return env_value
    return request_config.provider


def _load_ib_connection_settings() -> IbConnectionSettings:
    host = require_env("IB_HOST")
    port = require_int("IB_PORT")
    client_id = require_int("IB_CLIENT_ID")
    return IbConnectionSettings(host=host, port=port, client_id=client_id)


def is_daily_reset_window(now_local: datetime) -> bool:
    if now_local.weekday() == 5:
        return False
    current = now_local.time()
    return _DAILY_RESET_START <= current < _DAILY_RESET_END


def weekend_reset_window(
    now_et: datetime,
) -> tuple[datetime, datetime] | None:
    weekday = now_et.weekday()
    current = now_et.time()
    if weekday == 4 and current >= _WEEKEND_RESET_START:
        start_date = now_et.date()
    elif weekday == 5 and current < _WEEKEND_RESET_END:
        start_date = now_et.date() - timedelta(days=1)
    else:
        return None
    start_et = datetime.combine(start_date, _WEEKEND_RESET_START, tzinfo=_ET_TZ)
    return start_et, start_et + _WEEKEND_RESET_DURATION


def _warn_if_ibkr_reset_window() -> None:
    now_local = datetime.now(tz=_EUROPE_TZ)
    daily_active = is_daily_reset_window(now_local)
    weekend_window = weekend_reset_window(now_local.astimezone(_ET_TZ))
    if not daily_active and weekend_window is None:
        return

    details: list[str] = []
    if daily_active:
        start_local = datetime.combine(
            now_local.date(), _DAILY_RESET_START, tzinfo=_EUROPE_TZ
        )
        end_local = datetime.combine(
            now_local.date(), _DAILY_RESET_END, tzinfo=_EUROPE_TZ
        )
        tz_name = start_local.tzname() or "local"
        details.append(
            f"daily reset {start_local:%H:%M}-{end_local:%H:%M} {tz_name}"
        )
    if weekend_window is not None:
        start_et, end_et = weekend_window
        start_local = start_et.astimezone(_EUROPE_TZ)
        end_local = end_et.astimezone(_EUROPE_TZ)
        tz_name = start_local.tzname() or "local"
        details.append(
            f"weekend reset {start_local:%a %H:%M}-{end_local:%a %H:%M} {tz_name}"
        )
    warning_line = (
        "Warning: IBKR reset window active (local %s). "
        "API operations may fail or be unstable."
    )
    separator = "*" * 80
    logger.warning(
        "\n%s\n%s\n%s\n",
        separator,
        warning_line % "; ".join(details),
        separator,
    )


def _build_ib_provider(
    _request_config: HistoricalRequestConfig,
) -> HistoricalDataProvider:
    _warn_if_ibkr_reset_window()
    connection = _load_ib_connection_settings()
    max_parallel_requests = require_int("MAX_PARALLEL_REQUESTS")
    if max_parallel_requests < 1:
        raise EnvVarError(
            "MAX_PARALLEL_REQUESTS must be at least 1",
            context={
                "env_var": "MAX_PARALLEL_REQUESTS",
                "value": str(max_parallel_requests),
            },
        )
    return build_ib_provider(connection, max_parallel_requests)


ProviderBuilder = Callable[[HistoricalRequestConfig], HistoricalDataProvider]

_PROVIDER_BUILDERS: dict[str, ProviderBuilder] = {
    "ib": _build_ib_provider,
}


def build_historical_data_provider(
    request_config: HistoricalRequestConfig,
) -> HistoricalDataProvider:
    provider_name = _resolve_provider_name(request_config).lower()
    builder = _PROVIDER_BUILDERS.get(provider_name)
    if builder is not None:
        return builder(request_config)
    raise ConfigError(
        f"Unsupported historical data provider '{provider_name}'",
        context={"provider": provider_name},
    )
