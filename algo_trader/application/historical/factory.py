from __future__ import annotations

from typing import Callable

from algo_trader.domain import ConfigError, EnvVarError
from algo_trader.domain.market_data import HistoricalDataProvider
from algo_trader.infrastructure import optional_env, require_env, require_int
from algo_trader.providers.historical import IbConnectionSettings, build_ib_provider
from .config import HistoricalRequestConfig


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


def _build_ib_provider(
    _request_config: HistoricalRequestConfig,
) -> HistoricalDataProvider:
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
