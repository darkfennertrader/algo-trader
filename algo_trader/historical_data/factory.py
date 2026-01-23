from __future__ import annotations

from algo_trader.historical_data.config import HistoricalRequestConfig
from algo_trader.historical_data.env import optional_env
from algo_trader.historical_data.protocols import HistoricalDataProvider
from algo_trader.historical_data.providers.ib.provider import build_ib_provider


def _resolve_provider_name(request_config: HistoricalRequestConfig) -> str:
    env_value = optional_env("HISTORICAL_DATA_PROVIDER")
    if env_value:
        return env_value
    return request_config.provider


def build_historical_data_provider(
    request_config: HistoricalRequestConfig,
) -> HistoricalDataProvider:
    provider_name = _resolve_provider_name(request_config).lower()
    if provider_name == "ib":
        return build_ib_provider()
    raise ValueError(f"Unsupported historical data provider '{provider_name}'")
