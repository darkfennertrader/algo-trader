from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from algo_trader.domain import ConfigError
from algo_trader.infrastructure import require_env
from algo_trader.providers.exogenous import FredSeriesProvider
from .config import FredRequestConfig, FredSeriesConfig


class ExogenousDataProvider(Protocol):
    def name(self) -> str:
        ...

    def fetch_series(
        self,
        *,
        series: FredSeriesConfig,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        ...


ProviderBuilder = Callable[[FredRequestConfig], ExogenousDataProvider]


def _build_fred_provider(_config: FredRequestConfig) -> ExogenousDataProvider:
    api_key = require_env("FRED_API_KEY")
    return FredSeriesProvider(api_key=api_key)


_PROVIDER_BUILDERS: dict[str, ProviderBuilder] = {
    "fred": _build_fred_provider,
}


def build_exogenous_provider(config: FredRequestConfig) -> ExogenousDataProvider:
    builder = _PROVIDER_BUILDERS.get(config.provider)
    if builder is not None:
        return builder(config)
    raise ConfigError(
        f"Unsupported exogenous provider '{config.provider}'",
        context={"provider": config.provider},
    )
