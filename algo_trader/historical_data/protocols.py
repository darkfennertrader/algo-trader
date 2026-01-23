from __future__ import annotations

from typing import Mapping, Protocol

import pandas as pd

from algo_trader.historical_data.models import HistoricalDataRequest, HistoricalDataResult


class HistoricalDataProvider(Protocol):
    def name(self) -> str:
        """Identifier for the provider implementation."""
        ...

    def fetch(self, request: HistoricalDataRequest) -> HistoricalDataResult:
        """Return historical bars for the requested symbols."""
        ...
