from __future__ import annotations

from typing import Protocol

from .models import HistoricalDataRequest, HistoricalDataResult


class HistoricalDataProvider(Protocol):
    def name(self) -> str:
        """Identifier for the provider implementation."""
        ...

    def fetch(self, request: HistoricalDataRequest) -> HistoricalDataResult:
        """Return historical bars for the requested symbols."""
        ...
