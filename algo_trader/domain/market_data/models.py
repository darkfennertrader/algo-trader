from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Sequence, TypeAlias


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    sec_type: str
    currency: str
    exchange: str
    what_to_show: str
    asset_class: str


@dataclass(frozen=True)
class HistoricalDataRequest:
    tickers: Sequence[TickerConfig]
    duration: str
    bar_size: str
    end_date_time: str
    window_label: str | None = None


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


BarSeries: TypeAlias = Sequence[Bar]


@dataclass(frozen=True)
class RequestOutcome:
    symbol: str
    bars: int = 0
    error_code: int | None = None
    error_message: str | None = None

    def with_error(self, error_code: int, message: str | None) -> "RequestOutcome":
        return RequestOutcome(
            symbol=self.symbol,
            bars=self.bars,
            error_code=error_code,
            error_message=message,
        )

    def with_bars(self, bars: int) -> "RequestOutcome":
        return RequestOutcome(
            symbol=self.symbol,
            bars=bars,
            error_code=self.error_code,
            error_message=self.error_message,
        )


@dataclass(frozen=True)
class HistoricalDataResult:
    bars_by_symbol: Mapping[str, BarSeries]
    outcomes: Mapping[str, RequestOutcome]
