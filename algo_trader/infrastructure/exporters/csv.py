from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from algo_trader.domain import ExportError
from algo_trader.domain.market_data import (
    BarSeries,
    HistoricalDataRequest,
    HistoricalDataResult,
    TickerConfig,
)
from algo_trader.infrastructure.data import symbol_directory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CsvExportSettings:
    output_root: Path
    year: int
    month: int


class CsvHistoricalDataExporter:
    def __init__(self, output_root: Path, year: int, month: int) -> None:
        self._settings = CsvExportSettings(
            output_root=output_root,
            year=year,
            month=month,
        )
        self._ensure_output_root()

    def name(self) -> str:
        return "csv"

    def export(
        self,
        request: HistoricalDataRequest,
        result: HistoricalDataResult,
    ) -> None:
        for ticker in request.tickers:
            bars = result.bars_by_symbol.get(ticker.symbol, [])
            self._export_symbol(ticker, bars)

    def _ensure_output_root(self) -> None:
        path = self._settings.output_root
        if path.exists() and not path.is_dir():
            raise ExportError(
                "DATA_SOURCE must be a directory",
                context={"path": str(path)},
            )
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise ExportError(
                "Failed to prepare DATA_SOURCE directory",
                context={"path": str(path)},
            ) from exc

    def _export_symbol(self, ticker: TickerConfig, bars: BarSeries) -> None:
        symbol_dir = symbol_directory(ticker)
        year_dir = (
            self._settings.output_root
            / symbol_dir
            / f"{self._settings.year:04d}"
        )
        file_path = year_dir / (
            f"hist_data_{self._settings.year:04d}-{self._settings.month:02d}.csv"
        )
        try:
            year_dir.mkdir(parents=True, exist_ok=True)
            frame = _bars_to_frame(bars)
            frame.to_csv(file_path, index=False)
        except Exception as exc:
            raise ExportError(
                "Failed to export CSV",
                context={
                    "symbol": ticker.symbol,
                    "symbol_dir": symbol_dir,
                    "path": str(file_path),
                },
            ) from exc

        logger.info(
            "Saved CSV symbol=%s downloaded_bars=%s",
            symbol_dir,
            len(bars),
        )


def _bars_to_frame(bars: BarSeries) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
    timestamps = pd.to_datetime(
        [bar_item.timestamp for bar_item in bars],
        utc=True,
        errors="raise",
    )
    data = {
        "Datetime": timestamps,
        "Open": [bar_item.open for bar_item in bars],
        "High": [bar_item.high for bar_item in bars],
        "Low": [bar_item.low for bar_item in bars],
        "Close": [bar_item.close for bar_item in bars],
        "Volume": [bar_item.volume for bar_item in bars],
    }
    return pd.DataFrame(data)
