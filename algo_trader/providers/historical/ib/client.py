from __future__ import annotations

import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Mapping

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from ibapi.server_versions import MAX_CLIENT_VER

from algo_trader.domain import ProviderError
from algo_trader.domain.market_data import (
    Bar,
    HistoricalDataRequest,
    RequestOutcome,
    TickerConfig,
)

logger = logging.getLogger(__name__)

WARNING_ONLY_CODES = {2176}


@dataclass
class IbRequestRegistry:
    inflight_requests: threading.Semaphore
    done_events: Dict[int, threading.Event] = field(default_factory=dict)
    request_symbols: Dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, req_id: int, symbol: str) -> None:
        with self._lock:
            self.done_events[req_id] = threading.Event()
            self.request_symbols[req_id] = symbol

    def mark_done(self, req_id: int) -> None:
        done_event = None
        with self._lock:
            done_event = self.done_events.get(req_id)
        if done_event and not done_event.is_set():
            done_event.set()
            self.inflight_requests.release()

    def wait_all(self) -> None:
        for event in list(self.done_events.values()):
            event.wait()


@dataclass(frozen=True)
class IbHistoricalRequestContext:
    app: "IbTradeApp"
    registry: IbRequestRegistry
    duration: str
    bar_size: str
    end_date_time: str


class IbTradeApp(EWrapper, EClient):
    def __init__(self, registry: IbRequestRegistry) -> None:
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self._registry = registry
        self._data_lock = threading.Lock()
        self._outcome_lock = threading.Lock()
        self._ready_event = threading.Event()
        self.request_outcomes: Dict[int, RequestOutcome] = {}

    def connectAck(self) -> None:
        logger.info("IB API connection acknowledged.")
        self._ready_event.set()

    def nextValidId(self, orderId: int) -> None:
        logger.info("IB API next valid order id=%s", orderId)
        self._ready_event.set()

    def wait_ready(self, timeout: float) -> bool:
        return self._ready_event.wait(timeout)

    def historicalData(self, reqId: int, bar: BarData) -> None:  # pylint: disable=disallowed-name
        row = {
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
        }
        with self._data_lock:
            existing = self.data.get(reqId)
            if existing is None:
                self.data[reqId] = pd.DataFrame([row])
            else:
                self.data[reqId] = pd.concat(
                    (existing, pd.DataFrame([row])),
                    ignore_index=True,
                )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)
        logger.info(
            "HistoricalDataEnd req_id=%s start=%s end=%s",
            reqId,
            start,
            end,
        )
        self._registry.mark_done(reqId)

    def error(self, reqId: int, *args: object) -> None:
        error_time, error_code, error_string, advanced = _parse_error_args(
            args, req_id=reqId
        )
        if reqId < 0:
            logger.info(
                "IB notification req_id=%s code=%s message=%s reject=%s error_time=%s",
                reqId,
                error_code,
                error_string,
                advanced,
                error_time,
            )
            return

        if error_code in WARNING_ONLY_CODES:
            logger.warning(
                "Historical data warning req_id=%s code=%s message=%s reject=%s error_time=%s",
                reqId,
                error_code,
                error_string,
                advanced,
                error_time,
            )
            return

        logger.error(
            "Historical data error req_id=%s code=%s message=%s reject=%s error_time=%s",
            reqId,
            error_code,
            error_string,
            advanced,
            error_time,
        )
        self.record_error(reqId, error_code, error_string)
        self._registry.mark_done(reqId)

    def snapshot(self, req_id: int) -> pd.DataFrame:
        with self._data_lock:
            frame = self.data.get(req_id, pd.DataFrame())
            return frame.copy()

    def register_outcome(self, req_id: int, symbol: str) -> None:
        with self._outcome_lock:
            self.request_outcomes[req_id] = RequestOutcome(symbol=symbol)

    def record_error(
        self, req_id: int, error_code: int, message: str | None
    ) -> None:
        with self._outcome_lock:
            outcome = self.request_outcomes.get(req_id)
            if outcome is None:
                return
            self.request_outcomes[req_id] = outcome.with_error(
                error_code=error_code, message=message
            )

    def record_bars(self, req_id: int, bar_count: int) -> None:
        with self._outcome_lock:
            outcome = self.request_outcomes.get(req_id)
            if outcome is None:
                return
            self.request_outcomes[req_id] = outcome.with_bars(bar_count)


def build_contract(ticker: TickerConfig) -> Contract:
    contract = Contract()
    contract.symbol = ticker.symbol
    contract.secType = ticker.sec_type
    contract.currency = ticker.currency
    contract.exchange = ticker.exchange
    return contract


def _parse_error_args(
    args: tuple[object, ...], req_id: int
) -> tuple[int, int, str, str | None]:
    if len(args) < 3:
        raise ProviderError(
            f"IB error callback missing arguments req_id={req_id} args={args}",
            context={"req_id": str(req_id)},
        )

    error_time = _require_int_arg(args[0], "errorTime", req_id)
    error_code = _require_int_arg(args[1], "errorCode", req_id)
    error_string = _require_str_arg(args[2], "errorString", req_id)
    advanced = (
        _optional_str_arg(args[3], "advancedOrderRejectJson", req_id)
        if len(args) > 3
        else None
    )
    return error_time, error_code, error_string, advanced


def _require_int_arg(value: object, field: str, req_id: int) -> int:
    if not isinstance(value, int):
        raise ProviderError(
            f"IB error callback {field} must be int req_id={req_id} value={value}",
            context={"req_id": str(req_id), "field": field},
        )
    return value


def _require_str_arg(value: object, field: str, req_id: int) -> str:
    if not isinstance(value, str):
        raise ProviderError(
            f"IB error callback {field} must be str req_id={req_id} value={value}",
            context={"req_id": str(req_id), "field": field},
        )
    return value


def _optional_str_arg(value: object, field: str, req_id: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderError(
            f"IB error callback {field} must be str req_id={req_id} value={value}",
            context={"req_id": str(req_id), "field": field},
        )
    return value


def _submit_historical_request(
    context: IbHistoricalRequestContext, req_id: int, ticker: TickerConfig
) -> None:
    context.registry.register(req_id, ticker.symbol)
    context.app.register_outcome(req_id, ticker.symbol)

    context.registry.inflight_requests.acquire()
    try:
        contract = build_contract(ticker)
        context.app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=context.end_date_time,
            durationStr=context.duration,
            barSizeSetting=context.bar_size,
            whatToShow=ticker.what_to_show,
            useRTH=1,
            formatDate=1,
            keepUpToDate=0,
            chartOptions=[],
        )
        logger.info(
            "Submitted historical request req_id=%s symbol=%s",
            req_id,
            ticker.symbol,
        )
    except Exception as exc:
        context.registry.mark_done(req_id)
        logger.exception(
            "Failed to submit historical request req_id=%s symbol=%s",
            req_id,
            ticker.symbol,
        )
        raise ProviderError(
            f"Failed to submit historical request for {ticker.symbol}",
            context={"symbol": ticker.symbol, "req_id": str(req_id)},
        ) from exc


def request_historical_batch(
    app: IbTradeApp,
    request: HistoricalDataRequest,
    registry: IbRequestRegistry,
    max_parallel_requests: int,
) -> None:
    if max_parallel_requests < 1:
        raise ProviderError(
            "max_parallel_requests must be at least 1",
            context={"value": str(max_parallel_requests)},
        )
    if not request.tickers:
        logger.info("No tickers supplied; skipping historical requests.")
        return

    context = IbHistoricalRequestContext(
        app=app,
        registry=registry,
        duration=request.duration,
        bar_size=request.bar_size,
        end_date_time=request.end_date_time,
    )
    worker_count = min(len(request.tickers), max_parallel_requests * 2)
    req_ids = itertools.count(1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for ticker in request.tickers:
            futures.append(
                executor.submit(
                    _submit_historical_request, context, next(req_ids), ticker
                )
            )

        for future in futures:
            future.result()

    registry.wait_all()


def _frame_to_bars(frame: pd.DataFrame) -> list[Bar]:
    bars: list[Bar] = []
    for _, row in frame.iterrows():
        bars.append(
            Bar(
                timestamp=str(row["Date"]),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
        )
    return bars


def build_symbol_bars(
    app: IbTradeApp, request_symbols: Mapping[int, str]
) -> Dict[str, list[Bar]]:
    bars_by_symbol: Dict[str, list[Bar]] = {}
    for req_id, symbol in request_symbols.items():
        frame = app.snapshot(req_id)
        bars = _frame_to_bars(frame) if not frame.empty else []
        bars_by_symbol[symbol] = bars
        app.record_bars(req_id, len(bars))
    return bars_by_symbol


def build_symbol_outcomes(
    app: IbTradeApp, request_symbols: Mapping[int, str]
) -> Dict[str, RequestOutcome]:
    outcomes: Dict[str, RequestOutcome] = {}
    for req_id, symbol in request_symbols.items():
        outcome = app.request_outcomes.get(req_id)
        if outcome is None:
            outcome = RequestOutcome(symbol=symbol)
        outcomes[symbol] = outcome
    return outcomes


def log_ib_client_version() -> None:
    logger.info("IB API MAX_CLIENT_VER=%s", MAX_CLIENT_VER)
