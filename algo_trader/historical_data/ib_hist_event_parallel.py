from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Mapping, Sequence, Tuple

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

GLOBAL_LIMIT = 60
GLOBAL_WINDOW_SECONDS = 600.0
CONTRACT_LIMIT = 5
CONTRACT_WINDOW_SECONDS = 2.0
IDENTICAL_GAP_SECONDS = 15.0


class RequestPacer:
    """Throttle historical requests to stay within IB pacing limits."""

    def __init__(self) -> None:
        self._global_times: Deque[float] = deque()
        self._contract_times: Dict[Tuple[str, str, str], Deque[float]] = {}
        self._last_identical: Dict[Tuple[str, str, str, str, str], float] = {}
        self._lock = threading.Lock()

    def throttle(
        self,
        contract: Contract,
        duration: str,
        bar_size: str,
        what_to_show: str,
    ) -> None:
        with self._lock:
            now = time.time()
            ident_key = (
                contract.symbol,
                contract.exchange,
                contract.secType,
                duration,
                bar_size,
            )
            last = self._last_identical.get(ident_key)
            if last is not None:
                remaining = IDENTICAL_GAP_SECONDS - (now - last)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_identical[ident_key] = time.time()

            self._wait_for_slot(
                self._global_times, GLOBAL_WINDOW_SECONDS, GLOBAL_LIMIT
            )
            contract_key = (
                contract.symbol,
                contract.exchange,
                contract.secType,
            )
            self._wait_for_slot(
                self._contract_times.setdefault(contract_key, deque()),
                CONTRACT_WINDOW_SECONDS,
                CONTRACT_LIMIT,
            )

    def _wait_for_slot(
        self, times: Deque[float], window_seconds: float, limit: int
    ) -> None:
        now = time.time()
        self._prune(times, now, window_seconds)
        if len(times) >= limit:
            wait_seconds = window_seconds - (now - times[0])
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            now = time.time()
            self._prune(times, now, window_seconds)
        times.append(time.time())

    @staticmethod
    def _prune(times: Deque[float], now: float, window_seconds: float) -> None:
        while times and now - times[0] > window_seconds:
            times.popleft()


class TradeApp(EWrapper, EClient):
    def __init__(self, done_events: Dict[int, threading.Event]) -> None:
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self._done_events = done_events
        self._data_lock = threading.Lock()

    # pylint: disable=disallowed-name  # matches EWrapper signature
    def historicalData(self, reqId: int, bar: BarData) -> None:
        row = {
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume,
        }
        with self._data_lock:
            if reqId not in self.data:
                self.data[reqId] = pd.DataFrame([row])
            else:
                self.data[reqId] = pd.concat(
                    (self.data[reqId], pd.DataFrame([row])),
                    ignore_index=True,
                )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)
        done_event = self._done_events.get(reqId)
        if done_event:
            done_event.set()

    def snapshot(self, req_id: int) -> pd.DataFrame:
        with self._data_lock:
            frame = self.data.get(req_id, pd.DataFrame())
            return frame.copy()


def build_contract(
    symbol: str,
    sec_type: str = "STK",
    currency: str = "USD",
    exchange: str = "ISLAND",
) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract


def request_historical_data(
    app: TradeApp,
    pacer: RequestPacer,
    req_id: int,
    contract: Contract,
    duration: str,
    bar_size: str,
    what_to_show: str = "ADJUSTED_LAST",
) -> None:
    pacer.throttle(contract, duration, bar_size, what_to_show)
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=1,
        formatDate=1,
        keepUpToDate=0,
        chartOptions=[],
    )


def request_historical_batch(
    app: TradeApp,
    pacer: RequestPacer,
    symbols: Sequence[str],
    duration: str,
    bar_size: str,
    done_events: Dict[int, threading.Event],
    request_symbols: Dict[int, str],
) -> None:
    for req_id, symbol in enumerate(symbols):
        done_events[req_id] = threading.Event()
        request_symbols[req_id] = symbol
        contract = build_contract(symbol)
        request_historical_data(
            app, pacer, req_id, contract, duration, bar_size
        )

    for event in done_events.values():
        event.wait()


def build_symbol_frames(
    app: TradeApp, request_symbols: Mapping[int, str]
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for req_id, symbol in request_symbols.items():
        frame = app.snapshot(req_id)
        if not frame.empty:
            frame = frame.set_index("Date")
        frames[symbol] = frame
    return frames


def main() -> None:
    tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
    ]
    duration = "252 D"
    bar_size = "1 hour"

    done_events: Dict[int, threading.Event] = {}
    request_symbols: Dict[int, str] = {}
    pacer = RequestPacer()
    app = TradeApp(done_events)

    app.connect(host="127.0.0.1", port=7497, clientId=23)
    con_thread = threading.Thread(
        target=app.run, name="ib-event-loop", daemon=False
    )
    con_thread.start()

    try:
        request_historical_batch(
            app=app,
            pacer=pacer,
            symbols=tickers,
            duration=duration,
            bar_size=bar_size,
            done_events=done_events,
            request_symbols=request_symbols,
        )
    finally:
        app.disconnect()
        con_thread.join(timeout=5.0)

    historical_frames = build_symbol_frames(app, request_symbols)
    for symbol, frame in historical_frames.items():
        print(f"{symbol}: {len(frame)} bars")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nIt took {(time.time() -start)/1000:.2f} sec.")
