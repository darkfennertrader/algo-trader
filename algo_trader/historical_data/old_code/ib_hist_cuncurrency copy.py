from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Mapping, Sequence

import pandas as pd
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

load_dotenv()


# Maximum number of in-flight historical requests for a single client id.
try:
    MAX_PARALLEL_REQUESTS = int(os.environ["MAX_PARALLEL_REQUESTS"])
except KeyError as exc:
    raise ValueError("MAX_PARALLEL_REQUESTS must be set in .env") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class TradeApp(EWrapper, EClient):
    def __init__(
        self,
        done_events: Dict[int, threading.Event],
        inflight_requests: threading.Semaphore,
    ) -> None:
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self._done_events = done_events
        self._data_lock = threading.Lock()
        self._inflight_requests = inflight_requests

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
        self._mark_done(reqId)

    def error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str | None = None,
    ) -> None:
        if reqId >= 0:
            logger.error(
                "Historical data error req_id=%s code=%s message=%s reject=%s",
                reqId,
                errorCode,
                errorString,
                advancedOrderRejectJson,
            )
            self._mark_done(reqId)

    def snapshot(self, req_id: int) -> pd.DataFrame:
        with self._data_lock:
            frame = self.data.get(req_id, pd.DataFrame())
            return frame.copy()

    def _mark_done(self, req_id: int) -> None:
        done_event = self._done_events.get(req_id)
        if done_event and not done_event.is_set():
            done_event.set()
            self._inflight_requests.release()


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


def _submit_historical_request(
    app: TradeApp,
    inflight_requests: threading.Semaphore,
    done_events: Dict[int, threading.Event],
    request_symbols: Dict[int, str],
    req_id: int,
    symbol: str,
    duration: str,
    bar_size: str,
    what_to_show: str,
    mapping_lock: threading.Lock,
) -> None:
    with mapping_lock:
        done_events[req_id] = threading.Event()
        request_symbols[req_id] = symbol

    inflight_requests.acquire()
    try:
        contract = build_contract(symbol)
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
        logger.info(
            "Submitted historical request req_id=%s symbol=%s", req_id, symbol
        )
    except Exception:
        done_event = done_events[req_id]
        done_event.set()
        inflight_requests.release()
        raise


def request_historical_batch(
    app: TradeApp,
    symbols: Sequence[str],
    duration: str,
    bar_size: str,
    done_events: Dict[int, threading.Event],
    request_symbols: Dict[int, str],
    inflight_requests: threading.Semaphore,
    max_parallel_requests: int = MAX_PARALLEL_REQUESTS,
    what_to_show: str = "ADJUSTED_LAST",
) -> None:
    if max_parallel_requests < 1:
        raise ValueError("max_parallel_requests must be at least 1")
    if not symbols:
        logger.info("No symbols supplied; skipping historical requests.")
        return

    req_id_lock = threading.Lock()
    req_ids = itertools.count(1)
    mapping_lock = threading.Lock()

    def next_req_id() -> int:
        with req_id_lock:
            return next(req_ids)

    worker_count = min(len(symbols), max_parallel_requests * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for symbol in symbols:
            req_id = next_req_id()
            executor.submit(
                _submit_historical_request,
                app,
                inflight_requests,
                done_events,
                request_symbols,
                req_id,
                symbol,
                duration,
                bar_size,
                what_to_show,
                mapping_lock,
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


def main(max_parallel_requests: int = MAX_PARALLEL_REQUESTS) -> None:
    tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        # "TSLA",
        # "INTC",
        # "AMD",
        # "NFLX",
        # "ORCL",
        # "IBM",
        # "CRM",
        # "UBER",
        # "ABNB",
        # "SHOP",
        # "PYPL",
        # "SNOW",
        # "ADBE",
        # "QCOM",
        # "BA",
        # "DIS",
    ]
    duration = "252 D"
    bar_size = "1 hour"

    inflight_requests = threading.Semaphore(max_parallel_requests)
    done_events: Dict[int, threading.Event] = {}
    request_symbols: Dict[int, str] = {}
    app = TradeApp(done_events, inflight_requests)

    app.connect(host="127.0.0.1", port=7497, clientId=23)
    con_thread = threading.Thread(
        target=app.run, name="ib-event-loop", daemon=True
    )
    con_thread.start()
    time.sleep(1.0)

    start = time.time()
    try:
        request_historical_batch(
            app=app,
            symbols=tickers,
            duration=duration,
            bar_size=bar_size,
            done_events=done_events,
            request_symbols=request_symbols,
            inflight_requests=inflight_requests,
            max_parallel_requests=max_parallel_requests,
        )
    finally:
        app.disconnect()
        con_thread.join(timeout=5.0)

    historical_frames = build_symbol_frames(app, request_symbols)
    elapsed = time.time() - start
    for symbol, frame in historical_frames.items():
        logger.info("Symbol %s fetched %s bars", symbol, len(frame))
    logger.info(
        "Completed %s requests in %.2f seconds (max_parallel_requests=%s)",
        len(historical_frames),
        elapsed,
        max_parallel_requests,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nIt took {(time.time()- start)/1000:.2f} sec.")
