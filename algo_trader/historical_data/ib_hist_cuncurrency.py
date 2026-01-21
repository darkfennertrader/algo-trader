from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
import yaml
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from ibapi.server_versions import MAX_CLIENT_VER

load_dotenv()


print("MAX_CLIENT_VER =", MAX_CLIENT_VER)


# Paths for ticker configuration.
CONFIG_ENV_VAR = "TICKER_CONFIG_PATH"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "tickers.yml"
)
WARNING_ONLY_CODES = {2176}


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


def _coerce_to_str(value: Any) -> str:
    """Convert config value to string while tolerating missing or None values."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _resolve_value(
    data: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
) -> str:
    if key in data:
        return _coerce_to_str(data.get(key))
    return _coerce_to_str(defaults.get(key))


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    sec_type: str
    currency: str
    exchange: str

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        defaults: Mapping[str, Any],
        config_path: Path,
    ) -> "TickerConfig":
        del config_path  # kept for signature compatibility; validation removed.
        mapping: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
        resolved_defaults: Mapping[str, Any] = (
            defaults if isinstance(defaults, Mapping) else {}
        )
        return cls(
            symbol=_resolve_value(mapping, resolved_defaults, "symbol"),
            sec_type=_resolve_value(mapping, resolved_defaults, "sec_type"),
            currency=_resolve_value(mapping, resolved_defaults, "currency"),
            exchange=_resolve_value(mapping, resolved_defaults, "exchange"),
        )


@dataclass(frozen=True)
class HistoricalRequestConfig:
    tickers: Sequence[TickerConfig]
    duration: str
    bar_size: str
    what_to_show: str

    @classmethod
    def load(cls, config_path: Path) -> "HistoricalRequestConfig":
        if not config_path.exists():
            example_path = config_path.with_name(
                f"{config_path.stem}.example{config_path.suffix}"
            )
            raise FileNotFoundError(
                f"Ticker config not found at {config_path}. "
                f"Copy {example_path} and customize tickers."
            )
        raw_text = config_path.read_text(encoding="utf-8")
        try:
            raw_config: Any = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML content in {config_path}") from exc

        config_mapping: Mapping[str, Any] = (
            raw_config if isinstance(raw_config, Mapping) else {}
        )

        defaults_raw = config_mapping.get("defaults")
        defaults: Mapping[str, Any] = (
            defaults_raw if isinstance(defaults_raw, Mapping) else {}
        )

        tickers_raw = config_mapping.get("tickers")
        ticker_entries = tickers_raw if isinstance(tickers_raw, list) else []

        tickers = [
            TickerConfig.from_mapping(entry, defaults, config_path)
            for entry in ticker_entries
        ]

        duration = _coerce_to_str(config_mapping.get("duration"))
        bar_size = _coerce_to_str(config_mapping.get("bar_size"))
        what_to_show = _coerce_to_str(config_mapping.get("what_to_show"))

        return cls(
            tickers=tickers,
            duration=duration,
            bar_size=bar_size,
            what_to_show=what_to_show,
        )


@dataclass(frozen=True)
class RequestOutcome:
    symbol: str
    bars: int = 0
    error_code: int | None = None
    error_message: str | None = None

    def with_error(
        self, error_code: int, message: str | None
    ) -> "RequestOutcome":
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


def resolve_config_path() -> Path:
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


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
        self._outcome_lock = threading.Lock()
        self.request_outcomes: Dict[int, RequestOutcome] = {}

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
        errorTime: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str | None = None,
    ) -> None:
        if reqId < 0:
            logger.info(
                "IB notification req_id=%s code=%s message=%s reject=%s error_time=%s",
                reqId,
                errorCode,
                errorString,
                advancedOrderRejectJson,
                errorTime,
            )
            return

        if errorCode in WARNING_ONLY_CODES:
            logger.warning(
                "Historical data warning req_id=%s code=%s message=%s reject=%s error_time=%s",
                reqId,
                errorCode,
                errorString,
                advancedOrderRejectJson,
                errorTime,
            )
            return

        logger.error(
            "Historical data error req_id=%s code=%s message=%s reject=%s error_time=%s",
            reqId,
            errorCode,
            errorString,
            advancedOrderRejectJson,
            errorTime,
        )
        self.record_error(reqId, errorCode, errorString)
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


def _submit_historical_request(
    app: TradeApp,
    inflight_requests: threading.Semaphore,
    done_events: Dict[int, threading.Event],
    request_symbols: Dict[int, str],
    req_id: int,
    ticker: TickerConfig,
    duration: str,
    bar_size: str,
    what_to_show: str,
    mapping_lock: threading.Lock,
) -> None:
    with mapping_lock:
        done_events[req_id] = threading.Event()
        request_symbols[req_id] = ticker.symbol
        app.register_outcome(req_id, ticker.symbol)

    inflight_requests.acquire()
    try:
        contract = build_contract(ticker)
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
            "Submitted historical request req_id=%s symbol=%s",
            req_id,
            ticker.symbol,
        )
    except Exception:
        done_event = done_events[req_id]
        done_event.set()
        inflight_requests.release()
        raise


def request_historical_batch(
    app: TradeApp,
    tickers: Sequence[TickerConfig],
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
    if not tickers:
        logger.info("No tickers supplied; skipping historical requests.")
        return

    req_id_lock = threading.Lock()
    req_ids = itertools.count(1)
    mapping_lock = threading.Lock()

    def next_req_id() -> int:
        with req_id_lock:
            return next(req_ids)

    worker_count = min(len(tickers), max_parallel_requests * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for ticker in tickers:
            req_id = next_req_id()
            executor.submit(
                _submit_historical_request,
                app,
                inflight_requests,
                done_events,
                request_symbols,
                req_id,
                ticker,
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
        app.record_bars(req_id, len(frame))
    return frames


def main(max_parallel_requests: int = MAX_PARALLEL_REQUESTS) -> None:
    config_path = resolve_config_path()
    logger.info("Loading ticker config from %s", config_path)
    request_config = HistoricalRequestConfig.load(config_path)

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
            tickers=request_config.tickers,
            duration=request_config.duration,
            bar_size=request_config.bar_size,
            done_events=done_events,
            request_symbols=request_symbols,
            inflight_requests=inflight_requests,
            max_parallel_requests=max_parallel_requests,
            what_to_show=request_config.what_to_show,
        )
    finally:
        app.disconnect()
        con_thread.join(timeout=5.0)

    historical_frames = build_symbol_frames(app, request_symbols)
    elapsed = time.time() - start
    for req_id, outcome in app.request_outcomes.items():
        if outcome.error_code is not None:
            logger.error(
                "Request failed req_id=%s symbol=%s code=%s message=%s",
                req_id,
                outcome.symbol,
                outcome.error_code,
                outcome.error_message,
            )
            continue

        logger.info(
            "Symbol %s fetched %s bars (req_id=%s)",
            outcome.symbol,
            outcome.bars,
            req_id,
        )
        if outcome.bars == 0:
            logger.warning(
                "Request completed but returned zero bars req_id=%s symbol=%s",
                req_id,
                outcome.symbol,
            )
        logger.info(
            "Completed %s requests in %.2f seconds (max_parallel_requests=%s)",
            len(historical_frames),
            elapsed,
            max_parallel_requests,
        )
    summary_lines = ["Bar counts:"]
    for ticker in request_config.tickers:
        outcome = next(
            (
                item
                for item in app.request_outcomes.values()
                if item.symbol == ticker.symbol
            ),
            None,
        )
        bars = outcome.bars if outcome else 0
        summary_lines.append(f"{ticker.symbol}: {bars}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nIt took {(time.time()- start):.2f} sec.")
