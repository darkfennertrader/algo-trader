from __future__ import annotations

import itertools
import logging
import os
import re
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


def _require_non_empty(value: Any, field: str, config_path: Path) -> str:
    text = _coerce_to_str(value)
    if not text:
        raise ValueError(f"{field} is required in {config_path}")
    return text


def _parse_timestamp(value: str, field: str, config_path: Path) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field} '{value}' in {config_path}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Invalid {field} '{value}' in {config_path}")
    return timestamp


def _format_end_time(timestamp: pd.Timestamp) -> str:
    base = timestamp.strftime("%Y%m%d %H:%M:%S")
    tz_name = timestamp.tzname()
    if tz_name:
        return f"{base} {tz_name}"
    return base


def _compute_duration_from_range(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, config_path: Path
) -> str:
    delta = end_ts - start_ts
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        raise ValueError(
            f"end_time must be after start_time in {config_path} "
            f"(difference was {seconds} seconds)"
        )
    return f"{seconds} S"


FX_PAIR_PATTERN = re.compile(r"^([A-Z]{3})\.([A-Z]{3})$")


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    sec_type: str
    currency: str
    exchange: str
    what_to_show: str
    asset_class: str


@dataclass(frozen=True)
class HistoricalRequestConfig:
    tickers: Sequence[TickerConfig]
    duration: str
    bar_size: str
    start_time: str
    end_time: str
    config_path: Path

    def resolve_window(self) -> tuple[str, str]:
        """
        Returns (end_date_time, duration).
        When start_time and end_time are both set, duration is derived from them.
        Otherwise duration from the config is used with end_time (or IB default now).
        """
        if self.start_time and not self.end_time:
            raise ValueError(
                f"start_time requires end_time in {self.config_path}; "
                "provide both or omit start_time."
            )
        if self.start_time and self.end_time:
            start_ts = _parse_timestamp(
                self.start_time, "start_time", self.config_path
            )
            end_ts = _parse_timestamp(
                self.end_time, "end_time", self.config_path
            )
            derived_duration = _compute_duration_from_range(
                start_ts, end_ts, self.config_path
            )
            return _format_end_time(end_ts), derived_duration

        end_date_time = ""
        if self.end_time:
            end_ts = _parse_timestamp(
                self.end_time, "end_time", self.config_path
            )
            end_date_time = _format_end_time(end_ts)
        duration_value = _require_non_empty(
            self.duration, "duration", self.config_path
        )
        return end_date_time, duration_value

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

        start_time = _coerce_to_str(config_mapping.get("start_time"))
        end_time = _coerce_to_str(config_mapping.get("end_time"))

        tickers: list[TickerConfig] = []
        for asset_name in ("stocks", "forex", "indices", "commodities"):
            asset_section = config_mapping.get(asset_name)
            if not isinstance(asset_section, Mapping):
                continue
            tickers.extend(
                _load_asset_tickers(
                    asset_name=asset_name,
                    section=asset_section,
                    config_path=config_path,
                )
            )

        duration = _coerce_to_str(config_mapping.get("duration"))
        bar_size = _coerce_to_str(config_mapping.get("bar_size"))

        return cls(
            tickers=tickers,
            duration=duration,
            bar_size=bar_size,
            start_time=start_time,
            end_time=end_time,
            config_path=config_path,
        )


def _load_asset_tickers(
    asset_name: str, section: Mapping[str, Any], config_path: Path
) -> list[TickerConfig]:
    defaults_raw = section.get("contract_defaults")
    defaults: Mapping[str, Any] = (
        defaults_raw if isinstance(defaults_raw, Mapping) else {}
    )
    sec_type_default = _require_non_empty(
        defaults.get("sec_type"),
        f"{asset_name}.contract_defaults.sec_type",
        config_path,
    )
    exchange_default = _require_non_empty(
        defaults.get("exchange"),
        f"{asset_name}.contract_defaults.exchange",
        config_path,
    )
    currency_default = _coerce_to_str(defaults.get("currency"))
    default_what_to_show = _require_non_empty(
        defaults.get("what_to_show"),
        f"{asset_name}.contract_defaults.what_to_show",
        config_path,
    )

    raw_tickers = section.get("tickers")
    if raw_tickers is None:
        return []
    if not isinstance(raw_tickers, list):
        raise ValueError(f"{asset_name}.tickers must be a list in {config_path}")

    parsed: list[TickerConfig] = []
    for entry in raw_tickers:
        if asset_name == "forex":
            symbol, currency = _parse_fx_entry(entry, config_path, asset_name)
        elif isinstance(entry, Mapping):
            symbol = _require_non_empty(
                entry.get("symbol"),
                f"{asset_name}.tickers.symbol",
                config_path,
            )
            currency = _coerce_to_str(entry.get("currency"))
            if not currency:
                currency = currency_default
            currency = _require_non_empty(
                currency,
                f"{asset_name}.tickers.currency",
                config_path,
            )
        elif isinstance(entry, str):
            symbol = _require_non_empty(entry, f"{asset_name}.tickers.symbol", config_path)
            currency = _require_non_empty(
                currency_default,
                f"{asset_name}.contract_defaults.currency",
                config_path,
            )
        else:
            raise ValueError(
                f"Unsupported ticker entry in {asset_name} for {config_path}: {entry}"
            )

        if isinstance(entry, Mapping):
            what_to_show = _coerce_to_str(entry.get("what_to_show"))
            sec_type = _coerce_to_str(entry.get("sec_type")) or sec_type_default
            exchange = _coerce_to_str(entry.get("exchange")) or exchange_default
        else:
            what_to_show = ""
            sec_type = sec_type_default
            exchange = exchange_default

        parsed.append(
            TickerConfig(
                symbol=symbol,
                sec_type=sec_type,
                currency=currency,
                exchange=exchange,
                what_to_show=what_to_show or default_what_to_show,
                asset_class=asset_name,
            )
        )

    return parsed


def _parse_fx_entry(
    entry: Any, config_path: Path, asset_name: str
) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        symbol = _require_non_empty(
            entry.get("symbol"),
            f"{asset_name}.tickers.symbol",
            config_path,
        )
        currency = _require_non_empty(
            entry.get("currency"),
            f"{asset_name}.tickers.currency",
            config_path,
        )
        return symbol, currency

    if isinstance(entry, str):
        cleaned = entry.strip().upper().replace("/", ".")
        match = FX_PAIR_PATTERN.match(cleaned) or re.match(
            r"^([A-Z]{3})([A-Z]{3})$", cleaned
        )
        if not match:
            raise ValueError(
                f"{asset_name} tickers must use 'BASE.QUOTE' format (e.g., EUR.USD) "
                f"in {config_path}"
            )
        base, quote = match.groups()
        return base, quote

    raise ValueError(
        f"Unsupported ticker entry in {asset_name} for {config_path}: {entry}"
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
    end_date_time: str,
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
            endDateTime=end_date_time,
            durationStr=duration,
            barSizeSetting=bar_size,
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
    end_date_time: str,
    done_events: Dict[int, threading.Event],
    request_symbols: Dict[int, str],
    inflight_requests: threading.Semaphore,
    max_parallel_requests: int = MAX_PARALLEL_REQUESTS,
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
                end_date_time,
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
    config_path = DEFAULT_CONFIG_PATH
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
    end_date_time, duration = request_config.resolve_window()
    try:
        request_historical_batch(
            app=app,
            tickers=request_config.tickers,
            duration=duration,
            bar_size=request_config.bar_size,
            end_date_time=end_date_time,
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
