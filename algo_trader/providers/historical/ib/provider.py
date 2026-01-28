from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from algo_trader.domain import ProviderConnectionError, ProviderError
from algo_trader.infrastructure import log_boundary
from algo_trader.domain.market_data import (
    HistoricalDataRequest,
    HistoricalDataResult,
)
from .client import (
    IbRequestRegistry,
    IbTradeApp,
    build_symbol_bars,
    build_symbol_outcomes,
    log_ib_client_version,
    request_historical_batch,
)

logger = logging.getLogger(__name__)

IB_CONNECT_TIMEOUT_SECONDS = 5.0
IB_CONNECT_MAX_ATTEMPTS = 3
IB_CONNECT_BACKOFF_SECONDS = (30.0, 60.0)
IB_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
IB_HEARTBEAT_INTERVAL_SECONDS = 30.0
IB_RESTART_MAX_ATTEMPTS = 3
IB_RESTART_BACKOFF_SECONDS = (30.0, 60.0)

@dataclass(frozen=True)
class IbConnectionSettings:
    host: str
    port: int
    client_id: int


class IbHeartbeatMonitor:
    def __init__(
        self,
        app: IbTradeApp,
        connection: IbConnectionSettings,
        disconnect_event: threading.Event,
        interval_seconds: float = IB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._app = app
        self._connection = connection
        self._disconnect_event = disconnect_event
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ib-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=IB_THREAD_JOIN_TIMEOUT_SECONDS)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            if self._disconnect_event.is_set():
                return
            try:
                connected = self._app.isConnected()
            except Exception as exc:
                logger.warning(
                    "IB heartbeat check failed; treating as disconnected "
                    "host=%s port=%s client_id=%s",
                    self._connection.host,
                    self._connection.port,
                    self._connection.client_id,
                    exc_info=exc,
                )
                self._disconnect_event.set()
                return
            if not connected:
                logger.warning(
                    "IB heartbeat detected disconnect host=%s port=%s client_id=%s",
                    self._connection.host,
                    self._connection.port,
                    self._connection.client_id,
                )
                self._disconnect_event.set()
                return


class IbHistoricalDataProvider:
    def __init__(
        self, connection: IbConnectionSettings, max_parallel_requests: int
    ) -> None:
        self._connection = connection
        self._max_parallel_requests = max_parallel_requests

    def name(self) -> str:
        return "ib"

    def _connect_with_retry(
        self, registry: IbRequestRegistry
    ) -> tuple[IbTradeApp, threading.Thread]:
        for attempt in range(1, IB_CONNECT_MAX_ATTEMPTS + 1):
            app = IbTradeApp(registry)
            log_ib_client_version()

            app.connect(
                host=self._connection.host,
                port=self._connection.port,
                clientId=self._connection.client_id,
            )
            con_thread = threading.Thread(
                target=app.run, name="ib-event-loop", daemon=True
            )
            con_thread.start()

            if app.wait_ready(timeout=IB_CONNECT_TIMEOUT_SECONDS):
                return app, con_thread

            logger.warning(
                "IB connection attempt failed attempt=%s max_attempts=%s "
                "host=%s port=%s client_id=%s",
                attempt,
                IB_CONNECT_MAX_ATTEMPTS,
                self._connection.host,
                self._connection.port,
                self._connection.client_id,
            )
            app.disconnect()
            con_thread.join(timeout=IB_THREAD_JOIN_TIMEOUT_SECONDS)

            if attempt < IB_CONNECT_MAX_ATTEMPTS:
                delay = IB_CONNECT_BACKOFF_SECONDS[attempt - 1]
                logger.warning(
                    "Retrying IB connection after %s seconds attempt=%s",
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

        raise ProviderError(
            "Timed out waiting for IB API connection.",
            context={
                "host": self._connection.host,
                "port": str(self._connection.port),
                "client_id": str(self._connection.client_id),
                "attempts": str(IB_CONNECT_MAX_ATTEMPTS),
            },
        )

    @log_boundary(
        "provider.ib.fetch",
        context=lambda self, request: {
            "provider": self.name(),
            "tickers": str(len(request.tickers)),
        },
    )
    def fetch(self, request: HistoricalDataRequest) -> HistoricalDataResult:
        last_error: ProviderConnectionError | None = None
        for attempt in range(1, IB_RESTART_MAX_ATTEMPTS + 1):
            registry = IbRequestRegistry(
                inflight_requests=threading.Semaphore(
                    self._max_parallel_requests
                )
            )
            app, con_thread = self._connect_with_retry(registry)
            disconnect_event = app.disconnect_event()
            monitor = IbHeartbeatMonitor(
                app=app,
                connection=self._connection,
                disconnect_event=disconnect_event,
            )
            monitor.start()
            try:
                completed = request_historical_batch(
                    app=app,
                    request=request,
                    registry=registry,
                    max_parallel_requests=self._max_parallel_requests,
                    abort_event=disconnect_event,
                )
                if not completed:
                    raise ProviderConnectionError(
                        "IB connection lost during historical data download.",
                        context=self._connection_context(),
                    )
                bars_by_symbol = build_symbol_bars(
                    app, registry.request_symbols
                )
                outcomes = build_symbol_outcomes(app, registry.request_symbols)
                return HistoricalDataResult(
                    bars_by_symbol=bars_by_symbol, outcomes=outcomes
                )
            except ProviderConnectionError as exc:
                last_error = exc
                logger.warning(
                    "IB connection lost; restarting download attempt=%s "
                    "max_attempts=%s window=%s host=%s port=%s client_id=%s",
                    attempt,
                    IB_RESTART_MAX_ATTEMPTS,
                    self._window_label(request),
                    self._connection.host,
                    self._connection.port,
                    self._connection.client_id,
                )
            except ProviderError as exc:
                if disconnect_event.is_set():
                    last_error = ProviderConnectionError(
                        "IB connection lost during historical data download.",
                        context=self._connection_context(),
                    )
                    logger.warning(
                        "IB connection lost during request; restarting "
                        "attempt=%s max_attempts=%s window=%s "
                        "host=%s port=%s client_id=%s",
                        attempt,
                        IB_RESTART_MAX_ATTEMPTS,
                        self._window_label(request),
                        self._connection.host,
                        self._connection.port,
                        self._connection.client_id,
                        exc_info=exc,
                    )
                else:
                    raise
            finally:
                monitor.stop()
                app.disconnect()
                con_thread.join(timeout=IB_THREAD_JOIN_TIMEOUT_SECONDS)

            if attempt < IB_RESTART_MAX_ATTEMPTS:
                delay = IB_RESTART_BACKOFF_SECONDS[attempt - 1]
                logger.warning(
                    "Retrying historical download window=%s after %s seconds attempt=%s",
                    self._window_label(request),
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise ProviderConnectionError(
            "IB connection lost during historical data download.",
            context=self._connection_context(),
        )

    def _connection_context(self) -> dict[str, str]:
        return {
            "host": self._connection.host,
            "port": str(self._connection.port),
            "client_id": str(self._connection.client_id),
        }

    def _window_label(self, request: HistoricalDataRequest) -> str:
        if request.window_label:
            return request.window_label
        if request.end_date_time:
            return request.end_date_time
        return request.duration


def build_ib_provider(
    connection: IbConnectionSettings, max_parallel_requests: int
) -> IbHistoricalDataProvider:
    logger.info(
        "Configuring IB provider host=%s port=%s client_id=%s max_parallel_requests=%s",
        connection.host,
        connection.port,
        connection.client_id,
        max_parallel_requests,
    )
    return IbHistoricalDataProvider(connection, max_parallel_requests)
