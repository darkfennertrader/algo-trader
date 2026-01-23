from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from algo_trader.historical_data.env import optional_env, optional_int, require_int
from algo_trader.historical_data.models import (
    HistoricalDataRequest,
    HistoricalDataResult,
)
from algo_trader.historical_data.providers.ib.client import (
    IbRequestRegistry,
    IbTradeApp,
    build_symbol_frames,
    build_symbol_outcomes,
    log_ib_client_version,
    request_historical_batch,
)

logger = logging.getLogger(__name__)

DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 7497
DEFAULT_IB_CLIENT_ID = 23


@dataclass(frozen=True)
class IbConnectionSettings:
    host: str
    port: int
    client_id: int


class IbHistoricalDataProvider:
    def __init__(
        self, connection: IbConnectionSettings, max_parallel_requests: int
    ) -> None:
        self._connection = connection
        self._max_parallel_requests = max_parallel_requests

    def name(self) -> str:
        return "ib"

    def fetch(self, request: HistoricalDataRequest) -> HistoricalDataResult:
        registry = IbRequestRegistry(
            inflight_requests=threading.Semaphore(self._max_parallel_requests)
        )
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
        time.sleep(1.0)

        try:
            request_historical_batch(
                app=app,
                request=request,
                registry=registry,
                max_parallel_requests=self._max_parallel_requests,
            )
        finally:
            app.disconnect()
            con_thread.join(timeout=5.0)

        frames = build_symbol_frames(app, registry.request_symbols)
        outcomes = build_symbol_outcomes(app, registry.request_symbols)
        return HistoricalDataResult(frames=frames, outcomes=outcomes)


def _load_ib_connection_settings() -> IbConnectionSettings:
    host = optional_env("IB_HOST") or DEFAULT_IB_HOST
    port = optional_int("IB_PORT") or DEFAULT_IB_PORT
    client_id = optional_int("IB_CLIENT_ID") or DEFAULT_IB_CLIENT_ID
    return IbConnectionSettings(host=host, port=port, client_id=client_id)


def build_ib_provider() -> IbHistoricalDataProvider:
    max_parallel_requests = require_int("MAX_PARALLEL_REQUESTS")
    connection = _load_ib_connection_settings()
    logger.info(
        "Configuring IB provider host=%s port=%s client_id=%s max_parallel_requests=%s",
        connection.host,
        connection.port,
        connection.client_id,
        max_parallel_requests,
    )
    return IbHistoricalDataProvider(connection, max_parallel_requests)
