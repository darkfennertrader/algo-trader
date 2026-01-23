from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from algo_trader.domain import ProviderError
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

    @log_boundary(
        "provider.ib.fetch",
        context=lambda self, request: {
            "provider": self.name(),
            "tickers": str(len(request.tickers)),
        },
    )
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
        try:
            if not app.wait_ready(timeout=IB_CONNECT_TIMEOUT_SECONDS):
                raise ProviderError(
                    "Timed out waiting for IB API connection.",
                    context={
                        "host": self._connection.host,
                        "port": str(self._connection.port),
                    },
                )
            request_historical_batch(
                app=app,
                request=request,
                registry=registry,
                max_parallel_requests=self._max_parallel_requests,
            )
        finally:
            app.disconnect()
            con_thread.join(timeout=5.0)

        bars_by_symbol = build_symbol_bars(app, registry.request_symbols)
        outcomes = build_symbol_outcomes(app, registry.request_symbols)
        return HistoricalDataResult(
            bars_by_symbol=bars_by_symbol, outcomes=outcomes
        )


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
