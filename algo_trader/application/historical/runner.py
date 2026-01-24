from __future__ import annotations

import logging
import time
from pathlib import Path

from algo_trader.infrastructure import configure_logging, log_boundary
from algo_trader.domain.market_data import HistoricalDataResult
from .config import HistoricalRequestConfig
from .exporter_factory import build_historical_data_exporter
from .factory import build_historical_data_provider

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "tickers.yml"
)

logger = logging.getLogger(__name__)


def _run_context(config_path: Path | None) -> dict[str, str]:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    return {"config_path": str(resolved_path)}


@log_boundary("historical.run", context=_run_context)
def run(config_path: Path | None = None) -> HistoricalDataResult:
    config = HistoricalRequestConfig.load(config_path or DEFAULT_CONFIG_PATH)
    logger.info("Loading ticker config from %s", config.config_path)

    request = config.to_request()
    provider = build_historical_data_provider(config)
    exporter = build_historical_data_exporter(config)
    start = time.time()
    result = provider.fetch(request)
    logger.info("***********")
    exporter.export(request, result)
    _log_outcomes(config, result)
    elapsed = time.time() - start
    logger.info("Download completed in %.2f seconds", elapsed)
    return result


def _log_outcomes(
    config: HistoricalRequestConfig,
    result: HistoricalDataResult,
) -> None:
    if not config.tickers:
        logger.info("No tickers supplied; nothing to report.")
        return

    outcomes = result.outcomes
    for outcome in outcomes.values():
        if outcome.error_code is not None:
            logger.error(
                "Request failed symbol=%s code=%s message=%s",
                outcome.symbol,
                outcome.error_code,
                outcome.error_message,
            )



def main() -> None:
    configure_logging()
    start = time.time()
    run()
    logger.info("Total runtime %.2f seconds", time.time() - start)
