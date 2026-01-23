from __future__ import annotations

import logging
import time
from pathlib import Path

from algo_trader.infrastructure import configure_logging, log_boundary
from algo_trader.domain.market_data import HistoricalDataResult, RequestOutcome
from .config import HistoricalRequestConfig
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
    config = HistoricalRequestConfig.load(
        config_path or DEFAULT_CONFIG_PATH
    )
    logger.info("Loading ticker config from %s", config.config_path)

    request = config.to_request()
    provider = build_historical_data_provider(config)
    start = time.time()
    result = provider.fetch(request)
    elapsed = time.time() - start
    _log_outcomes(config, result, elapsed)
    return result


def _log_outcomes(
    config: HistoricalRequestConfig,
    result: HistoricalDataResult,
    elapsed: float,
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

    logger.info(
        "Completed %s requests in %.2f seconds",
        len(result.bars_by_symbol),
        elapsed,
    )
    for ticker in config.tickers:
        outcome = outcomes.get(
            ticker.symbol, RequestOutcome(symbol=ticker.symbol)
        )
        logger.info(
            "Symbol %s fetched %s bars",
            outcome.symbol,
            outcome.bars,
        )
        if outcome.bars == 0:
            logger.warning(
                "Request completed but returned zero bars symbol=%s",
                outcome.symbol,
            )


def main() -> None:
    configure_logging()
    start = time.time()
    run()
    logger.info("Total runtime %.2f seconds", time.time() - start)
