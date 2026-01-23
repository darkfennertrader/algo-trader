from __future__ import annotations

import argparse
import logging
from pathlib import Path
from functools import partial
from typing import Sequence

from algo_trader.historical_data.runner import (
    DEFAULT_CONFIG_PATH,
    configure_logging,
    run,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algotrader",
        description="Algo Trader command line interface.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Path to ticker config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "historical",
        help="Download historical data.",
    )

    subparsers.add_parser(
        "backtest",
        help="Run backtests (not implemented yet).",
    )

    return parser


def _run_historical(config_path: Path | None) -> int:
    run(config_path=config_path)
    return 0


def _run_backtest() -> int:
    logger.error("Backtest command not implemented yet.")
    return 1


def _run_pipeline() -> int:
    logger.info("Pipeline entrypoint not implemented yet.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = getattr(args, "config", None)
    handlers = {
        None: _run_pipeline,
        "historical": partial(_run_historical, config_path),
        "backtest": _run_backtest,
    }
    handler = handlers.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        return 2
    return handler()


if __name__ == "__main__":
    raise SystemExit(main())
