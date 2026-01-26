from __future__ import annotations

import argparse
import logging
from pathlib import Path
from functools import partial
from typing import Sequence

from algo_trader.domain import AlgoTraderError
from algo_trader.infrastructure import log_boundary
from algo_trader.application.historical import (
    DEFAULT_CONFIG_PATH,
    configure_logging,
    run,
)
from algo_trader.application.data_cleaning import run as run_data_cleaning
from algo_trader.application.data_processing import run as run_data_processing
from algo_trader.cli.wizard import run as run_wizard

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

    data_cleaning_parser = subparsers.add_parser(
        "data_cleaning",
        help="Clean hourly data into daily returns.",
    )
    data_cleaning_parser.add_argument(
        "--start",
        help="Start month in YYYY-MM format (inclusive).",
    )
    data_cleaning_parser.add_argument(
        "--end",
        help="End month in YYYY-MM format (inclusive).",
    )
    data_cleaning_parser.add_argument(
        "--return-type",
        choices=["simple", "log"],
        default="simple",
        help="Return type to compute.",
    )
    data_cleaning_parser.add_argument(
        "--assets",
        help="Comma-separated list of assets to process.",
    )

    data_processing_parser = subparsers.add_parser(
        "data_processing",
        help="Preprocess returns data.",
    )
    data_processing_parser.add_argument(
        "--preprocessor",
        default="identity",
        help="Name of the preprocessor to apply.",
    )
    data_processing_parser.add_argument(
        "--preprocessor-arg",
        action="append",
        help="Preprocessor argument in key=value format (repeatable).",
    )

    subparsers.add_parser(
        "wizard",
        help="Interactive wizard to build CLI commands.",
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


def _run_data_cleaning(
    *,
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> int:
    run_data_cleaning(
        config_path=config_path,
        start=start,
        end=end,
        return_type=return_type,
        assets=assets,
    )
    return 0


def _run_data_processing(
    *,
    preprocessor: str,
    preprocessor_args: list[str] | None,
) -> int:
    run_data_processing(
        preprocessor_name=preprocessor, preprocessor_args=preprocessor_args
    )
    return 0


def _run_wizard() -> int:
    return run_wizard()


def _cli_context(argv: Sequence[str] | None) -> dict[str, str]:
    if not argv:
        return {}
    return {"argv": " ".join(argv)}


@log_boundary("cli.dispatch", context=_cli_context)
def _dispatch(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = getattr(args, "config", None)
    handlers = {
        None: _run_pipeline,
        "historical": partial(_run_historical, config_path),
        "backtest": _run_backtest,
        "data_cleaning": partial(
            _run_data_cleaning,
            config_path=config_path,
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
            return_type=getattr(args, "return_type", "simple"),
            assets=getattr(args, "assets", None),
        ),
        "data_processing": partial(
            _run_data_processing,
            preprocessor=getattr(args, "preprocessor", ""),
            preprocessor_args=getattr(args, "preprocessor_arg", None),
        ),
        "wizard": _run_wizard,
    }
    handler = handlers.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        return 2
    try:
        return handler()
    except AlgoTraderError as exc:
        if not getattr(exc, "_logged", False):
            if exc.context:
                logger.error("Error: %s context=%s", exc, exc.context)
            else:
                logger.error("Error: %s", exc)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    return _dispatch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
