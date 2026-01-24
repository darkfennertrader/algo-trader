from __future__ import annotations

import logging
import re
from datetime import date
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, cast

import pandas as pd

from algo_trader.application.historical import (
    DEFAULT_CONFIG_PATH,
    HistoricalRequestConfig,
)
from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.domain.market_data import TickerConfig
from algo_trader.infrastructure import log_boundary, require_env
from algo_trader.infrastructure.data import (
    ReturnType,
    ReturnsSource,
    ReturnsSourceConfig,
    symbol_directory,
)

YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)

_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")


def _run_context(
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> dict[str, str]:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    return {
        "config_path": str(resolved_path),
        "start": start or "",
        "end": end or "",
        "return_type": return_type,
        "assets": assets or "",
    }


@log_boundary("data_cleaning.run", context=_run_context)
def run(
    *,
    config_path: Path | None,
    start: str | None,
    end: str | None,
    return_type: str,
    assets: str | None,
) -> Path:
    asset_list = _resolve_asset_list(config_path, assets)
    if not asset_list:
        raise ConfigError("No assets available for data_cleaning")
    logger.info("Using %s assets for data_cleaning", len(asset_list))

    data_source, data_lake = _resolve_data_paths()
    start_month, end_month = _parse_month_range(start, end)
    return_type_value = _normalize_return_type(return_type)

    returns = _load_returns(
        base_dir=data_source,
        assets=asset_list,
        return_type=return_type_value,
        start=start_month,
        end=end_month,
    )
    output_path = _write_returns(returns, data_lake)
    _write_metadata(
        returns=returns,
        data_lake=data_lake,
        assets=asset_list,
        return_type=return_type_value,
    )
    logger.info(
        "Saved returns CSV path=%s rows=%s assets=%s",
        output_path,
        len(returns),
        len(returns.columns),
    )
    return output_path


def _resolve_asset_list(
    config_path: Path | None, assets: str | None
) -> list[str]:
    assets_override = _parse_assets(assets)
    tickers: Iterable[TickerConfig] = []
    if assets_override is None:
        config = HistoricalRequestConfig.load(config_path or DEFAULT_CONFIG_PATH)
        tickers = config.tickers
    return _resolve_assets(assets_override, tickers)


def _parse_assets(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    assets = [item.strip() for item in raw.split(",") if item.strip()]
    if not assets:
        raise ConfigError("assets must contain at least one symbol")
    return _dedupe_assets(assets)


def _resolve_assets(
    assets_override: list[str] | None,
    tickers: Iterable[TickerConfig],
) -> list[str]:
    if assets_override is not None:
        return _dedupe_assets(assets_override)
    assets = [symbol_directory(ticker) for ticker in tickers]
    return _dedupe_assets(assets)


def _dedupe_assets(assets: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for asset in assets:
        normalized = asset.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _parse_month(value: str | None, field: str) -> YearMonth | None:
    if value is None or not value.strip():
        return None
    match = _MONTH_PATTERN.match(value.strip())
    if not match:
        raise ConfigError(
            f"{field} must be in YYYY-MM format (received '{value}')"
        )
    return int(match.group("year")), int(match.group("month"))


def _parse_month_range(
    start: str | None, end: str | None
) -> tuple[YearMonth | None, YearMonth | None]:
    start_month = _parse_month(start, "start")
    end_month = _parse_month(end, "end")
    _validate_month_window(start_month, end_month)
    return start_month, end_month


def _validate_month_window(
    start: YearMonth | None, end: YearMonth | None
) -> None:
    if start and end and start > end:
        raise ConfigError("start must be before or equal to end")


def _resolve_data_paths() -> tuple[Path, Path]:
    data_source = Path(require_env("DATA_SOURCE")).expanduser()
    data_lake = Path(require_env("DATA_LAKE_SOURCE")).expanduser()
    _validate_directory(data_source, "DATA_SOURCE")
    _validate_directory(data_lake, "DATA_LAKE_SOURCE")
    if not data_source.exists():
        raise DataSourceError(
            "DATA_SOURCE does not exist",
            context={"path": str(data_source)},
        )
    return data_source, data_lake


def _validate_directory(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise DataSourceError(
            f"{label} must be a directory",
            context={"path": str(path)},
        )


def _load_returns(
    *,
    base_dir: Path,
    assets: list[str],
    return_type: ReturnType,
    start: YearMonth | None,
    end: YearMonth | None,
) -> pd.DataFrame:
    source = ReturnsSource(
        ReturnsSourceConfig(
            base_dir=base_dir,
            assets=assets,
            return_type=return_type,
            start=start,
            end=end,
        )
    )
    return source.get_returns_frame()


def _write_returns(returns: pd.DataFrame, data_lake: Path) -> Path:
    output_dir = _versioned_output_dir(data_lake, date.today())
    output_path = output_dir / "returns.csv"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        returns.to_csv(output_path)
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write returns CSV",
            context={"path": str(output_path)},
        ) from exc
    return output_path


def _write_metadata(
    *,
    returns: pd.DataFrame,
    data_lake: Path,
    assets: list[str],
    return_type: ReturnType,
) -> Path:
    output_dir = _versioned_output_dir(data_lake, date.today())
    output_path = output_dir / "returns_meta.json"
    metadata = _build_metadata(returns, assets, return_type)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        raise DataProcessingError(
            "Failed to write returns metadata",
            context={"path": str(output_path)},
        ) from exc
    return output_path


def _build_metadata(
    returns: pd.DataFrame,
    assets: list[str],
    return_type: ReturnType,
) -> OrderedDict[str, object]:
    index = _normalize_index_dates(returns.index)
    start_date = index[0] if index else ""
    end_date = index[-1] if index else ""
    missing_by_asset: dict[str, dict[str, object]] = {}
    for asset in assets:
        if asset not in returns.columns:
            missing_by_asset[asset] = {
                "missing_dates": index,
                "missing_count": len(index),
            }
            continue
        series = returns[asset]
        missing_mask = series.isna()
        missing_dates = [
            date_str
            for date_str, is_missing in zip(index, missing_mask, strict=False)
            if is_missing
        ]
        missing_by_asset[asset] = {
            "missing_dates": missing_dates,
            "missing_count": len(missing_dates),
        }
    return OrderedDict(
        [
            ("assets", assets),
            ("return_type", return_type),
            ("start_date", start_date),
            ("end_date", end_date),
            ("missing_by_asset", missing_by_asset),
        ]
    )


def _normalize_index_dates(index: pd.Index) -> list[str]:
    if index.empty:
        return []
    if isinstance(index, pd.DatetimeIndex):
        return [item.strftime("%Y-%m-%d") for item in index]
    return [str(item) for item in index]


def _versioned_output_dir(root: Path, run_date: date) -> Path:
    year, week = _year_week(run_date)
    return root / f"{year:04d}-{week:02d}"


def _year_week(run_date: date) -> tuple[int, int]:
    week = (run_date.timetuple().tm_yday - 1) // 7 + 1
    return run_date.year, min(week, 52)


def _normalize_return_type(raw: str) -> ReturnType:
    normalized = raw.strip().lower()
    if normalized not in {"simple", "log"}:
        raise ConfigError(
            f"return_type must be 'simple' or 'log' (received '{raw}')"
        )
    return cast(ReturnType, normalized)
