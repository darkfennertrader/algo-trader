from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd
import torch

from algo_trader.application.historical import (
    DEFAULT_CONFIG_PATH,
    HistoricalRequestConfig,
)
from algo_trader.domain import ConfigError, DataProcessingError, DataSourceError
from algo_trader.domain.market_data import TickerConfig
from algo_trader.infrastructure import (
    ErrorPolicy,
    FileOutputWriter,
    OutputNames,
    OutputPaths,
    OutputWriter,
    build_weekly_output_paths,
    ensure_directory,
    log_boundary,
    require_env,
    write_csv,
)
from algo_trader.infrastructure.paths import format_tilde_path
from algo_trader.infrastructure.data import (
    ReturnFrequency,
    ReturnType,
    ReturnsSource,
    ReturnsSourceConfig,
    require_utc_hourly_index,
    symbol_directory,
    timestamps_to_epoch_hours,
    write_tensor_bundle,
)
from .missing_data import MissingDataSummary, build_missing_data_summary

YearMonth = tuple[int, int]

logger = logging.getLogger(__name__)

_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")
_OUTPUT_NAMES = OutputNames(
    output_name="returns.csv",
    metadata_name="returns_meta.json",
)
_RETURN_FREQUENCY: ReturnFrequency = "weekly"
_WEEKLY_OHLC_NAME = "weekly_ohlc.csv"
_WEEKLY_OHLC_META_NAME = "weekly_ohlc_meta.json"
_DAILY_OHLC_NAME = "daily_ohlc.csv"
_DAILY_OHLC_META_NAME = "daily_ohlc_meta.json"
_RETURN_TENSOR_NAME = "return_tensor.pt"
_CHECK_AVERAGE_NAME = "check_average.json"
_TENSOR_METADATA_NAME = "tensor_metadata.json"
_TENSOR_SCALE = 1_000_000
_TENSOR_TIMESTAMP_UNIT = "epoch_hours"
_TENSOR_TIMEZONE = "UTC"
_TENSOR_VALUE_DTYPE = "int64"


@dataclass(frozen=True)
class MetadataContext:
    returns: pd.DataFrame
    assets: list[str]
    return_profile: "ReturnProfile"
    source_dir: Path
    destination_dir: Path
    monthly_avg_close_by_asset: dict[str, dict[str, Decimal]] = field(
        default_factory=dict
    )
    tensor_info: ReturnTensorInfo | None = None


@dataclass(frozen=True)
class ReturnTensorBundle:
    values: torch.Tensor
    timestamps: torch.Tensor
    missing_mask: torch.Tensor
    assets: list[str]


@dataclass(frozen=True)
class ReturnTensorInfo:
    path: Path
    assets: list[str]
    scale: int
    timestamp_unit: str
    timezone: str
    dtype: str
    missing_mask: bool = True


@dataclass(frozen=True)
class RunConfig:
    assets: list[str]
    data_source: Path
    data_lake: Path
    start: YearMonth | None
    end: YearMonth | None
    return_type: ReturnType


@dataclass(frozen=True)
class RunRequest:
    config_path: Path | None
    start: str | None
    end: str | None
    return_type: str
    assets: str | None


@dataclass(frozen=True)
class ReturnProfile:
    return_type: ReturnType
    return_frequency: ReturnFrequency


@dataclass(frozen=True)
class ReturnData:
    returns: pd.DataFrame
    monthly_avg_close_by_asset: dict[str, dict[str, Decimal]]
    weekly_ohlc: pd.DataFrame
    weekly_ohlc_time_meta: dict[str, pd.DataFrame]
    daily_ohlc: pd.DataFrame
    missing_summary: MissingDataSummary


def _run_context(request: RunRequest) -> dict[str, str]:
    resolved_path = request.config_path or DEFAULT_CONFIG_PATH
    return {
        "config_path": str(resolved_path),
        "start": request.start or "",
        "end": request.end or "",
        "return_type": request.return_type,
        "assets": request.assets or "",
    }


@log_boundary("data_cleaning.run", context=_run_context)
def run(
    *,
    request: RunRequest,
) -> Path:
    total_start = time.perf_counter()
    config = _resolve_run_config(request)
    workers = _resolve_parallel_workers(len(config.assets))
    logger.info(
        "Using %s assets for data_cleaning workers=%s",
        len(config.assets),
        workers,
    )

    phase_start = time.perf_counter()
    return_data = _load_return_data(config)
    logger.info(
        "Data cleaning loaded inputs duration=%.2fs",
        time.perf_counter() - phase_start,
    )

    phase_start = time.perf_counter()
    output_paths = _prepare_output_paths(config.data_lake, date.today())
    writer = _build_output_writer()
    output_path = _write_returns(
        return_data.returns, output_paths.output_path, writer
    )
    logger.info(
        "Data cleaning wrote returns duration=%.2fs",
        time.perf_counter() - phase_start,
    )

    phase_start = time.perf_counter()
    weekly_ohlc_path = _write_weekly_ohlc(
        return_data.weekly_ohlc, output_paths.output_dir
    )
    daily_ohlc_path = _write_daily_ohlc(
        return_data.daily_ohlc, output_paths.output_dir
    )
    _write_weekly_ohlc_metadata(
        return_data.weekly_ohlc_time_meta,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    _write_daily_ohlc_metadata(
        return_data.missing_summary,
        source_dir=config.data_source,
        destination_dir=output_paths.output_dir,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    logger.info(
        "Data cleaning wrote OHLC duration=%.2fs",
        time.perf_counter() - phase_start,
    )

    phase_start = time.perf_counter()
    tensor_info = _write_return_tensor(
        return_data.returns, output_paths.output_dir
    )
    metadata_context = MetadataContext(
        returns=return_data.returns,
        assets=config.assets,
        return_profile=ReturnProfile(
            return_type=config.return_type,
            return_frequency=_RETURN_FREQUENCY,
        ),
        source_dir=config.data_source,
        destination_dir=output_paths.output_dir,
        monthly_avg_close_by_asset=return_data.monthly_avg_close_by_asset,
        tensor_info=tensor_info,
    )
    _write_check_average(
        metadata_context,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    _write_tensor_metadata(
        metadata_context,
        output_dir=output_paths.output_dir,
        writer=writer,
    )
    _write_metadata(
        metadata_context,
        metadata_path=output_paths.metadata_path,
        writer=writer,
    )
    logger.info(
        "Data cleaning wrote metadata duration=%.2fs",
        time.perf_counter() - phase_start,
    )
    logger.info(
        "Saved returns CSV path=%s rows=%s assets=%s",
        output_path,
        len(return_data.returns),
        len(return_data.returns.columns),
    )
    logger.info(
        "Saved weekly OHLC CSV path=%s rows=%s assets=%s",
        weekly_ohlc_path,
        len(return_data.weekly_ohlc),
        len(return_data.weekly_ohlc.columns),
    )
    logger.info(
        "Saved daily OHLC CSV path=%s rows=%s assets=%s",
        daily_ohlc_path,
        len(return_data.daily_ohlc),
        len(return_data.daily_ohlc.columns),
    )
    duration = time.perf_counter() - total_start
    logger.info("----- TOTAL -----")
    logger.info(
        "Data cleaning completed duration=%.2fm",
        duration / 60.0,
    )
    return output_path


def _resolve_run_config(request: RunRequest) -> RunConfig:
    asset_list = _resolve_asset_list(
        request.config_path, request.assets
    )
    if not asset_list:
        raise ConfigError("No assets available for data_cleaning")
    data_source, data_lake = _resolve_data_paths()
    start_month, end_month = _parse_month_range(
        request.start, request.end
    )
    return_type_value = _normalize_return_type(request.return_type)
    return RunConfig(
        assets=asset_list,
        data_source=data_source,
        data_lake=data_lake,
        start=start_month,
        end=end_month,
        return_type=return_type_value,
    )


def _load_return_data(config: RunConfig) -> ReturnData:
    source = _build_returns_source(config)
    returns = source.get_returns_frame()
    daily_prices = source.get_daily_price_series()
    monthly_avg_close_by_asset = _monthly_average_closes(daily_prices)
    weekly_ohlc, weekly_ohlc_time_meta = source.get_weekly_ohlc_bundle()
    hourly_ohlc = source.get_hourly_ohlc_data()
    daily_ohlc = source.get_daily_ohlc_frame()
    missing_summary = build_missing_data_summary(
        hourly_ohlc, assets=config.assets
    )
    return ReturnData(
        returns=returns,
        monthly_avg_close_by_asset=monthly_avg_close_by_asset,
        weekly_ohlc=weekly_ohlc,
        weekly_ohlc_time_meta=weekly_ohlc_time_meta,
        daily_ohlc=daily_ohlc,
        missing_summary=missing_summary,
    )


def _resolve_asset_list(
    config_path: Path | None, assets: str | None
) -> list[str]:
    assets_override = _parse_assets(assets)
    tickers: Iterable[TickerConfig] = []
    if assets_override is None:
        config = HistoricalRequestConfig.load(
            config_path or DEFAULT_CONFIG_PATH
        )
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
    if start is None or not start.strip():
        raise ConfigError("start is required for data_cleaning")
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


def _build_returns_source(config: RunConfig) -> ReturnsSource:
    workers = _resolve_parallel_workers(len(config.assets))
    return ReturnsSource(
        ReturnsSourceConfig(
            base_dir=config.data_source,
            assets=config.assets,
            return_type=config.return_type,
            start=config.start,
            end=config.end,
            workers=workers,
        )
    )


def _resolve_parallel_workers(asset_count: int) -> int:
    if asset_count <= 1:
        return 1
    count = os.cpu_count() or 1
    return min(asset_count, max(1, count - 1))


def _prepare_output_paths(data_lake: Path, run_date: date) -> OutputPaths:
    output_paths = build_weekly_output_paths(
        data_lake, run_date, _OUTPUT_NAMES
    )
    ensure_directory(
        output_paths.output_dir,
        error_type=DataProcessingError,
        invalid_message="Data cleaning output path must be a directory",
        create_message="Failed to prepare data cleaning output directory",
    )
    return output_paths


def _build_output_writer() -> OutputWriter:
    return FileOutputWriter(
        data_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write returns CSV",
        ),
        metadata_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write returns metadata",
        ),
    )


def _write_returns(
    returns: pd.DataFrame,
    output_path: Path,
    writer: OutputWriter,
) -> Path:
    writer.write_frame(returns, output_path)
    return output_path


def _write_weekly_ohlc(
    weekly_ohlc: pd.DataFrame,
    output_dir: Path,
) -> Path:
    require_utc_hourly_index(
        weekly_ohlc.index, label="weekly_ohlc", timezone=_TENSOR_TIMEZONE
    )
    path = output_dir / _WEEKLY_OHLC_NAME
    write_csv(
        weekly_ohlc,
        path,
        error_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write weekly OHLC CSV",
        ),
    )
    return path


def _write_daily_ohlc(
    daily_ohlc: pd.DataFrame,
    output_dir: Path,
) -> Path:
    if not isinstance(daily_ohlc.index, pd.DatetimeIndex):
        raise DataProcessingError(
            "daily_ohlc index must be datetime",
            context={"index_type": type(daily_ohlc.index).__name__},
        )
    path = output_dir / _DAILY_OHLC_NAME
    write_csv(
        daily_ohlc,
        path,
        error_policy=ErrorPolicy(
            error_type=DataProcessingError,
            message="Failed to write daily OHLC CSV",
        ),
    )
    return path


def _write_weekly_ohlc_metadata(
    time_meta_by_asset: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_weekly_ohlc_metadata(time_meta_by_asset)
    path = output_dir / _WEEKLY_OHLC_META_NAME
    writer.write_metadata(payload, path)
    return path


def _write_daily_ohlc_metadata(
    missing_summary: MissingDataSummary,
    *,
    source_dir: Path,
    destination_dir: Path,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_daily_ohlc_metadata(
        missing_summary,
        source_dir=source_dir,
        destination_dir=destination_dir,
    )
    path = output_dir / _DAILY_OHLC_META_NAME
    writer.write_metadata(payload, path)
    return path


def _build_daily_ohlc_metadata(
    missing_summary: MissingDataSummary,
    *,
    source_dir: Path,
    destination_dir: Path,
) -> OrderedDict[str, object]:
    missing_by_asset = _build_missing_by_asset_payload(missing_summary)
    return OrderedDict(
        [
            ("run_at", _format_run_at_local(datetime.now(timezone.utc))),
            ("source", format_tilde_path(source_dir)),
            ("destination", format_tilde_path(destination_dir)),
            ("missing_by_asset", missing_by_asset),
        ]
    )


def _build_weekly_ohlc_metadata(
    time_meta_by_asset: dict[str, pd.DataFrame],
) -> OrderedDict[str, object]:
    metadata: OrderedDict[str, object] = OrderedDict()
    for asset, frame in time_meta_by_asset.items():
        asset_records: list[OrderedDict[str, str]] = []
        if not frame.empty:
            frame = frame.copy()
            for week_start, row in frame.iterrows():
                record = OrderedDict()
                week_start_value = row.get("open_time")
                week_end_value = (
                    row.get("week_end")
                    if "week_end" in row
                    else None
                )
                record["week_start"] = _format_timestamp(week_start_value)
                record["week_end"] = _format_timestamp(week_end_value)
                record["open_time"] = _format_timestamp(
                    row.get("open_time")
                )
                record["high_time"] = _format_timestamp(
                    row.get("high_time")
                )
                record["low_time"] = _format_timestamp(
                    row.get("low_time")
                )
                record["close_time"] = _format_timestamp(
                    row.get("close_time")
                )
                asset_records.append(record)
        metadata[asset] = asset_records
    return metadata


def _build_missing_by_asset_payload(
    missing_summary: MissingDataSummary,
) -> OrderedDict[str, object]:
    payload: OrderedDict[str, object] = OrderedDict()
    for asset, missing_timestamps in missing_summary.missing_by_asset.items():
        missing_data = [
            _format_timestamp(timestamp) for timestamp in missing_timestamps
        ]
        month_counts = missing_summary.missing_counts_by_month.get(
            asset, {}
        )
        payload[asset] = OrderedDict(
            [
                ("missing_data", missing_data),
                ("missing_count_by_month", month_counts),
            ]
        )
    return payload


def _format_timestamp(value: object) -> str:
    if value is None or value is pd.NaT:
        return ""
    if isinstance(value, pd.Timestamp):
        timestamp = value
    elif isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
    else:
        return ""
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_run_at_local(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local = value.astimezone()
    tz_name = local.tzname()
    if tz_name:
        return f"{local.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}"
    offset = local.strftime("%z")
    if offset:
        return f"{local.strftime('%Y-%m-%d %H:%M:%S')} {offset}"
    return local.strftime("%Y-%m-%d %H:%M:%S")


def _write_metadata(
    context: MetadataContext,
    *,
    metadata_path: Path,
    writer: OutputWriter,
) -> Path:
    metadata = _build_metadata(context)
    writer.write_metadata(metadata, metadata_path)
    return metadata_path


def _build_metadata(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    index = _normalize_index_dates(context.returns.index)
    start_date = index[0] if index else ""
    end_date = index[-1] if index else ""
    missing_by_asset: dict[str, dict[str, object]] = {}
    missing_weeks_by_asset = _build_missing_weeks_by_asset(context)
    zero_returns_by_asset: dict[str, dict[str, object]] = {}
    for asset in context.assets:
        if asset not in context.returns.columns:
            missing_by_asset[asset] = {
                "missing_dates": index,
                "missing_count": len(index),
            }
            zero_returns_by_asset[asset] = {
                "zero_dates": [],
                "zero_count": 0,
            }
            continue
        series = context.returns[asset]
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
        zero_mask = series.eq(0) & series.notna()
        zero_dates = [
            date_str
            for date_str, is_zero in zip(index, zero_mask, strict=False)
            if is_zero
        ]
        zero_returns_by_asset[asset] = {
            "zero_dates": zero_dates,
            "zero_count": len(zero_dates),
        }
    return OrderedDict(
        [
            ("assets", context.assets),
            ("return_type", context.return_profile.return_type),
            ("return_frequency", context.return_profile.return_frequency),
            ("start_date", start_date),
            ("end_date", end_date),
            ("missing_by_asset", missing_by_asset),
            ("missing_weeks_by_asset", missing_weeks_by_asset),
            ("zero_returns_by_asset", zero_returns_by_asset),
            ("run_at", _format_run_at_local(datetime.now(timezone.utc))),
            ("source", format_tilde_path(context.source_dir)),
            ("destination", format_tilde_path(context.destination_dir)),
        ]
    )


def _write_check_average(
    context: MetadataContext,
    *,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_check_average(context)
    path = output_dir / _CHECK_AVERAGE_NAME
    writer.write_metadata(payload, path)
    return path


def _build_check_average(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    return OrderedDict(
        [
            (
                "monthly_avg_close_by_asset",
                context.monthly_avg_close_by_asset,
            ),
            ("run_at", _format_run_at_local(datetime.now(timezone.utc))),
        ]
    )


def _write_tensor_metadata(
    context: MetadataContext,
    *,
    output_dir: Path,
    writer: OutputWriter,
) -> Path:
    payload = _build_tensor_metadata(context)
    path = output_dir / _TENSOR_METADATA_NAME
    writer.write_metadata(payload, path)
    return path


def _build_tensor_metadata(
    context: MetadataContext,
) -> OrderedDict[str, object]:
    return OrderedDict(
        [
            ("tensor", _tensor_metadata(context)),
            ("run_at", _format_run_at_local(datetime.now(timezone.utc))),
        ]
    )


def _build_missing_weeks_by_asset(
    context: MetadataContext,
) -> dict[str, dict[str, object]]:
    week_labels = _iso_week_labels(context.returns.index)
    unique_weeks = _unique_ordered(week_labels)
    missing_weeks_by_asset: dict[str, dict[str, object]] = {}
    for asset in context.assets:
        missing_weeks = _asset_missing_weeks(
            context.returns, asset, week_labels, unique_weeks
        )
        missing_weeks_by_asset[asset] = {
            "missing_weeks": missing_weeks,
            "missing_count": len(missing_weeks),
        }
    return missing_weeks_by_asset


def _asset_missing_weeks(
    returns: pd.DataFrame,
    asset: str,
    week_labels: list[str],
    unique_weeks: list[str],
) -> list[str]:
    if returns.empty or asset not in returns.columns:
        return unique_weeks
    series = returns[asset]
    if series.empty:
        return unique_weeks
    if not week_labels:
        return []
    week_index = pd.Series(week_labels, index=series.index)
    grouped = series.groupby(week_index, sort=False)
    missing_weeks: list[str] = []
    for week, group in grouped:
        if group.isna().all():
            missing_weeks.append(str(week))
    return missing_weeks


def _iso_week_labels(index: pd.Index) -> list[str]:
    if index.empty:
        return []
    if isinstance(index, pd.DatetimeIndex):
        iso = index.isocalendar()
        years = iso["year"].astype(str)
        weeks = iso["week"].astype(str).str.zfill(2)
        return [
            f"{year}-W{week}"
            for year, week in zip(years, weeks, strict=False)
        ]
    return [str(item) for item in index]


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _monthly_average_closes(
    daily_prices: dict[str, pd.Series],
) -> dict[str, dict[str, Decimal]]:
    monthly_averages: dict[str, dict[str, Decimal]] = {}
    for asset, series in daily_prices.items():
        if series.empty:
            monthly_averages[asset] = {}
            continue
        cleaned = series.dropna()
        if cleaned.empty:
            monthly_averages[asset] = {}
            continue
        monthly = cleaned.resample("MS").mean()
        monthly_averages[asset] = {}
        for timestamp, value in monthly.items():
            if pd.notna(value):
                month_key = cast(pd.Timestamp, timestamp).strftime("%Y-%m")
                monthly_averages[asset][month_key] = _to_decimal(value)
    return monthly_averages


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _normalize_index_dates(index: pd.Index) -> list[str]:
    if index.empty:
        return []
    if isinstance(index, pd.DatetimeIndex):
        return [_format_timestamp(item) for item in index]
    return [str(item) for item in index]


def _normalize_return_type(raw: str) -> ReturnType:
    normalized = raw.strip().lower()
    if normalized not in {"simple", "log"}:
        raise ConfigError(
            f"return_type must be 'simple' or 'log' (received '{raw}')"
        )
    return cast(ReturnType, normalized)


def _year_week(run_date: date) -> tuple[int, int]:
    iso = run_date.isocalendar()
    return iso.year, iso.week


def _write_return_tensor(
    returns: pd.DataFrame, output_dir: Path
) -> ReturnTensorInfo:
    bundle = _build_return_tensor_bundle(returns, scale=_TENSOR_SCALE)
    tensor_path = output_dir / _RETURN_TENSOR_NAME
    write_tensor_bundle(
        tensor_path,
        values=bundle.values,
        timestamps=bundle.timestamps,
        missing_mask=bundle.missing_mask,
        error_message="Failed to write return tensor",
    )
    return ReturnTensorInfo(
        path=tensor_path,
        assets=bundle.assets,
        scale=_TENSOR_SCALE,
        timestamp_unit=_TENSOR_TIMESTAMP_UNIT,
        timezone=_TENSOR_TIMEZONE,
        dtype=_TENSOR_VALUE_DTYPE,
        missing_mask=True,
    )


def _build_return_tensor_bundle(
    returns: pd.DataFrame, *, scale: int
) -> ReturnTensorBundle:
    if returns.empty:
        raise DataProcessingError(
            "Returns frame is empty",
            context={"rows": "0"},
        )
    index = require_utc_hourly_index(
        returns.index, label="Returns", timezone=_TENSOR_TIMEZONE
    )
    assets = [str(asset) for asset in returns.columns]
    values = returns.to_numpy(dtype=float)
    missing_mask = np.isnan(values)
    safe_values = np.where(missing_mask, 0.0, values)
    scaled = np.rint(safe_values * scale).astype("int64")
    timestamps = timestamps_to_epoch_hours(index)
    return ReturnTensorBundle(
        values=torch.as_tensor(scaled, dtype=torch.int64),
        timestamps=torch.as_tensor(timestamps, dtype=torch.int64),
        missing_mask=torch.as_tensor(missing_mask, dtype=torch.bool),
        assets=assets,
    )




def _tensor_metadata(
    context: MetadataContext,
) -> dict[str, object] | None:
    if context.tensor_info is None:
        return None
    rows, cols = context.returns.shape
    return {
        "path": format_tilde_path(context.tensor_info.path),
        "assets": context.tensor_info.assets,
        "scale": context.tensor_info.scale,
        "dtype": context.tensor_info.dtype,
        "timestamp_unit": context.tensor_info.timestamp_unit,
        "timezone": context.tensor_info.timezone,
        "missing_mask": context.tensor_info.missing_mask,
        "values_shape": [rows, cols],
        "timestamps_shape": [rows],
        "missing_mask_shape": [rows, cols],
    }
