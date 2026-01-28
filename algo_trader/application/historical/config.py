from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from algo_trader.domain import ConfigError
from algo_trader.domain.market_data import HistoricalDataRequest, TickerConfig

DEFAULT_PROVIDER = "ib"
FX_PAIR_PATTERN = re.compile(r"^([A-Z]{3})\.([A-Z]{3})$")
_DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")


def _coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _require_non_empty(value: Any, field: str, config_path: Path) -> str:
    text = _coerce_to_str(value)
    if not text:
        raise ConfigError(f"{field} is required in {config_path}")
    return text


def _parse_timestamp(value: str, field: str, config_path: Path) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value)
    except Exception as exc:
        raise ConfigError(f"Invalid {field} '{value}' in {config_path}") from exc
    if pd.isna(timestamp):
        raise ConfigError(f"Invalid {field} '{value}' in {config_path}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp


def _is_date_only(value: str) -> bool:
    return bool(_DATE_ONLY_PATTERN.match(value.strip()))


def _is_hour_bar_size(bar_size: str) -> bool:
    return "hour" in bar_size.lower()


def _adjust_end_timestamp(
    end_time: str, end_ts: pd.Timestamp, bar_size: str
) -> pd.Timestamp:
    if _is_date_only(end_time) and _is_hour_bar_size(bar_size):
        return end_ts + pd.Timedelta(days=1)
    return end_ts


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
    seconds_exact = delta.total_seconds()
    if seconds_exact <= 0:
        raise ConfigError(
            f"end_time must be after start_time in {config_path} "
            f"(difference was {seconds_exact} seconds)"
        )
    if not seconds_exact.is_integer():
        raise ConfigError(
            "start_time and end_time must align to whole seconds in "
            f"{config_path}"
        )
    seconds = int(seconds_exact)
    if seconds <= 86400:
        return f"{seconds} S"
    if seconds % 86400 != 0:
        raise ConfigError(
            "start_time and end_time must align to whole days for ranges "
            f"over 1 day in {config_path}"
        )
    days = seconds // 86400
    return f"{days} D"


def _parse_month(value: str, field: str, config_path: Path) -> tuple[int, int]:
    match = _MONTH_PATTERN.match(value.strip())
    if not match:
        raise ConfigError(
            f"Invalid {field} '{value}' in {config_path}; "
            "expected YYYY-MM."
        )
    return int(match.group("year")), int(match.group("month"))


@dataclass(frozen=True)
class HistoricalWindowConfig:
    duration: str
    bar_size: str
    start_time: str
    end_time: str
    month: str

    def resolve_window(self, config_path: Path) -> tuple[str, str]:
        if self.month:
            if self.start_time or self.end_time:
                raise ConfigError(
                    "month cannot be combined with start_time or end_time in "
                    f"{config_path}"
                )
            year, month = _parse_month(self.month, "month", config_path)
            start_ts = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            end_ts = (start_ts + pd.offsets.MonthBegin(1)).normalize()
            derived_duration = _compute_duration_from_range(
                start_ts, end_ts, config_path
            )
            return _format_end_time(end_ts), derived_duration
        if self.start_time and not self.end_time:
            raise ConfigError(
                f"start_time requires end_time in {config_path}; "
                "provide both or omit start_time."
            )
        if self.start_time and self.end_time:
            start_ts = _parse_timestamp(
                self.start_time, "start_time", config_path
            )
            end_ts = _parse_timestamp(self.end_time, "end_time", config_path)
            end_ts = _adjust_end_timestamp(
                self.end_time, end_ts, self.bar_size
            )
            derived_duration = _compute_duration_from_range(
                start_ts, end_ts, config_path
            )
            return _format_end_time(end_ts), derived_duration

        end_date_time = ""
        if self.end_time:
            end_ts = _parse_timestamp(self.end_time, "end_time", config_path)
            end_ts = _adjust_end_timestamp(
                self.end_time, end_ts, self.bar_size
            )
            end_date_time = _format_end_time(end_ts)
        duration_value = _require_non_empty(
            self.duration, "duration", config_path
        )
        return end_date_time, duration_value

    def resolve_export_month(self, config_path: Path) -> tuple[int, int]:
        if self.month:
            if self.start_time or self.end_time:
                raise ConfigError(
                    "month cannot be combined with start_time or end_time in "
                    f"{config_path}"
                )
            return _parse_month(self.month, "month", config_path)
        if not self.start_time or not self.end_time:
            raise ConfigError(
                "CSV export requires start_time and end_time in "
                f"{config_path}"
            )
        start_ts = _parse_timestamp(
            self.start_time, "start_time", config_path
        )
        end_ts = _parse_timestamp(self.end_time, "end_time", config_path)
        end_ts = _adjust_end_timestamp(
            self.end_time, end_ts, self.bar_size
        )
        _compute_duration_from_range(start_ts, end_ts, config_path)
        if start_ts != start_ts.normalize():
            raise ConfigError(
                "CSV export requires start_time at 00:00:00 for "
                f"{config_path}"
            )
        expected_end = (start_ts + pd.offsets.MonthBegin(1)).normalize()
        if end_ts != expected_end:
            raise ConfigError(
                "CSV export requires end_time to be the first day of the "
                "next month at 00:00:00 in "
                f"{config_path}"
            )
        return start_ts.year, start_ts.month


@dataclass(frozen=True)
class HistoricalRequestConfig:
    tickers: Sequence[TickerConfig]
    window: HistoricalWindowConfig
    provider: str
    config_path: Path

    def resolve_window(self) -> tuple[str, str]:
        return self.window.resolve_window(self.config_path)

    def resolve_export_month(self) -> tuple[int, int]:
        return self.window.resolve_export_month(self.config_path)

    def to_request(self) -> HistoricalDataRequest:
        end_date_time, duration = self.resolve_window()
        window_label = self.window.month or None
        return HistoricalDataRequest(
            tickers=self.tickers,
            duration=duration,
            bar_size=self.window.bar_size,
            end_date_time=end_date_time,
            window_label=window_label,
        )

    @classmethod
    def load(cls, config_path: Path) -> "HistoricalRequestConfig":
        config_mapping = _load_yaml_mapping(config_path)
        window = _load_window_config(config_mapping, config_path)
        provider = _load_provider(config_mapping)
        tickers = _load_tickers_from_config(config_mapping, config_path)
        return cls(
            tickers=tickers,
            window=window,
            provider=provider,
            config_path=config_path,
        )


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        example_path = config_path.with_name(
            f"{config_path.stem}.example{config_path.suffix}"
        )
        raise ConfigError(
            f"Ticker config not found at {config_path}. "
            f"Copy {example_path} and customize tickers."
        )
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        raw_config: Any = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML content in {config_path}") from exc

    if not isinstance(raw_config, Mapping):
        raise ConfigError(
            f"Ticker config must be a mapping in {config_path}"
        )
    return raw_config


def _load_provider(config_mapping: Mapping[str, Any]) -> str:
    provider = _coerce_to_str(config_mapping.get("provider"))
    return provider or DEFAULT_PROVIDER


def _load_window_config(
    config_mapping: Mapping[str, Any], config_path: Path
) -> HistoricalWindowConfig:
    if "start_time" in config_mapping or "end_time" in config_mapping:
        raise ConfigError(
            "start_time/end_time are not supported in ticker config; "
            "use month (YYYY-MM) or duration."
        )
    return HistoricalWindowConfig(
        duration=_coerce_to_str(config_mapping.get("duration")),
        bar_size=_coerce_to_str(config_mapping.get("bar_size")),
        start_time="",
        end_time="",
        month=_coerce_to_str(config_mapping.get("month")),
    )


def _load_tickers_from_config(
    config_mapping: Mapping[str, Any], config_path: Path
) -> list[TickerConfig]:
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
    return tickers


@dataclass(frozen=True)
class ContractDefaults:
    sec_type: str
    exchange: str
    currency: str
    what_to_show: str


def _load_asset_tickers(
    asset_name: str, section: Mapping[str, Any], config_path: Path
) -> list[TickerConfig]:
    defaults = _load_contract_defaults(asset_name, section, config_path)
    raw_tickers = section.get("tickers")
    if raw_tickers is None:
        return []
    if not isinstance(raw_tickers, list):
        raise ConfigError(
            f"{asset_name}.tickers must be a list in {config_path}"
        )

    parsed: list[TickerConfig] = []
    for entry in raw_tickers:
        parsed.append(
            _parse_asset_ticker_entry(
                asset_name=asset_name,
                entry=entry,
                defaults=defaults,
                config_path=config_path,
            )
        )

    return parsed


def _load_contract_defaults(
    asset_name: str, section: Mapping[str, Any], config_path: Path
) -> ContractDefaults:
    defaults_raw = section.get("contract_defaults")
    defaults: Mapping[str, Any] = (
        defaults_raw if isinstance(defaults_raw, Mapping) else {}
    )
    sec_type = _require_non_empty(
        defaults.get("sec_type"),
        f"{asset_name}.contract_defaults.sec_type",
        config_path,
    )
    exchange = _require_non_empty(
        defaults.get("exchange"),
        f"{asset_name}.contract_defaults.exchange",
        config_path,
    )
    currency = _coerce_to_str(defaults.get("currency"))
    what_to_show = _require_non_empty(
        defaults.get("what_to_show"),
        f"{asset_name}.contract_defaults.what_to_show",
        config_path,
    )
    return ContractDefaults(
        sec_type=sec_type,
        exchange=exchange,
        currency=currency,
        what_to_show=what_to_show,
    )


def _parse_asset_ticker_entry(
    asset_name: str,
    entry: Any,
    defaults: ContractDefaults,
    config_path: Path,
) -> TickerConfig:
    if asset_name == "forex":
        symbol, currency = _parse_fx_entry(entry, config_path, asset_name)
        return TickerConfig(
            symbol=symbol,
            sec_type=defaults.sec_type,
            currency=currency,
            exchange=defaults.exchange,
            what_to_show=defaults.what_to_show,
            asset_class=asset_name,
        )

    if isinstance(entry, Mapping):
        symbol = _require_non_empty(
            entry.get("symbol"),
            f"{asset_name}.tickers.symbol",
            config_path,
        )
        currency = _coerce_to_str(entry.get("currency")) or defaults.currency
        currency = _require_non_empty(
            currency,
            f"{asset_name}.tickers.currency",
            config_path,
        )
        what_to_show = _coerce_to_str(entry.get("what_to_show")) or defaults.what_to_show
        sec_type = _coerce_to_str(entry.get("sec_type")) or defaults.sec_type
        exchange = _coerce_to_str(entry.get("exchange")) or defaults.exchange
        return TickerConfig(
            symbol=symbol,
            sec_type=sec_type,
            currency=currency,
            exchange=exchange,
            what_to_show=what_to_show,
            asset_class=asset_name,
        )

    if isinstance(entry, str):
        symbol = _require_non_empty(entry, f"{asset_name}.tickers.symbol", config_path)
        currency = _require_non_empty(
            defaults.currency,
            f"{asset_name}.contract_defaults.currency",
            config_path,
        )
        return TickerConfig(
            symbol=symbol,
            sec_type=defaults.sec_type,
            currency=currency,
            exchange=defaults.exchange,
            what_to_show=defaults.what_to_show,
            asset_class=asset_name,
        )

    raise ConfigError(
        f"Unsupported ticker entry in {asset_name} for {config_path}: {entry}"
    )


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
            raise ConfigError(
                f"{asset_name} tickers must use 'BASE.QUOTE' format (e.g., EUR.USD) "
                f"in {config_path}"
            )
        base, quote = match.groups()
        return base, quote

    raise ConfigError(
        f"Unsupported ticker entry in {asset_name} for {config_path}: {entry}"
    )
