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
    return timestamp


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
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        raise ConfigError(
            f"end_time must be after start_time in {config_path} "
            f"(difference was {seconds} seconds)"
        )
    return f"{seconds} S"


@dataclass(frozen=True)
class HistoricalRequestConfig:
    tickers: Sequence[TickerConfig]
    duration: str
    bar_size: str
    start_time: str
    end_time: str
    provider: str
    config_path: Path

    def resolve_window(self) -> tuple[str, str]:
        if self.start_time and not self.end_time:
            raise ConfigError(
                f"start_time requires end_time in {self.config_path}; "
                "provide both or omit start_time."
            )
        if self.start_time and self.end_time:
            start_ts = _parse_timestamp(
                self.start_time, "start_time", self.config_path
            )
            end_ts = _parse_timestamp(self.end_time, "end_time", self.config_path)
            derived_duration = _compute_duration_from_range(
                start_ts, end_ts, self.config_path
            )
            return _format_end_time(end_ts), derived_duration

        end_date_time = ""
        if self.end_time:
            end_ts = _parse_timestamp(self.end_time, "end_time", self.config_path)
            end_date_time = _format_end_time(end_ts)
        duration_value = _require_non_empty(
            self.duration, "duration", self.config_path
        )
        return end_date_time, duration_value

    def to_request(self) -> HistoricalDataRequest:
        end_date_time, duration = self.resolve_window()
        return HistoricalDataRequest(
            tickers=self.tickers,
            duration=duration,
            bar_size=self.bar_size,
            end_date_time=end_date_time,
        )

    @classmethod
    def load(cls, config_path: Path) -> "HistoricalRequestConfig":
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

        config_mapping: Mapping[str, Any] = (
            raw_config if isinstance(raw_config, Mapping) else {}
        )

        start_time = _coerce_to_str(config_mapping.get("start_time"))
        end_time = _coerce_to_str(config_mapping.get("end_time"))
        provider = _coerce_to_str(config_mapping.get("provider")) or DEFAULT_PROVIDER

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

        duration = _coerce_to_str(config_mapping.get("duration"))
        bar_size = _coerce_to_str(config_mapping.get("bar_size"))

        return cls(
            tickers=tickers,
            duration=duration,
            bar_size=bar_size,
            start_time=start_time,
            end_time=end_time,
            provider=provider,
            config_path=config_path,
        )


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
