from pathlib import Path

import pytest

from algo_trader.application.historical import (
    HistoricalRequestConfig,
    HistoricalWindowConfig,
)
from algo_trader.domain import ConfigError
from algo_trader.domain.market_data import (
    Bar,
    HistoricalDataRequest,
    HistoricalDataResult,
    TickerConfig,
)
from algo_trader.infrastructure.exporters import CsvHistoricalDataExporter


def _build_config(
    start_time: str,
    end_time: str,
    bar_size: str = "1 min",
    month: str = "",
) -> HistoricalRequestConfig:
    ticker = TickerConfig(
        symbol="AUD",
        sec_type="CASH",
        currency="CAD",
        exchange="IDEALPRO",
        what_to_show="MIDPOINT",
        asset_class="forex",
    )
    window = HistoricalWindowConfig(
        duration="1 D",
        bar_size=bar_size,
        start_time=start_time,
        end_time=end_time,
        month=month,
    )
    return HistoricalRequestConfig(
        tickers=[ticker],
        window=window,
        provider="ib",
        config_path=Path("config/tickers.yml"),
    )


def test_resolve_export_month_requires_window() -> None:
    config = _build_config("", "")
    with pytest.raises(ConfigError):
        config.resolve_export_month()


def test_resolve_export_month_requires_full_month_window() -> None:
    config = _build_config(
        "2023-01-01 00:00:00+00:00",
        "2023-01-31 00:00:00+00:00",
    )
    with pytest.raises(ConfigError):
        config.resolve_export_month()


def test_load_rejects_start_end_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "tickers.yml"
    config_path.write_text(
        'start_time: "2023-01-01"\n'
        'end_time: "2023-01-02"\n'
        'duration: "1 D"\n'
        'bar_size: "1 min"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        HistoricalRequestConfig.load(config_path)


def test_load_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "tickers.yml"
    config_path.write_text("- foo\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        HistoricalRequestConfig.load(config_path)


def test_load_accepts_month_only(tmp_path: Path) -> None:
    config_path = tmp_path / "tickers.yml"
    config_path.write_text(
        'month: "2023-01"\n'
        'duration: "1 D"\n'
        'bar_size: "1 min"\n',
        encoding="utf-8",
    )
    config = HistoricalRequestConfig.load(config_path)
    assert config.window.month == "2023-01"
    assert config.window.start_time == ""
    assert config.window.end_time == ""


def test_resolve_export_month_accepts_month_only() -> None:
    config = _build_config("", "", month="2023-01")
    assert config.resolve_export_month() == (2023, 1)


def test_resolve_export_month_returns_year_month() -> None:
    config = _build_config(
        "2023-01-01 00:00:00+00:00",
        "2023-02-01 00:00:00+00:00",
    )
    assert config.resolve_export_month() == (2023, 1)


def test_resolve_window_uses_seconds_for_sub_day() -> None:
    config = _build_config(
        "2023-01-01 00:00:00",
        "2023-01-01 12:00:00",
    )
    end_date_time, duration = config.resolve_window()
    assert duration == "43200 S"
    assert end_date_time == "20230101 12:00:00 UTC"


def test_resolve_window_uses_days_for_multi_day() -> None:
    config = _build_config(
        "2023-01-01",
        "2023-01-03",
    )
    end_date_time, duration = config.resolve_window()
    assert duration == "2 D"
    assert end_date_time == "20230103 00:00:00 UTC"


def test_resolve_window_uses_month_only() -> None:
    config = _build_config("", "", month="2023-12")
    end_date_time, duration = config.resolve_window()
    assert duration == "31 D"
    assert end_date_time == "20240101 00:00:00 UTC"


def test_resolve_window_rejects_month_with_start_end() -> None:
    config = _build_config(
        "2023-01-01",
        "2023-01-03",
        month="2023-01",
    )
    with pytest.raises(ConfigError):
        config.resolve_window()


def test_resolve_window_rejects_partial_days_over_one_day() -> None:
    config = _build_config(
        "2023-01-01 00:00:00",
        "2023-01-02 12:00:00",
    )
    with pytest.raises(ConfigError):
        config.resolve_window()


def test_resolve_window_extends_date_only_end_for_hour_bars() -> None:
    config = _build_config(
        "2023-01-01",
        "2023-01-31",
        bar_size="1 hour",
    )
    end_date_time, duration = config.resolve_window()
    assert duration == "31 D"
    assert end_date_time == "20230201 00:00:00 UTC"


def test_csv_exporter_writes_csv(tmp_path: Path) -> None:
    exporter = CsvHistoricalDataExporter(
        output_root=tmp_path,
        year=2023,
        month=1,
    )
    ticker = TickerConfig(
        symbol="AUD",
        sec_type="CASH",
        currency="CAD",
        exchange="IDEALPRO",
        what_to_show="MIDPOINT",
        asset_class="forex",
    )
    request = HistoricalDataRequest(
        tickers=[ticker],
        duration="1 D",
        bar_size="1 min",
        end_date_time="",
    )
    result = HistoricalDataResult(
        bars_by_symbol={
            "AUD": [
                Bar(
                    timestamp="2023-01-02 00:00:00",
                    open=1.0,
                    high=1.2,
                    low=0.9,
                    close=1.1,
                    volume=10.0,
                ),
                Bar(
                    timestamp="2023-01-02 00:01:00",
                    open=1.1,
                    high=1.3,
                    low=1.0,
                    close=1.2,
                    volume=11.0,
                ),
            ]
        },
        outcomes={},
    )

    exporter.export(request, result)

    csv_path = tmp_path / "AUD.CAD" / "2023" / "hist_data_2023-01.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Datetime,Open,High,Low,Close,Volume"
    assert lines[1].startswith("2023-01-02 00:00:00")


def test_csv_exporter_writes_commodities_pair_dir(
    tmp_path: Path,
) -> None:
    exporter = CsvHistoricalDataExporter(
        output_root=tmp_path,
        year=2023,
        month=1,
    )
    ticker = TickerConfig(
        symbol="XAGUSD",
        sec_type="CFD",
        currency="USD",
        exchange="SMART",
        what_to_show="MIDPOINT",
        asset_class="commodities",
    )
    request = HistoricalDataRequest(
        tickers=[ticker],
        duration="1 D",
        bar_size="1 min",
        end_date_time="",
    )
    result = HistoricalDataResult(
        bars_by_symbol={
            "XAGUSD": [
                Bar(
                    timestamp="2023-01-02 00:00:00",
                    open=20.0,
                    high=20.2,
                    low=19.9,
                    close=20.1,
                    volume=5.0,
                )
            ]
        },
        outcomes={},
    )

    exporter.export(request, result)

    csv_path = tmp_path / "XAG.USD" / "2023" / "hist_data_2023-01.csv"
    assert csv_path.exists()
