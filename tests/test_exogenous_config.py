from pathlib import Path

import pytest

from algo_trader.application.exogenous import FredRequestConfig
from algo_trader.domain import ConfigError


def test_fred_config_loads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-01-01"\n'
        'end_date: "2020-12-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "market_risk"\n'
        '    units: "lin"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n',
        encoding="utf-8",
    )

    config = FredRequestConfig.load(config_path)

    assert config.provider == "fred"
    assert config.start_date == "2020-01-01"
    assert config.end_date == "2020-12-31"
    assert len(config.series) == 1
    assert config.series[0].series_id == "VIXCLS"
    assert config.series[0].dir_name == "market_risk"


def test_fred_config_accepts_nested_dir_name(tmp_path: Path) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-01-01"\n'
        'end_date: "2020-12-31"\n'
        "series:\n"
        '  - id: "IR3TIB01USM156N"\n'
        '    dir_name: "carry/USD"\n',
        encoding="utf-8",
    )

    config = FredRequestConfig.load(config_path)

    assert config.series[0].dir_name == "carry/USD"


def test_fred_config_rejects_invalid_date_order(tmp_path: Path) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-12-31"\n'
        'end_date: "2020-01-01"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "market_risk"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        FredRequestConfig.load(config_path)


def test_fred_config_rejects_invalid_dir_name(tmp_path: Path) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-01-01"\n'
        'end_date: "2020-12-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "../unsafe"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        FredRequestConfig.load(config_path)


def test_fred_config_rejects_absolute_dir_name(tmp_path: Path) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2020-01-01"\n'
        'end_date: "2020-12-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "/unsafe"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        FredRequestConfig.load(config_path)
