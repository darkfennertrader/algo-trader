from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.data_cleaning import runner as data_cleaning_runner
from algo_trader.infrastructure.data import ReturnsSource, ReturnsSourceConfig


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    frame = pd.DataFrame(rows, columns=["Datetime", "Close"])
    frame.to_csv(path, index=False)


def test_returns_source_dedup_keeps_last(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_csv(
        asset_dir / "hist_data_2024-01.csv",
        [
            ("2024-01-01 00:00:00", 100.0),
            ("2024-01-01 23:00:00", 110.0),
        ],
    )
    _write_csv(
        asset_dir / "hist_data_2024-02.csv",
        [
            ("2024-01-01 23:00:00", 120.0),
            ("2024-01-02 23:00:00", 130.0),
        ],
    )

    source = ReturnsSource(
        ReturnsSourceConfig(
            base_dir=tmp_path,
            assets=["EUR.USD"],
            return_type="simple",
        )
    )
    returns = source.get_returns_frame()

    assert list(returns.columns) == ["EUR.USD"]
    assert returns.index.equals(pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))
    assert returns.iloc[0, 0] == pytest.approx((130.0 / 120.0) - 1.0)


def test_returns_source_month_filter_is_inclusive(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_csv(
        asset_dir / "hist_data_2024-01.csv",
        [
            ("2024-01-31 23:00:00", 100.0),
            ("2024-02-01 23:00:00", 110.0),
            ("2024-02-02 23:00:00", 121.0),
        ],
    )

    source = ReturnsSource(
        ReturnsSourceConfig(
            base_dir=tmp_path,
            assets=["EUR.USD"],
            return_type="simple",
            start=(2024, 2),
            end=(2024, 2),
        )
    )
    returns = source.get_returns_frame()

    assert returns.index.equals(pd.DatetimeIndex([pd.Timestamp("2024-02-02")]))
    assert returns.iloc[0, 0] == pytest.approx(0.1)


def test_year_week_uses_iso_week_53() -> None:
    year, week = data_cleaning_runner._year_week(date(2020, 12, 31))
    assert (year, week) == (2020, 53)


def test_build_metadata_includes_source_destination_after_run_at() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1, 0.2]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        ),
    )
    source_dir = Path.home() / "data_source"
    destination_dir = Path.home() / "data_lake" / "2024-01"

    metadata = data_cleaning_runner._build_metadata(
        data_cleaning_runner.MetadataContext(
            returns=frame,
            assets=["EUR.USD"],
            return_type="simple",
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    )

    keys = list(metadata.keys())
    run_at_index = keys.index("run_at")
    assert keys[run_at_index + 1] == "source"
    assert keys[run_at_index + 2] == "destination"
    assert metadata["source"].startswith("~")
    assert metadata["destination"].startswith("~")
