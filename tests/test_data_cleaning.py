from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
