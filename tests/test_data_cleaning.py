from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.data_cleaning import (
    build_missing_data_summary,
    runner as data_cleaning_runner,
)
from algo_trader.domain import DataProcessingError
from algo_trader.infrastructure.data import ReturnsSource, ReturnsSourceConfig


def _write_ohlc_csv(
    path: Path, rows: list[tuple[str, float, float, float, float]]
) -> None:
    frame = pd.DataFrame(
        rows, columns=["Datetime", "Open", "High", "Low", "Close"]
    )
    frame.to_csv(path, index=False)


def test_returns_source_dedup_keeps_last(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_ohlc_csv(
        asset_dir / "hist_data_2024-01.csv",
        [
            ("2024-01-01 09:00:00", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-01 09:00:00", 120.0, 120.0, 120.0, 120.0),
            ("2024-01-05 16:00:00", 130.0, 130.0, 130.0, 130.0),
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
    assert returns.index.equals(
        pd.DatetimeIndex([pd.Timestamp("2024-01-05 16:00:00")])
    )
    assert returns.iloc[0, 0] == pytest.approx((130.0 / 120.0) - 1.0)


def test_returns_source_month_filter_is_inclusive(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_ohlc_csv(
        asset_dir / "hist_data_2024-02.csv",
        [
            ("2024-01-31 23:00:00", 100.0, 100.0, 100.0, 100.0),
            ("2024-02-01 09:00:00", 110.0, 110.0, 110.0, 110.0),
            ("2024-02-02 16:00:00", 121.0, 121.0, 121.0, 121.0),
            ("2024-02-05 09:00:00", 100.0, 100.0, 100.0, 100.0),
            ("2024-02-09 16:00:00", 110.0, 110.0, 110.0, 110.0),
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

    assert returns.index.equals(
        pd.DatetimeIndex([pd.Timestamp("2024-02-09 16:00:00")])
    )
    assert returns.iloc[0, 0] == pytest.approx(0.1)


def test_returns_source_weekly_uses_week_bounds(tmp_path: Path) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_ohlc_csv(
        asset_dir / "hist_data_2024-01.csv",
        [
            ("2024-01-01 09:00:00", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03 12:00:00", 115.0, 115.0, 115.0, 115.0),
            ("2024-01-05 16:00:00", 120.0, 120.0, 120.0, 120.0),
            ("2024-01-08 09:00:00", 130.0, 130.0, 130.0, 130.0),
            ("2024-01-09 10:00:00", 125.0, 125.0, 125.0, 125.0),
            ("2024-01-12 16:00:00", 143.0, 143.0, 143.0, 143.0),
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

    assert returns.index.equals(
        pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-05 16:00:00"),
                pd.Timestamp("2024-01-12 16:00:00"),
            ]
        )
    )
    assert returns.iloc[0, 0] == pytest.approx(0.2)
    assert returns.iloc[1, 0] == pytest.approx(0.1)


def test_returns_source_daily_ohlc_uses_day_end(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "EUR.USD" / "2024"
    asset_dir.mkdir(parents=True)
    _write_ohlc_csv(
        asset_dir / "hist_data_2024-01.csv",
        [
            ("2024-01-01 09:00:00", 100.0, 105.0, 99.0, 104.0),
            ("2024-01-01 16:00:00", 110.0, 115.0, 108.0, 112.0),
            ("2024-01-02 09:00:00", 120.0, 125.0, 119.0, 121.0),
            ("2024-01-02 16:00:00", 130.0, 135.0, 128.0, 132.0),
        ],
    )

    source = ReturnsSource(
        ReturnsSourceConfig(
            base_dir=tmp_path,
            assets=["EUR.USD"],
            return_type="simple",
        )
    )
    daily_ohlc = source.get_daily_ohlc_frame()

    assert daily_ohlc.index.equals(
        pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 16:00:00"),
                pd.Timestamp("2024-01-02 16:00:00"),
            ]
        )
    )
    assert daily_ohlc.loc[
        pd.Timestamp("2024-01-01 16:00:00"), ("EUR.USD", "Open")
    ] == pytest.approx(100.0)
    assert daily_ohlc.loc[
        pd.Timestamp("2024-01-01 16:00:00"), ("EUR.USD", "High")
    ] == pytest.approx(115.0)
    assert daily_ohlc.loc[
        pd.Timestamp("2024-01-01 16:00:00"), ("EUR.USD", "Low")
    ] == pytest.approx(99.0)
    assert daily_ohlc.loc[
        pd.Timestamp("2024-01-01 16:00:00"), ("EUR.USD", "Close")
    ] == pytest.approx(112.0)


def test_build_missing_data_summary_tracks_missing_days() -> None:
    index_a = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 09:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
            pd.Timestamp("2024-01-03 12:00:00", tz="UTC"),
        ]
    )
    frame_a = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [100.0, 101.0, 102.0],
            "Low": [100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0],
        },
        index=index_a,
    )
    index_b = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 09:00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 10:00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 16:00:00", tz="UTC"),
            pd.Timestamp("2024-01-03 12:00:00", tz="UTC"),
        ]
    )
    frame_b = pd.DataFrame(
        {
            "Open": [200.0, 201.0, 202.0, 203.0],
            "High": [200.0, 201.0, 202.0, 203.0],
            "Low": [200.0, 201.0, 202.0, 203.0],
            "Close": [200.0, 201.0, 202.0, 203.0],
        },
        index=index_b,
    )

    summary = build_missing_data_summary(
        {"EUR.USD": frame_a, "GBP.USD": frame_b},
        assets=["EUR.USD", "GBP.USD"],
    )

    assert summary.missing_by_asset["EUR.USD"] == [
        pd.Timestamp("2024-01-02 16:00:00", tz="UTC")
    ]
    assert summary.missing_by_asset["GBP.USD"] == []
    assert summary.missing_counts_by_month["EUR.USD"] == {"2024-01": 1}
    assert summary.missing_counts_by_month["GBP.USD"] == {"2024-01": 0}


def test_year_week_uses_iso_week_53() -> None:
    year, week = data_cleaning_runner._year_week(date(2020, 12, 31))
    assert (year, week) == (2020, 53)


def test_build_metadata_includes_source_destination_after_run_at() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1, 0.2]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-02 00:00:00", tz="UTC"),
            ]
        ),
    )
    source_dir = Path.home() / "data_source"
    destination_dir = Path.home() / "data_lake" / "2024-01"
    tensor_path = destination_dir / "return_tensor.pt"

    metadata = data_cleaning_runner._build_metadata(
        data_cleaning_runner.MetadataContext(
            returns=frame,
            assets=["EUR.USD"],
            return_profile=data_cleaning_runner.ReturnProfile(
                return_type="simple",
                return_frequency="weekly",
            ),
            source_dir=source_dir,
            destination_dir=destination_dir,
            tensor_info=data_cleaning_runner.ReturnTensorInfo(
                path=tensor_path,
                assets=["EUR.USD"],
                scale=1_000_000,
                timestamp_unit="epoch_hours",
                timezone="UTC",
                dtype="int64",
            ),
        )
    )

    keys = list(metadata.keys())
    run_at_index = keys.index("run_at")
    assert keys[run_at_index + 1] == "source"
    assert keys[run_at_index + 2] == "destination"
    assert metadata["source"].startswith("~")
    assert metadata["destination"].startswith("~")
    assert metadata["return_frequency"] == "weekly"
    assert metadata["missing_weeks_by_asset"] == {
        "EUR.USD": {"missing_weeks": [], "missing_count": 0}
    }
    assert "tensor" not in metadata


def test_build_check_average_payload() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")]
        ),
    )
    context = data_cleaning_runner.MetadataContext(
        returns=frame,
        assets=["EUR.USD"],
        return_profile=data_cleaning_runner.ReturnProfile(
            return_type="simple",
            return_frequency="weekly",
        ),
        source_dir=Path.home(),
        destination_dir=Path.home(),
        monthly_avg_close_by_asset={"EUR.USD": {"2024-01": Decimal("1.1")}},
    )

    payload = data_cleaning_runner._build_check_average(context)

    assert list(payload.keys()) == [
        "monthly_avg_close_by_asset",
        "run_at",
    ]
    assert payload["monthly_avg_close_by_asset"] == {
        "EUR.USD": {"2024-01": Decimal("1.1")}
    }


def test_build_tensor_metadata_payload() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")]
        ),
    )
    context = data_cleaning_runner.MetadataContext(
        returns=frame,
        assets=["EUR.USD"],
        return_profile=data_cleaning_runner.ReturnProfile(
            return_type="simple",
            return_frequency="weekly",
        ),
        source_dir=Path.home(),
        destination_dir=Path.home(),
        tensor_info=data_cleaning_runner.ReturnTensorInfo(
            path=Path.home() / "return_tensor.pt",
            assets=["EUR.USD"],
            scale=1_000_000,
            timestamp_unit="epoch_hours",
            timezone="UTC",
            dtype="int64",
        ),
    )

    payload = data_cleaning_runner._build_tensor_metadata(context)

    assert list(payload.keys()) == ["tensor", "run_at"]
    assert payload["tensor"] is not None


def test_build_metadata_includes_missing_weeks_by_asset() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [float("nan"), 0.1]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-08 00:00:00", tz="UTC"),
            ]
        ),
    )
    metadata = data_cleaning_runner._build_metadata(
        data_cleaning_runner.MetadataContext(
            returns=frame,
            assets=["EUR.USD"],
            return_profile=data_cleaning_runner.ReturnProfile(
                return_type="simple",
                return_frequency="weekly",
            ),
            source_dir=Path.home(),
            destination_dir=Path.home(),
        )
    )

    assert metadata["missing_weeks_by_asset"] == {
        "EUR.USD": {
            "missing_weeks": ["2024-W01"],
            "missing_count": 1,
        }
    }


def test_build_return_tensor_bundle_scales_and_masks() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1, 0.2], "GBP.USD": [float("nan"), 0.3]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
            ]
        ),
    )

    bundle = data_cleaning_runner._build_return_tensor_bundle(
        frame, scale=1_000_000
    )

    assert bundle.values.tolist() == [
        [100000, 0],
        [200000, 300000],
    ]
    assert bundle.missing_mask.tolist() == [
        [False, True],
        [False, False],
    ]
    epoch_hours = [
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC").value
        // 3_600_000_000_000,
        pd.Timestamp("2024-01-01 01:00:00", tz="UTC").value
        // 3_600_000_000_000,
    ]
    assert bundle.timestamps.tolist() == epoch_hours


def test_build_return_tensor_bundle_rejects_naive_index() -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00")]),
    )

    with pytest.raises(DataProcessingError):
        data_cleaning_runner._build_return_tensor_bundle(
            frame, scale=1_000_000
        )
