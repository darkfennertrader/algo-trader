from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.exogenous_cleaning import (
    RunRequest,
    runner as exogenous_cleaning_runner,
)
from algo_trader.domain import DataProcessingError


def _write_returns(path: Path, timestamps: list[str]) -> None:
    frame = pd.DataFrame(
        {"EUR.USD": [0.1 for _ in timestamps]},
        index=pd.to_datetime(timestamps, utc=True),
    )
    frame.to_csv(path)


def _write_raw_series(
    path: Path, rows: list[tuple[str, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["date", "value"])
    frame.to_csv(path, index=False)


def test_exogenous_cleaning_runner_writes_aligned_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2023-01-01"\n'
        'end_date: "2024-01-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    family_key: "equity_implied_vol"\n'
        '    alias: "vix_us"\n'
        '    dir_name: "equity_implied_vol"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '  - id: "IR3TIB01USM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_usd"\n'
        '    dir_name: "carry/USD"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "both"\n'
        "families:\n"
        '  - key: "equity_implied_vol"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '    channel: "mean"\n'
        '  - key: "carry"\n'
        '    priority: "core"\n'
        '    future_role: "both"\n'
        '    channel: "mean"\n',
        encoding="utf-8",
    )
    raw_root = tmp_path / "exogenous_feats"
    data_lake = tmp_path / "data_lake"
    version_dir = data_lake / "2024-10"
    version_dir.mkdir(parents=True)
    _write_returns(
        version_dir / "returns.csv",
        [
            "2024-01-05 16:00:00",
            "2024-01-12 16:00:00",
            "2024-01-19 16:00:00",
        ],
    )
    _write_raw_series(
        raw_root / "fred" / "equity_implied_vol" / "VIXCLS.csv",
        [
            ("2024-01-05", 14.0),
            ("2024-01-12", 15.0),
            ("2024-01-19", 16.0),
        ],
    )
    _write_raw_series(
        raw_root / "fred" / "carry" / "USD" / "IR3TIB01USM156N.csv",
        [("2023-12-31", 5.0)],
    )
    monkeypatch.setenv("EXOGENOUS_FEATURES_SOURCE", str(raw_root))
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))

    output_path = exogenous_cleaning_runner.run(
        request=RunRequest(config_path=config_path)
    )

    assert output_path == version_dir / "exogenous" / "exogenous_cleaned.csv"
    output_frame = pd.read_csv(output_path, index_col=0, parse_dates=[0])
    assert list(output_frame.columns) == [
        "fred__equity_implied_vol__vix_us",
        "fred__carry__rate_3m_usd",
    ]
    assert list(output_frame.index) == list(
        pd.to_datetime(
            ["2024-01-12 16:00:00+00:00", "2024-01-19 16:00:00+00:00"]
        )
    )
    assert output_frame.iloc[:, 0].tolist() == [15.0, 16.0]
    assert output_frame.iloc[:, 1].tolist() == [5.0, 5.0]

    metadata = json.loads(
        (version_dir / "exogenous" / "exogenous_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["version_label"] == "2024-10"
    assert metadata["features"] == 2


def test_exogenous_cleaning_runner_drops_optional_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2024-01-01"\n'
        'end_date: "2024-01-31"\n'
        "cleaning:\n"
        "  fill_policy:\n"
        '    method: "forward_fill_only"\n'
        "    allow_backfill: false\n"
        "    weekly_max_ffill_weeks: 0\n"
        "    monthly_max_ffill_weeks: 0\n"
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "equity_implied_vol"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '  - id: "MOVE"\n'
        '    dir_name: "rates_implied_vol"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "optional"\n',
        encoding="utf-8",
    )
    raw_root = tmp_path / "exogenous_feats"
    data_lake = tmp_path / "data_lake"
    version_dir = data_lake / "2024-10"
    version_dir.mkdir(parents=True)
    _write_returns(
        version_dir / "returns.csv",
        [
            "2024-01-05 16:00:00",
            "2024-01-12 16:00:00",
            "2024-01-19 16:00:00",
        ],
    )
    _write_raw_series(
        raw_root / "fred" / "equity_implied_vol" / "VIXCLS.csv",
        [
            ("2024-01-05", 14.0),
            ("2024-01-12", 15.0),
            ("2024-01-19", 16.0),
        ],
    )
    _write_raw_series(
        raw_root / "fred" / "rates_implied_vol" / "MOVE.csv",
        [("2024-01-05", 120.0)],
    )
    monkeypatch.setenv("EXOGENOUS_FEATURES_SOURCE", str(raw_root))
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))

    output_path = exogenous_cleaning_runner.run(
        request=RunRequest(config_path=config_path)
    )

    output_frame = pd.read_csv(output_path, index_col=0, parse_dates=[0])
    assert list(output_frame.columns) == ["fred__equity_implied_vol__VIXCLS"]
    metadata = json.loads(
        (version_dir / "exogenous" / "exogenous_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(metadata["dropped_features"]) == 1
    assert metadata["dropped_features"][0]["series_id"] == "MOVE"
    assert metadata["dropped_features"][0]["drop_reason"] == "missing_after_alignment"


def test_exogenous_cleaning_runner_fails_on_missing_core_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "fred_config.yml"
    config_path.write_text(
        'provider: "fred"\n'
        'start_date: "2024-01-01"\n'
        'end_date: "2024-01-31"\n'
        "series:\n"
        '  - id: "VIXCLS"\n'
        '    dir_name: "equity_implied_vol"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '  - id: "DTWEXBGS"\n'
        '    dir_name: "broad_USD_factor"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n',
        encoding="utf-8",
    )
    raw_root = tmp_path / "exogenous_feats"
    data_lake = tmp_path / "data_lake"
    version_dir = data_lake / "2024-10"
    version_dir.mkdir(parents=True)
    _write_returns(
        version_dir / "returns.csv",
        ["2024-01-05 16:00:00", "2024-01-12 16:00:00"],
    )
    _write_raw_series(
        raw_root / "fred" / "equity_implied_vol" / "VIXCLS.csv",
        [("2024-01-05", 14.0), ("2024-01-12", 15.0)],
    )
    monkeypatch.setenv("EXOGENOUS_FEATURES_SOURCE", str(raw_root))
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))

    with pytest.raises(DataProcessingError, match="Core exogenous series is unavailable"):
        exogenous_cleaning_runner.run(
            request=RunRequest(config_path=config_path)
        )
