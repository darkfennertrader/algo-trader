from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.exogenous_feature_engineering import (
    RunRequest,
    runner as exogenous_feature_engineering_runner,
)


def _write_returns(
    path: Path, timestamps: list[str], *, assets: list[str]
) -> None:
    frame = pd.DataFrame(
        {
            asset: [0.1 + index * 0.01 for index, _ in enumerate(timestamps)]
            for asset in assets
        },
        index=pd.to_datetime(timestamps, utc=True),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _write_exogenous_cleaned(
    path: Path,
    timestamps: list[str],
) -> None:
    frame = pd.DataFrame(
        {
            "fred__equity_implied_vol__vix_us": [14.0, 15.0, 16.0],
            "fred__broad_usd_factor__usd_broad": [120.0, 121.0, 122.0],
            "fred__credit_spreads_risk__credit_us_hy_oas": [3.0, 3.1, 3.2],
            "fred__credit_spreads_risk__credit_eur_hy_oas": [2.8, 2.9, 3.0],
            "fred__carry__rate_3m_usd": [5.0, 5.0, 5.1],
            "fred__carry__rate_3m_eur": [3.0, 3.1, 3.2],
            "fred__carry__rate_3m_aud": [4.2, 4.3, 4.4],
            "fred__carry__rate_3m_cad": [4.0, 4.1, 4.2],
        },
        index=pd.to_datetime(timestamps, utc=True),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def test_exogenous_feature_engineering_runner_writes_asset_and_global_blocks(
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
        '  - id: "DTWEXBGS"\n'
        '    family_key: "broad_usd_factor"\n'
        '    alias: "usd_broad"\n'
        '    dir_name: "broad_USD_factor"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '  - id: "BAMLH0A0HYM2"\n'
        '    family_key: "credit_spreads_risk"\n'
        '    alias: "credit_us_hy_oas"\n'
        '    dir_name: "credit_spreads_risk"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '  - id: "BAMLHE00EHYIOAS"\n'
        '    family_key: "credit_spreads_risk"\n'
        '    alias: "credit_eur_hy_oas"\n'
        '    dir_name: "credit_spreads_risk"\n'
        '    frequency: "w"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '  - id: "IR3TIB01USM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_usd"\n'
        '    currency: "USD"\n'
        '    dir_name: "carry/USD"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '  - id: "IR3TIB01EZM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_eur"\n'
        '    currency: "EUR"\n'
        '    dir_name: "carry/EUR"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '  - id: "IR3TIB01AUM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_aud"\n'
        '    currency: "AUD"\n'
        '    dir_name: "carry/AUD"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '  - id: "IR3TIB01CAM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_cad"\n'
        '    currency: "CAD"\n'
        '    dir_name: "carry/CAD"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        "families:\n"
        '  - key: "equity_implied_vol"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '    channel: "mean"\n'
        '  - key: "broad_usd_factor"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '    channel: "mean"\n'
        '  - key: "credit_spreads_risk"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '    channel: "mean"\n'
        '  - key: "carry"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '    channel: "mean"\n',
        encoding="utf-8",
    )
    data_lake = tmp_path / "data_lake"
    feature_store = tmp_path / "feature_store"
    version_dir = data_lake / "2024-10"
    _write_returns(
        version_dir / "returns.csv",
        [
            "2024-01-05 16:00:00",
            "2024-01-12 16:00:00",
            "2024-01-19 16:00:00",
            "2024-01-26 16:00:00",
        ],
        assets=["AUD.CAD", "EUR.USD", "IBUS500"],
    )
    _write_exogenous_cleaned(
        version_dir / "exogenous" / "exogenous_cleaned.csv",
        [
            "2024-01-12 16:00:00",
            "2024-01-19 16:00:00",
            "2024-01-26 16:00:00",
        ],
    )
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))
    monkeypatch.setenv("FEATURE_STORE_SOURCE", str(feature_store))

    output_paths = exogenous_feature_engineering_runner.run(
        request=RunRequest(config_path=config_path)
    )

    assert output_paths == [
        feature_store / "2024-10" / "exogenous" / "asset" / "features.csv",
        feature_store / "2024-10" / "exogenous" / "global" / "features.csv",
    ]

    asset_frame = pd.read_csv(
        output_paths[0],
        index_col=0,
        parse_dates=[0],
        header=[0, 1],
        skip_blank_lines=False,
        na_values=[""],
    )
    assert list(asset_frame.columns) == [
        ("AUD.CAD", "carry_3m_diff"),
        ("EUR.USD", "carry_3m_diff"),
        ("IBUS500", "carry_3m_diff"),
    ]
    assert pd.isna(asset_frame[("AUD.CAD", "carry_3m_diff")].iloc[0])
    assert asset_frame[("AUD.CAD", "carry_3m_diff")].iloc[1:].tolist() == pytest.approx(
        [0.2, 0.2, 0.2]
    )
    assert pd.isna(asset_frame[("EUR.USD", "carry_3m_diff")].iloc[0])
    assert asset_frame[("EUR.USD", "carry_3m_diff")].iloc[1:].tolist() == pytest.approx(
        [-2.0, -1.9, -1.9]
    )
    assert asset_frame[("IBUS500", "carry_3m_diff")].isna().all()

    global_frame = pd.read_csv(
        output_paths[1],
        index_col=0,
        parse_dates=[0],
        skip_blank_lines=False,
        na_values=[""],
    )
    assert list(global_frame.columns) == [
        "log_vix_us",
        "log_usd_broad",
        "log_credit_us_hy_oas",
        "log_credit_eur_hy_oas",
    ]
    assert global_frame.index.tolist() == list(
        pd.to_datetime(
            [
                "2024-01-05 16:00:00+00:00",
                "2024-01-12 16:00:00+00:00",
                "2024-01-19 16:00:00+00:00",
                "2024-01-26 16:00:00+00:00",
            ]
        )
    )
    assert global_frame.iloc[0].isna().all()

    asset_metadata = json.loads(
        (
            feature_store / "2024-10" / "exogenous" / "asset" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert asset_metadata["feature_block"] == "asset"
    assert asset_metadata["feature_names"] == ["carry_3m_diff"]

    global_metadata = json.loads(
        (
            feature_store / "2024-10" / "exogenous" / "global" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert global_metadata["feature_block"] == "global"
    assert global_metadata["feature_names"] == [
        "log_vix_us",
        "log_usd_broad",
        "log_credit_us_hy_oas",
        "log_credit_eur_hy_oas",
    ]


def test_exogenous_feature_engineering_runner_uses_full_returns_window(
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
        '    currency: "USD"\n'
        '    dir_name: "carry/USD"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '  - id: "IR3TIB01EZM156N"\n'
        '    family_key: "carry"\n'
        '    alias: "rate_3m_eur"\n'
        '    currency: "EUR"\n'
        '    dir_name: "carry/EUR"\n'
        '    frequency: "m"\n'
        '    aggregation_method: "eop"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        "families:\n"
        '  - key: "equity_implied_vol"\n'
        '    priority: "core"\n'
        '    future_role: "global"\n'
        '    channel: "mean"\n'
        '  - key: "carry"\n'
        '    priority: "core"\n'
        '    future_role: "asset"\n'
        '    channel: "mean"\n',
        encoding="utf-8",
    )
    data_lake = tmp_path / "data_lake"
    feature_store = tmp_path / "feature_store"
    version_dir = data_lake / "2024-10"
    _write_returns(
        version_dir / "returns.csv",
        [
            "2024-01-05 16:00:00",
            "2024-01-12 16:00:00",
            "2024-01-19 16:00:00",
            "2024-01-26 16:00:00",
        ],
        assets=["EUR.USD"],
    )
    cleaned = pd.DataFrame(
        {
            "fred__equity_implied_vol__vix_us": [14.0, 15.0, 16.0],
            "fred__carry__rate_3m_usd": [5.0, 5.0, 5.1],
            "fred__carry__rate_3m_eur": [3.0, 3.1, 3.2],
        },
        index=pd.to_datetime(
            [
                "2024-01-12 16:00:00",
                "2024-01-19 16:00:00",
                "2024-01-26 16:00:00",
            ],
            utc=True,
        ),
    )
    cleaned_path = version_dir / "exogenous" / "exogenous_cleaned.csv"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path)
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))
    monkeypatch.setenv("FEATURE_STORE_SOURCE", str(feature_store))

    output_paths = exogenous_feature_engineering_runner.run(
        request=RunRequest(config_path=config_path)
    )

    asset_frame = pd.read_csv(
        output_paths[0],
        index_col=0,
        parse_dates=[0],
        header=[0, 1],
        skip_blank_lines=False,
        na_values=[""],
    )
    assert asset_frame.index.tolist() == list(
        pd.to_datetime(
            [
                "2024-01-05 16:00:00+00:00",
                "2024-01-12 16:00:00+00:00",
                "2024-01-19 16:00:00+00:00",
                "2024-01-26 16:00:00+00:00",
            ]
        )
    )
    assert asset_frame.iloc[0].isna().all()
    assert not asset_frame.iloc[1].isna().all()

    asset_metadata = json.loads(
        (
            feature_store / "2024-10" / "exogenous" / "asset" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert "requested_start_date" not in asset_metadata
    assert "requested_end_date" not in asset_metadata
