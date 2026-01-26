from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from algo_trader.application.data_processing import runner as data_processing_runner
from algo_trader.domain import ConfigError


def _write_returns(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "EUR.USD": [0.01, 0.02],
            "GBP.USD": [0.03, 0.04],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02"],
            utc=True,
        ),
    )
    frame.to_csv(path)


def test_resolve_latest_directory_selects_lexicographic_max(
    tmp_path: Path,
) -> None:
    (tmp_path / "2023-52").mkdir()
    (tmp_path / "2024-01").mkdir()
    (tmp_path / "2024-10").mkdir()
    (tmp_path / "notes").mkdir()

    latest = data_processing_runner._resolve_latest_directory(tmp_path)

    assert latest.name == "2024-10"


def test_parse_preprocessor_args_requires_key_value() -> None:
    with pytest.raises(ConfigError):
        data_processing_runner._parse_preprocessor_args(["missing_equals"])


def test_run_writes_processed_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "2024-05"
    data_dir.mkdir()
    input_path = data_dir / "returns.csv"
    _write_returns(input_path)

    monkeypatch.setenv("DATA_LAKE_SOURCE", str(tmp_path))
    feature_store = tmp_path / "feature_store"
    feature_store.mkdir()
    monkeypatch.setenv("FEATURE_STORE_SOURCE", str(feature_store))

    output_path = data_processing_runner.run(preprocessor_name="identity")

    assert (
        output_path
        == feature_store / "identity" / "debug" / "2024-05" / "processed.csv"
    )
    metadata_path = (
        feature_store / "identity" / "debug" / "2024-05" / "metadata.json"
    )
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["input_path"] == str(input_path)
    assert metadata["output_path"] == str(output_path)
    output_frame = data_processing_runner._load_returns(output_path)
    input_frame = data_processing_runner._load_returns(input_path)
    pd.testing.assert_frame_equal(output_frame, input_frame)
