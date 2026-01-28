from __future__ import annotations

import json
from pathlib import Path
from decimal import Decimal

import pandas as pd
import pytest

from algo_trader.application import modeling


def _write_prepared(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "EUR.USD": [0.01, 0.02, 0.03],
            "GBP.USD": [0.04, 0.05, 0.06],
        },
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            utc=True,
        ),
    )
    frame.to_csv(path)


def test_run_writes_params_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_store = tmp_path / "feature_store"
    input_dir = feature_store / "identity" / "debug" / "2024-05"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "processed.csv"
    _write_prepared(input_path)

    model_store = tmp_path / "model_store"
    model_store.mkdir()

    monkeypatch.setenv("FEATURE_STORE_SOURCE", str(feature_store))
    monkeypatch.setenv("MODEL_STORE_SOURCE", str(model_store))

    output_path = modeling.run(
        model_name="normal",
        guide_name="normal_mean_field",
        options=modeling.InferenceOptions(
            steps=1,
            learning_rate=Decimal("0.01"),
            seed=0,
        ),
        data=modeling.DataSelection(
            preprocessor_name="identity",
            pipeline="debug",
        ),
    )

    assert (
        output_path
        == model_store
        / "normal"
        / "normal_mean_field"
        / "debug"
        / "2024-05"
        / "params.csv"
    )
    assert output_path.exists()
    params_frame = pd.read_csv(output_path)
    assert {"param", "index", "value"}.issubset(params_frame.columns)

    metadata_path = output_path.parent / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["model"] == "normal"
    assert metadata["guide"] == "normal_mean_field"
