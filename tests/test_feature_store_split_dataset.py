from __future__ import annotations

import json
from pathlib import Path

import torch

from algo_trader.domain.simulation import DataConfig
from algo_trader.infrastructure.data import load_feature_store_split_dataset


def _write_tensor_bundle(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_feature_store_split_dataset_aligns_asset_global_and_targets(
    tmp_path: Path, monkeypatch
) -> None:
    feature_store = tmp_path / "feature_store"
    data_lake = tmp_path / "data_lake"
    version = "2024-10"

    asset_timestamps = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    exogenous_timestamps = torch.tensor([20, 30, 40], dtype=torch.int64)
    returns_timestamps = torch.tensor([10, 20, 30, 40], dtype=torch.int64)

    _write_tensor_bundle(
        feature_store / version / "momentum" / "features_tensor.pt",
        {
            "values": torch.tensor(
                [
                    [[1.0], [2.0]],
                    [[3.0], [4.0]],
                    [[5.0], [6.0]],
                    [[7.0], [8.0]],
                ],
                dtype=torch.float64,
            ),
            "timestamps": asset_timestamps,
            "missing_mask": torch.zeros((4, 2, 1), dtype=torch.bool),
        },
    )
    (feature_store / version / "momentum" / "metadata.json").write_text(
        json.dumps({"feature_names": ["mom_4w"]}),
        encoding="utf-8",
    )

    _write_tensor_bundle(
        feature_store / version / "exogenous" / "asset" / "features_tensor.pt",
        {
            "values": torch.tensor(
                [
                    [[0.1], [0.2]],
                    [[0.3], [0.4]],
                    [[0.5], [0.6]],
                ],
                dtype=torch.float64,
            ),
            "timestamps": exogenous_timestamps,
            "missing_mask": torch.zeros((3, 2, 1), dtype=torch.bool),
        },
    )
    (feature_store / version / "exogenous" / "asset" / "metadata.json").write_text(
        json.dumps({"feature_names": ["carry_3m_diff"]}),
        encoding="utf-8",
    )

    _write_tensor_bundle(
        feature_store / version / "exogenous" / "global" / "features_tensor.pt",
        {
            "values": torch.tensor(
                [
                    [10.0, 20.0],
                    [11.0, 21.0],
                    [12.0, 22.0],
                ],
                dtype=torch.float64,
            ),
            "timestamps": exogenous_timestamps,
            "missing_mask": torch.zeros((3, 2), dtype=torch.bool),
        },
    )
    (feature_store / version / "exogenous" / "global" / "metadata.json").write_text(
        json.dumps({"feature_names": ["log_vix_us", "log_usd_broad"]}),
        encoding="utf-8",
    )

    _write_tensor_bundle(
        data_lake / version / "return_tensor.pt",
        {
            "values": torch.tensor(
                [
                    [0.01, 0.02],
                    [0.03, 0.04],
                    [0.05, 0.06],
                    [0.07, 0.08],
                ],
                dtype=torch.float64,
            ),
            "timestamps": returns_timestamps,
        },
    )
    (data_lake / version / "returns_meta.json").write_text(
        json.dumps({"assets": ["EUR.USD", "AUD.CAD"]}),
        encoding="utf-8",
    )

    monkeypatch.setenv("FEATURE_STORE_SOURCE", str(feature_store))
    monkeypatch.setenv("DATA_LAKE_SOURCE", str(data_lake))

    dataset = load_feature_store_split_dataset(
        config=DataConfig(dataset_params={}),
        device="cpu",
    )

    assert tuple(dataset.data.shape) == (2, 2, 2)
    assert tuple(dataset.targets.shape) == (2, 2)
    assert tuple(dataset.global_data.shape) == (2, 2)  # type: ignore[union-attr]
    assert dataset.features == ["momentum::mom_4w", "exogenous::asset::carry_3m_diff"]
    assert list(dataset.global_features) == ["log_vix_us", "log_usd_broad"]
    assert list(dataset.assets) == ["EUR.USD", "AUD.CAD"]
    assert list(dataset.dates) == [20, 30]
    assert torch.allclose(
        dataset.data[:, 0, :],
        torch.tensor([[3.0, 0.1], [5.0, 0.3]], dtype=torch.float64),
    )
    assert torch.allclose(
        dataset.targets,
        torch.tensor([[0.05, 0.06], [0.07, 0.08]], dtype=torch.float64) / 1_000_000.0,
    )
    assert torch.allclose(
        dataset.global_data,
        torch.tensor([[10.0, 20.0], [11.0, 21.0]], dtype=torch.float64),
    )
