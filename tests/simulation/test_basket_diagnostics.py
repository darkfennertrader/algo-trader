from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from algo_trader.application.simulation.basket_diagnostics import (
    write_global_basket_diagnostics,
    write_postprocess_basket_diagnostics,
)


def test_write_postprocess_basket_diagnostics_creates_expected_outputs(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _write_candidate_payload(tmp_path, outer_k=7, candidate_id=3, offset=0.0)

    write_postprocess_basket_diagnostics(
        base_dir=tmp_path,
        outer_k=7,
        candidate_id=3,
    )

    output_dir = tmp_path / "inner" / "outer_7" / "postprocessing" / "basket_diagnostics"
    payload = json.loads((output_dir / "basket_scores.json").read_text(encoding="utf-8"))
    histogram = pd.read_csv(output_dir / "pit_histogram_broad_mixed.csv")

    assert (output_dir / "basket_scores.csv").exists()
    assert (output_dir / "pit_histogram_us_index.csv").exists()
    assert payload["best_candidate_id"] == 3
    assert payload["basket_definitions"]["broad_mixed"]["status"] == "ok"
    assert payload["basket_diagnostics"]["broad_mixed"]["n_assets"] == 6.0
    assert payload["basket_diagnostics"]["swiss_index"]["n_assets"] == 1.0
    assert "IBUS500" in payload["basket_definitions"]["broad_mixed"]["weights"]
    assert float(histogram["probability"].sum()) == 1.0


def test_write_postprocess_basket_diagnostics_skips_unavailable_basket(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path, include_xau=False)
    _write_candidate_payload(
        tmp_path,
        outer_k=7,
        candidate_id=3,
        offset=0.0,
        include_xau=False,
    )

    write_postprocess_basket_diagnostics(
        base_dir=tmp_path,
        outer_k=7,
        candidate_id=3,
    )

    output_dir = tmp_path / "inner" / "outer_7" / "postprocessing" / "basket_diagnostics"
    payload = json.loads((output_dir / "basket_scores.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(output_dir / "basket_scores.csv")

    broad_mixed = payload["basket_definitions"]["broad_mixed"]
    broad_mixed_row = scores.loc[scores["basket"] == "broad_mixed"].iloc[0]

    assert broad_mixed["status"] == "unavailable"
    assert broad_mixed["assets_missing"] == ["XAUUSD"]
    assert pd.isna(payload["basket_diagnostics"]["broad_mixed"]["crps"])
    assert broad_mixed_row["status"] == "unavailable"
    assert pd.isna(broad_mixed_row["crps"])
    assert not (output_dir / "pit_histogram_broad_mixed.csv").exists()


def test_write_global_basket_diagnostics_creates_manifest(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _write_candidate_payload(tmp_path, outer_k=7, candidate_id=3, offset=0.0)
    _write_candidate_payload(tmp_path, outer_k=8, candidate_id=3, offset=0.2)

    write_global_basket_diagnostics(
        base_dir=tmp_path,
        outer_ids=[7, 8],
        candidate_id=3,
    )

    output_dir = tmp_path / "outer" / "diagnostics" / "basket_diagnostics"
    payload = json.loads((output_dir / "basket_scores.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "aggregate_manifest.json").read_text(encoding="utf-8"))
    histogram = pd.read_csv(output_dir / "pit_histogram_us_index.csv")

    assert (output_dir / "basket_scores.csv").exists()
    assert payload["outer_ids"] == [7, 8]
    assert manifest["scope"] == "selected_candidate_basket_diagnostics"
    assert payload["basket_definitions"]["us_minus_europe"]["status"] == "unavailable"
    assert not (output_dir / "pit_histogram_us_minus_europe.csv").exists()
    assert int(histogram.loc[0, "n_outer_folds"]) == 2


def _write_inputs(base_dir: Path, *, include_xau: bool = True) -> None:
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    columns = [
        "timestamp",
        "IBUS30",
        "IBUS500",
        "IBUST100",
        "IBEU50",
        "IBFR40",
        "IBGB100",
        "IBCH20",
        "EUR.USD",
        "USD.JPY",
    ]
    if include_xau:
        columns.append("XAUUSD")
    frame = pd.DataFrame(
        columns=columns,
    )
    frame.to_csv(inputs_dir / "targets.csv", index=False)


def _write_candidate_payload(
    base_dir: Path,
    *,
    outer_k: int,
    candidate_id: int,
    offset: float,
    include_xau: bool = True,
) -> None:
    candidates_dir = (
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "candidates"
    )
    candidates_dir.mkdir(parents=True)
    weight_values = [1.0, 1.2, 0.8, -0.9, -1.1, -0.7, 0.4, 0.2, -0.2]
    if include_xau:
        weight_values.append(0.6)
    weights = torch.tensor(weight_values)
    sample_multipliers = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    true_multipliers = torch.tensor([0.5, -0.5, 1.5, -1.5], dtype=torch.float64) + offset
    z_samples = sample_multipliers[:, None, None] * weights[None, None, :]
    z_samples = z_samples.repeat(1, 4, 1)
    z_true = true_multipliers[:, None] * weights[None, :]
    payload = {
        "z_true": z_true.to(dtype=torch.float64),
        "z_samples": z_samples.to(dtype=torch.float64),
        "scale": torch.ones(len(weight_values), dtype=torch.float64),
        "test_idx": [0, 1, 2, 3],
    }
    torch.save(
        payload,
        candidates_dir / f"candidate_{candidate_id:04d}_split_0000.pt",
    )
