from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from algo_trader.application.simulation.residual_dependence_diagnostics import (
    write_global_residual_dependence,
    write_postprocess_residual_dependence,
)


def test_write_postprocess_residual_dependence_outputs_expected_files(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _write_candidate_payload(tmp_path, outer_k=7, candidate_id=3, offset=0.0)

    write_postprocess_residual_dependence(
        base_dir=tmp_path,
        outer_k=7,
        candidate_id=3,
    )

    output_dir = tmp_path / "inner" / "outer_7" / "postprocessing" / "residual_dependence"
    summary = json.loads(
        (output_dir / "residual_dependence_summary.json").read_text(encoding="utf-8")
    )
    pairwise = pd.read_csv(output_dir / "index_pairwise_residual_structure.csv")

    assert (output_dir / "residual_dependence_summary.csv").exists()
    assert (output_dir / "index_residual_corr_matrix.csv").exists()
    assert (output_dir / "index_whitened_residual_corr_matrix.csv").exists()
    assert summary["blocks"]["indices"]["hard_assets_present"] == ["IBEU50", "IBUS30"]
    assert summary["blocks"]["full"]["summary"]["n_assets"] == 3.0
    assert pairwise.loc[0, "left_asset"] == "IBEU50"
    assert pairwise.loc[0, "right_asset"] == "IBUS30"
    assert float(pairwise.loc[0, "residual_corr"]) > 0.99


def test_write_global_residual_dependence_aggregates_outer_folds(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _write_candidate_payload(tmp_path, outer_k=7, candidate_id=3, offset=0.0)
    _write_candidate_payload(tmp_path, outer_k=8, candidate_id=3, offset=0.2)

    write_global_residual_dependence(
        base_dir=tmp_path,
        outer_ids=[7, 8],
        candidate_id=3,
    )

    output_dir = tmp_path / "outer" / "diagnostics" / "residual_dependence"
    summary = json.loads(
        (output_dir / "residual_dependence_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    pairwise = pd.read_csv(output_dir / "index_pairwise_residual_structure.csv")

    assert (output_dir / "residual_dependence_summary.csv").exists()
    assert (output_dir / "index_residual_corr_matrix.csv").exists()
    assert summary["outer_ids"] == [7, 8]
    assert manifest["scope"] == "selected_candidate_residual_dependence"
    assert int(pairwise.loc[0, "n_outer_folds"]) == 2


def _write_inputs(base_dir: Path) -> None:
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        columns=["timestamp", "IBEU50", "IBUS30", "EUR.USD"],
    )
    frame.to_csv(inputs_dir / "targets.csv", index=False)


def _write_candidate_payload(
    base_dir: Path,
    *,
    outer_k: int,
    candidate_id: int,
    offset: float,
) -> None:
    candidates_dir = (
        base_dir / "inner" / f"outer_{outer_k}" / "postprocessing" / "candidates"
    )
    candidates_dir.mkdir(parents=True)
    samples = torch.tensor(
        [
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            [[-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
            [[1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0]],
            [[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    z_true = torch.tensor(
        [
            [1.0 + offset, 1.0 + offset, 0.5 + offset],
            [-1.0 - offset, -1.0 - offset, -0.5 - offset],
            [2.0 + offset, 2.0 + offset, 1.0 + offset],
            [-2.0 - offset, -2.0 - offset, -1.0 - offset],
        ],
        dtype=torch.float64,
    )
    payload = {
        "z_true": z_true,
        "z_samples": samples,
        "scale": torch.ones(3, dtype=torch.float64),
        "test_idx": [0, 1, 2, 3],
    }
    torch.save(
        payload,
        candidates_dir / f"candidate_{candidate_id:04d}_split_0000.pt",
    )
