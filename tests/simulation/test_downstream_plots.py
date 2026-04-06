from __future__ import annotations

import json
from pathlib import Path

from algo_trader.application.simulation.downstream_plots import (
    write_downstream_plots,
)


def test_write_downstream_plots_creates_downstream_charts(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "simulation_run"
    downstream_dir = base_dir / "downstream"
    downstream_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "inputs" / "data_source.json").write_text(
        json.dumps(
            {
                "version_label": "2026-14",
                "return_type": "log",
                "return_frequency": "weekly",
                "data_lake_dir": "/tmp/data_lake/2026-14",
            }
        ),
        encoding="utf-8",
    )
    (downstream_dir / "stitched_returns.csv").write_text(
        "\n".join(
            [
                "timestamp,primary,equal_weight",
                "2024-01-05,0.01,0.005",
                "2024-01-12,0.02,0.004",
            ]
        ),
        encoding="utf-8",
    )

    write_downstream_plots(base_dir=base_dir, dataset_params={})

    assert (
        downstream_dir / "plots" / "cumulative_returns.png"
    ).exists()
    assert (
        downstream_dir / "plots" / "underwater_drawdown.png"
    ).exists()
