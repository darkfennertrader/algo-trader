from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algo_trader.application.simulation.downstream_outputs import (
    write_downstream_outputs,
)


def test_write_downstream_outputs_creates_stitched_portfolio_files(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "simulation_run"
    outer_results = [
        {
            "outer_k_test": 0,
            "portfolios": {
                "primary": {
                    "timestamps": [473448, 473616],
                    "gross_returns": [0.02, 0.01],
                    "net_returns": [0.019, 0.009],
                    "costs": [0.001, 0.001],
                    "turnover": [0.0, 0.1],
                    "weights": [
                        np.array([0.6, 0.4]),
                        np.array([0.7, 0.3]),
                    ],
                },
                "equal_weight": {
                    "timestamps": [473448, 473616],
                    "gross_returns": [0.01, 0.015],
                    "net_returns": [0.01, 0.015],
                    "costs": [0.0, 0.0],
                    "turnover": [0.0, 0.0],
                    "weights": [
                        np.array([0.5, 0.5]),
                        np.array([0.5, 0.5]),
                    ],
                },
            },
        },
        {
            "outer_k_test": 1,
            "portfolios": {
                "primary": {
                    "timestamps": [473784],
                    "gross_returns": [0.03],
                    "net_returns": [0.028],
                    "costs": [0.002],
                    "turnover": [0.2],
                    "weights": [np.array([0.8, 0.2])],
                },
                "equal_weight": {
                    "timestamps": [473784],
                    "gross_returns": [0.02],
                    "net_returns": [0.02],
                    "costs": [0.0],
                    "turnover": [0.0],
                    "weights": [np.array([0.5, 0.5])],
                },
            },
        },
    ]

    write_downstream_outputs(
        base_dir=base_dir,
        outer_results=outer_results,
        assets=("A", "B"),
    )

    downstream_dir = base_dir / "downstream"
    primary_returns = pd.read_csv(
        downstream_dir / "portfolios" / "primary" / "weekly_returns.csv"
    )
    primary_weights = pd.read_csv(
        downstream_dir / "portfolios" / "primary" / "weights.csv"
    )
    stitched_returns = pd.read_csv(
        downstream_dir / "stitched_returns.csv"
    )
    manifest = json.loads(
        (downstream_dir / "portfolio_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert list(primary_returns.columns) == [
        "timestamp",
        "outer_k",
        "gross_return",
        "net_return",
        "cost",
        "turnover",
    ]
    assert len(primary_returns) == 3
    assert len(primary_weights) == 6
    assert primary_returns["timestamp"].tolist() == [
        "2024-01-05",
        "2024-01-12",
        "2024-01-19",
    ]
    assert list(stitched_returns.columns) == [
        "timestamp",
        "equal_weight",
        "primary",
    ]
    assert manifest["portfolio_names"] == ["equal_weight", "primary"]
