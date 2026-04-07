from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..artifacts import resolve_relative_simulation_output_dir


def walkforward_dir(base_dir: Path) -> Path:
    return base_dir


def seed_stability_dir(base_dir: Path) -> Path:
    return base_dir / "seed_stability"


def resolve_portfolio_base_dir(
    *,
    source_dir: Path,
    portfolio_output_path: str | None,
    dataset_params: Mapping[str, Any],
) -> Path:
    del dataset_params
    if portfolio_output_path is None:
        return source_dir / "walkforward"
    return resolve_relative_simulation_output_dir(
        output_path=portfolio_output_path
    )
