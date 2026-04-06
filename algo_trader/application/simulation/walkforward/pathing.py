from __future__ import annotations

from pathlib import Path


def walkforward_dir(base_dir: Path) -> Path:
    return base_dir / "walkforward"


def seed_stability_dir(base_dir: Path) -> Path:
    return walkforward_dir(base_dir) / "seed_stability"
