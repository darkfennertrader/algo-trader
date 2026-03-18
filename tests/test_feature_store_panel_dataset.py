from __future__ import annotations

from pathlib import Path

from algo_trader.infrastructure.data.feature_store_panel_dataset import (
    _resolve_group_names,
)


def test_resolve_group_names_skips_container_directories(tmp_path: Path) -> None:
    version_dir = tmp_path / "2024-10"
    momentum_dir = version_dir / "momentum"
    momentum_dir.mkdir(parents=True)
    (momentum_dir / "features_tensor.pt").write_bytes(b"tensor")
    (momentum_dir / "metadata.json").write_text("{}", encoding="utf-8")
    nested_container = version_dir / "exogenous"
    (nested_container / "asset").mkdir(parents=True)
    groups = _resolve_group_names(
        type(
            "_Params",
            (),
            {"groups": None},
        )(),
        version_dir,
    )
    assert groups == ["momentum"]
