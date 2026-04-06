from __future__ import annotations

import json
from pathlib import Path


def write_log_weekly_data_source_metadata(base_dir: Path) -> None:
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "data_source.json").write_text(
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
