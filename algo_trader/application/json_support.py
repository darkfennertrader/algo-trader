from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algo_trader.domain import SimulationError


def write_json_file(
    *,
    path: Path,
    payload: Any,
    message: str,
) -> None:
    try:
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        raise SimulationError(message, context={"path": str(path)}) from exc


__all__ = ["write_json_file"]
