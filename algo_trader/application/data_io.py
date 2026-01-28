from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo_trader.domain import DataSourceError


def read_indexed_csv(
    path: Path, *, missing_message: str, read_message: str
) -> pd.DataFrame:
    if not path.exists():
        raise DataSourceError(
            missing_message,
            context={"path": str(path)},
        )
    try:
        return pd.read_csv(path, index_col=0, parse_dates=[0])
    except Exception as exc:
        raise DataSourceError(
            read_message,
            context={"path": str(path)},
        ) from exc
