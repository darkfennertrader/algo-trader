from __future__ import annotations

from typing import Mapping, Protocol

import pandas as pd


class Preprocessor(Protocol):
    def process(
        self, data: pd.DataFrame, *, params: Mapping[str, str]
    ) -> pd.DataFrame:
        """Transform a returns matrix into a processed form."""
        ...
