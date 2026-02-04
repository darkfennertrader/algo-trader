from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonSpec:
    days: int
    weeks: int
