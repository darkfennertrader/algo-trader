from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PosteriorConfidencePolicyConfig:
    score_name: str
    block_scope: str
    score_threshold: float


__all__ = ["PosteriorConfidencePolicyConfig"]
