from __future__ import annotations

from pathlib import Path

import torch

from algo_trader.application.simulation.model_selection import (
    _complexity_scores_post_tune,
)
from algo_trader.domain.simulation import (
    CandidateSpec,
    ModelSelectionComplexity,
    ModelSelectionConfig,
)


def test_posterior_l1_complexity_prefers_more_shrunk_candidate(
    tmp_path: Path,
) -> None:
    debug_dir = (
        tmp_path / "inner" / "outer_7" / "postprocessing" / "debug"
    )
    debug_dir.mkdir(parents=True)
    _write_debug_payload(
        debug_dir / "candidate_0000_split_0000_state.pt",
        weight_scale=0.01,
    )
    _write_debug_payload(
        debug_dir / "candidate_0001_split_0000_state.pt",
        weight_scale=0.10,
    )

    scores = _complexity_scores_post_tune(
        base_dir=tmp_path,
        outer_k=7,
        candidates=(
            CandidateSpec(candidate_id=0, params={}),
            CandidateSpec(candidate_id=1, params={}),
        ),
        model_selection=ModelSelectionConfig(
            complexity=ModelSelectionComplexity(method="posterior_l1")
        ),
    )

    assert scores[0] < scores[1]


def _write_debug_payload(path: Path, *, weight_scale: float) -> None:
    payload = {
        "structural_posterior_means": {
            "alpha": torch.full((2,), 0.5),
            "sigma_idio": torch.full((2,), 0.2),
            "w": torch.full((2, 3), weight_scale),
            "beta": torch.full((2, 1), weight_scale),
            "B": torch.full((2, 2), weight_scale),
            "s_u_mean": torch.tensor(0.1),
        }
    }
    torch.save(payload, path)
