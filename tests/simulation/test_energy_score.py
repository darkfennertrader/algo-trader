import pytest
import torch

from algo_trader.application.simulation.metrics import build_metric_scorer


def test_energy_score_simple_case() -> None:
    spec = {"metric_name": "energy_score"}
    scorer = build_metric_scorer(spec, scope="inner")
    y_true = torch.tensor([[0.0]])
    samples = torch.tensor([[[1.0]], [[3.0]]])
    pred = {
        "samples": samples,
        "energy_score": {
            "scale": torch.ones(1),
            "whitener": torch.eye(1),
        },
    }
    score = scorer(y_true, pred, spec)
    assert score == pytest.approx(-1.5)


def test_energy_score_accepts_single_time_sample() -> None:
    spec = {"metric_name": "energy_score"}
    scorer = build_metric_scorer(spec, scope="inner")
    y_true = torch.tensor([0.0])
    samples = torch.tensor([[1.0], [3.0]])
    pred = {
        "samples": samples,
        "energy_score": {
            "scale": torch.ones(1),
            "whitener": torch.eye(1),
        },
    }
    score = scorer(y_true, pred, spec)
    assert score == pytest.approx(-1.5)
