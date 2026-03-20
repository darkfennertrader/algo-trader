import pyro
from pyro import poutine
import torch

from algo_trader.pipeline.stages.modeling.factor.learning.guide_v1_l10_online_filtering_amortized import (
    OnlineFilteringAmortizedGuide,
)
from algo_trader.pipeline.stages.modeling.factor.learning.model_v1_l10_online_filtering_sv_scale_mixture_global import (
    Batch,
    FilteringState,
    OnlineFilteringSVScaleMixtureGlobalModel,
)
from algo_trader.pipeline.stages.modeling.factor.learning.predict_v1_l10_online_filtering import (
    NextStepInputs,
    sample_next_step_predictive,
)


def test_l10_amortized_guide_builds_filtering_state() -> None:
    pyro.clear_param_store()
    batch = Batch(
        X_asset=torch.randn((2, 3, 4), dtype=torch.float32),
        X_global=torch.randn((2, 2), dtype=torch.float32),
        y=torch.randn((2, 3), dtype=torch.float32),
        filtering_state=FilteringState(
            h_loc=torch.tensor(0.1),
            h_scale=torch.tensor(0.2),
            steps_seen=10,
        ),
    )
    guide = OnlineFilteringAmortizedGuide(factor_count=2, hidden_dim=8)

    with poutine.trace() as trace:
        guide(batch)

    assert "h_1" in trace.trace.nodes
    assert "h_2" in trace.trace.nodes
    assert "v" in trace.trace.nodes

    filtering_state = guide.build_filtering_state(batch)
    structural = guide.structural_posterior_means(batch)

    assert filtering_state.steps_seen == 12
    assert filtering_state.h_loc.ndim == 0
    assert filtering_state.h_scale.ndim == 0
    assert structural.alpha.shape == (3,)
    assert structural.w.shape == (3, 4)
    assert structural.beta.shape == (3, 2)
    assert structural.B.shape == (3, 2)


def test_l10_predictive_helper_returns_horizon_one_distribution() -> None:
    pyro.clear_param_store()
    batch = Batch(
        X_asset=torch.randn((3, 2, 3), dtype=torch.float32),
        X_global=torch.randn((3, 2), dtype=torch.float32),
        y=torch.randn((3, 2), dtype=torch.float32),
    )
    guide = OnlineFilteringAmortizedGuide(factor_count=2, hidden_dim=8)
    guide(batch)
    model = OnlineFilteringSVScaleMixtureGlobalModel(factor_count=2)

    result = sample_next_step_predictive(
        model=model,
        guide=guide,
        fitted_batch=batch,
        next_inputs=NextStepInputs(
            X_asset=torch.randn((2, 3), dtype=torch.float32),
            X_global=torch.randn((2,), dtype=torch.float32),
        ),
        num_samples=32,
    )

    assert result.samples.shape == (32, 2)
    assert result.mean.shape == (2,)
    assert result.covariance.shape == (2, 2)
    assert result.filtering_state.steps_seen == 3
