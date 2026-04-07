from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
import torch

from algo_trader.application.simulation import hooks, outer_walk_forward, runner
from algo_trader.application.simulation.feature_panel_data import (
    FeaturePanelData,
)
from algo_trader.domain.simulation import (
    AllocationConfig,
    AllocationFamilyConfig,
    CleaningSpec,
    OuterFold,
    PreprocessSpec,
    ScalingSpec,
)


def test_build_portfolio_specs_for_outer_includes_baselines() -> None:
    build_portfolio_specs = getattr(runner, "_build_portfolio_specs_for_outer")
    allocation = AllocationConfig(
        primary=AllocationFamilyConfig(
            family="long_only",
            params={"gross_exposure": 1.0, "max_weight": 0.25},
        ),
        baselines=(
            AllocationFamilyConfig(family="equal_weight"),
            AllocationFamilyConfig(family="random", params={"random_seed": 7}),
        ),
    )

    portfolios = build_portfolio_specs(allocation, {})

    assert [item.name for item in portfolios] == [
        "primary",
        "equal_weight",
        "random",
    ]
    assert portfolios[0].allocation["family"] == "long_only"
    assert portfolios[1].allocation["family"] == "equal_weight"
    assert portfolios[2].allocation["family"] == "random"


def test_evaluate_outer_walk_forward_runs_primary_and_baseline() -> None:
    simulation_hooks = hooks.SimulationHooks(
        fit_model=_fit_model,
        predict=_predict,
        score=_score,
        allocate=hooks.default_hooks().allocate,
        compute_pnl=hooks.default_hooks().compute_pnl,
    )

    result, cleaning = outer_walk_forward.evaluate_outer_walk_forward(
        context=_outer_context(),
        best_config={"training": {"method": "online_filtering"}},
        hooks=simulation_hooks,
    )

    assert cleaning is not None
    assert result["portfolio_primary"] == "primary"
    assert len(result["pnl"]) == 2
    assert len(result["weights"]) == 2
    assert len(result["timestamps"]) == 2
    assert len(result["gross_returns"]) == 2
    assert len(result["net_returns"]) == 2
    assert len(result["costs"]) == 2
    assert len(result["turnover"]) == 2
    portfolios = cast(Mapping[str, Mapping[str, list[Any]]], result["portfolios"])
    assert set(portfolios) == {"primary", "equal_weight"}
    assert len(portfolios["primary"]["pnl"]) == 2
    assert len(portfolios["equal_weight"]["pnl"]) == 2
    assert len(portfolios["primary"]["weights"]) == 2
    assert len(portfolios["equal_weight"]["weights"]) == 2
    assert len(portfolios["primary"]["timestamps"]) == 2
    assert len(portfolios["primary"]["gross_returns"]) == 2
    assert len(portfolios["primary"]["net_returns"]) == 2
    assert len(portfolios["primary"]["costs"]) == 2
    assert len(portfolios["primary"]["turnover"]) == 2


def _outer_context() -> outer_walk_forward.OuterEvaluationContext:
    preprocess_spec = PreprocessSpec(
        cleaning=CleaningSpec(
            min_usable_ratio=0.0,
            min_variance=0.0,
            max_abs_corr=1.0,
            corr_subsample=None,
        ),
        scaling=ScalingSpec(mad_eps=1e-6),
    )
    return outer_walk_forward.OuterEvaluationContext(
        panel=FeaturePanelData(
            X=torch.tensor(
                [
                    [[1.0], [0.5]],
                    [[1.1], [0.4]],
                    [[1.2], [0.3]],
                    [[1.3], [0.2]],
                ],
                dtype=torch.float32,
            ),
            M=torch.zeros((4, 2, 1), dtype=torch.bool),
            X_global=None,
            M_global=None,
            global_feature_names=(),
        ),
        y=torch.tensor(
            [
                [0.01, 0.00],
                [0.02, 0.01],
                [0.03, -0.01],
                [0.01, 0.02],
            ],
            dtype=torch.float32,
        ),
        timestamps=(0, 1, 2, 3),
        outer_fold=OuterFold(
            k_test=0,
            train_idx=np.array([0, 1], dtype=int),
            test_idx=np.array([2, 3], dtype=int),
            inner_group_ids=[0],
        ),
        preprocess_spec=preprocess_spec,
        num_pp_samples=8,
        portfolios=(
            outer_walk_forward.PortfolioSpec(
                name="primary",
                allocation={
                    "family": "long_only",
                    "min_weight": 0.0,
                    "max_weight": 0.75,
                },
                cost={},
            ),
            outer_walk_forward.PortfolioSpec(
                name="equal_weight",
                allocation={"family": "equal_weight"},
                cost={},
            ),
        ),
        assets=("A", "B"),
        execution_mode="walkforward",
    )


def _fit_model(
    X_train: torch.Tensor,
    X_train_global: torch.Tensor | None,
    y_train: torch.Tensor,
    config: Mapping[str, Any],
    init_state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    del X_train, X_train_global, y_train, config, init_state
    return {"state_id": 1}


def _predict(
    X_pred: torch.Tensor,
    X_pred_global: torch.Tensor | None,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    num_samples: int,
) -> Mapping[str, Any]:
    del X_pred, X_pred_global, state, config
    mean = torch.tensor([0.10, 0.02], dtype=torch.float32)
    covariance = torch.diag(torch.tensor([0.04, 0.01], dtype=torch.float32))
    samples = mean.repeat(num_samples, 1)
    return {"mean": mean, "covariance": covariance, "samples": samples}


def _score(
    y_true: torch.Tensor,
    pred: Mapping[str, Any],
    score_spec: Mapping[str, Any],
) -> float:
    del y_true, pred, score_spec
    return 0.0
