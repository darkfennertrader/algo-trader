from pathlib import Path

import pytest

from algo_trader.application.simulation import config as simulation_config
from algo_trader.domain import ConfigError


def test_build_training_config_parses_online_filtering() -> None:
    training = simulation_config._build_training_config(  # pylint: disable=protected-access
        {
            "training": {
                "method": "online_filtering",
                "target_normalization": False,
                "log_prob_scaling": False,
                "online_filtering": {"steps_per_observation": 3},
                "svi_shared": {
                    "learning_rate": 0.01,
                    "grad_accum_steps": 1,
                },
                "tbptt": {"num_steps": 200},
            }
        },
        Path("simulation.yml"),
    )

    assert training.method == "online_filtering"
    assert training.online_filtering.steps_per_observation == 3
    assert training.tbptt.num_steps == 200


def test_build_model_config_parses_optional_predictor() -> None:
    model = simulation_config._build_model_config(  # pylint: disable=protected-access
        {
            "model": {
                "model_name": "factor_model_l10_online_filtering",
                "guide_name": "factor_guide_l10_online_filtering",
                "predict_name": "factor_predict_l10_online_filtering",
                "predict_params": {},
            }
        },
        Path("simulation.yml"),
    )

    assert model.predict_name == "factor_predict_l10_online_filtering"
    assert dict(model.predict_params) == {}


def test_build_preprocess_config_parses_exogenous_mask_flag() -> None:
    spec = simulation_config._build_preprocess_spec(  # pylint: disable=protected-access
        {
            "preprocessing": {
                "append_mask_as_features": False,
                "append_exogenous_mask_as_features": True,
            }
        },
        Path("simulation.yml"),
    )

    assert not spec.scaling.inputs.append_mask_as_features
    assert spec.scaling.inputs.append_exogenous_mask_as_features


def test_build_model_selection_complexity_parses_posterior_l1() -> None:
    selection = simulation_config._build_model_selection_config(  # pylint: disable=protected-access
        {
            "model_selection": {
                "complexity": {"method": "posterior_l1", "seed": 7}
            }
        },
        Path("simulation.yml"),
    )

    assert selection.complexity.method == "posterior_l1"
    assert selection.complexity.seed == 7


def test_build_training_config_rejects_online_filtering_target_norm() -> None:
    with pytest.raises(ConfigError):
        simulation_config._build_training_config(  # pylint: disable=protected-access
            {
                "training": {
                    "method": "online_filtering",
                    "target_normalization": True,
                    "log_prob_scaling": False,
                    "svi_shared": {"grad_accum_steps": 1},
                    "tbptt": {},
                }
            },
            Path("simulation.yml"),
        )


def test_build_training_config_rejects_online_filtering_grad_accum() -> None:
    with pytest.raises(ConfigError):
        simulation_config._build_training_config(  # pylint: disable=protected-access
            {
                "training": {
                    "method": "online_filtering",
                    "target_normalization": False,
                    "log_prob_scaling": True,
                    "svi_shared": {"grad_accum_steps": 2},
                    "tbptt": {},
                }
            },
            Path("simulation.yml"),
        )


def test_build_allocation_config_parses_random_long_short_bounded_net() -> None:
    allocation = simulation_config._build_allocation_config(  # pylint: disable=protected-access
        {
            "allocation": {
                "spec": {
                    "method": "random",
                    "portfolio_style": "long_short_bounded_net",
                    "random_seed": 7,
                    "gross_exposure": 1.0,
                }
            }
        },
        Path("simulation.yml"),
    )

    assert allocation.spec["method"] == "random"
    assert allocation.spec["portfolio_style"] == "long_short_bounded_net"


def test_build_allocation_config_rejects_unsupported_risk_budgeting_style() -> None:
    with pytest.raises(ConfigError):
        simulation_config._build_allocation_config(  # pylint: disable=protected-access
            {
                "allocation": {
                    "spec": {
                        "method": "skfolio_risk_budgeting",
                        "portfolio_style": "long_short_bounded_net",
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_build_allocation_config_parses_de_risked_flat_portfolio() -> None:
    allocation = simulation_config._build_allocation_config(  # pylint: disable=protected-access
        {
            "allocation": {
                "spec": {
                    "method": "de_risked",
                    "portfolio_style": "long_only",
                    "gross_exposure": 0.0,
                }
            }
        },
        Path("simulation.yml"),
    )

    assert allocation.spec["method"] == "de_risked"
    assert allocation.spec["gross_exposure"] == 0.0
