from pathlib import Path

import pytest

from algo_trader.application.simulation import config as simulation_config
from algo_trader.application.simulation import config_tuning
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
    assert not dict(model.predict_params)


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


def test_build_model_selection_calibration_parses_custom_values() -> None:
    selection = simulation_config._build_model_selection_config(  # pylint: disable=protected-access
        {
            "model_selection": {
                "calibration": {
                    "top_k": 2,
                    "coverage_levels": [0.5, 0.8, 0.95],
                    "mean_abs_weight": 2.0,
                    "max_abs_weight": 3.0,
                    "pit_weight": 4.0,
                }
            }
        },
        Path("simulation.yml"),
    )

    assert selection.calibration.top_k == 2
    assert selection.calibration.coverage_levels == (0.5, 0.8, 0.95)
    assert selection.calibration.mean_abs_weight == 2.0
    assert selection.calibration.max_abs_weight == 3.0
    assert selection.calibration.pit_weight == 4.0


def test_build_model_selection_parses_basket_aware_mode() -> None:
    selection = simulation_config._build_model_selection_config(  # pylint: disable=protected-access
        {
            "model_selection": {
                "mode": "basket_aware",
                "basket": {
                    "baskets": [
                        "us_index",
                        "europe_index",
                        "us_minus_europe",
                        "index_equal_weight",
                    ],
                    "mean_abs_weight": 2.0,
                    "max_abs_weight": 3.0,
                    "pit_weight": 4.0,
                },
            }
        },
        Path("simulation.yml"),
    )

    assert selection.mode == "basket_aware"
    assert selection.basket.baskets == (
        "us_index",
        "europe_index",
        "us_minus_europe",
        "index_equal_weight",
    )
    assert selection.basket.mean_abs_weight == 2.0
    assert selection.basket.max_abs_weight == 3.0
    assert selection.basket.pit_weight == 4.0


def test_build_flags_parses_execution_mode() -> None:
    build_flags = getattr(simulation_config, "_build_flags")
    flags = build_flags(
        {"execution": {"mode": "walkforward"}},
        Path("simulation.yml"),
    )

    assert flags.execution_mode == "walkforward"


def test_build_flags_parses_posterior_signal_execution_mode() -> None:
    build_flags = getattr(simulation_config, "_build_flags")
    flags = build_flags(
        {"execution": {"mode": "posterior_signal"}},
        Path("simulation.yml"),
    )

    assert flags.execution_mode == "posterior_signal"


def test_build_flags_normalizes_outer_evaluation_alias() -> None:
    build_flags = getattr(simulation_config, "_build_flags")
    flags = build_flags(
        {"execution": {"mode": "outer_evaluation"}},
        Path("simulation.yml"),
    )

    assert flags.execution_mode == "walkforward"


def test_build_flags_rejects_invalid_execution_mode() -> None:
    build_flags = getattr(simulation_config, "_build_flags")

    with pytest.raises(ConfigError):
        build_flags(
            {"execution": {"mode": "bad_mode"}},
            Path("simulation.yml"),
        )


def test_build_flags_rejects_stop_after() -> None:
    build_flags = getattr(simulation_config, "_build_flags")

    with pytest.raises(ConfigError):
        build_flags(
            {
                "execution": {"mode": "full"},
                "stop_after": "inner",
            },
            Path("simulation.yml"),
        )


def test_build_walkforward_config_parses_flat_seed_schema() -> None:
    build_walkforward = getattr(
        simulation_config, "_build_walkforward_config"
    )
    config = build_walkforward(
        {
            "walkforward": {
                "num_seeds": 3,
                "seeds": [11, 13, 17],
                "max_parallel_seeds_per_gpu": 2,
            }
        },
        Path("simulation.yml"),
    )

    assert config.num_seeds == 3
    assert config.seeds == (11, 13, 17)
    assert config.max_parallel_seeds_per_gpu == 2


def test_build_walkforward_config_rejects_mismatched_seed_count() -> None:
    build_walkforward = getattr(
        simulation_config, "_build_walkforward_config"
    )

    with pytest.raises(ConfigError):
        build_walkforward(
            {
                "walkforward": {
                    "num_seeds": 2,
                    "seeds": [7, 19, 43],
                }
            },
            Path("simulation.yml"),
        )


def test_build_walkforward_config_rejects_removed_nested_keys() -> None:
    build_walkforward = getattr(
        simulation_config, "_build_walkforward_config"
    )

    with pytest.raises(ConfigError):
        build_walkforward(
            {
                "walkforward": {
                    "runtime": {
                        "parallelize_seeds": True,
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_build_data_config_parses_portfolio_output_path() -> None:
    build_data = getattr(simulation_config, "_build_data_config")
    config = build_data(
        {
            "data": {
                "simulation_output_path": (
                    "older experiments/v3/v3_l1_unified_5y_bugfixed"
                ),
                "portfolio_output_path": "portfolio/herc",
                "posterior_output_path": "v4_l1_11y_basket_aware",
            }
        },
        Path("simulation.yml"),
    )

    assert (
        config.simulation_output_path
        == "older experiments/v3/v3_l1_unified_5y_bugfixed"
    )
    assert config.portfolio_output_path == "portfolio/herc"
    assert config.posterior_output_path == "v4_l1_11y_basket_aware"


def test_build_data_config_rejects_invalid_simulation_output_path() -> None:
    build_data = getattr(simulation_config, "_build_data_config")

    with pytest.raises(ConfigError):
        build_data(
            {
                "data": {
                    "simulation_output_path": "../older experiments/v3/model",
                }
            },
            Path("simulation.yml"),
        )


def test_build_data_config_rejects_invalid_portfolio_output_path() -> None:
    build_data = getattr(simulation_config, "_build_data_config")

    with pytest.raises(ConfigError):
        build_data(
            {
                "data": {
                    "simulation_output_path": "validated_model",
                    "portfolio_output_path": "../portfolio/herc",
                }
            },
            Path("simulation.yml"),
        )


def test_build_data_config_rejects_invalid_posterior_output_path() -> None:
    build_data = getattr(simulation_config, "_build_data_config")

    with pytest.raises(ConfigError):
        build_data(
            {
                "data": {
                    "simulation_output_path": "validated_model",
                    "posterior_output_path": "../posterior_signal/model_a",
                }
            },
            Path("simulation.yml"),
        )


def test_build_tuning_config_parses_ray_early_stopping() -> None:
    tuning = config_tuning.build_tuning_config(
        {
            "tuning": {
                "engine": "ray",
                "num_samples": 4,
                "space": [],
                "ray": {
                    "early_stopping": {
                        "enabled": True,
                        "method": "median",
                        "grace_period": 16,
                        "min_samples_required": 4,
                    }
                },
            }
        },
        Path("simulation.yml"),
    )

    assert tuning.ray.early_stopping.enabled
    assert tuning.ray.early_stopping.method == "median"
    assert tuning.ray.early_stopping.grace_period == 16
    assert tuning.ray.early_stopping.min_samples_required == 4


def test_build_tuning_config_rejects_invalid_ray_early_stopping_method() -> None:
    with pytest.raises(ConfigError):
        config_tuning.build_tuning_config(
            {
                "tuning": {
                    "engine": "ray",
                    "num_samples": 4,
                    "space": [],
                    "ray": {
                        "early_stopping": {
                            "enabled": True,
                            "method": "asha",
                        }
                    },
                }
            },
            Path("simulation.yml"),
        )


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


def test_load_config_parses_primary_and_baselines() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )
    config = build_allocation_config(
        {
            "allocation": {
                "primary": {
                    "family": "long_only",
                    "min_weight": 0.0,
                    "max_weight": 0.25,
                },
                "baselines": [
                    {"family": "equal_weight"},
                    {"family": "random", "random_seed": 7},
                ],
            }
        },
        Path("simulation.yml"),
    )

    assert config.primary.family == "long_only"
    assert config.primary.params["max_weight"] == 0.25
    assert len(config.baselines) == 2
    assert config.baselines[0].family == "equal_weight"
    assert config.baselines[1].family == "random"


def test_load_config_rejects_long_only_gross_exposure() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )

    with pytest.raises(ConfigError):
        build_allocation_config(
            {
                "allocation": {
                    "primary": {
                        "family": "long_only",
                        "gross_exposure": 1.0,
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_load_config_rejects_long_only_use_previous_weights() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )

    with pytest.raises(ConfigError):
        build_allocation_config(
            {
                "allocation": {
                    "primary": {
                        "family": "long_only",
                        "use_previous_weights": True,
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_load_config_parses_herc_primary() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )
    config = build_allocation_config(
        {
            "allocation": {
                "primary": {
                    "family": "herc",
                    "risk_measure": "cvar",
                    "distance_estimator": "pearson",
                    "min_weight": 0.0,
                    "max_weight": 0.20,
                    "transaction_costs": 0.0,
                    "use_previous_weights": False,
                },
                "baselines": [{"family": "equal_weight"}],
            }
        },
        Path("simulation.yml"),
    )

    assert config.primary.family == "herc"
    assert config.primary.params["risk_measure"] == "cvar"
    assert config.primary.params["distance_estimator"] == "pearson"


def test_load_config_rejects_invalid_herc_distance_estimator() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )

    with pytest.raises(ConfigError):
        build_allocation_config(
            {
                "allocation": {
                    "primary": {
                        "family": "herc",
                        "distance_estimator": "kendall",
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_load_config_parses_schur_primary() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )
    config = build_allocation_config(
        {
            "allocation": {
                "primary": {
                    "family": "schur",
                    "gamma": 0.5,
                    "distance_estimator": "pearson",
                    "transaction_costs": 0.0,
                    "use_previous_weights": False,
                },
                "baselines": [{"family": "equal_weight"}],
            }
        },
        Path("simulation.yml"),
    )

    assert config.primary.family == "schur"
    assert config.primary.params["gamma"] == 0.5
    assert config.primary.params["distance_estimator"] == "pearson"


def test_load_config_rejects_invalid_schur_gamma() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )

    with pytest.raises(ConfigError):
        build_allocation_config(
            {
                "allocation": {
                    "primary": {
                        "family": "schur",
                        "gamma": 1.5,
                    }
                }
            },
            Path("simulation.yml"),
        )


def test_load_config_rejects_long_only_negative_min_weight() -> None:
    build_allocation_config = getattr(
        simulation_config, "_build_allocation_config"
    )

    with pytest.raises(ConfigError):
        build_allocation_config(
            {
                "allocation": {
                    "primary": {
                        "family": "long_only",
                        "min_weight": -0.1,
                    }
                }
            },
            Path("simulation.yml"),
        )
