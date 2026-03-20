from pathlib import Path

import pytest

from algo_trader.application.simulation import config as simulation_config
from algo_trader.domain import ConfigError


def test_build_training_config_parses_online_filtering() -> None:
    training = simulation_config._build_training_config(  # pylint: disable=protected-access
        {
            "training": {
                "method": "online_filtering",
                "target_normalization": True,
                "log_prob_scaling": False,
                "online_filtering": {"steps_per_observation": 3},
                "svi": {
                    "num_steps": 200,
                    "learning_rate": 0.01,
                    "grad_accum_steps": 1,
                },
            }
        },
        Path("simulation.yml"),
    )

    assert training.method == "online_filtering"
    assert training.online_filtering.steps_per_observation == 3
    assert training.svi.num_steps == 200


def test_build_training_config_rejects_online_filtering_grad_accum() -> None:
    with pytest.raises(ConfigError):
        simulation_config._build_training_config(  # pylint: disable=protected-access
            {
                "training": {
                    "method": "online_filtering",
                    "target_normalization": False,
                    "log_prob_scaling": True,
                    "svi": {"grad_accum_steps": 2},
                }
            },
            Path("simulation.yml"),
        )
