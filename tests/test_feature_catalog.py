from pathlib import Path

import pytest

from algo_trader.application.feature_catalog import (
    DEFAULT_FEATURE_CONFIG_PATH,
    FeatureCatalogConfig,
)
from algo_trader.application.feature_engineering import runner as feature_runner
from algo_trader.domain import ConfigError
from tests.support import DEFAULT_EXOGENOUS_GLOBAL_FEATURES


def test_feature_catalog_loads_default_config() -> None:
    catalog = FeatureCatalogConfig.load()

    assert DEFAULT_FEATURE_CONFIG_PATH == Path(
        "/home/ray/projects/algo-trader/config/features.yml"
    )
    assert catalog.technical.ordered_group_names() == (
        "momentum",
        "mean_reversion",
        "breakout",
        "cross_sectional",
        "volatility",
        "seasonal",
        "regime",
    )
    assert catalog.exogenous.asset_features == ("carry_3m_diff",)
    assert catalog.exogenous.global_features == DEFAULT_EXOGENOUS_GLOBAL_FEATURES


def test_feature_engineering_rejects_cli_feature_override() -> None:
    with pytest.raises(
        ConfigError,
        match="feature selection is config-driven; edit config/features.yml",
    ):
        feature_runner.run(
            request=feature_runner.RunRequest(
                groups=["momentum"], features=["vol_scaled_momentum"]
            )
        )
