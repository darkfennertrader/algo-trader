from __future__ import annotations

import logging
from dataclasses import replace

from algo_trader.domain.simulation import PanelDataset, SimulationConfig
from algo_trader.infrastructure.data import build_synthetic_panel_dataset

SMOKE_TEST_T = 8
SMOKE_TEST_A = 3
SMOKE_TEST_F = 5
SMOKE_TEST_SEED = 123
SMOKE_TEST_SVI_STEPS = 2
SMOKE_TEST_NUM_PARTICLES = 1
SMOKE_TEST_PRED_SAMPLES = 1
SMOKE_TEST_WARMUP_LEN = 1
SMOKE_TEST_GROUP_LEN = 2
SMOKE_TEST_HORIZON = 1
SMOKE_TEST_EMBARGO_LEN = 0
SMOKE_TEST_Q = 1
SMOKE_TEST_MAX_INNER_COMBOS = 1
SMOKE_TEST_OUTER_LAST_N = 1

logger = logging.getLogger(__name__)


def is_smoke_test_enabled(config: SimulationConfig) -> bool:
    return config.flags.smoke_test_enabled


def apply_smoke_test_overrides(
    config: SimulationConfig,
) -> SimulationConfig:
    if not is_smoke_test_enabled(config):
        return config
    logger.info(
        "Smoke test enabled: using synthetic data and fast SVI overrides."
    )
    flags = replace(config.flags, stop_after="inner")
    tuning = replace(
        config.modeling.tuning,
        engine="local",
        space=tuple(),
        num_samples=1,
    )
    training_svi = replace(
        config.modeling.training.svi,
        num_steps=SMOKE_TEST_SVI_STEPS,
        num_elbo_particles=SMOKE_TEST_NUM_PARTICLES,
        log_every=None,
    )
    training = replace(config.modeling.training, svi=training_svi)
    modeling = replace(config.modeling, training=training, tuning=tuning)
    model_selection = replace(
        config.evaluation.model_selection, enable=False
    )
    predictive = replace(
        config.evaluation.predictive,
        num_samples_inner=SMOKE_TEST_PRED_SAMPLES,
        num_samples_outer=SMOKE_TEST_PRED_SAMPLES,
    )
    evaluation = replace(
        config.evaluation,
        model_selection=model_selection,
        predictive=predictive,
    )
    cv_window = replace(
        config.cv.window,
        warmup_len=SMOKE_TEST_WARMUP_LEN,
        group_len=SMOKE_TEST_GROUP_LEN,
    )
    cv_leakage = replace(
        config.cv.leakage,
        horizon=SMOKE_TEST_HORIZON,
        embargo_len=SMOKE_TEST_EMBARGO_LEN,
    )
    cv_cpcv = replace(
        config.cv.cpcv,
        q=SMOKE_TEST_Q,
        max_inner_combos=SMOKE_TEST_MAX_INNER_COMBOS,
    )
    cv = replace(
        config.cv,
        window=cv_window,
        leakage=cv_leakage,
        cpcv=cv_cpcv,
        exclude_warmup=True,
    )
    outer = replace(
        config.outer,
        test_group_ids=None,
        last_n=SMOKE_TEST_OUTER_LAST_N,
    )
    return replace(
        config,
        flags=flags,
        modeling=modeling,
        evaluation=evaluation,
        cv=cv,
        outer=outer,
    )


def build_smoke_test_dataset(device: str) -> PanelDataset:
    return build_synthetic_panel_dataset(
        T=SMOKE_TEST_T,
        A=SMOKE_TEST_A,
        F=SMOKE_TEST_F,
        seed=SMOKE_TEST_SEED,
        device=device,
    )
