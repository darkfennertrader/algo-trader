from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from algo_trader.domain import ModelSelectionError
from algo_trader.infrastructure import log_boundary

from .config import DEFAULT_CONFIG_PATH, load_config
from .experiment import build_experiment
from .registry import default_registries

logger = logging.getLogger(__name__)


def _run_context(config_path: Path | None) -> Mapping[str, str]:
    return {"config": str(config_path or DEFAULT_CONFIG_PATH)}


@log_boundary("model_selection.run", context=_run_context)
def run(*, config_path: Path | None = None) -> Mapping[str, float]:
    config = load_config(config_path)
    registries = default_registries()
    experiment = build_experiment(config, registries=registries)
    results = experiment.run()
    if not results:
        raise ModelSelectionError(
            "Model selection completed with no metrics; check your CV setup."
        )
    logger.info("Model selection results=%s", results)
    return results
