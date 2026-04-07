from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Protocol, Sequence

import pandas as pd
import torch

from algo_trader.domain import ConfigError, SimulationError
from algo_trader.domain.simulation import PredictionPacket


class _FittedSkfolioModel(Protocol):
    weights_: Any

    def fit(self, X: Any) -> Any:
        ...


@dataclass(frozen=True)
class TensorTarget:
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class SkfolioRuntimeConfig:
    risk_measure: str
    min_weight: float | None
    max_weight: float | None
    transaction_costs: float
    previous_weights: torch.Tensor | None
    use_previous_weights: bool
    tensor_target: TensorTarget


def allocate_herc(
    *,
    prediction: PredictionPacket,
    runtime_config: SkfolioRuntimeConfig,
    distance_estimator: str,
) -> torch.Tensor:
    scenarios = _extract_predictive_scenarios(prediction)
    frame = _build_scenario_frame(scenarios, prediction.asset_names)
    model = _build_herc_model(
        runtime_config=runtime_config,
        distance_estimator=distance_estimator,
    )
    try:
        model.fit(frame)
    except Exception as exc:  # pragma: no cover - library failure path
        raise SimulationError(
            "skfolio HERC allocation failed",
            context={"error": str(exc)},
        ) from exc
    return _model_weights(model, runtime_config=runtime_config)


def allocate_risk_budgeting(
    *,
    prediction: PredictionPacket,
    runtime_config: SkfolioRuntimeConfig,
) -> torch.Tensor:
    scenarios = _extract_predictive_scenarios(prediction)
    frame = _build_scenario_frame(scenarios, prediction.asset_names)
    model = _build_risk_budgeting_model(
        runtime_config=runtime_config,
    )
    try:
        model.fit(frame)
    except Exception as exc:  # pragma: no cover - library failure path
        raise SimulationError(
            "skfolio RiskBudgeting allocation failed",
            context={"error": str(exc)},
        ) from exc
    return _model_weights(model, runtime_config=runtime_config)


def _extract_predictive_scenarios(
    prediction: PredictionPacket,
) -> torch.Tensor:
    samples = prediction.samples
    if not isinstance(samples, torch.Tensor):
        raise ConfigError(
            "allocation with skfolio requires predictive samples"
        )
    return samples.detach()


def _build_scenario_frame(
    scenarios: torch.Tensor,
    asset_names: Sequence[str],
) -> pd.DataFrame:
    columns = list(asset_names)
    if not columns:
        columns = [f"asset_{idx}" for idx in range(int(scenarios.shape[-1]))]
    if len(columns) != int(scenarios.shape[-1]):
        raise ConfigError(
            "allocation asset_names length must match predictive asset dimension"
        )
    values = scenarios.to(device="cpu", dtype=torch.float32).numpy()
    return pd.DataFrame(values, columns=columns)


def _build_herc_model(
    *,
    runtime_config: SkfolioRuntimeConfig,
    distance_estimator: str,
    ) -> _FittedSkfolioModel:
    skfolio_module = importlib.import_module("skfolio")
    optimization_module = importlib.import_module("skfolio.optimization")
    estimator_cls = optimization_module.HierarchicalEqualRiskContribution
    return estimator_cls(
        risk_measure=_resolve_risk_measure(
            risk_measure=runtime_config.risk_measure,
            risk_measure_enum=skfolio_module.RiskMeasure,
            extra_risk_measure_enum=skfolio_module.ExtraRiskMeasure,
        ),
        distance_estimator=_build_distance_estimator(distance_estimator),
        min_weights=_min_weight(runtime_config),
        max_weights=_max_weight(runtime_config),
        transaction_costs=runtime_config.transaction_costs,
        previous_weights=_serialize_previous_weights(
            runtime_config=runtime_config,
        ),
        raise_on_failure=True,
    )


def _build_risk_budgeting_model(
    *,
    runtime_config: SkfolioRuntimeConfig,
) -> _FittedSkfolioModel:
    skfolio_module = importlib.import_module("skfolio")
    optimization_module = importlib.import_module("skfolio.optimization")
    estimator_cls = optimization_module.RiskBudgeting
    return estimator_cls(
        risk_measure=_resolve_risk_measure(
            risk_measure=runtime_config.risk_measure,
            risk_measure_enum=skfolio_module.RiskMeasure,
            extra_risk_measure_enum=skfolio_module.ExtraRiskMeasure,
        ),
        min_weights=_min_weight(runtime_config),
        max_weights=_max_weight(runtime_config),
        transaction_costs=runtime_config.transaction_costs,
        previous_weights=_serialize_previous_weights(
            runtime_config=runtime_config,
        ),
        raise_on_failure=True,
    )


def _resolve_risk_measure(
    *,
    risk_measure: str,
    risk_measure_enum: object,
    extra_risk_measure_enum: object,
) -> object:
    name = risk_measure.strip().upper()
    if hasattr(risk_measure_enum, name):
        return getattr(risk_measure_enum, name)
    if hasattr(extra_risk_measure_enum, name):
        return getattr(extra_risk_measure_enum, name)
    valid = sorted(
        [
            item.lower()
            for item in dir(risk_measure_enum)
            if item.isupper()
        ]
        + [
            item.lower()
            for item in dir(extra_risk_measure_enum)
            if item.isupper()
        ]
    )
    raise ConfigError(
        "allocation.spec.risk_measure must be one of "
        + ", ".join(valid)
    )


def _build_distance_estimator(name: str) -> object:
    distance_module = importlib.import_module("skfolio.distance")
    normalized = name.strip().lower()
    if normalized == "pearson":
        return distance_module.PearsonDistance()
    if normalized == "mutual_information":
        return distance_module.MutualInformation()
    raise ConfigError(
        "allocation.spec.distance_estimator must be pearson or "
        "mutual_information"
    )


def _serialize_previous_weights(
    *,
    runtime_config: SkfolioRuntimeConfig,
) -> object | None:
    previous_weights = runtime_config.previous_weights
    if not runtime_config.use_previous_weights or previous_weights is None:
        return None
    return (
        previous_weights.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )


def _model_weights(
    model: _FittedSkfolioModel,
    *,
    runtime_config: SkfolioRuntimeConfig,
) -> torch.Tensor:
    return torch.as_tensor(
        model.weights_,
        device=runtime_config.tensor_target.device,
        dtype=runtime_config.tensor_target.dtype,
    )


def _min_weight(runtime_config: SkfolioRuntimeConfig) -> float:
    if runtime_config.min_weight is None:
        return 0.0
    return runtime_config.min_weight


def _max_weight(runtime_config: SkfolioRuntimeConfig) -> float:
    if runtime_config.max_weight is None:
        return 1.0
    return runtime_config.max_weight


__all__ = [
    "SkfolioRuntimeConfig",
    "TensorTarget",
    "allocate_herc",
    "allocate_risk_budgeting",
]
