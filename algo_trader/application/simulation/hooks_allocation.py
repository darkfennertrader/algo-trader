from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Literal, Mapping, Sequence, cast

import pandas as pd
import torch

from algo_trader.domain import ConfigError, SimulationError

AllocatorMethod = Literal[
    "equal_weight",
    "random",
    "de_risked",
    "skfolio_risk_budgeting",
]
PortfolioStyle = Literal[
    "long_only",
    "long_short_bounded_net",
    "factor_neutral_long_short",
]

_VALID_METHODS = {
    "equal_weight",
    "random",
    "de_risked",
    "skfolio_risk_budgeting",
}
_VALID_STYLES = {
    "long_only",
    "long_short_bounded_net",
    "factor_neutral_long_short",
}


@dataclass(frozen=True)
class _SkfolioOptions:
    risk_measure: str
    use_previous_weights: bool
    transaction_costs: float
    min_weight: float | None
    max_weight: float | None


@dataclass(frozen=True)
class _AllocationParams:
    method: AllocatorMethod
    portfolio_style: PortfolioStyle
    gross_exposure: float
    random_seed: int | None
    skfolio: _SkfolioOptions


@dataclass(frozen=True)
class _TensorSpec:
    device: torch.device
    dtype: torch.dtype


def _allocate_weights(
    pred: Mapping[str, Any],
    alloc_spec: Mapping[str, Any],
    w_prev: torch.Tensor | None = None,
    asset_names: Sequence[str] = (),
) -> torch.Tensor:
    params = _parse_allocation_params(alloc_spec)
    n_assets = _resolve_n_assets(pred, alloc_spec, asset_names)
    tensor_spec = _resolve_tensor_spec(pred)
    if params.method == "equal_weight":
        return _allocate_equal_weight(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    if params.method == "random":
        return _allocate_random(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    if params.method == "de_risked":
        return _allocate_de_risked(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    return _allocate_skfolio_risk_budgeting(
        pred=pred,
        asset_names=asset_names,
        w_prev=w_prev,
        params=params,
        tensor_spec=tensor_spec,
    )


def _parse_allocation_params(
    alloc_spec: Mapping[str, Any],
) -> _AllocationParams:
    method = str(alloc_spec.get("method", "equal_weight")).strip().lower()
    if method not in _VALID_METHODS:
        raise ConfigError(
            "allocation.spec.method must be equal_weight, random, "
            "de_risked, or skfolio_risk_budgeting"
        )
    portfolio_style = str(
        alloc_spec.get("portfolio_style", "long_only")
    ).strip().lower()
    if portfolio_style not in _VALID_STYLES:
        raise ConfigError(
            "allocation.spec.portfolio_style must be long_only, "
            "long_short_bounded_net, or factor_neutral_long_short"
        )
    gross_default = 0.10 if method == "de_risked" else 1.0
    gross_exposure = float(alloc_spec.get("gross_exposure", gross_default))
    if gross_exposure < 0.0:
        raise ConfigError("allocation.spec.gross_exposure must be >= 0")
    if gross_exposure == 0.0 and method != "de_risked":
        raise ConfigError(
            "allocation.spec.gross_exposure must be > 0 unless "
            "allocation.spec.method=de_risked"
        )
    random_seed = _optional_int(alloc_spec.get("random_seed"), "random_seed")
    risk_measure = str(alloc_spec.get("risk_measure", "cvar")).strip().lower()
    use_previous_weights = bool(alloc_spec.get("use_previous_weights", False))
    transaction_costs = float(alloc_spec.get("transaction_costs", 0.0))
    min_weight = _optional_float(alloc_spec.get("min_weight"), "min_weight")
    max_weight = _optional_float(alloc_spec.get("max_weight"), "max_weight")
    return _AllocationParams(
        method=cast(AllocatorMethod, method),
        portfolio_style=cast(PortfolioStyle, portfolio_style),
        gross_exposure=gross_exposure,
        random_seed=random_seed,
        skfolio=_SkfolioOptions(
            risk_measure=risk_measure,
            use_previous_weights=use_previous_weights,
            transaction_costs=transaction_costs,
            min_weight=min_weight,
            max_weight=max_weight,
        ),
    )


def _optional_int(raw: Any, field: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"allocation.spec.{field} must be an integer")
    return int(raw)


def _optional_float(raw: Any, field: str) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"allocation.spec.{field} must be numeric") from exc


def _resolve_n_assets(
    pred: Mapping[str, Any],
    alloc_spec: Mapping[str, Any],
    asset_names: Sequence[str],
) -> int:
    if asset_names:
        return len(asset_names)
    samples = pred.get("samples")
    if isinstance(samples, torch.Tensor):
        return int(samples.shape[-1])
    mean = pred.get("mean")
    if isinstance(mean, torch.Tensor):
        return int(mean.shape[-1])
    return int(alloc_spec.get("n_assets", 1))


def _resolve_tensor_spec(pred: Mapping[str, Any]) -> _TensorSpec:
    for key in ("mean", "samples", "covariance"):
        value = pred.get(key)
        if isinstance(value, torch.Tensor):
            return _TensorSpec(device=value.device, dtype=value.dtype)
    return _TensorSpec(device=torch.device("cpu"), dtype=torch.float32)


def _allocate_equal_weight(
    *,
    n_assets: int,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    style = params.portfolio_style
    gross_exposure = params.gross_exposure
    if style == "long_only":
        return torch.full(
            (n_assets,),
            gross_exposure / n_assets,
            device=tensor_spec.device,
            dtype=tensor_spec.dtype,
        )
    if style == "factor_neutral_long_short":
        raise ConfigError(
            "allocation.spec.portfolio_style=factor_neutral_long_short "
            "requires a factor-constrained allocator backend"
        )
    return _bounded_net_equal_weights(
        n_assets=n_assets,
        gross_exposure=gross_exposure,
        tensor_spec=tensor_spec,
    )


def _bounded_net_equal_weights(
    *,
    n_assets: int,
    gross_exposure: float,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if n_assets < 2:
        raise ConfigError(
            "allocation.spec.portfolio_style=long_short_bounded_net "
            "requires at least 2 assets"
        )
    positive_indices = list(range(0, n_assets, 2))
    negative_indices = list(range(1, n_assets, 2))
    if not negative_indices:
        raise ConfigError(
            "allocation.spec.portfolio_style=long_short_bounded_net "
            "requires at least 2 assets"
        )
    weights = torch.zeros(
        (n_assets,),
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )
    pos_weight = gross_exposure * 0.5 / len(positive_indices)
    neg_weight = gross_exposure * 0.5 / len(negative_indices)
    weights[positive_indices] = pos_weight
    weights[negative_indices] = -neg_weight
    return weights


def _allocate_random(
    *,
    n_assets: int,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    style = params.portfolio_style
    gross_exposure = params.gross_exposure
    generator = _build_generator(params.random_seed)
    if style == "long_only":
        return _random_simplex(
            n_assets=n_assets,
            total=gross_exposure,
            generator=generator,
            tensor_spec=tensor_spec,
        )
    if style == "factor_neutral_long_short":
        raise ConfigError(
            "allocation.spec.portfolio_style=factor_neutral_long_short "
            "requires a factor-constrained allocator backend"
        )
    return _random_bounded_net(
        n_assets=n_assets,
        gross_exposure=gross_exposure,
        generator=generator,
        tensor_spec=tensor_spec,
    )


def _allocate_de_risked(
    *,
    n_assets: int,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=de_risked currently supports only "
            "allocation.spec.portfolio_style=long_only"
        )
    if params.gross_exposure == 0.0:
        return torch.zeros(
            (n_assets,),
            device=tensor_spec.device,
            dtype=tensor_spec.dtype,
        )
    return _allocate_equal_weight(
        n_assets=n_assets,
        params=params,
        tensor_spec=tensor_spec,
    )


def _build_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _random_simplex(
    *,
    n_assets: int,
    total: float,
    generator: torch.Generator | None,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    raw = torch.rand((n_assets,), generator=generator, device="cpu", dtype=torch.float32)
    weights = raw / raw.sum().clamp_min(1e-12)
    return weights.to(
        device=tensor_spec.device, dtype=tensor_spec.dtype
    ) * total


def _random_bounded_net(
    *,
    n_assets: int,
    gross_exposure: float,
    generator: torch.Generator | None,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if n_assets < 2:
        raise ConfigError(
            "allocation.spec.portfolio_style=long_short_bounded_net "
            "requires at least 2 assets"
        )
    num_long = n_assets // 2
    num_short = n_assets - num_long
    if num_long == 0 or num_short == 0:
        raise ConfigError(
            "allocation.spec.portfolio_style=long_short_bounded_net "
            "requires at least 2 assets"
        )
    perm = torch.randperm(n_assets, generator=generator, device="cpu")
    weights = torch.zeros((n_assets,), device="cpu", dtype=torch.float32)
    long_idx = perm[:num_long]
    short_idx = perm[num_long:]
    weights[long_idx] = _random_simplex(
        n_assets=num_long,
        total=gross_exposure * 0.5,
        generator=generator,
        tensor_spec=_TensorSpec(
            device=torch.device("cpu"), dtype=torch.float32
        ),
    )
    weights[short_idx] = -_random_simplex(
        n_assets=num_short,
        total=gross_exposure * 0.5,
        generator=generator,
        tensor_spec=_TensorSpec(
            device=torch.device("cpu"), dtype=torch.float32
        ),
    )
    return weights.to(
        device=tensor_spec.device, dtype=tensor_spec.dtype
    )


def _allocate_skfolio_risk_budgeting(
    *,
    pred: Mapping[str, Any],
    asset_names: Sequence[str],
    w_prev: torch.Tensor | None,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=skfolio_risk_budgeting currently supports "
            "only allocation.spec.portfolio_style=long_only"
        )
    scenarios = _extract_predictive_scenarios(pred)
    frame = _build_scenario_frame(scenarios, asset_names)
    model = _build_risk_budgeting_model(params=params, w_prev=w_prev)
    try:
        model.fit(frame)
    except Exception as exc:  # pragma: no cover - library failure path
        raise SimulationError(
            "skfolio RiskBudgeting allocation failed",
            context={"error": str(exc)},
        ) from exc
    weights = torch.as_tensor(
        model.weights_,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )
    return weights


def _extract_predictive_scenarios(pred: Mapping[str, Any]) -> torch.Tensor:
    samples = pred.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ConfigError(
            "allocation with skfolio_risk_budgeting requires pred['samples']"
        )
    if samples.ndim == 2:
        return samples.detach()
    if samples.ndim == 3:
        return samples[:, 0, :].detach()
    raise ConfigError(
        "pred['samples'] must have shape (num_samples, n_assets) or "
        "(num_samples, horizon, n_assets)"
    )


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


def _build_risk_budgeting_model(
    *,
    params: _AllocationParams,
    w_prev: torch.Tensor | None,
) -> Any:
    risk_measure_enum = importlib.import_module("skfolio").RiskMeasure
    risk_budgeting_cls = importlib.import_module(
        "skfolio.optimization"
    ).RiskBudgeting
    risk_measure = _resolve_risk_measure(
        params.skfolio.risk_measure, risk_measure_enum
    )
    previous_weights = None
    if params.skfolio.use_previous_weights and w_prev is not None:
        previous_weights = (
            w_prev.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
    min_weight = (
        0.0
        if params.skfolio.min_weight is None
        else params.skfolio.min_weight
    )
    max_weight = (
        1.0
        if params.skfolio.max_weight is None
        else params.skfolio.max_weight
    )
    return risk_budgeting_cls(
        risk_measure=risk_measure,
        min_weights=min_weight,
        max_weights=max_weight,
        transaction_costs=params.skfolio.transaction_costs,
        previous_weights=previous_weights,
        raise_on_failure=True,
    )


def _resolve_risk_measure(raw: str, enum_cls: Any) -> Any:
    name = raw.strip().upper()
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    valid = sorted(name for name in dir(enum_cls) if name.isupper())
    raise ConfigError(
        "allocation.spec.risk_measure must be one of "
        + ", ".join(valid)
    )


def _allocate_weights_stub(
    pred: Mapping[str, Any],
    alloc_spec: Mapping[str, Any],
    w_prev: torch.Tensor | None = None,
    asset_names: Sequence[str] = (),
) -> torch.Tensor:
    _ = (pred, w_prev, asset_names)
    n_assets = int(alloc_spec.get("n_assets", 1))
    return torch.full((n_assets,), 1.0 / max(n_assets, 1))


__all__ = [
    "_allocate_weights",
    "_allocate_weights_stub",
]
