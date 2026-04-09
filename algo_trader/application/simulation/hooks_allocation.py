from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.domain.simulation import (
    AllocationRequest,
    AllocationResult,
    PredictionPacket,
)
from .allocation_common import (
    VALID_ALLOCATION_FAMILIES,
    VALID_PORTFOLIO_STYLES,
    VALID_SKFOLIO_DISTANCE_ESTIMATORS,
    optional_float_value,
)
from .skfolio_allocators import (
    SkfolioRuntimeConfig,
    TensorTarget,
    allocate_herc,
    allocate_risk_budgeting,
    allocate_schur,
)

AllocatorMethod = str
PortfolioStyle = str
_DEFAULT_RANDOM_BASELINE_SEED = 17


@dataclass(frozen=True)
class _SkfolioOptions:
    risk_measure: str
    distance_estimator: str
    gamma: float
    use_previous_weights: bool
    transaction_costs: float


@dataclass(frozen=True)
class _AllocationParams:
    method: AllocatorMethod
    portfolio_style: PortfolioStyle
    gross_exposure: float
    random_seed: int | None
    min_weight: float | None
    max_weight: float | None
    skfolio: _SkfolioOptions


@dataclass(frozen=True)
class _TensorSpec:
    device: torch.device
    dtype: torch.dtype


def _allocate_weights(
    request: AllocationRequest,
) -> AllocationResult:
    params = _parse_allocation_params(request.allocation_spec)
    prediction = request.prediction
    n_assets = _resolve_n_assets(prediction, request.allocation_spec)
    tensor_spec = _resolve_tensor_spec(prediction)
    if params.method == "long_only":
        weights = _allocate_long_only(
            prediction=prediction,
            params=params,
            tensor_spec=tensor_spec,
        )
    elif params.method == "equal_weight":
        weights = _allocate_equal_weight(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    elif params.method == "random":
        weights = _allocate_random(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    elif params.method == "de_risked":
        weights = _allocate_de_risked(
            n_assets=n_assets,
            params=params,
            tensor_spec=tensor_spec,
        )
    elif params.method == "herc":
        weights = _allocate_herc(
            prediction=prediction,
            previous_weights=request.previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        )
    elif params.method == "schur":
        weights = _allocate_schur(
            prediction=prediction,
            previous_weights=request.previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        )
    else:
        weights = _allocate_skfolio_risk_budgeting(
            prediction=prediction,
            previous_weights=request.previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        )
    return _build_allocation_result(
        prediction=prediction,
        previous_weights=request.previous_weights,
        weights=weights,
    )


def _parse_allocation_params(
    alloc_spec: Mapping[str, Any],
) -> _AllocationParams:
    method = _resolve_allocation_method(alloc_spec)
    if method not in VALID_ALLOCATION_FAMILIES:
        raise ConfigError(
            "allocation family must be long_only, equal_weight, random, "
            "de_risked, herc, schur, or skfolio_risk_budgeting"
        )
    portfolio_style = _resolve_portfolio_style(alloc_spec, method)
    if portfolio_style not in VALID_PORTFOLIO_STYLES:
        raise ConfigError(
            "allocation.spec.portfolio_style must be long_only, "
            "long_short_bounded_net, or factor_neutral_long_short"
        )
    gross_exposure = _resolve_gross_exposure(alloc_spec, method)
    if gross_exposure < 0.0:
        raise ConfigError("allocation.spec.gross_exposure must be >= 0")
    if gross_exposure == 0.0 and method not in {"de_risked", "long_only"}:
        raise ConfigError(
            "allocation.spec.gross_exposure must be > 0 unless "
            "allocation family is de_risked"
        )
    random_seed = _resolve_random_seed(alloc_spec, method)
    risk_measure = _resolve_risk_measure_name(alloc_spec, method)
    distance_estimator = _resolve_distance_estimator_name(alloc_spec, method)
    gamma = _resolve_gamma(alloc_spec, method)
    use_previous_weights = _resolve_use_previous_weights(alloc_spec, method)
    transaction_costs = float(alloc_spec.get("transaction_costs", 0.0))
    min_weight = _optional_float(alloc_spec.get("min_weight"), "min_weight")
    max_weight = _optional_float(alloc_spec.get("max_weight"), "max_weight")
    return _AllocationParams(
        method=cast(AllocatorMethod, method),
        portfolio_style=cast(PortfolioStyle, portfolio_style),
        gross_exposure=gross_exposure,
        random_seed=random_seed,
        min_weight=min_weight,
        max_weight=max_weight,
        skfolio=_SkfolioOptions(
            risk_measure=risk_measure,
            distance_estimator=distance_estimator,
            gamma=gamma,
            use_previous_weights=use_previous_weights,
            transaction_costs=transaction_costs,
        ),
    )


def _resolve_allocation_method(alloc_spec: Mapping[str, Any]) -> str:
    raw = alloc_spec.get("family", alloc_spec.get("method", "equal_weight"))
    return str(raw).strip().lower()


def _resolve_portfolio_style(
    alloc_spec: Mapping[str, Any], method: str
) -> str:
    if method == "long_only":
        raw = str(alloc_spec.get("portfolio_style", "long_only")).strip().lower()
        if raw != "long_only":
            raise ConfigError(
                "allocation.spec.family=long_only supports only "
                "allocation.spec.portfolio_style=long_only"
            )
        return raw
    return str(alloc_spec.get("portfolio_style", "long_only")).strip().lower()


def _optional_int(raw: Any, field: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"allocation.spec.{field} must be an integer")
    return int(raw)


def _resolve_random_seed(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> int | None:
    random_seed = _optional_int(alloc_spec.get("random_seed"), "random_seed")
    if method != "random" or random_seed is not None:
        return random_seed
    return _DEFAULT_RANDOM_BASELINE_SEED


def _optional_float(raw: Any, field: str) -> float | None:
    return optional_float_value(raw, location=f"allocation.spec.{field}")


def _resolve_gross_exposure(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> float:
    if method == "long_only":
        if "gross_exposure" in alloc_spec:
            raise ConfigError(
                "allocation.spec.gross_exposure is not supported for "
                "allocation.spec.family=long_only"
            )
        return 1.0
    gross_default = 0.10 if method == "de_risked" else 1.0
    return float(alloc_spec.get("gross_exposure", gross_default))


def _resolve_use_previous_weights(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> bool:
    if method == "long_only" and "use_previous_weights" in alloc_spec:
        raise ConfigError(
            "allocation.spec.use_previous_weights is not supported for "
            "allocation.spec.family=long_only"
        )
    return bool(alloc_spec.get("use_previous_weights", False))


def _resolve_risk_measure_name(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> str:
    default = "variance" if method == "herc" else "cvar"
    return str(alloc_spec.get("risk_measure", default)).strip().lower()


def _resolve_distance_estimator_name(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> str:
    if method not in {"herc", "schur"}:
        return "pearson"
    raw = str(alloc_spec.get("distance_estimator", "pearson")).strip().lower()
    if raw not in VALID_SKFOLIO_DISTANCE_ESTIMATORS:
        raise ConfigError(
            "allocation.spec.distance_estimator must be pearson or "
            "mutual_information"
        )
    return raw


def _resolve_gamma(
    alloc_spec: Mapping[str, Any],
    method: str,
) -> float:
    if method != "schur":
        return 0.5
    raw = _optional_float(alloc_spec.get("gamma"), "gamma")
    if raw is None:
        return 0.5
    if raw < 0.0 or raw > 1.0:
        raise ConfigError("allocation.spec.gamma must be between 0 and 1")
    return raw


def _resolve_n_assets(
    prediction: PredictionPacket,
    alloc_spec: Mapping[str, Any],
) -> int:
    if prediction.asset_names:
        return len(prediction.asset_names)
    return int(alloc_spec.get("n_assets", 1))


def _resolve_tensor_spec(prediction: PredictionPacket) -> _TensorSpec:
    return _TensorSpec(
        device=prediction.mu.device,
        dtype=prediction.mu.dtype,
    )


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


def _allocate_long_only(
    *,
    prediction: PredictionPacket,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.family=long_only supports only "
            "allocation.spec.portfolio_style=long_only"
        )
    if params.gross_exposure == 0.0:
        return torch.zeros(
            prediction.mu.shape,
            device=tensor_spec.device,
            dtype=tensor_spec.dtype,
        )
    scores = _long_only_scores(prediction)
    min_weight = 0.0 if params.min_weight is None else params.min_weight
    max_weight = params.gross_exposure if params.max_weight is None else params.max_weight
    return _capped_long_only_weights(
        scores=scores,
        gross_exposure=params.gross_exposure,
        min_weight=min_weight,
        max_weight=max_weight,
        tensor_spec=tensor_spec,
    )


def _long_only_scores(prediction: PredictionPacket) -> torch.Tensor:
    variance = prediction.covariance.diag().clamp_min(1e-12)
    risk = variance.sqrt()
    scores = torch.relu(prediction.mu / risk)
    if torch.count_nonzero(scores) == 0:
        return torch.ones_like(prediction.mu)
    return scores


def _capped_long_only_weights(
    *,
    scores: torch.Tensor,
    gross_exposure: float,
    min_weight: float,
    max_weight: float,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    n_assets = int(scores.shape[0])
    if min_weight < 0.0:
        raise ConfigError("allocation.spec.min_weight must be >= 0")
    if max_weight <= 0.0:
        raise ConfigError("allocation.spec.max_weight must be > 0")
    if min_weight > max_weight:
        raise ConfigError(
            "allocation.spec.min_weight must be <= allocation.spec.max_weight"
        )
    if n_assets * min_weight > gross_exposure + 1e-12:
        raise ConfigError(
            "allocation.spec.min_weight is too large for the long_only gross_exposure"
        )
    if n_assets * max_weight < gross_exposure - 1e-12:
        raise ConfigError(
            "allocation.spec.max_weight is too small for the long_only gross_exposure"
        )
    weights = torch.full(
        (n_assets,),
        min_weight,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )
    remaining = gross_exposure - float(n_assets * min_weight)
    if remaining <= 1e-12:
        return weights
    capacities = torch.full(
        (n_assets,),
        max_weight - min_weight,
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )
    active = capacities > 1e-12
    working_scores = scores.to(device=tensor_spec.device, dtype=tensor_spec.dtype)
    while remaining > 1e-12 and bool(torch.any(active)):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        active_scores = working_scores[active_idx]
        proposal = _long_only_proposal(
            active_scores=active_scores,
            remaining=remaining,
            count=int(active_idx.numel()),
            tensor_spec=tensor_spec,
        )
        allocation = torch.minimum(proposal, capacities[active_idx])
        weights[active_idx] = weights[active_idx] + allocation
        capacities[active_idx] = capacities[active_idx] - allocation
        remaining -= float(allocation.sum().item())
        active = capacities > 1e-12
    return weights


def _long_only_proposal(
    *,
    active_scores: torch.Tensor,
    remaining: float,
    count: int,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    score_sum = float(active_scores.sum().item())
    if score_sum <= 1e-12:
        return torch.full(
            (count,),
            remaining / count,
            device=tensor_spec.device,
            dtype=tensor_spec.dtype,
        )
    return active_scores * (remaining / score_sum)


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


def _allocate_herc(
    *,
    prediction: PredictionPacket,
    previous_weights: torch.Tensor | None,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=herc currently supports only "
            "allocation.spec.portfolio_style=long_only"
        )
    return allocate_herc(
        prediction=prediction,
        runtime_config=_build_skfolio_runtime_config(
            previous_weights=previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        ),
        distance_estimator=params.skfolio.distance_estimator,
    )


def _allocate_schur(
    *,
    prediction: PredictionPacket,
    previous_weights: torch.Tensor | None,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=schur currently supports only "
            "allocation.spec.portfolio_style=long_only"
        )
    return allocate_schur(
        prediction=prediction,
        runtime_config=_build_skfolio_runtime_config(
            previous_weights=previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        ),
        gamma=params.skfolio.gamma,
        distance_estimator=params.skfolio.distance_estimator,
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
    prediction: PredictionPacket,
    previous_weights: torch.Tensor | None,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> torch.Tensor:
    if params.portfolio_style != "long_only":
        raise ConfigError(
            "allocation.spec.method=skfolio_risk_budgeting currently supports "
            "only allocation.spec.portfolio_style=long_only"
        )
    return allocate_risk_budgeting(
        prediction=prediction,
        runtime_config=_build_skfolio_runtime_config(
            previous_weights=previous_weights,
            params=params,
            tensor_spec=tensor_spec,
        ),
    )


def _build_skfolio_runtime_config(
    *,
    previous_weights: torch.Tensor | None,
    params: _AllocationParams,
    tensor_spec: _TensorSpec,
) -> SkfolioRuntimeConfig:
    return SkfolioRuntimeConfig(
        risk_measure=params.skfolio.risk_measure,
        min_weight=params.min_weight,
        max_weight=params.max_weight,
        transaction_costs=params.skfolio.transaction_costs,
        previous_weights=previous_weights,
        use_previous_weights=params.skfolio.use_previous_weights,
        tensor_target=TensorTarget(
            device=tensor_spec.device,
            dtype=tensor_spec.dtype,
        ),
    )


def _allocate_weights_stub(
    request: AllocationRequest,
) -> AllocationResult:
    n_assets = int(request.allocation_spec.get("n_assets", 1))
    tensor_spec = _resolve_tensor_spec(request.prediction)
    weights = torch.full(
        (n_assets,),
        1.0 / max(n_assets, 1),
        device=tensor_spec.device,
        dtype=tensor_spec.dtype,
    )
    return _build_allocation_result(
        prediction=request.prediction,
        previous_weights=request.previous_weights,
        weights=weights,
    )


def _build_allocation_result(
    *,
    prediction: PredictionPacket,
    previous_weights: torch.Tensor | None,
    weights: torch.Tensor,
) -> AllocationResult:
    expected_return = _expected_return(weights, prediction.mu)
    expected_risk = _expected_risk(weights, prediction.covariance)
    turnover = _turnover(weights, previous_weights)
    return AllocationResult(
        rebalance_index=prediction.rebalance_index,
        rebalance_timestamp=prediction.rebalance_timestamp,
        asset_names=prediction.asset_names,
        weights=weights,
        expected_return=expected_return,
        expected_risk=expected_risk,
        turnover=turnover,
    )


def _expected_return(
    weights: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    weights_cpu = _as_cpu_vector(weights)
    mu_cpu = _as_cpu_vector(mu)
    return torch.dot(weights_cpu, mu_cpu)


def _expected_risk(
    weights: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    weights_cpu = _as_cpu_vector(weights)
    covariance_cpu = covariance.detach().to(device="cpu", dtype=torch.float64)
    risk = weights_cpu.matmul(covariance_cpu).matmul(weights_cpu)
    return risk.clamp_min(0.0).sqrt()


def _turnover(
    weights: torch.Tensor,
    previous_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if previous_weights is None:
        return None
    weights_cpu = _as_cpu_vector(weights)
    previous_weights_cpu = _as_cpu_vector(previous_weights)
    return torch.abs(weights_cpu - previous_weights_cpu).sum()


def _as_cpu_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)


__all__ = [
    "_allocate_weights",
    "_allocate_weights_stub",
]
