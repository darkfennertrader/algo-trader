from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping

import pandas as pd
import torch

from algo_trader.domain import ConfigError, DataProcessingError
from .validation import validate_no_unknown_params
from .zscore import ZScorePreprocessor

_ALLOWED_PARAMS = {"k", "variance", "start_date", "end_date", "missing"}


@dataclass(frozen=True)
class PCAConfig:
    k: int | None
    variance: Decimal | None
    zscore_params: Mapping[str, str]


@dataclass(frozen=True)
class PCAResult:
    factors: pd.DataFrame
    loadings: pd.DataFrame
    eigenvalues: pd.DataFrame
    selected_k: int
    variance_target: Decimal | None


@dataclass(frozen=True)
class EigenDecomposition:
    values: torch.Tensor
    vectors: torch.Tensor
    explained: torch.Tensor
    cumulative: torch.Tensor


class PCAPreprocessor:
    def __init__(self, zscore: ZScorePreprocessor | None = None) -> None:
        self._zscore = zscore or ZScorePreprocessor()
        self._result: PCAResult | None = None

    def process(
        self, data: pd.DataFrame, *, params: Mapping[str, str]
    ) -> pd.DataFrame:
        config = _parse_config(params)
        standardized = self._zscore.process(data, params=config.zscore_params)
        result = _compute_pca(standardized, config)
        self._result = result
        return result.factors

    def result(self) -> PCAResult:
        if self._result is None:
            raise DataProcessingError(
                "PCA artifacts are not available before processing",
                context={"preprocessor": "pca"},
            )
        return self._result


def _parse_config(params: Mapping[str, str]) -> PCAConfig:
    validate_no_unknown_params(params, allowed=_ALLOWED_PARAMS)
    k = _parse_k(params.get("k"))
    variance = _parse_variance(params.get("variance"))
    if k is not None and variance is not None:
        raise ConfigError(
            "Provide either k or variance, not both",
            context={"k": str(k), "variance": str(variance)},
        )
    if k is None and variance is None:
        raise ConfigError("Either k or variance must be provided")
    zscore_params = {
        key: value
        for key, value in params.items()
        if key in {"start_date", "end_date", "missing"}
    }
    return PCAConfig(k=k, variance=variance, zscore_params=zscore_params)


def _parse_k(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "k must be an integer",
            context={"value": value},
        ) from exc
    if parsed <= 0:
        raise ConfigError(
            "k must be a positive integer",
            context={"value": value},
        )
    return parsed


def _parse_variance(value: str | None) -> Decimal | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ConfigError(
            "variance must be a number between 0 and 1",
            context={"value": value},
        ) from exc
    if parsed <= Decimal("0") or parsed > Decimal("1"):
        raise ConfigError(
            "variance must be between 0 and 1",
            context={"value": value},
        )
    return parsed


def _compute_pca(frame: pd.DataFrame, config: PCAConfig) -> PCAResult:
    _validate_frame(frame)
    tensor = _frame_to_tensor(frame)
    eigen = _eigendecomposition(tensor)
    selected_k = _select_k(config, eigen.cumulative, eigen.vectors.shape[1])
    return _build_pca_result(
        frame=frame,
        tensor=tensor,
        eigen=eigen,
        selected_k=selected_k,
        variance_target=config.variance,
    )


def _validate_frame(frame: pd.DataFrame) -> None:
    if frame.shape[0] < 2:
        raise DataProcessingError(
            "PCA requires at least two rows",
            context={"rows": str(len(frame))},
        )
    if frame.shape[1] < 1:
        raise DataProcessingError(
            "PCA requires at least one column",
            context={"columns": str(len(frame.columns))},
        )


def _frame_to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    values = frame.to_numpy(dtype=float, copy=True)
    return torch.as_tensor(values, dtype=torch.float64)


def _eigendecomposition(tensor: torch.Tensor) -> EigenDecomposition:
    cov = (tensor.T @ tensor) / (tensor.shape[0] - 1)
    values, vectors = _sort_eigenpairs(*torch.linalg.eigh(cov))  # pylint: disable=not-callable
    total_variance = float(torch.sum(values))
    if total_variance <= 0:
        raise DataProcessingError(
            "PCA requires positive total variance",
            context={"variance": str(total_variance)},
        )
    explained = values / total_variance
    cumulative = torch.cumsum(explained, dim=0)
    return EigenDecomposition(
        values=values,
        vectors=vectors,
        explained=explained,
        cumulative=cumulative,
    )


def _build_pca_result(
    *,
    frame: pd.DataFrame,
    tensor: torch.Tensor,
    eigen: EigenDecomposition,
    selected_k: int,
    variance_target: Decimal | None,
) -> PCAResult:
    factor_columns = _factor_columns(selected_k)
    loadings = eigen.vectors[:, :selected_k]
    factors = tensor @ loadings
    return PCAResult(
        factors=_frame_from_tensor(
            factors, index=frame.index, columns=factor_columns
        ),
        loadings=_frame_from_tensor(
            loadings, index=frame.columns, columns=factor_columns
        ),
        eigenvalues=_eigenvalues_frame(eigen),
        selected_k=selected_k,
        variance_target=variance_target,
    )


def _select_k(
    config: PCAConfig, cumulative: torch.Tensor, max_components: int
) -> int:
    if config.k is not None:
        if config.k > max_components:
            raise ConfigError(
                "k cannot exceed number of columns",
                context={
                    "k": str(config.k),
                    "columns": str(max_components),
                },
            )
        return config.k
    target = config.variance
    if target is None:
        raise ConfigError("variance is required when k is not provided")
    meets = torch.nonzero(
        cumulative >= float(target),
        as_tuple=False,
    )
    if meets.numel() == 0:
        return max_components
    return int(meets[0].item()) + 1


def _sort_eigenpairs(
    values: torch.Tensor, vectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.argsort(values, descending=True)
    return values[indices], vectors[:, indices]


def _frame_from_tensor(
    tensor: torch.Tensor, *, index: pd.Index, columns: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        tensor.cpu().numpy(),
        index=index,
        columns=columns,
    )


def _eigenvalues_frame(
    eigen: EigenDecomposition,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eigenvalue": eigen.values.cpu().numpy(),
            "explained_variance": eigen.explained.cpu().numpy(),
            "cumulative_variance": eigen.cumulative.cpu().numpy(),
        },
        index=_pc_index(len(eigen.values)),
    )


def _factor_columns(k: int) -> list[str]:
    return [f"factor_{index}" for index in range(1, k + 1)]


def _pc_index(count: int) -> list[str]:
    return [f"pc_{index}" for index in range(1, count + 1)]
