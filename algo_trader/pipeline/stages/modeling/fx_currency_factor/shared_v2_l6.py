from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState

from .guide_v2_l2 import _resolve_anchor_currency, _resolve_currency_names


@dataclass(frozen=True)
class RuntimeObservations:
    y_input: torch.Tensor
    y_obs: torch.Tensor | None
    time_mask: torch.BoolTensor | None
    obs_scale: float | None


@dataclass(frozen=True)
class RuntimeCurrencyMetadata:
    exposure_matrix: torch.Tensor
    currency_names: tuple[str, ...]
    anchor_currency: str
    pair_components: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class V2L6RuntimeBatch:
    X_asset: torch.Tensor
    X_global: torch.Tensor
    observations: RuntimeObservations
    currency: RuntimeCurrencyMetadata
    filtering_state: FilteringState | None = None

    @property
    def y_input(self) -> torch.Tensor:
        return self.observations.y_input

    @property
    def y_obs(self) -> torch.Tensor | None:
        return self.observations.y_obs

    @property
    def time_mask(self) -> torch.BoolTensor | None:
        return self.observations.time_mask

    @property
    def obs_scale(self) -> float | None:
        return self.observations.obs_scale

    @property
    def exposure_matrix(self) -> torch.Tensor:
        return self.currency.exposure_matrix

    @property
    def currency_names(self) -> tuple[str, ...]:
        return self.currency.currency_names

    @property
    def anchor_currency(self) -> str:
        return self.currency.anchor_currency

    @property
    def pair_components(self) -> tuple[tuple[str, str], ...]:
        return self.currency.pair_components

    @property
    def currency_count(self) -> int:
        return int(self.exposure_matrix.shape[1])


@dataclass(frozen=True)
class StructuralTensorMeans:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    gamma_currency: torch.Tensor
    B_currency_broad: torch.Tensor
    B_currency_cross: torch.Tensor


@dataclass(frozen=True)
class RegimePosteriorMeans:
    s_u_broad_mean: torch.Tensor
    s_u_cross_mean: torch.Tensor


@dataclass(frozen=True)
class CurrencyPosteriorMetadata:
    currency_names: tuple[str, ...]
    anchor_currency: str


@dataclass(frozen=True)
class StructuralPosteriorMeans:
    tensors: StructuralTensorMeans
    regime: RegimePosteriorMeans
    metadata: CurrencyPosteriorMetadata

    @property
    def alpha(self) -> torch.Tensor:
        return self.tensors.alpha

    @property
    def sigma_idio(self) -> torch.Tensor:
        return self.tensors.sigma_idio

    @property
    def w(self) -> torch.Tensor:
        return self.tensors.w

    @property
    def gamma_currency(self) -> torch.Tensor:
        return self.tensors.gamma_currency

    @property
    def B_currency_broad(self) -> torch.Tensor:
        return self.tensors.B_currency_broad

    @property
    def B_currency_cross(self) -> torch.Tensor:
        return self.tensors.B_currency_cross

    @property
    def s_u_broad_mean(self) -> torch.Tensor:
        return self.regime.s_u_broad_mean

    @property
    def s_u_cross_mean(self) -> torch.Tensor:
        return self.regime.s_u_cross_mean

    @property
    def currency_names(self) -> tuple[str, ...]:
        return self.metadata.currency_names

    @property
    def anchor_currency(self) -> str:
        return self.metadata.anchor_currency

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "gamma_currency": self.gamma_currency.detach(),
            "B_currency_broad": self.B_currency_broad.detach(),
            "B_currency_cross": self.B_currency_cross.detach(),
            "s_u_broad_mean": self.s_u_broad_mean.detach(),
            "s_u_cross_mean": self.s_u_cross_mean.detach(),
            "currency_names": tuple(self.currency_names),
            "anchor_currency": self.anchor_currency,
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeans":
        tensor_keys = (
            "alpha",
            "sigma_idio",
            "w",
            "gamma_currency",
            "B_currency_broad",
            "B_currency_cross",
            "s_u_broad_mean",
            "s_u_cross_mean",
        )
        values: dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ConfigError(
                    "structural_posterior_means must include tensor entries",
                    context={"field": key},
                )
            values[key] = value.detach()
        currency_names = _resolve_currency_names(payload.get("currency_names"))
        anchor_currency = _resolve_anchor_currency(
            payload.get("anchor_currency"),
            currency_names=currency_names,
        )
        return cls(
            tensors=StructuralTensorMeans(
                alpha=values["alpha"],
                sigma_idio=values["sigma_idio"],
                w=values["w"],
                gamma_currency=values["gamma_currency"],
                B_currency_broad=values["B_currency_broad"],
                B_currency_cross=values["B_currency_cross"],
            ),
            regime=RegimePosteriorMeans(
                s_u_broad_mean=values["s_u_broad_mean"],
                s_u_cross_mean=values["s_u_cross_mean"],
            ),
            metadata=CurrencyPosteriorMetadata(
                currency_names=currency_names,
                anchor_currency=anchor_currency,
            ),
        )


def coerce_two_state_tensor(
    value: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(2)
    if tensor.numel() != 2:
        raise ConfigError("v2_l6 filtering state expects 1 or 2 latent values")
    return tensor
