from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.factor.guide_l11 import FilteringState
from algo_trader.pipeline.stages.modeling.runtime_support import (
    RuntimeObservations,
    require_tensor_entry,
)

FX_CLASS_ID = 0
INDEX_CLASS_ID = 1
COMMODITY_CLASS_ID = 2
_STATE_COUNT = 4


def classify_asset_name(name: str) -> int:
    normalized = str(name).strip().upper()
    if _is_fx_name(normalized):
        return FX_CLASS_ID
    if normalized.startswith(("XAU", "XAG", "XPT", "XPD")):
        return COMMODITY_CLASS_ID
    return INDEX_CLASS_ID


def _is_fx_name(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and all(len(part) == 3 and part.isalpha() for part in parts)


def build_asset_class_ids(
    asset_names: tuple[str, ...], *, device: torch.device
) -> torch.LongTensor:
    tensor = torch.tensor(
        [classify_asset_name(name) for name in asset_names],
        device=device,
        dtype=torch.long,
    )
    return cast(torch.LongTensor, tensor)


def asset_class_mask(
    class_ids: torch.LongTensor, *, class_id: int, dtype: torch.dtype
) -> torch.Tensor:
    return (class_ids == class_id).to(dtype=dtype)


@dataclass(frozen=True)
class RuntimeAssetMetadata:
    asset_names: tuple[str, ...]
    class_ids: torch.LongTensor

    @property
    def fx_mask(self) -> torch.BoolTensor:
        return cast(torch.BoolTensor, self.class_ids == FX_CLASS_ID)

    @property
    def index_mask(self) -> torch.BoolTensor:
        return cast(torch.BoolTensor, self.class_ids == INDEX_CLASS_ID)

    @property
    def commodity_mask(self) -> torch.BoolTensor:
        return cast(torch.BoolTensor, self.class_ids == COMMODITY_CLASS_ID)


@dataclass(frozen=True)
class FactorCountConfig:
    global_factor_count: int = 1
    fx_broad_factor_count: int = 1
    fx_cross_factor_count: int = 1
    index_factor_count: int = 1
    commodity_factor_count: int = 1


@dataclass(frozen=True)
class PanelDimensions:
    T: int
    A: int
    F: int
    G: int


@dataclass(frozen=True)
class V3L1UnifiedRuntimeBatch:
    X_asset: torch.Tensor
    X_global: torch.Tensor
    observations: RuntimeObservations
    assets: RuntimeAssetMetadata
    filtering_state: FilteringState | None = None


@dataclass(frozen=True)
class MeanTensorMeans:
    alpha: torch.Tensor
    sigma_idio: torch.Tensor
    w: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class CovarianceLoadings:
    B_global: torch.Tensor
    B_fx_broad: torch.Tensor
    B_fx_cross: torch.Tensor
    B_index: torch.Tensor
    B_commodity: torch.Tensor


@dataclass(frozen=True)
class StructuralTensorMeans:
    mean: MeanTensorMeans
    loadings: CovarianceLoadings


@dataclass(frozen=True)
class RegimePosteriorMeans:
    s_u_fx_broad_mean: torch.Tensor
    s_u_fx_cross_mean: torch.Tensor
    s_u_index_mean: torch.Tensor
    s_u_commodity_mean: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.s_u_fx_broad_mean,
                self.s_u_fx_cross_mean,
                self.s_u_index_mean,
                self.s_u_commodity_mean,
            ]
        )


@dataclass(frozen=True)
class StructuralPosteriorMeans:
    tensors: StructuralTensorMeans
    regime: RegimePosteriorMeans

    @property
    def alpha(self) -> torch.Tensor:
        return self.tensors.mean.alpha

    @property
    def sigma_idio(self) -> torch.Tensor:
        return self.tensors.mean.sigma_idio

    @property
    def w(self) -> torch.Tensor:
        return self.tensors.mean.w

    @property
    def beta(self) -> torch.Tensor:
        return self.tensors.mean.beta

    @property
    def B_global(self) -> torch.Tensor:
        return self.tensors.loadings.B_global

    @property
    def B_fx_broad(self) -> torch.Tensor:
        return self.tensors.loadings.B_fx_broad

    @property
    def B_fx_cross(self) -> torch.Tensor:
        return self.tensors.loadings.B_fx_cross

    @property
    def B_index(self) -> torch.Tensor:
        return self.tensors.loadings.B_index

    @property
    def B_commodity(self) -> torch.Tensor:
        return self.tensors.loadings.B_commodity

    @property
    def s_u_fx_broad_mean(self) -> torch.Tensor:
        return self.regime.s_u_fx_broad_mean

    @property
    def s_u_fx_cross_mean(self) -> torch.Tensor:
        return self.regime.s_u_fx_cross_mean

    @property
    def s_u_index_mean(self) -> torch.Tensor:
        return self.regime.s_u_index_mean

    @property
    def s_u_commodity_mean(self) -> torch.Tensor:
        return self.regime.s_u_commodity_mean

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "alpha": self.alpha.detach(),
            "sigma_idio": self.sigma_idio.detach(),
            "w": self.w.detach(),
            "beta": self.beta.detach(),
            "B_global": self.B_global.detach(),
            "B_fx_broad": self.B_fx_broad.detach(),
            "B_fx_cross": self.B_fx_cross.detach(),
            "B_index": self.B_index.detach(),
            "B_commodity": self.B_commodity.detach(),
            "s_u_fx_broad_mean": self.s_u_fx_broad_mean.detach(),
            "s_u_fx_cross_mean": self.s_u_fx_cross_mean.detach(),
            "s_u_index_mean": self.s_u_index_mean.detach(),
            "s_u_commodity_mean": self.s_u_commodity_mean.detach(),
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeans":
        tensor_keys = (
            "alpha",
            "sigma_idio",
            "w",
            "beta",
            "B_global",
            "B_fx_broad",
            "B_fx_cross",
            "B_index",
            "B_commodity",
            "s_u_fx_broad_mean",
            "s_u_fx_cross_mean",
            "s_u_index_mean",
            "s_u_commodity_mean",
        )
        values = {
            key: _require_tensor(payload, key).detach() for key in tensor_keys
        }
        return cls(
            tensors=StructuralTensorMeans(
                mean=MeanTensorMeans(
                    alpha=values["alpha"],
                    sigma_idio=values["sigma_idio"],
                    w=values["w"],
                    beta=values["beta"],
                ),
                loadings=CovarianceLoadings(
                    B_global=values["B_global"],
                    B_fx_broad=values["B_fx_broad"],
                    B_fx_cross=values["B_fx_cross"],
                    B_index=values["B_index"],
                    B_commodity=values["B_commodity"],
                ),
            ),
            regime=RegimePosteriorMeans(
                s_u_fx_broad_mean=values["s_u_fx_broad_mean"],
                s_u_fx_cross_mean=values["s_u_fx_cross_mean"],
                s_u_index_mean=values["s_u_index_mean"],
                s_u_commodity_mean=values["s_u_commodity_mean"],
            ),
        )


def _require_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    return require_tensor_entry(payload, key)


def coerce_four_state_tensor(
    value: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(_STATE_COUNT)
    if tensor.numel() != _STATE_COUNT:
        raise ConfigError("v3_l1_unified filtering state expects 1 or 4 latent values")
    return tensor


def build_factor_count_config(
    values: Mapping[str, Any], *, base: FactorCountConfig
) -> FactorCountConfig:
    return FactorCountConfig(
        global_factor_count=int(
            values.get("global_factor_count", base.global_factor_count)
        ),
        fx_broad_factor_count=int(
            values.get("fx_broad_factor_count", base.fx_broad_factor_count)
        ),
        fx_cross_factor_count=int(
            values.get("fx_cross_factor_count", base.fx_cross_factor_count)
        ),
        index_factor_count=int(
            values.get("index_factor_count", base.index_factor_count)
        ),
        commodity_factor_count=int(
            values.get("commodity_factor_count", base.commodity_factor_count)
        ),
    )


def build_panel_dimensions(batch: V3L1UnifiedRuntimeBatch) -> PanelDimensions:
    return PanelDimensions(
        T=int(batch.X_asset.shape[0]),
        A=int(batch.X_asset.shape[1]),
        F=int(batch.X_asset.shape[2]),
        G=int(batch.X_global.shape[1]),
    )
