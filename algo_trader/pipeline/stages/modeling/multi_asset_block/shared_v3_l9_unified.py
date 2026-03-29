from __future__ import annotations
# pylint: disable=duplicate-code

import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.modeling.runtime_support import require_tensor_entry

from .shared_v3_l1_unified import (
    CovarianceLoadings,
    MeanTensorMeans,
    RuntimeAssetMetadata,
    StructuralTensorMeans,
)

_STATE_COUNT = 6
_US_EU_REGION_STATE_INDEX = 3
_EU_CORE_STATE_INDEX = 4
_COMMODITY_STATE_INDEX = 5
_STATE_SITE_NAMES = (
    "h_fx_broad",
    "h_fx_cross",
    "h_index",
    "h_index_region_us_eu",
    "h_index_region_eu_core_vs_uk_ch",
    "h_commodity",
)


def v3_l9_state_count() -> int:
    return _STATE_COUNT


def v3_l9_us_eu_region_state_index() -> int:
    return _US_EU_REGION_STATE_INDEX


def v3_l9_eu_core_state_index() -> int:
    return _EU_CORE_STATE_INDEX


def v3_l9_commodity_state_index() -> int:
    return _COMMODITY_STATE_INDEX


def v3_l9_state_site_names() -> tuple[str, ...]:
    return _STATE_SITE_NAMES


def coerce_v3_l9_state_tensor(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(_STATE_COUNT)
    if tensor.numel() == _STATE_COUNT:
        return tensor
    if tensor.numel() == _STATE_COUNT - 1:
        prefix = tensor[:_US_EU_REGION_STATE_INDEX + 1]
        suffix = tensor[_US_EU_REGION_STATE_INDEX + 1 :]
        pad = torch.zeros(1, device=device, dtype=dtype)
        return torch.cat([prefix, pad, suffix], dim=0)
    if tensor.numel() == _STATE_COUNT - 2:
        prefix = tensor[:_US_EU_REGION_STATE_INDEX]
        suffix = tensor[_US_EU_REGION_STATE_INDEX:]
        pad = torch.zeros(2, device=device, dtype=dtype)
        return torch.cat([prefix, pad, suffix], dim=0)
    raise ConfigError("v3_l9_unified filtering state expects 1, 4, 5, or 6 latent values")


def build_dynamic_us_europe_region_block(
    *,
    assets: RuntimeAssetMetadata,
    region_state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    exposure = _build_us_europe_region_exposure(
        asset_names=assets.asset_names,
        class_ids=assets.class_ids,
        dtype=dtype,
    )
    return _build_region_block(exposure=exposure, region_state=region_state)


def build_dynamic_europe_core_region_block(
    *,
    assets: RuntimeAssetMetadata,
    region_state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    reference = _build_us_europe_region_exposure(
        asset_names=assets.asset_names,
        class_ids=assets.class_ids,
        dtype=dtype,
    )
    exposure = _build_europe_core_region_exposure(
        asset_names=assets.asset_names,
        class_ids=assets.class_ids,
        dtype=dtype,
        reference=reference,
    )
    return _build_region_block(exposure=exposure, region_state=region_state)


def _build_region_block(
    *, exposure: torch.Tensor, region_state: torch.Tensor
) -> torch.Tensor:
    if region_state.ndim == 0:
        return exposure * region_state.reshape(1, 1)
    if region_state.ndim in (1, 2):
        scale = region_state.reshape(-1, 1, 1)
        return exposure.unsqueeze(0) * scale
    raise ConfigError("v3_l9_unified region state must be rank 0, 1, or 2")


def _build_us_europe_region_exposure(
    *,
    asset_names: tuple[str, ...],
    class_ids: torch.LongTensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = _build_index_weights(
        asset_names=asset_names,
        class_ids=class_ids,
        dtype=dtype,
        bucket_weights={
            "US": 1.0,
            "DE": -1.0,
            "ES": -1.0,
            "EU": -1.0,
            "FR": -1.0,
            "NL": -1.0,
            "GB": -1.0,
        },
    )
    return _center_normalize_exposure(weights)


def _build_europe_core_region_exposure(
    *,
    asset_names: tuple[str, ...],
    class_ids: torch.LongTensor,
    dtype: torch.dtype,
    reference: torch.Tensor,
) -> torch.Tensor:
    weights = _build_index_weights(
        asset_names=asset_names,
        class_ids=class_ids,
        dtype=dtype,
        bucket_weights={
            "DE": 1.0,
            "ES": 1.0,
            "EU": 1.0,
            "FR": 1.0,
            "NL": 1.0,
            "GB": -1.0,
            "CH": -1.0,
        },
    )
    return _center_normalize_exposure(weights, reference=reference.squeeze(-1))


def _build_index_weights(
    *,
    asset_names: tuple[str, ...],
    class_ids: torch.LongTensor,
    dtype: torch.dtype,
    bucket_weights: Mapping[str, float],
) -> torch.Tensor:
    weights = torch.zeros((len(asset_names),), device=class_ids.device, dtype=dtype)
    for index, (name, class_id) in enumerate(
        zip(asset_names, class_ids.tolist(), strict=True)
    ):
        if int(class_id) != 1:
            continue
        code = _index_code(name)
        if code is not None and code in bucket_weights:
            weights[index] = bucket_weights[code]
    return weights


def _index_code(name: str) -> str | None:
    normalized = str(name).strip().upper()
    match = re.match(r"^IB([A-Z]+)\d", normalized)
    if match is None:
        return None
    code = match.group(1)
    if code.startswith("US"):
        return "US"
    return code


def _center_normalize_exposure(
    weights: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    active = weights != 0
    if not bool(active.any()):
        return weights.unsqueeze(-1)
    centered = weights.clone()
    centered[active] = centered[active] - centered[active].mean()
    if reference is not None and float(reference.norm()) > 0.0:
        denominator = reference.square().sum().clamp_min(
            torch.finfo(reference.dtype).eps
        )
        projection = (centered * reference).sum() / denominator
        centered = centered - projection * reference
    norm = centered.norm()
    if float(norm) <= 0.0:
        return torch.zeros_like(weights).unsqueeze(-1)
    return (centered / norm).unsqueeze(-1)


@dataclass(frozen=True)
class RegimePosteriorMeansV3L9:
    s_u_fx_broad_mean: torch.Tensor
    s_u_fx_cross_mean: torch.Tensor
    s_u_index_mean: torch.Tensor
    s_u_index_region_us_eu_mean: torch.Tensor
    s_u_index_region_eu_core_vs_uk_ch_mean: torch.Tensor
    s_u_commodity_mean: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.s_u_fx_broad_mean,
                self.s_u_fx_cross_mean,
                self.s_u_index_mean,
                self.s_u_index_region_us_eu_mean,
                self.s_u_index_region_eu_core_vs_uk_ch_mean,
                self.s_u_commodity_mean,
            ]
        )


@dataclass(frozen=True)
class StructuralPosteriorMeansV3L9:
    tensors: StructuralTensorMeans
    regime: RegimePosteriorMeansV3L9

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
    def B_index_static(self) -> torch.Tensor:
        return self.tensors.loadings.B_index_static

    @property
    def index_group_scale(self) -> torch.Tensor:
        return self.tensors.loadings.index_group_scale

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
    def s_u_index_region_us_eu_mean(self) -> torch.Tensor:
        return self.regime.s_u_index_region_us_eu_mean

    @property
    def s_u_index_region_eu_core_vs_uk_ch_mean(self) -> torch.Tensor:
        return self.regime.s_u_index_region_eu_core_vs_uk_ch_mean

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
            "B_index_static": self.B_index_static.detach(),
            "index_group_scale": self.index_group_scale.detach(),
            "B_commodity": self.B_commodity.detach(),
            "s_u_fx_broad_mean": self.s_u_fx_broad_mean.detach(),
            "s_u_fx_cross_mean": self.s_u_fx_cross_mean.detach(),
            "s_u_index_mean": self.s_u_index_mean.detach(),
            "s_u_index_region_us_eu_mean": self.s_u_index_region_us_eu_mean.detach(),
            "s_u_index_region_eu_core_vs_uk_ch_mean": (
                self.s_u_index_region_eu_core_vs_uk_ch_mean.detach()
            ),
            "s_u_commodity_mean": self.s_u_commodity_mean.detach(),
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "StructuralPosteriorMeansV3L9":
        tensor_keys = (
            "alpha",
            "sigma_idio",
            "w",
            "beta",
            "B_global",
            "B_fx_broad",
            "B_fx_cross",
            "B_index",
            "B_index_static",
            "index_group_scale",
            "B_commodity",
            "s_u_fx_broad_mean",
            "s_u_fx_cross_mean",
            "s_u_index_mean",
            "s_u_index_region_us_eu_mean",
            "s_u_index_region_eu_core_vs_uk_ch_mean",
            "s_u_commodity_mean",
        )
        values = {key: _require_tensor(payload, key).detach() for key in tensor_keys}
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
                    B_index_static=values["B_index_static"],
                    index_group_scale=values["index_group_scale"],
                    B_commodity=values["B_commodity"],
                ),
            ),
            regime=RegimePosteriorMeansV3L9(
                s_u_fx_broad_mean=values["s_u_fx_broad_mean"],
                s_u_fx_cross_mean=values["s_u_fx_cross_mean"],
                s_u_index_mean=values["s_u_index_mean"],
                s_u_index_region_us_eu_mean=values["s_u_index_region_us_eu_mean"],
                s_u_index_region_eu_core_vs_uk_ch_mean=values[
                    "s_u_index_region_eu_core_vs_uk_ch_mean"
                ],
                s_u_commodity_mean=values["s_u_commodity_mean"],
            ),
        )


def _require_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    return require_tensor_entry(payload, key)


__all__ = [
    "RegimePosteriorMeansV3L9",
    "StructuralPosteriorMeansV3L9",
    "build_dynamic_europe_core_region_block",
    "build_dynamic_us_europe_region_block",
    "coerce_v3_l9_state_tensor",
    "v3_l9_commodity_state_index",
    "v3_l9_eu_core_state_index",
    "v3_l9_state_count",
    "v3_l9_state_site_names",
    "v3_l9_us_eu_region_state_index",
]
