from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pyro
import torch
from torch import nn

from algo_trader.application.historical import HistoricalRequestConfig
from algo_trader.domain import ConfigError
from algo_trader.infrastructure.data import symbol_directory
from algo_trader.pipeline.stages.modeling.batch_utils import resolve_batch_shape
from algo_trader.pipeline.stages.modeling.protocols import ModelBatch, PyroGuide
from algo_trader.pipeline.stages.modeling.registry_core import register_guide

from .guide_l11 import (
    FactorGuideL11OnlineFiltering,
    FilteringState,
    Level11GuideConfig as Level15GuideConfig,
    Level11RuntimeBatch as Level15RuntimeBatch,
    RegimeEncoder as _BaseRegimeEncoder,
    _GuideContext,
    _LocalGuideSites,
    _build_context,
    _build_guide_config as _build_level15_guide_config,
    _encoder_features as _encoder_features_l11,
    _initial_prior_message as _initial_prior_message_l11,
    _next_steps_seen,
    _propagated_message as _propagated_message_l11,
    _resolve_filtering_state,
    _resolve_time_mask,
    _resolve_y_input,
    _sample_global_sites,
    _sample_regime_sites,
    _sample_scale_sites,
)
from .guide_l13 import (
    _build_gain_inputs,
    _build_gain_local_sites,
    _GainLocalEncodingPlan,
    _decode_gain_components,
    _unbound_gain_slices,
)

_MIN_SCALE = 1e-4
_MAX_H_GAIN = 0.35
_MAX_H_LOG_SCALE_STEP = 0.40
_MAX_H_SCALE = 0.25
_REPO_ROOT = Path(__file__).resolve().parents[5]
_TICKERS_CONFIG_PATH = _REPO_ROOT / "config" / "tickers.yml"


class RegimeEncoder(_BaseRegimeEncoder):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        layers = list(self._network.children())
        layers[-1] = nn.Linear(hidden_dim, 5)
        self._network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._network(inputs)


def build_level15_runtime_batch(batch: ModelBatch) -> Level15RuntimeBatch:
    shape = resolve_batch_shape(batch)
    X_asset = batch.X_asset if batch.X_asset is not None else batch.X
    if X_asset is None:
        raise ConfigError(
            "Level 15 online-filtering runtime requires batch.X_asset"
        )
    if X_asset.ndim != 3:
        raise ConfigError("batch.X_asset must have shape [T, A, F]")
    if batch.X_global is None:
        raise ConfigError(
            "Level 15 online-filtering runtime requires batch.X_global"
        )
    if batch.X_global.ndim != 2:
        raise ConfigError("batch.X_global must have shape [T, G]")
    if int(batch.X_global.shape[0]) != shape.T:
        raise ConfigError("batch.X_global and targets must align on T")
    _validated_fx_asset_names(batch.asset_names, expected=shape.A)
    return Level15RuntimeBatch(
        X_asset=X_asset.to(device=shape.device, dtype=shape.dtype),
        X_global=batch.X_global.to(device=shape.device, dtype=shape.dtype),
        y_input=_resolve_y_input(shape),
        y_obs=shape.y_obs,
        time_mask=_resolve_time_mask(batch, shape.T, shape.A),
        obs_scale=batch.obs_scale,
        filtering_state=_resolve_filtering_state(
            batch.filtering_state,
            device=shape.device,
            dtype=shape.dtype,
        ),
    )


@dataclass
class FactorGuideL15OnlineFiltering(FactorGuideL11OnlineFiltering):
    config: Level15GuideConfig

    def __call__(self, batch: ModelBatch) -> None:
        runtime_batch, local = self._build_local_sites(batch)
        _sample_regime_sites(local)
        _sample_scale_sites(local)

    def build_filtering_state(self, batch: ModelBatch) -> FilteringState:
        runtime_batch, local = self._build_local_sites(
            batch, use_posterior_summary=True
        )
        return FilteringState(
            h_loc=local.h_loc[-1].detach(),
            h_scale=local.h_scale[-1].detach(),
            steps_seen=_next_steps_seen(runtime_batch),
        )

    def _require_encoder(self, context: _GuideContext) -> RegimeEncoder:
        if self._encoder is None:
            self._encoder = RegimeEncoder(
                input_dim=context.encoder_input_dim,
                hidden_dim=self.config.hidden_dim,
            ).to(device=context.device, dtype=context.dtype)
            self._encoder_input_dim = context.encoder_input_dim
        if self._encoder_input_dim != context.encoder_input_dim:
            raise ConfigError(
                "Level 15 guide encoder input dimension changed across calls"
            )
        return cast(RegimeEncoder, self._encoder)

    def _build_local_sites(
        self,
        batch: ModelBatch,
        *,
        use_posterior_summary: bool = False,
    ) -> tuple[Level15RuntimeBatch, _LocalGuideSites]:
        runtime_batch = build_level15_runtime_batch(batch)
        context = _build_context(runtime_batch)
        encoder = self._require_encoder(context)
        if not use_posterior_summary:
            pyro.module("factor_l15_online_filtering_encoder", encoder)
            structural = _sample_global_sites(
                context=context,
                factor_count=self.config.factor_count,
            )
        else:
            structural = self.structural_posterior_means()
        local = cast(
            _LocalGuideSites,
            _build_gain_local_sites(
                inputs=cast(
                    Any,
                    _build_gain_inputs(
                        context=cast(Any, context),
                        encoder=cast(Any, encoder),
                        s_u_mean=structural.s_u_mean,
                        phi=self.config.phi,
                        eps=self.config.eps,
                    ),
                ),
                plan=cast(
                    Any,
                    _GainLocalEncodingPlan(
                        decode_step=lambda encoded, prior_loc, prior_scale: _decode_local_parameters(
                            encoded=encoded,
                            prior_loc=prior_loc,
                            prior_scale=prior_scale,
                        ),
                        initial_prior_message=cast(
                            Any, _initial_prior_message_l11
                        ),
                        encoder_features=cast(Any, _encoder_features_l11),
                        propagated_message=cast(
                            Any, _propagated_message_l11
                        ),
                    ),
                ),
            ),
        )
        return runtime_batch, local


@register_guide("factor_guide_l15_online_filtering")
def build_factor_guide_l15_online_filtering(
    params: Mapping[str, Any]
) -> PyroGuide:
    return FactorGuideL15OnlineFiltering(
        config=cast(Level15GuideConfig, _build_level15_guide_config(params))
    )


def _decode_local_parameters(
    *,
    encoded: torch.Tensor,
    prior_loc: torch.Tensor,
    prior_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _decode_gain_components(
        slices=_unbound_gain_slices(encoded),
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def _validated_fx_asset_names(
    asset_names: Sequence[str] | None, *, expected: int
) -> tuple[str, ...]:
    if asset_names is None:
        raise ConfigError(
            "Level 15 requires batch.asset_names for FX-only validation"
        )
    names = tuple(str(name) for name in asset_names)
    if len(names) != expected:
        raise ConfigError("batch.asset_names must align with the asset dimension")
    fx_names = _configured_fx_asset_names()
    invalid = [name for name in names if name not in fx_names]
    if invalid:
        raise ConfigError(
            "Level 15 is FX-only and received non-FX assets",
            context={"assets": ", ".join(sorted(invalid))},
        )
    return names


@lru_cache(maxsize=1)
def _configured_fx_asset_names() -> frozenset[str]:
    config = HistoricalRequestConfig.load(_TICKERS_CONFIG_PATH)
    return frozenset(
        symbol_directory(ticker) for ticker in config.tickers
    )
