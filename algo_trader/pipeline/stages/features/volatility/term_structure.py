from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from algo_trader.pipeline.stages.features.horizons import HorizonSpec

LogRatio = Callable[[pd.Series, pd.Series], pd.Series]
RequireWeeks = Callable[[Sequence[HorizonSpec], Sequence[int], str], list[HorizonSpec]]


@dataclass(frozen=True)
class TermStructurePair:
    short_spec: HorizonSpec
    long_spec: HorizonSpec
    name: str


@dataclass(frozen=True)
class TermStructureContext:
    cc_pairs: Sequence[TermStructurePair]
    atr_pairs: Sequence[TermStructurePair]
    log_ratio: LogRatio
    vol_cc_name: Callable[[int], str]
    atrp_name: Callable[[int], str]


@dataclass(frozen=True)
class TermStructureSpec:
    feature_name: str
    short_weeks: int
    long_weeks: int


def build_cc_pairs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    *,
    require_weeks: RequireWeeks,
) -> list[TermStructurePair]:
    specs = [
        TermStructureSpec(
            feature_name="vol_ts_cc_1w_4w",
            short_weeks=1,
            long_weeks=4,
        ),
        TermStructureSpec(
            feature_name="vol_ts_cc_4w_12w",
            short_weeks=4,
            long_weeks=12,
        ),
        TermStructureSpec(
            feature_name="vol_ts_cc_4w_26w",
            short_weeks=4,
            long_weeks=26,
        ),
    ]
    pairs: list[TermStructurePair] = []
    for spec in specs:
        pairs.extend(
            _pair(
                horizons,
                feature_set,
                require_weeks=require_weeks,
                spec=spec,
            )
        )
    return pairs


def build_atr_pairs(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    *,
    require_weeks: RequireWeeks,
) -> list[TermStructurePair]:
    return _pair(
        horizons,
        feature_set,
        require_weeks=require_weeks,
        spec=TermStructureSpec(
            feature_name="vol_ts_atr_4w_12w",
            short_weeks=4,
            long_weeks=12,
        ),
    )


def build_features(
    vol_cc_weekly: Mapping[str, pd.Series],
    atrp_weekly: Mapping[str, pd.Series],
    context: TermStructureContext,
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    feature_data.update(
        _from_pairs(
            vol_cc_weekly,
            context.cc_pairs,
            log_ratio=context.log_ratio,
            name_for=lambda spec: context.vol_cc_name(spec.weeks),
        )
    )
    feature_data.update(
        _from_pairs(
            atrp_weekly,
            context.atr_pairs,
            log_ratio=context.log_ratio,
            name_for=lambda spec: context.atrp_name(spec.weeks),
        )
    )
    return feature_data


def _from_pairs(
    source: Mapping[str, pd.Series],
    pairs: Sequence[TermStructurePair],
    *,
    log_ratio: LogRatio,
    name_for: Callable[[HorizonSpec], str],
) -> dict[str, pd.Series]:
    feature_data: dict[str, pd.Series] = {}
    for pair in pairs:
        short_series = source.get(name_for(pair.short_spec))
        long_series = source.get(name_for(pair.long_spec))
        if short_series is None or long_series is None:
            feature_data[pair.name] = pd.Series(dtype=float)
            continue
        feature_data[pair.name] = log_ratio(short_series, long_series)
    return feature_data


def _pair(
    horizons: Sequence[HorizonSpec],
    feature_set: set[str],
    *,
    require_weeks: RequireWeeks,
    spec: TermStructureSpec,
) -> list[TermStructurePair]:
    if spec.feature_name not in feature_set:
        return []
    short_spec = require_weeks(
        horizons, (spec.short_weeks,), spec.feature_name
    )[0]
    long_spec = require_weeks(
        horizons, (spec.long_weeks,), spec.feature_name
    )[0]
    return [
        TermStructurePair(
            short_spec=short_spec,
            long_spec=long_spec,
            name=spec.feature_name,
        )
    ]
