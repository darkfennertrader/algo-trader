from __future__ import annotations

from typing import Sequence

FX_CLASS_ID = 0
INDEX_CLASS_ID = 1
COMMODITY_CLASS_ID = 2

FULL_BLOCK = "full"
FX_BLOCK = "fx"
INDICES_BLOCK = "indices"
COMMODITIES_BLOCK = "commodities"
BLOCK_ORDER = (FX_BLOCK, INDICES_BLOCK, COMMODITIES_BLOCK, FULL_BLOCK)


def classify_asset_name(name: str) -> int:
    normalized = str(name).strip().upper()
    if _is_commodity_name(normalized):
        return COMMODITY_CLASS_ID
    if _is_fx_name(normalized):
        return FX_CLASS_ID
    return INDEX_CLASS_ID


def build_asset_block_index_map(
    asset_names: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    fx: list[int] = []
    indices: list[int] = []
    commodities: list[int] = []
    for idx, asset_name in enumerate(asset_names):
        class_id = classify_asset_name(asset_name)
        if class_id == FX_CLASS_ID:
            fx.append(idx)
        elif class_id == INDEX_CLASS_ID:
            indices.append(idx)
        else:
            commodities.append(idx)
    return {
        FX_BLOCK: tuple(fx),
        INDICES_BLOCK: tuple(indices),
        COMMODITIES_BLOCK: tuple(commodities),
        FULL_BLOCK: tuple(range(len(asset_names))),
    }


def _is_fx_name(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and all(len(part) == 3 and part.isalpha() for part in parts)


def _is_commodity_name(name: str) -> bool:
    return name.startswith(("XAU", "XAG", "XPT", "XPD"))
