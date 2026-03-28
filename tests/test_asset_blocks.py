from algo_trader.domain import (
    COMMODITY_CLASS_ID,
    FX_CLASS_ID,
    INDEX_CLASS_ID,
    build_asset_block_index_map,
    classify_asset_name,
)


def test_classify_asset_name_treats_spot_metals_as_commodities() -> None:
    assert classify_asset_name("XAU.USD") == COMMODITY_CLASS_ID
    assert classify_asset_name("xag.usd") == COMMODITY_CLASS_ID
    assert classify_asset_name("EUR.USD") == FX_CLASS_ID
    assert classify_asset_name("IBUS500") == INDEX_CLASS_ID


def test_build_asset_block_index_map_places_spot_metals_in_commodity_block() -> None:
    block_map = build_asset_block_index_map(
        ["IBUS500", "XAU.USD", "XAG.USD", "EUR.USD"]
    )

    assert block_map["indices"] == (0,)
    assert block_map["commodities"] == (1, 2)
    assert block_map["fx"] == (3,)
