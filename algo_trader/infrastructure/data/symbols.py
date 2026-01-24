from __future__ import annotations

from algo_trader.domain.market_data import TickerConfig


def symbol_directory(ticker: TickerConfig) -> str:
    if ticker.asset_class in {"forex", "commodities"}:
        return join_symbol_currency(ticker.symbol, ticker.currency)
    return ticker.symbol


def join_symbol_currency(symbol: str, currency: str) -> str:
    if "." in symbol:
        return symbol
    if currency and symbol.endswith(currency) and len(symbol) > len(currency):
        base = symbol[: -len(currency)]
        return f"{base}.{currency}"
    if currency:
        return f"{symbol}.{currency}"
    return symbol
