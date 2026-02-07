from __future__ import annotations

from algo_trader.infrastructure import OutputNames

_WEEKLY_OHLC_NAME = "weekly_ohlc.csv"
_DAILY_OHLC_NAME = "daily_ohlc.csv"
_OUTPUT_NAMES = OutputNames(
    output_name="features.csv",
    metadata_name="metadata.json",
)
_TENSOR_NAME = "features_tensor.pt"
_GOODNESS_NAME = "goodness.json"
_FREQUENCY = "weekly"
_TRADING_DAYS_PER_WEEK = 5
_TENSOR_TIMESTAMP_UNIT = "epoch_hours"
_TENSOR_TIMEZONE = "UTC"
_TENSOR_VALUE_DTYPE = "float64"
_ALL_GROUP = "all"
_CROSS_SECTIONAL_GROUP = "cross_sectional"
