"""The ``PriceSource`` protocol and the frame validation every source must pass.

A price source hands back one daily bar per row: a tz-naive, sorted, unique
``DatetimeIndex`` named ``date`` and a float64 ``close`` column with no gaps. The
other OHLCV columns are optional. Everything downstream (labelling, backtests,
reports) assumes this shape, so ``validate_price_frame`` is the single place it is
enforced.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Column order every source emits; ``close`` is mandatory, the rest are optional.
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
# Index name shared with tests/fixtures/spy_10y.parquet so frames are interchangeable.
INDEX_NAME = "date"


@runtime_checkable
class PriceSource(Protocol):
    """Anything that can serve validated daily bars for a symbol."""

    name: str  # "yfinance" | "csv" | "alpaca"

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame: ...


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assert ``df`` has the PriceSource schema and return it unchanged.

    Raises ``ValueError`` naming the first problem found: index type or name, timezone,
    ordering, duplicate dates, missing or non-float64 ``close``, NaN closes.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"expected a pandas DataFrame, got {type(df).__name__}")
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"price frame index must be a DatetimeIndex, got {type(index).__name__}")
    if index.name != INDEX_NAME:
        raise ValueError(f"price frame index must be named {INDEX_NAME!r}, got {index.name!r}")
    if index.tz is not None:
        raise ValueError(f"price frame index must be tz-naive, got tz={index.tz}")
    if not index.is_monotonic_increasing:
        raise ValueError("price frame index must be sorted ascending by date")
    if not index.is_unique:
        duplicated = index[index.duplicated()]
        raise ValueError(f"price frame index has duplicate dates, first: {duplicated[0].date()}")
    if "close" not in df.columns:
        raise ValueError(f"price frame must have a 'close' column, saw {list(df.columns)}")
    close = df["close"]
    if close.dtype != np.float64:
        raise ValueError(f"price frame 'close' must be float64, got {close.dtype}")
    if close.isna().any():
        first_nan = close.index[close.isna()][0]
        raise ValueError(f"price frame 'close' has NaN values, first at {first_nan.date()}")
    return df


def close_series(df: pd.DataFrame, name: str | None = None) -> pd.Series:
    """Return the float64 ``close`` column of ``df`` as a Series, optionally renamed."""
    if "close" not in df.columns:
        raise ValueError(f"frame has no 'close' column, saw {list(df.columns)}")
    close = df["close"].astype("float64")
    if name is not None:
        close = close.rename(name)
    return close
