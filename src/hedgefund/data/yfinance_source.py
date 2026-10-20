"""Daily bars from Yahoo Finance via ``yfinance``.

``fetch_ticker`` is a port of the companion script's fetch: the same ``end``/``start``
computation, ``auto_adjust=True``, MultiIndex flattening, and retry on empty data. The
retry count and backoff are parameters and the sleep is injected so tests never wait.
``yfinance`` is imported lazily inside the download so importing this module is free and
tests can substitute a fake module.

Yahoo is rate-limited and flaky; prefer the parquet cache (Phase 5) and never call this
from a test.
"""

import time
from collections.abc import Callable

import pandas as pd

from hedgefund.data.base import INDEX_NAME, PRICE_COLUMNS, validate_price_frame

# History length when fetching a ticker; matches config/default.yaml data.years.
DEFAULT_YEARS = 10
# Total download attempts before giving up; the companion script tried twice.
DEFAULT_RETRIES = 2
# Pause between attempts; Yahoo rate limits clear in well under a minute.
DEFAULT_BACKOFF_SECONDS = 30.0
# yfinance's ``end`` is exclusive, so an inclusive window asks for one more day.
ONE_DAY = pd.Timedelta(days=1)


def default_window(years: int = DEFAULT_YEARS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` exactly as the companion script computes them."""
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = end - pd.DateOffset(years=years)
    return start, end


def download_frame(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download raw daily bars for ``ticker``, retrying on empty data or errors.

    ``start`` and ``end`` are passed to yfinance as ISO dates (``end`` exclusive). Column
    MultiIndexes are flattened to the price level. Raises ``ValueError`` naming the ticker
    when every attempt comes back empty.
    """
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")
    import yfinance as yf

    df = pd.DataFrame()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            last_error = exc
            df = pd.DataFrame()
        if df is not None and not df.empty:
            break
        if attempt < retries:
            sleep(backoff_seconds)

    if df is None or df.empty:
        detail = f" Last error: {last_error}" if last_error is not None else ""
        raise ValueError(
            f"yfinance returned empty data for {ticker} after {retries} attempt(s). "
            f"Yahoo may be rate-limiting. Try again in a few minutes.{detail}"
        )

    # Some yfinance versions return a MultiIndex column frame.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ticker(
    ticker: str,
    years: int = DEFAULT_YEARS,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.Series:
    """Fetch a daily close series via yfinance, companion-faithful.

    Returns the adjusted ``Close`` column with NaNs dropped, named ``ticker``, over the
    last ``years`` years ending today (UTC).
    """
    start, end = default_window(years)
    df = download_frame(
        ticker, start, end, retries=retries, backoff_seconds=backoff_seconds, sleep=sleep
    )
    close = df["Close"].dropna()
    close.name = ticker
    return close


def normalise_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn a raw yfinance frame into the PriceSource schema used by the frozen fixture."""
    df = df.rename(columns=lambda c: str(c).lower())
    present = [c for c in PRICE_COLUMNS if c in df.columns]
    if "close" not in present:
        raise ValueError(f"yfinance frame for {ticker} has no close column: {list(df.columns)}")
    df = df[present].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = INDEX_NAME
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["close"])
    for col in df.columns:
        if col == "volume" and not df[col].isna().any():
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].astype("float64")
    return df


class YFinanceSource:
    """``PriceSource`` backed by yfinance daily bars."""

    name = "yfinance"

    def __init__(
        self,
        years: int = DEFAULT_YEARS,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if years < 1:
            raise ValueError(f"years must be at least 1, got {years}")
        self.years = years
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a validated OHLCV frame for ``symbol``.

        With no ``start``/``end`` the window is the companion default: the last ``years``
        years ending today. Explicit bounds slice inclusively, so ``end`` is the last bar
        returned when Yahoo has it.
        """
        default_start, default_end = default_window(self.years)
        request_start = default_start if start is None else pd.Timestamp(start)
        request_end = default_end if end is None else pd.Timestamp(end) + ONE_DAY
        if request_start >= request_end:
            raise ValueError(
                f"start {request_start.date()} must be before end {request_end.date()}"
            )

        raw = download_frame(
            symbol,
            request_start,
            request_end,
            retries=self.retries,
            backoff_seconds=self.backoff_seconds,
            sleep=self.sleep,
        )
        frame = normalise_frame(raw, symbol)
        if start is not None or end is not None:
            frame = frame.loc[start:end]
        if frame.empty:
            raise ValueError(f"yfinance returned no {symbol} rows between {start} and {end}.")
        return validate_price_frame(frame)
