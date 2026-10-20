"""Daily stock and crypto bars from Alpaca's market-data API, paper keys only.

``AlpacaSource`` implements ``PriceSource`` on top of ``alpaca-py``. The library is
optional and imported lazily inside the methods that need it, so importing this module
costs nothing and ``alpaca_available()`` reports whether it can be used at all.

Credentials come from ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` in the environment,
read at call time and handed straight to the alpaca-py client. They are never stored on
this object, logged, written to config or cache, or echoed in an error message.

This module only *reads* bars. The only trading host it names is the paper endpoint, and
nothing here (or anywhere in this repo) places an order.

Timestamps
----------
Alpaca stamps a daily bar at the session's midnight in America/New_York, which the API
returns as 04:00 UTC (EDT) or 05:00 UTC (EST); crypto daily bars carry the same stamp.
``bars_to_frame`` converts the UTC timestamp to New York time, keeps the calendar date,
and drops the timezone, so a bar for 2024-07-01 lands on ``Timestamp("2024-07-01")``
exactly as yfinance's row for that session does. That is what makes an Alpaca frame and
a yfinance frame for the same symbol align row for row. A stamp that is not New York
midnight is refused rather than rounded: dating it would be a silent lookahead.

Completed bars only
-------------------
Alpaca serves the bar for the *current* New York day while it is still forming, and
``CachedSource`` keeps whatever it is handed for the life of the key, so a partial close
must never reach the cache. ``get_daily`` therefore drops every bar whose New York day
has not ended, judged against an injectable ``now``:

* crypto trades around the clock, so the bar for today's New York date is always open
  and is dropped; the last complete bar is yesterday's (New York).
* stocks close at 16:00 New York; today's bar is kept only once ``now`` is at or past the
  close, otherwise it is dropped too.
"""

import os
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pandas as pd

from hedgefund.data.base import INDEX_NAME, PRICE_COLUMNS, validate_price_frame
from hedgefund.data.yfinance_source import DEFAULT_YEARS, ONE_DAY, default_window

# Environment variables alpaca-py itself reads; this repo reads them the same way and
# nowhere else (never from config, cache, fixtures, or CLI flags).
ENV_KEY_ID = "APCA_API_KEY_ID"
ENV_SECRET_KEY = "APCA_API_SECRET_KEY"
# The only trading host this repo ever names. Nothing here calls it; it exists so a
# future paper-account helper has one constant to import and no reason to type a URL.
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
# Read-only market data. Alpaca serves historical bars from one host for paper and live
# keys alike; there is no separate paper data host.
DATA_URL = "https://data.alpaca.markets"
# IEX is the free equity feed available to every paper account; "sip" needs a paid
# subscription and is the only other sensible choice.
DEFAULT_FEED = "iex"
# Splits and dividends, so closes match yfinance's auto_adjust=True convention.
DEFAULT_ADJUSTMENT = "all"
# Daily bars are the only bar size the platform consumes (plan.md non-goals: intraday).
BAR_TIMEFRAME = "Day"
# Session midnight lives in this zone; Alpaca's daily stamps are its UTC image.
SESSION_TZ = "America/New_York"
# Regular US equity session close (New York); a stock bar is complete from this time on.
STOCK_SESSION_CLOSE = pd.Timedelta(hours=16)
# Alpaca's symbol separator for crypto pairs; yfinance uses "-" for the same pair.
ALPACA_PAIR_SEPARATOR = "/"
YFINANCE_PAIR_SEPARATOR = "-"
# Quote currencies that mark a yfinance-style ticker as a crypto pair. A 3-5 letter base
# covers every major coin ("BTC", "DOGE", "MATIC") and excludes equity tickers.
_CRYPTO_PAIR = re.compile(r"^[A-Z0-9]{3,5}-(USD|USDT|USDC)$")


class AlpacaCredentialsError(LookupError):
    """Raised when an Alpaca credential is missing from the environment.

    The message names the variable that is missing and nothing else: no value of either
    variable ever appears in it.
    """


def read_credentials(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Return ``(key_id, secret_key)`` from ``environ`` (default ``os.environ``).

    Read at call time so a key exported after import is seen. Raises
    ``AlpacaCredentialsError`` naming the first variable that is unset or empty.
    """
    source = os.environ if environ is None else environ
    values: list[str] = []
    for name in (ENV_KEY_ID, ENV_SECRET_KEY):
        value = source.get(name, "")
        if not value:
            raise AlpacaCredentialsError(
                f"{name} is not set. Export your Alpaca *paper* API credentials as "
                f"{ENV_KEY_ID} and {ENV_SECRET_KEY} before using the alpaca source."
            )
        values.append(value)
    return values[0], values[1]


def alpaca_available() -> dict[str, Any]:
    """Report whether ``alpaca-py`` can be imported, in the optional-dependency shape."""
    try:
        import alpaca
    except ImportError as exc:
        return {
            "available": False,
            "reason": f"alpaca-py is not installed (install with `uv sync --extra alpaca`): {exc}",
        }
    return {"available": True, "version": getattr(alpaca, "__version__", None)}


def is_crypto_symbol(symbol: str) -> bool:
    """True for a crypto pair in Alpaca ("BTC/USD") or yfinance ("BTC-USD") spelling."""
    text = symbol.strip().upper()
    if ALPACA_PAIR_SEPARATOR in text:
        return True
    return _CRYPTO_PAIR.match(text) is not None


def to_alpaca_symbol(symbol: str) -> str:
    """Return ``symbol`` as Alpaca spells it: pairs use ``/``, everything is uppercased."""
    text = symbol.strip().upper()
    if not text:
        raise ValueError("symbol must be a non-empty string")
    if is_crypto_symbol(text) and ALPACA_PAIR_SEPARATOR not in text:
        base, _, quote = text.rpartition(YFINANCE_PAIR_SEPARATOR)
        return f"{base}{ALPACA_PAIR_SEPARATOR}{quote}"
    return text


def _timestamp_level(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop alpaca-py's ``symbol`` index level, selecting ``symbol`` if several are present."""
    index = bars.index
    if not isinstance(index, pd.MultiIndex):
        return bars
    if index.nlevels != 2:
        raise ValueError(
            f"alpaca bars for {symbol} have a {index.nlevels}-level index; "
            "expected (symbol, timestamp)"
        )
    symbols = index.get_level_values(0).unique()
    if len(symbols) == 1:
        return bars.droplevel(0)
    wanted = to_alpaca_symbol(symbol)
    if wanted not in symbols:
        raise ValueError(f"alpaca bars do not contain {wanted!r}; saw {list(symbols)}")
    return bars.xs(wanted, level=0)


def bars_to_frame(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert alpaca-py's ``BarSet.df`` into the ``PriceSource`` schema.

    Input: a frame indexed by ``(symbol, timestamp)`` or by ``timestamp`` alone, with
    tz-aware UTC timestamps and columns ``open/high/low/close/volume`` (plus
    ``trade_count`` and ``vwap``, which are dropped). A tz-naive index is taken as UTC,
    which is how alpaca-py documents naive inputs.

    Output: a tz-naive ``DatetimeIndex`` named ``date`` holding the America/New_York
    calendar date of each bar (see the module docstring for why), sorted and unique,
    with float64 ``open/high/low/close/volume`` in that order. ``volume`` is float64
    rather than int64 because crypto volume is fractional. When two bars map to the same
    date the last one wins. Rows without a close are dropped. A stamp whose New York
    time-of-day is not 00:00 raises ``ValueError`` naming the symbol and the stamp.
    """
    frame = _timestamp_level(bars, symbol)
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"alpaca bars for {symbol} need a DatetimeIndex, got {type(index).__name__}"
        )
    stamps = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    session = stamps.tz_convert(SESSION_TZ)
    dates = session.normalize()
    off_midnight = session != dates
    if off_midnight.any():
        first = session[off_midnight][0]
        raise ValueError(
            f"alpaca bar for {symbol} stamped {first} is not New York midnight; refusing to "
            "date it (a daily bar must be stamped at the session's 00:00 America/New_York)"
        )
    dates = dates.tz_localize(None)

    frame = frame.rename(columns=lambda c: str(c).lower())
    present = [c for c in PRICE_COLUMNS if c in frame.columns]
    if "close" not in present:
        raise ValueError(f"alpaca bars for {symbol} have no close column: {list(frame.columns)}")
    out = frame[present].copy()
    out.index = pd.DatetimeIndex(dates, name=INDEX_NAME)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    for col in out.columns:
        out[col] = out[col].astype("float64")
    return out


def utc_bound(value: pd.Timestamp | str) -> pd.Timestamp:
    """Return ``value`` as a tz-aware UTC timestamp; naive inputs are taken as UTC."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def last_complete_date(now: pd.Timestamp, crypto: bool) -> pd.Timestamp:
    """Return the latest New York calendar date whose daily bar is complete at ``now``.

    Crypto trades 24/7, so today's (New York) bar is always still forming and the answer
    is yesterday. A stock bar is complete once the session has closed, so today's date is
    returned from ``STOCK_SESSION_CLOSE`` New York onwards and yesterday's before that.
    The result is tz-naive, matching the ``date`` index ``bars_to_frame`` produces.
    """
    session_now = utc_bound(now).tz_convert(SESSION_TZ)
    today = session_now.normalize()
    if crypto or session_now - today < STOCK_SESSION_CLOSE:
        today = today - ONE_DAY
    return today.tz_localize(None)


class AlpacaSource:
    """``PriceSource`` backed by Alpaca daily bars (stocks via IEX/SIP, crypto via US feed).

    ``stock_client`` and ``crypto_client`` are injectable so tests can hand in fakes that
    record the request and return an object with ``.df``. When they are ``None`` the real
    alpaca-py clients are built inside ``get_daily`` with credentials read from
    ``environ`` (default ``os.environ``) at that moment; the key values live only in the
    alpaca client, never as attributes of this object, and ``repr`` shows none of them.

    ``now`` fixes the clock used to decide which bars are complete (see
    ``last_complete_date``); ``None`` reads the wall clock on every call.
    """

    name = "alpaca"

    def __init__(
        self,
        *,
        years: int = DEFAULT_YEARS,
        feed: str = DEFAULT_FEED,
        adjustment: str = DEFAULT_ADJUSTMENT,
        environ: Mapping[str, str] | None = None,
        stock_client: Any = None,
        crypto_client: Any = None,
        now: pd.Timestamp | None = None,
    ) -> None:
        if years < 1:
            raise ValueError(f"years must be at least 1, got {years}")
        if not feed:
            raise ValueError("feed must be a non-empty string such as 'iex' or 'sip'")
        if not adjustment:
            raise ValueError("adjustment must be one of raw, split, dividend, all")
        self.years = years
        self.feed = feed
        self.adjustment = adjustment
        self._environ = environ
        self._stock_client = stock_client
        self._crypto_client = crypto_client
        self._now = None if now is None else utc_bound(now)

    def _clock(self) -> pd.Timestamp:
        return pd.Timestamp.now("UTC") if self._now is None else self._now

    def __repr__(self) -> str:
        return (
            f"AlpacaSource(years={self.years}, feed={self.feed!r}, adjustment={self.adjustment!r})"
        )

    # ------------------------------------------------------------------ clients
    def _stock(self) -> Any:
        if self._stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            key_id, secret_key = read_credentials(self._environ)
            self._stock_client = StockHistoricalDataClient(api_key=key_id, secret_key=secret_key)
        return self._stock_client

    def _crypto(self) -> Any:
        if self._crypto_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient

            key_id, secret_key = read_credentials(self._environ)
            self._crypto_client = CryptoHistoricalDataClient(api_key=key_id, secret_key=secret_key)
        return self._crypto_client

    # ------------------------------------------------------------------ requests
    def _build_request(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, crypto: bool
    ) -> Any:
        """Build the alpaca-py request, or a duck-typed stand-in when the library is absent.

        The stand-in only matters for injected clients (tests, or a caller with its own
        transport); without alpaca-py there is no real client to build anyway.
        """
        fields: dict[str, Any] = {
            "symbol_or_symbols": symbol,
            "start": start.to_pydatetime(),
            "end": end.to_pydatetime(),
        }
        try:
            from alpaca.data.enums import Adjustment, DataFeed
            from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            fields["timeframe"] = BAR_TIMEFRAME
            if not crypto:
                fields["adjustment"] = self.adjustment
                fields["feed"] = self.feed
            return SimpleNamespace(**fields)

        fields["timeframe"] = TimeFrame.Day
        if crypto:
            return CryptoBarsRequest(**fields)
        return StockBarsRequest(
            adjustment=Adjustment(self.adjustment), feed=DataFeed(self.feed), **fields
        )

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a validated OHLCV frame for ``symbol`` over an inclusive window.

        With no ``start``/``end`` the window is ``default_window(years)``, the same
        convention as ``YFinanceSource`` so ``CachedSource`` keys match across sources.
        Crypto pairs ("BTC-USD" or "BTC/USD") go to the crypto client, everything else to
        the stock client. Bounds may be tz-naive (taken as UTC) or tz-aware. Bars whose
        New York day has not ended at ``now`` are dropped (see ``last_complete_date``).
        Raises ``ValueError`` naming the symbol when no bars come back.
        """
        default_start, default_end = default_window(self.years)
        request_start = utc_bound(default_start if start is None else start)
        # ``end`` is inclusive; the bar for that date is stamped a few hours into it (UTC),
        # so ask up to the start of the following day. The default window already ends on
        # today's midnight, which excludes today's partial bar exactly as yfinance does.
        request_end = utc_bound(default_end) if end is None else utc_bound(end) + ONE_DAY
        if request_start >= request_end:
            raise ValueError(
                f"start {request_start.date()} must be before end {request_end.date()}"
            )

        alpaca_symbol = to_alpaca_symbol(symbol)
        crypto = is_crypto_symbol(alpaca_symbol)
        request = self._build_request(alpaca_symbol, request_start, request_end, crypto)
        if crypto:
            bars = self._crypto().get_crypto_bars(request)
        else:
            bars = self._stock().get_stock_bars(request)

        raw = getattr(bars, "df", bars)
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"alpaca returned no bars for {symbol} between {request_start.date()} and "
                f"{(request_end - ONE_DAY).date()} (feed={self.feed!r})."
            )
        frame = bars_to_frame(raw, alpaca_symbol)
        if start is not None or end is not None:
            slice_start = request_start.tz_localize(None)
            slice_end = (request_end - ONE_DAY).tz_localize(None)
            frame = frame.loc[slice_start:slice_end]
        complete_through = last_complete_date(self._clock(), crypto)
        frame = frame[frame.index <= complete_through]
        if frame.empty:
            raise ValueError(
                f"alpaca returned no completed {symbol} bars between {request_start.date()} "
                f"and {(request_end - ONE_DAY).date()} (last complete session "
                f"{complete_through.date()})."
            )
        return validate_price_frame(frame)
