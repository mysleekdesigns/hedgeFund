"""Tests for the Alpaca price source (Phase 7).

All offline. The alpaca-py clients are replaced by fakes that record the request they
were handed and return an object with ``.df`` shaped like ``BarSet.df``; the conftest
socket guard would fail any real call. The aligned-closes test feeds the frozen SPY
fixture through both the yfinance port and the Alpaca source and expects identical rows.
"""

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hedgefund.data.alpaca_source import (
    DATA_URL,
    DEFAULT_ADJUSTMENT,
    DEFAULT_FEED,
    ENV_KEY_ID,
    ENV_SECRET_KEY,
    PAPER_TRADING_URL,
    AlpacaCredentialsError,
    AlpacaSource,
    alpaca_available,
    bars_to_frame,
    is_crypto_symbol,
    last_complete_date,
    read_credentials,
    to_alpaca_symbol,
    utc_bound,
)
from hedgefund.data.base import PRICE_COLUMNS, PriceSource, validate_price_frame
from hedgefund.data.cache import CachedSource
from hedgefund.data.yfinance_source import YFinanceSource

FIXTURES = Path(__file__).parent / "fixtures"
SPY_PARQUET = FIXTURES / "spy_10y.parquet"

SENTINEL_KEY = "SENTINELKEY123"
SENTINEL_SECRET = "SENTINELSECRET456"
SENTINEL_ENV = {ENV_KEY_ID: SENTINEL_KEY, ENV_SECRET_KEY: SENTINEL_SECRET}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _session_stamps(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Stamp each date at New York midnight expressed in UTC, exactly as Alpaca does."""
    return pd.DatetimeIndex(dates).tz_localize("America/New_York").tz_convert("UTC")


def _alpaca_bars(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Re-shape a PriceSource frame into ``BarSet.df``: (symbol, timestamp) MultiIndex."""
    stamps = _session_stamps(frame.index)
    index = pd.MultiIndex.from_arrays(
        [np.repeat(symbol, len(frame)), stamps], names=["symbol", "timestamp"]
    )
    return pd.DataFrame(
        {
            "open": frame["open"].to_numpy(dtype="float64"),
            "high": frame["high"].to_numpy(dtype="float64"),
            "low": frame["low"].to_numpy(dtype="float64"),
            "close": frame["close"].to_numpy(dtype="float64"),
            "volume": frame["volume"].to_numpy(dtype="float64"),
            "trade_count": np.arange(len(frame), dtype="float64"),
            "vwap": frame["close"].to_numpy(dtype="float64"),
        },
        index=index,
    )


def _yf_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """The same bars in yfinance's raw download shape (capitalised columns, ``Date`` index)."""
    raw = frame.rename(columns=str.capitalize)
    raw.index = pd.DatetimeIndex(frame.index, name="Date")
    return raw


def _as_utc(value: Any) -> pd.Timestamp:
    """alpaca-py stores request bounds as naive UTC datetimes; compare them as UTC."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _small_frame(n: int = 5) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.date_range("2024-07-01", periods=n, freq="B"), name="date")
    base = np.linspace(100.0, 104.0, n)
    return pd.DataFrame(
        {
            "open": base - 0.5,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.arange(1, n + 1) * 1000,
        },
        index=index,
    )


class FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class FakeStockClient:
    """Stands in for ``StockHistoricalDataClient``; records requests, serves one frame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.requests: list[Any] = []

    def get_stock_bars(self, request: Any) -> FakeBarSet:
        self.requests.append(request)
        return FakeBarSet(self._df)


class FakeCryptoClient:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.requests: list[Any] = []

    def get_crypto_bars(self, request: Any) -> FakeBarSet:
        self.requests.append(request)
        return FakeBarSet(self._df)


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return pd.read_parquet(SPY_PARQUET)


@pytest.fixture
def fake_yf(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.DataFrame], list[dict[str, Any]]]:
    """Install a fake ``yfinance`` module serving ``raw``; return the recorded calls list."""

    def install(raw: pd.DataFrame) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def download(ticker: str, **kwargs: Any) -> pd.DataFrame:
            calls.append({"ticker": ticker, **kwargs})
            return raw

        module = types.ModuleType("yfinance")
        module.download = download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "yfinance", module)
        return calls

    return install


# --------------------------------------------------------------------------- #
# Constants and credentials
# --------------------------------------------------------------------------- #
def test_only_paper_trading_host_is_named() -> None:
    assert PAPER_TRADING_URL == "https://paper-api.alpaca.markets"
    assert DATA_URL == "https://data.alpaca.markets"
    assert "api.alpaca.markets" not in PAPER_TRADING_URL.replace("paper-api", "")


def test_read_credentials_returns_pair_from_mapping() -> None:
    assert read_credentials(SENTINEL_ENV) == (SENTINEL_KEY, SENTINEL_SECRET)


def test_read_credentials_reads_os_environ_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_KEY_ID, "k")
    monkeypatch.setenv(ENV_SECRET_KEY, "s")
    assert read_credentials() == ("k", "s")


@pytest.mark.parametrize("missing", [ENV_KEY_ID, ENV_SECRET_KEY])
def test_read_credentials_missing_var_names_it_without_echoing_values(missing: str) -> None:
    environ = dict(SENTINEL_ENV)
    del environ[missing]
    with pytest.raises(AlpacaCredentialsError) as info:
        read_credentials(environ)
    message = str(info.value)
    assert missing in message
    assert SENTINEL_KEY not in message
    assert SENTINEL_SECRET not in message


def test_read_credentials_empty_var_is_missing() -> None:
    with pytest.raises(AlpacaCredentialsError, match=ENV_SECRET_KEY):
        read_credentials({ENV_KEY_ID: "k", ENV_SECRET_KEY: ""})


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def test_alpaca_available_when_installed() -> None:
    assert alpaca_available()["available"] is True


def test_alpaca_available_reports_reason_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "alpaca", None)
    result = alpaca_available()
    assert result["available"] is False
    assert "alpaca-py" in result["reason"]


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("BTC/USD", True),
        ("BTC-USD", True),
        ("ETH/USD", True),
        ("eth-usd", True),
        ("DOGE-USDT", True),
        ("MATIC-USDC", True),
        ("SPY", False),
        ("BRK-B", False),
        ("USD", False),
    ],
)
def test_is_crypto_symbol(symbol: str, expected: bool) -> None:
    assert is_crypto_symbol(symbol) is expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("BTC-USD", "BTC/USD"),
        ("btc/usd", "BTC/USD"),
        ("eth-usdt", "ETH/USDT"),
        ("spy", "SPY"),
        ("BRK-B", "BRK-B"),
    ],
)
def test_to_alpaca_symbol(symbol: str, expected: str) -> None:
    assert to_alpaca_symbol(symbol) == expected


def test_to_alpaca_symbol_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        to_alpaca_symbol("  ")


# --------------------------------------------------------------------------- #
# bars_to_frame
# --------------------------------------------------------------------------- #
def test_bars_to_frame_converts_multiindex_to_schema() -> None:
    small = _small_frame()
    frame = bars_to_frame(_alpaca_bars("SPY", small), "SPY")
    assert validate_price_frame(frame) is frame
    assert list(frame.columns) == PRICE_COLUMNS
    assert frame.index.name == "date"
    assert frame.index.tz is None
    assert all(frame[c].dtype == "float64" for c in frame.columns)
    pd.testing.assert_index_equal(frame.index, small.index)


def test_bars_to_frame_maps_utc_stamp_to_new_york_date() -> None:
    # 2024-07-01 04:00 UTC is midnight EDT on 2024-07-01; 2024-12-02 05:00 UTC is midnight
    # EST on 2024-12-02. Both land on their session date, not the UTC date-minus-nothing.
    stamps = pd.DatetimeIndex(
        [pd.Timestamp("2024-07-01 04:00", tz="UTC"), pd.Timestamp("2024-12-02 05:00", tz="UTC")],
        name="timestamp",
    )
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=stamps)
    frame = bars_to_frame(bars, "SPY")
    assert frame.index.tolist() == [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-02")]


def test_bars_to_frame_keeps_last_duplicate_and_sorts() -> None:
    stamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-07-02 04:00", tz="UTC"),
            pd.Timestamp("2024-07-01 04:00", tz="UTC"),
            pd.Timestamp("2024-07-01 04:00", tz="UTC"),
        ]
    )
    bars = pd.DataFrame({"close": [3.0, 1.0, 1.5]}, index=stamps)
    frame = bars_to_frame(bars, "SPY")
    assert frame.index.tolist() == [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-02")]
    assert frame["close"].tolist() == [1.5, 3.0]


def test_bars_to_frame_selects_symbol_from_multi_symbol_frame() -> None:
    small = _small_frame(3)
    bars = pd.concat([_alpaca_bars("SPY", small), _alpaca_bars("QQQ", small * 2.0)])
    frame = bars_to_frame(bars, "QQQ")
    np.testing.assert_allclose(frame["close"], small["close"] * 2.0)


def test_bars_to_frame_without_close_raises() -> None:
    stamps = pd.DatetimeIndex([pd.Timestamp("2024-07-01 04:00", tz="UTC")])
    with pytest.raises(ValueError, match=r"SPY.*close"):
        bars_to_frame(pd.DataFrame({"vwap": [1.0]}, index=stamps), "SPY")


# --------------------------------------------------------------------------- #
# AlpacaSource
# --------------------------------------------------------------------------- #
def test_alpaca_source_satisfies_protocol() -> None:
    assert isinstance(AlpacaSource(stock_client=FakeStockClient(pd.DataFrame())), PriceSource)


def test_alpaca_source_rejects_bad_years() -> None:
    with pytest.raises(ValueError, match="years"):
        AlpacaSource(years=0)


def test_alpaca_and_yfinance_return_aligned_closes(
    spy: pd.DataFrame, fake_yf: Callable[[pd.DataFrame], list[dict[str, Any]]]
) -> None:
    """The plan item: both sources yield the same dates and closes for the same symbol."""
    start, end = spy.index[0], spy.index[-1]
    fake_yf(_yf_raw(spy))
    bars = _alpaca_bars("SPY", spy)
    # Ten years of sessions cross DST both ways, so both 04:00 and 05:00 UTC stamps occur.
    hours = set(bars.index.get_level_values("timestamp").hour)
    assert hours == {4, 5}
    stock = FakeStockClient(bars)

    from_yf = YFinanceSource().get_daily("SPY", start, end)
    from_alpaca = AlpacaSource(stock_client=stock, environ=SENTINEL_ENV).get_daily(
        "SPY", start, end
    )

    pd.testing.assert_index_equal(from_alpaca.index, from_yf.index)
    # Same input bars on both sides, so exact equality is the right tolerance: any
    # difference would be a conversion bug, not float noise.
    np.testing.assert_allclose(
        from_alpaca["close"].to_numpy(), from_yf["close"].to_numpy(), rtol=0, atol=0
    )
    assert len(from_alpaca) == len(spy)


def test_alpaca_source_stock_request_fields() -> None:
    small = _small_frame()
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    AlpacaSource(stock_client=stock, feed="sip", adjustment="split").get_daily(
        "spy", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-03")
    )
    request = stock.requests[0]
    assert request.symbol_or_symbols == "SPY"
    assert str(request.timeframe) == "1Day"
    assert request.adjustment == "split"
    assert request.feed == "sip"
    assert _as_utc(request.start) == pd.Timestamp("2024-07-01", tz="UTC")
    # Inclusive end: the 2024-07-03 bar is stamped 04:00 UTC, so ask through the next midnight.
    assert _as_utc(request.end) == pd.Timestamp("2024-07-04", tz="UTC")


def test_alpaca_source_defaults_match_module_constants() -> None:
    small = _small_frame()
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    AlpacaSource(stock_client=stock).get_daily("SPY", small.index[0], small.index[-1])
    request = stock.requests[0]
    assert request.adjustment == DEFAULT_ADJUSTMENT == "all"
    assert request.feed == DEFAULT_FEED == "iex"


def test_alpaca_source_explicit_window_is_inclusive() -> None:
    small = _small_frame(5)  # 2024-07-01 .. 2024-07-05
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    frame = AlpacaSource(stock_client=stock).get_daily(
        "SPY", pd.Timestamp("2024-07-02"), pd.Timestamp("2024-07-04")
    )
    assert frame.index[0] == pd.Timestamp("2024-07-02")
    assert frame.index[-1] == pd.Timestamp("2024-07-04")
    assert len(frame) == 3


def test_alpaca_source_default_window_is_years_apart() -> None:
    small = _small_frame()
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    AlpacaSource(stock_client=stock, years=3).get_daily("SPY")
    request = stock.requests[0]
    start, end = _as_utc(request.start), _as_utc(request.end)
    assert start == end - pd.DateOffset(years=3)
    # No explicit end: the window stops at today's UTC midnight, excluding today's partial
    # bar, exactly where the yfinance default window stops.
    assert end == pd.Timestamp.now("UTC").normalize()


def test_crypto_symbol_routes_to_crypto_client_only() -> None:
    small = _small_frame()
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    crypto = FakeCryptoClient(_alpaca_bars("BTC/USD", small))
    frame = AlpacaSource(stock_client=stock, crypto_client=crypto).get_daily(
        "BTC-USD", small.index[0], small.index[-1]
    )
    assert stock.requests == []
    assert len(crypto.requests) == 1
    request = crypto.requests[0]
    assert request.symbol_or_symbols == "BTC/USD"
    assert not hasattr(request, "adjustment") or request.adjustment is None
    assert validate_price_frame(frame) is frame


def test_empty_bars_raise_value_error_naming_symbol() -> None:
    stock = FakeStockClient(pd.DataFrame())
    with pytest.raises(ValueError, match="QQQ"):
        AlpacaSource(stock_client=stock).get_daily("QQQ", "2024-07-01", "2024-07-05")


def test_window_outside_bars_raises_value_error_naming_symbol() -> None:
    stock = FakeStockClient(_alpaca_bars("QQQ", _small_frame()))
    with pytest.raises(ValueError, match="QQQ"):
        AlpacaSource(stock_client=stock).get_daily("QQQ", "2020-01-01", "2020-01-05")


def test_secrets_never_leak(tmp_path: Path) -> None:
    small = _small_frame()
    source = AlpacaSource(
        stock_client=FakeStockClient(_alpaca_bars("SPY", small)), environ=SENTINEL_ENV
    )
    frame = source.get_daily("SPY", small.index[0], small.index[-1])

    text = repr(source) + frame.to_string() + str(frame.attrs)
    assert SENTINEL_KEY not in text
    assert SENTINEL_SECRET not in text

    empty = AlpacaSource(stock_client=FakeStockClient(pd.DataFrame()), environ=SENTINEL_ENV)
    with pytest.raises(ValueError) as info:
        empty.get_daily("SPY", small.index[0], small.index[-1])
    assert SENTINEL_KEY not in str(info.value)
    assert SENTINEL_SECRET not in str(info.value)

    cached = CachedSource(source, tmp_path)
    cached.get_daily("SPY", small.index[0], small.index[-1])
    written = list(tmp_path.rglob("*.parquet"))
    assert len(written) == 1
    blob = written[0].read_bytes()
    assert SENTINEL_KEY.encode() not in blob
    assert SENTINEL_SECRET.encode() not in blob
    assert "alpaca" in written[0].parent.name


def test_missing_credentials_surface_before_any_client_is_built() -> None:
    source = AlpacaSource(environ={})
    with pytest.raises(AlpacaCredentialsError, match=ENV_KEY_ID):
        source.get_daily("SPY", "2024-07-01", "2024-07-05")


def test_injected_client_works_without_alpaca_py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without alpaca-py the request is a plain namespace with the same fields."""
    for name in ("alpaca", "alpaca.data", "alpaca.data.enums", "alpaca.data.requests"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", None)
    small = _small_frame()
    stock = FakeStockClient(_alpaca_bars("SPY", small))
    frame = AlpacaSource(stock_client=stock).get_daily("SPY", small.index[0], small.index[-1])
    request = stock.requests[0]
    assert request.symbol_or_symbols == "SPY"
    assert request.timeframe == "Day"
    assert request.feed == DEFAULT_FEED
    assert _as_utc(request.start) == pd.Timestamp("2024-07-01", tz="UTC")
    assert validate_price_frame(frame) is frame


# --------------------------------------------------------------------------- #
# Completed bars only (reviewer finding 1)
# --------------------------------------------------------------------------- #
def _frame_ending_today(now_ny: pd.Timestamp, n: int = 5) -> pd.DataFrame:
    """``n`` daily bars whose last date is the New York calendar date of ``now_ny``."""
    today = now_ny.tz_convert("America/New_York").normalize().tz_localize(None)
    index = pd.DatetimeIndex(pd.date_range(end=today, periods=n, freq="D"), name="date")
    base = np.linspace(100.0, 104.0, n)
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base, "volume": base},
        index=index,
    )


@pytest.mark.parametrize(
    ("now", "crypto", "expected"),
    [
        # 02:00 UTC on 2 Jul is 22:00 NY on 1 Jul: crypto's 1 Jul bar is still open.
        (pd.Timestamp("2024-07-02 02:00", tz="UTC"), True, pd.Timestamp("2024-06-30")),
        # Crypto at 17:00 NY: today's bar is still open; yesterday is the last complete one.
        (pd.Timestamp("2024-07-02 17:00", tz="America/New_York"), True, pd.Timestamp("2024-07-01")),
        # Stocks before the close: today's bar is incomplete.
        (
            pd.Timestamp("2024-07-02 10:00", tz="America/New_York"),
            False,
            pd.Timestamp("2024-07-01"),
        ),
        # Stocks at or after the 16:00 close: today's bar is complete.
        (
            pd.Timestamp("2024-07-02 16:00", tz="America/New_York"),
            False,
            pd.Timestamp("2024-07-02"),
        ),
        (
            pd.Timestamp("2024-07-02 17:00", tz="America/New_York"),
            False,
            pd.Timestamp("2024-07-02"),
        ),
        # Naive ``now`` is taken as UTC: 2024-07-02 21:00 UTC = 17:00 NY.
        (pd.Timestamp("2024-07-02 21:00"), False, pd.Timestamp("2024-07-02")),
    ],
)
def test_last_complete_date(now: pd.Timestamp, crypto: bool, expected: pd.Timestamp) -> None:
    result = last_complete_date(now, crypto)
    assert result == expected
    assert result.tzinfo is None


def test_crypto_bar_for_open_new_york_day_is_dropped() -> None:
    now = pd.Timestamp("2024-07-02 02:00", tz="UTC")  # 22:00 NY on 2024-07-01
    bars = _frame_ending_today(now)  # 2024-06-27 .. 2024-07-01; the last one is still open
    crypto = FakeCryptoClient(_alpaca_bars("BTC/USD", bars))
    frame = AlpacaSource(crypto_client=crypto, now=now).get_daily(
        "BTC-USD", bars.index[0], bars.index[-1]
    )
    assert frame.index[-1] == pd.Timestamp("2024-06-30")
    assert len(frame) == len(bars) - 1


def test_stock_bar_for_today_is_kept_after_the_close() -> None:
    now = pd.Timestamp("2024-07-02 17:00", tz="America/New_York")
    bars = _frame_ending_today(now)
    stock = FakeStockClient(_alpaca_bars("SPY", bars))
    frame = AlpacaSource(stock_client=stock, now=now).get_daily(
        "SPY", bars.index[0], bars.index[-1]
    )
    assert frame.index[-1] == pd.Timestamp("2024-07-02")
    assert len(frame) == len(bars)


def test_stock_bar_for_today_is_dropped_before_the_close() -> None:
    now = pd.Timestamp("2024-07-02 10:00", tz="America/New_York")
    bars = _frame_ending_today(now)
    stock = FakeStockClient(_alpaca_bars("SPY", bars))
    frame = AlpacaSource(stock_client=stock, now=now).get_daily(
        "SPY", bars.index[0], bars.index[-1]
    )
    assert frame.index[-1] == pd.Timestamp("2024-07-01")
    assert len(frame) == len(bars) - 1


def test_only_incomplete_bars_raises_naming_symbol() -> None:
    now = pd.Timestamp("2024-07-02 10:00", tz="America/New_York")
    bars = _frame_ending_today(now, n=1)
    stock = FakeStockClient(_alpaca_bars("SPY", bars))
    with pytest.raises(ValueError, match=r"SPY.*last complete session 2024-07-01"):
        AlpacaSource(stock_client=stock, now=now).get_daily("SPY", bars.index[0], bars.index[-1])


def test_partial_bar_never_reaches_the_cache(tmp_path: Path) -> None:
    now = pd.Timestamp("2024-07-02 02:00", tz="UTC")
    bars = _frame_ending_today(now)
    crypto = FakeCryptoClient(_alpaca_bars("BTC/USD", bars))
    cached = CachedSource(AlpacaSource(crypto_client=crypto, now=now), tmp_path)
    cached.get_daily("BTC-USD", bars.index[0], bars.index[-1])
    written = pd.read_parquet(next(tmp_path.rglob("*.parquet")))
    assert written.index[-1] == pd.Timestamp("2024-06-30")


# --------------------------------------------------------------------------- #
# Off-midnight stamps are refused (reviewer finding 2)
# --------------------------------------------------------------------------- #
def test_bars_to_frame_refuses_stamp_that_is_not_new_york_midnight() -> None:
    stamps = pd.DatetimeIndex(
        [pd.Timestamp("2024-07-01 04:00", tz="UTC"), pd.Timestamp("2024-07-02 12:00", tz="UTC")]
    )
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=stamps)
    with pytest.raises(ValueError, match=r"SPY.*2024-07-02 08:00:00-04:00.*New York midnight"):
        bars_to_frame(bars, "SPY")


def test_bars_to_frame_refuses_utc_midnight_stamp() -> None:
    # 00:00 UTC is 20:00 NY the previous evening; dating it would shift the bar a day early.
    stamps = pd.DatetimeIndex([pd.Timestamp("2024-07-02 00:00", tz="UTC")])
    with pytest.raises(ValueError, match="New York midnight"):
        bars_to_frame(pd.DataFrame({"close": [1.0]}, index=stamps), "SPY")


# --------------------------------------------------------------------------- #
# tz-aware bounds (reviewer finding 3)
# --------------------------------------------------------------------------- #
def test_utc_bound_accepts_naive_and_aware() -> None:
    assert utc_bound("2024-07-01") == pd.Timestamp("2024-07-01", tz="UTC")
    assert utc_bound(pd.Timestamp("2024-07-01", tz="UTC")) == pd.Timestamp("2024-07-01", tz="UTC")
    assert utc_bound(pd.Timestamp("2024-06-30 20:00", tz="America/New_York")) == pd.Timestamp(
        "2024-07-01", tz="UTC"
    )


def test_tz_aware_and_naive_bounds_give_the_same_request_and_frame() -> None:
    small = _small_frame(5)
    naive_client = FakeStockClient(_alpaca_bars("SPY", small))
    aware_client = FakeStockClient(_alpaca_bars("SPY", small))
    naive = AlpacaSource(stock_client=naive_client).get_daily(
        "SPY", pd.Timestamp("2024-07-02"), pd.Timestamp("2024-07-04")
    )
    aware = AlpacaSource(stock_client=aware_client).get_daily(
        "SPY", pd.Timestamp("2024-07-02", tz="UTC"), pd.Timestamp("2024-07-04", tz="UTC")
    )
    assert _as_utc(naive_client.requests[0].start) == _as_utc(aware_client.requests[0].start)
    assert _as_utc(naive_client.requests[0].end) == _as_utc(aware_client.requests[0].end)
    pd.testing.assert_frame_equal(naive, aware)
