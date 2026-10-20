"""Tests for the parquet price cache and the directory CSV source.

All offline. A counting fake stands in for the network source, so the tests can prove
the cache layer alone decides whether the inner source is called.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.data.base import PriceSource, validate_price_frame
from hedgefund.data.cache import (
    DEFAULT_CACHE_DIR,
    CachedSource,
    CacheEvent,
    CacheMissError,
    cache_key,
    cache_path,
    file_fingerprint,
    frame_digest,
    read_cached,
    write_cached,
)
from hedgefund.data.csv_source import CsvDirSource
from hedgefund.data.yfinance_source import ONE_DAY, default_window

FIXTURES = Path(__file__).parent / "fixtures"
SPY_PARQUET = FIXTURES / "spy_10y.parquet"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-03-31")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _frame(n: int = 40, seed: int = 0, with_volume: bool = True) -> pd.DataFrame:
    """A validated OHLCV frame starting at START."""
    rng = np.random.default_rng(seed)
    # Real sources carry no ``freq``; parquet drops it too, so build the index without one.
    index = pd.DatetimeIndex(pd.date_range(START, periods=n, freq="B").to_numpy(), name="date")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    data = {
        "open": close * (1.0 - 0.001),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    }
    if with_volume:
        data["volume"] = (rng.integers(1_000, 10_000, n)).astype("int64")
    return validate_price_frame(pd.DataFrame(data, index=index))


class CountingSource:
    """Fake PriceSource that records every call and serves a deterministic frame."""

    def __init__(self, name: str = "fake", frame: pd.DataFrame | None = None) -> None:
        self.name = name
        self.frame = _frame() if frame is None else frame
        self.calls: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        frame = self.frame
        if start is not None or end is not None:
            frame = frame.loc[start:end]
        return frame.copy()


class FailingSource:
    """Fake PriceSource whose fetch always fails, like a rate-limited Yahoo."""

    name = "failing"

    def get_daily(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        raise ValueError(f"no data for {symbol}")


def _csv_text(frame: pd.DataFrame) -> str:
    out = frame.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False)


# --------------------------------------------------------------------------- #
# frame_digest
# --------------------------------------------------------------------------- #
def test_frame_digest_is_sha256_hex_and_deterministic() -> None:
    a = _frame()
    b = _frame()
    assert frame_digest(a) == frame_digest(b)
    assert len(frame_digest(a)) == 64
    assert all(c in "0123456789abcdef" for c in frame_digest(a))


def test_frame_digest_changes_with_one_close_value() -> None:
    a = _frame()
    b = a.copy()
    b.iloc[10, b.columns.get_loc("close")] += 1e-9
    assert frame_digest(a) != frame_digest(b)


def test_frame_digest_changes_with_one_date() -> None:
    a = _frame()
    b = a.copy()
    index = b.index.to_list()
    index[-1] = index[-1] + pd.Timedelta(days=1)
    b.index = pd.DatetimeIndex(index, name="date")
    assert frame_digest(a) != frame_digest(b)


def test_frame_digest_ignores_non_close_columns() -> None:
    """Only dates and closes feed the pipeline, so only they are hashed."""
    a = _frame()
    b = a.copy()
    b["volume"] = b["volume"] + 1
    assert frame_digest(a) == frame_digest(b)


def test_frame_digest_matches_known_value() -> None:
    """Pin the recipe: int64 nanoseconds of the index, then float64 closes.

    pandas 3 builds string-parsed indexes in microseconds; the digest must still hash
    nanoseconds so the same dates give the same digest whatever unit the index carries.
    """
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date")
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=index)
    seconds = np.array([1_704_153_600, 1_704_240_000], dtype="int64")  # 2024-01-02, 2024-01-03 UTC
    expected = hashlib.sha256()
    expected.update((seconds * 1_000_000_000).tobytes())
    expected.update(np.array([1.0, 2.0], dtype="float64").tobytes())
    assert frame_digest(df) == expected.hexdigest()
    assert frame_digest(df.set_index(index.as_unit("ns"))) == expected.hexdigest()


# --------------------------------------------------------------------------- #
# Keys and paths
# --------------------------------------------------------------------------- #
def test_cache_key_format() -> None:
    assert cache_key("yfinance", "SPY", START, END) == "yfinance/SPY_2024-01-01_2024-03-31"


def test_cache_key_sanitises_slash_in_symbol() -> None:
    key = cache_key("yfinance", "BTC/USD", START, END)
    assert key == "yfinance/BTC-USD_2024-01-01_2024-03-31"
    path = cache_path(Path("c"), "yfinance", "BTC/USD", START, END)
    assert path.parent == Path("c") / "yfinance"
    assert path.name == "BTC-USD_2024-01-01_2024-03-31.parquet"


def test_cache_key_rejects_empty_or_dot_components() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cache_key("yfinance", "", START, END)
    with pytest.raises(ValueError, match="not a usable file name"):
        cache_key("yfinance", "..", START, END)
    with pytest.raises(ValueError, match="start"):
        cache_key("yfinance", "SPY", END, START)


def test_cache_path_differs_by_symbol_range_and_source(tmp_path: Path) -> None:
    base = cache_path(tmp_path, "yfinance", "SPY", START, END)
    assert base == tmp_path / "yfinance" / "SPY_2024-01-01_2024-03-31.parquet"
    assert cache_path(tmp_path, "yfinance", "QQQ", START, END) != base
    assert cache_path(tmp_path, "yfinance", "SPY", START, END + pd.Timedelta(days=1)) != base
    assert cache_path(tmp_path, "yfinance", "SPY", START - pd.Timedelta(days=1), END) != base
    assert cache_path(tmp_path, "csv", "SPY", START, END) != base


def test_default_cache_dir_matches_config() -> None:
    import yaml

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())
    assert Path(cfg["data"]["cache_dir"]) == DEFAULT_CACHE_DIR


# --------------------------------------------------------------------------- #
# write_cached / read_cached
# --------------------------------------------------------------------------- #
def test_parquet_round_trip_is_exact_and_valid(tmp_path: Path) -> None:
    frame = _frame()
    path = tmp_path / "nested" / "dir" / "x.parquet"
    write_cached(path, frame)
    assert path.is_file()
    back = read_cached(path)
    pd.testing.assert_frame_equal(back, frame, check_exact=True)
    assert back.index.name == "date"
    assert back.index.tz is None
    assert back["close"].dtype == np.float64
    assert back["volume"].dtype == np.int64


def test_parquet_round_trip_of_fixture(tmp_path: Path) -> None:
    frame = validate_price_frame(pd.read_parquet(SPY_PARQUET))
    path = tmp_path / "spy.parquet"
    write_cached(path, frame)
    back = read_cached(path)
    pd.testing.assert_frame_equal(back, frame, check_exact=True)
    assert frame_digest(back) == frame_digest(frame)


def test_write_cached_rejects_invalid_frame_and_writes_nothing(tmp_path: Path) -> None:
    bad = _frame()
    bad.index.name = "Date"
    path = tmp_path / "bad.parquet"
    with pytest.raises(ValueError, match="named 'date'"):
        write_cached(path, bad)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_read_cached_validates(tmp_path: Path) -> None:
    index = pd.DatetimeIndex(pd.date_range(START, periods=3, freq="B"), name="date")
    pd.DataFrame({"close": [1, 2, 3]}, index=index).to_parquet(tmp_path / "int.parquet")
    with pytest.raises(ValueError, match="float64"):
        read_cached(tmp_path / "int.parquet")
    with pytest.raises(FileNotFoundError, match=r"missing\.parquet"):
        read_cached(tmp_path / "missing.parquet")


# --------------------------------------------------------------------------- #
# CachedSource
# --------------------------------------------------------------------------- #
def test_cached_source_is_a_price_source(tmp_path: Path) -> None:
    inner = CountingSource(name="yfinance")
    cached = CachedSource(inner, tmp_path)
    assert isinstance(cached, PriceSource)
    assert cached.name == "yfinance"
    assert isinstance(CsvDirSource(tmp_path), PriceSource)


def test_cold_then_warm_calls_inner_once(tmp_path: Path) -> None:
    inner = CountingSource(name="yfinance")
    cached = CachedSource(inner, tmp_path)

    cold = cached.get_daily("SPY", START, END)
    assert len(inner.calls) == 1
    assert inner.calls[0] == ("SPY", START, END)
    assert (cached.hits, cached.misses) == (0, 1)

    warm = cached.get_daily("SPY", START, END)
    assert len(inner.calls) == 1
    assert (cached.hits, cached.misses) == (1, 1)
    pd.testing.assert_frame_equal(warm, cold, check_exact=True)

    expected_path = tmp_path / "yfinance" / "SPY_2024-01-01_2024-03-31.parquet"
    assert expected_path.is_file()
    assert [e.hit for e in cached.events] == [False, True]
    assert all(isinstance(e, CacheEvent) for e in cached.events)
    assert cached.events[0].path == expected_path
    assert cached.events[1].path == expected_path
    assert cached.events[0].digest == cached.events[1].digest == frame_digest(cold)
    assert cached.events[0].rows == len(cold)
    assert cached.events[0].start == START and cached.events[0].end == END


def test_warm_cache_across_instances_never_calls_inner(tmp_path: Path) -> None:
    """A fresh process with a populated cache dir makes zero inner calls."""
    first_inner = CountingSource(name="yfinance")
    cold = CachedSource(first_inner, tmp_path).get_daily("SPY", START, END)

    second_inner = CountingSource(name="yfinance")
    second = CachedSource(second_inner, tmp_path)
    warm = second.get_daily("SPY", START, END)
    assert second_inner.calls == []
    assert (second.hits, second.misses) == (1, 0)
    pd.testing.assert_frame_equal(warm, cold, check_exact=True)


def test_none_bounds_resolve_to_default_window_ending_yesterday(tmp_path: Path) -> None:
    """The companion window is today-exclusive, so the inclusive cached end is yesterday."""
    years = 3
    inner = CountingSource(name="yfinance")
    cached = CachedSource(inner, tmp_path, years=years)
    start, today = default_window(years)
    end = today - ONE_DAY

    cached.get_daily("SPY")
    assert inner.calls == [("SPY", start, end)]
    assert end == pd.Timestamp.now("UTC").tz_localize(None).normalize() - ONE_DAY
    expected = cache_path(tmp_path, "yfinance", "SPY", start, end)
    assert cached.events[0].path == expected
    assert expected.is_file()
    assert cached.path_for("SPY") == expected

    cached.get_daily("SPY")
    assert len(inner.calls) == 1
    assert cached.hits == 1


def test_default_window_never_requests_or_stores_todays_bar(tmp_path: Path) -> None:
    """A partial intraday bar for today must not be frozen into the cache."""
    _, today = default_window(1)
    index = pd.DatetimeIndex(pd.date_range(end=today, periods=30, freq="D").to_numpy(), name="date")
    frame = validate_price_frame(
        pd.DataFrame({"close": np.linspace(100.0, 110.0, len(index))}, index=index)
    )
    assert frame.index[-1] == today
    inner = CountingSource(name="yfinance", frame=frame)
    cached = CachedSource(inner, tmp_path, years=1)

    out = cached.get_daily("SPY")
    assert inner.calls[0][2] == today - ONE_DAY
    assert out.index.max() == today - ONE_DAY
    stored = read_cached(cached.events[0].path)
    assert today not in stored.index
    assert cached.events[0].end == today - ONE_DAY


def test_different_symbols_ranges_and_sources_cache_separately(tmp_path: Path) -> None:
    yf = CountingSource(name="yfinance")
    csv = CountingSource(name="csv")
    cached_yf = CachedSource(yf, tmp_path)
    cached_csv = CachedSource(csv, tmp_path)

    cached_yf.get_daily("SPY", START, END)
    cached_yf.get_daily("QQQ", START, END)
    cached_yf.get_daily("SPY", START, END - pd.Timedelta(days=7))
    cached_csv.get_daily("SPY", START, END)
    assert len(yf.calls) == 3
    assert len(csv.calls) == 1
    paths = {e.path for e in cached_yf.events} | {e.path for e in cached_csv.events}
    assert len(paths) == 4
    assert cached_yf.misses == 3 and cached_csv.misses == 1


def test_symbol_with_slash_gets_safe_filename(tmp_path: Path) -> None:
    inner = CountingSource(name="yfinance")
    cached = CachedSource(inner, tmp_path)
    cached.get_daily("BTC/USD", START, END)
    path = cached.events[0].path
    assert path == tmp_path / "yfinance" / "BTC-USD_2024-01-01_2024-03-31.parquet"
    assert path.is_file()
    assert "/" not in path.name


def test_offline_miss_raises_cache_miss_error_naming_symbol_and_path(tmp_path: Path) -> None:
    inner = CountingSource(name="yfinance")
    cached = CachedSource(inner, tmp_path, offline=True)
    expected = cache_path(tmp_path, "yfinance", "SPY", START, END)
    with pytest.raises(CacheMissError) as info:
        cached.get_daily("SPY", START, END)
    assert "SPY" in str(info.value)
    assert str(expected) in str(info.value)
    assert isinstance(info.value, LookupError)
    assert inner.calls == []
    assert cached.events == []


def test_offline_hit_is_served(tmp_path: Path) -> None:
    online_inner = CountingSource(name="yfinance")
    cold = CachedSource(online_inner, tmp_path).get_daily("SPY", START, END)

    offline_inner = CountingSource(name="yfinance")
    offline = CachedSource(offline_inner, tmp_path, offline=True)
    warm = offline.get_daily("SPY", START, END)
    assert offline_inner.calls == []
    assert offline.hits == 1
    pd.testing.assert_frame_equal(warm, cold, check_exact=True)


def test_failed_fetch_writes_nothing_and_retries_next_time(tmp_path: Path) -> None:
    cached = CachedSource(FailingSource(), tmp_path)
    with pytest.raises(ValueError, match="no data for SPY"):
        cached.get_daily("SPY", START, END)
    assert cached.events == []
    assert not (tmp_path / "failing").exists()

    good = CountingSource(name="failing")
    recovered = CachedSource(good, tmp_path).get_daily("SPY", START, END)
    assert len(good.calls) == 1
    assert len(recovered) > 0


def test_cached_source_validates_inner_frames(tmp_path: Path) -> None:
    bad = _frame()
    bad["close"] = bad["close"].astype("float32")
    cached = CachedSource(CountingSource(name="yfinance", frame=bad), tmp_path)
    with pytest.raises(ValueError, match="float64"):
        cached.get_daily("SPY", START, END)
    assert cached.events == []


def test_cached_source_rejects_bad_years(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="years"):
        CachedSource(CountingSource(), tmp_path, years=0)


# --------------------------------------------------------------------------- #
# CsvDirSource
# --------------------------------------------------------------------------- #
def test_csv_dir_source_reads_symbol_file(tmp_path: Path) -> None:
    frame = _frame()
    (tmp_path / "SPY.csv").write_text(_csv_text(frame))
    source = CsvDirSource(tmp_path)
    assert source.name == "csv"
    assert source.path_for("SPY") == tmp_path / "SPY.csv"

    out = source.get_daily("SPY")
    pd.testing.assert_frame_equal(out, frame, check_exact=True)

    sliced = source.get_daily("SPY", START + pd.Timedelta(days=7), START + pd.Timedelta(days=14))
    assert sliced.index.min() >= START + pd.Timedelta(days=7)
    assert sliced.index.max() <= START + pd.Timedelta(days=14)
    assert 0 < len(sliced) < len(frame)


def test_csv_dir_source_missing_file_names_it(tmp_path: Path) -> None:
    source = CsvDirSource(tmp_path)
    with pytest.raises(ValueError) as info:
        source.get_daily("QQQ")
    assert "QQQ" in str(info.value)
    assert str(tmp_path / "QQQ.csv") in str(info.value)


def test_csv_dir_source_behind_cache(tmp_path: Path) -> None:
    """End to end: a CSV universe read through the cache hits on the second pass."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = _frame()
    csv = data_dir / "SPY.csv"
    csv.write_text(_csv_text(frame))
    cached = CachedSource(CsvDirSource(data_dir), tmp_path / "cache", years=1)

    first = cached.get_daily("SPY", START, END)
    second = cached.get_daily("SPY", START, END)
    assert (cached.hits, cached.misses) == (1, 1)
    pd.testing.assert_frame_equal(second, first, check_exact=True)
    pd.testing.assert_frame_equal(first, frame, check_exact=True)
    path = cached.events[0].path
    assert path.parent == tmp_path / "cache" / "csv"
    assert path.name == f"SPY_2024-01-01_2024-03-31_{file_fingerprint(csv)}.parquet"
    assert cached.events[1].path == path


def test_csv_fingerprint_is_short_sha256_of_bytes(tmp_path: Path) -> None:
    csv = tmp_path / "SPY.csv"
    csv.write_text(_csv_text(_frame()))
    source = CsvDirSource(tmp_path)
    fingerprint = source.fingerprint("SPY")
    assert fingerprint == hashlib.sha256(csv.read_bytes()).hexdigest()[:12]
    assert fingerprint == file_fingerprint(csv)
    assert source.fingerprint("QQQ") is None


def test_edited_csv_misses_with_a_new_path(tmp_path: Path) -> None:
    """Changing the CSV under the same window must not serve the stale parquet."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv = data_dir / "SPY.csv"
    csv.write_text(_csv_text(_frame(seed=0)))
    cached = CachedSource(CsvDirSource(data_dir), tmp_path / "cache", years=1)

    first = cached.get_daily("SPY", START, END)
    edited = _frame(seed=1)
    csv.write_text(_csv_text(edited))
    second = cached.get_daily("SPY", START, END)
    assert (cached.hits, cached.misses) == (0, 2)
    assert cached.events[1].path != cached.events[0].path
    assert cached.events[1].path.is_file() and cached.events[0].path.is_file()
    pd.testing.assert_frame_equal(second, edited, check_exact=True)
    assert frame_digest(second) != frame_digest(first)

    third = cached.get_daily("SPY", START, END)
    assert cached.hits == 1
    pd.testing.assert_frame_equal(third, edited, check_exact=True)


def test_fingerprint_absent_leaves_key_unchanged(tmp_path: Path) -> None:
    """Sources without ``fingerprint`` (yfinance) keep the plain key."""
    inner = CountingSource(name="yfinance")
    assert not hasattr(inner, "fingerprint")
    cached = CachedSource(inner, tmp_path)
    cached.get_daily("SPY", START, END)
    assert cached.events[0].path == tmp_path / "yfinance" / "SPY_2024-01-01_2024-03-31.parquet"


def test_fingerprint_must_be_str_or_none(tmp_path: Path) -> None:
    inner = CountingSource(name="odd")
    inner.fingerprint = lambda symbol: 42  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="fingerprint"):
        CachedSource(inner, tmp_path).get_daily("SPY", START, END)
