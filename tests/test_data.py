"""Tests for the data layer: PriceSource protocol, CSV loader, yfinance port.

All offline. yfinance is replaced by a fake module in ``sys.modules`` so the lazy import
inside the download picks it up; the conftest socket guard would fail any real call.
"""

import re
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.data.base import PRICE_COLUMNS, PriceSource, close_series, validate_price_frame
from hedgefund.data.csv_source import CsvSource, load_csv
from hedgefund.data.yfinance_source import YFinanceSource, fetch_ticker

FIXTURES = Path(__file__).parent / "fixtures"
SPY_PARQUET = FIXTURES / "spy_10y.parquet"
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _good_frame(n: int = 5) -> pd.DataFrame:
    """A frame that satisfies validate_price_frame."""
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="B"), name="date")
    return pd.DataFrame({"close": np.linspace(100.0, 101.0, n)}, index=index)


def _yf_flat_frame(n: int = 5) -> pd.DataFrame:
    """Shape of a yfinance download with flat columns."""
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="B"), name="Date")
    base = np.linspace(100.0, 104.0, n)
    return pd.DataFrame(
        {
            "Open": base - 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": np.arange(1, n + 1) * 1000,
        },
        index=index,
    )


def _yf_multiindex_frame(ticker: str, n: int = 5) -> pd.DataFrame:
    """Shape of a yfinance download with (Price, Ticker) MultiIndex columns."""
    flat = _yf_flat_frame(n)
    flat.columns = pd.MultiIndex.from_product([flat.columns, [ticker]], names=["Price", "Ticker"])
    return flat


class FakeYFinance:
    """Stands in for the yfinance module; serves frames from a queue and records calls."""

    def __init__(self, frames: list[pd.DataFrame | Exception]) -> None:
        self.frames = list(frames)
        self.calls: list[dict[str, object]] = []

    def download(self, ticker: str, **kwargs: object) -> pd.DataFrame:
        self.calls.append({"ticker": ticker, **kwargs})
        result = self.frames.pop(0) if self.frames else pd.DataFrame()
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_yf(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[pd.DataFrame | Exception]], FakeYFinance]:
    """Install a fake ``yfinance`` module and return the factory."""

    def install(frames: list[pd.DataFrame | Exception]) -> FakeYFinance:
        fake = FakeYFinance(frames)
        module = types.ModuleType("yfinance")
        module.download = fake.download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "yfinance", module)
        return fake

    return install


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return pd.read_parquet(SPY_PARQUET)


@pytest.fixture
def spy_csv(tmp_path: Path, spy: pd.DataFrame) -> Iterator[Path]:
    """The fixture close exported as ``date,close``, exactly how the gate feeds the companion."""
    path = tmp_path / "spy_10y.csv"
    spy["close"].to_csv(path, header=["close"], index_label="date")
    yield path


# --------------------------------------------------------------------------- #
# load_csv
# --------------------------------------------------------------------------- #
def test_load_csv_date_close(tmp_path: Path) -> None:
    path = _write(tmp_path, "prices.csv", "date,close\n2024-01-02,100.5\n2024-01-03,101.0\n")
    close = load_csv(path)
    assert close.name == "prices"
    assert isinstance(close.index, pd.DatetimeIndex)
    assert close.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert close.tolist() == [100.5, 101.0]


def test_load_csv_capitalised_adj_close(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "x.csv",
        "Date,Open,Adj Close\n2024-01-02,99.0,100.5\n2024-01-03,100.0,101.0\n",
    )
    close = load_csv(path)
    # "adj close" wins over the unrecognised "open" even though open comes first.
    assert close.tolist() == [100.5, 101.0]


def test_load_csv_close_beats_adj_close(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.csv", "date,adj close,close\n2024-01-02,1.0,2.0\n")
    assert load_csv(path).tolist() == [2.0]


def test_load_csv_single_unnamed_numeric_column(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.csv", "when,px\n2024-01-02,100.5\n2024-01-03,101.0\n")
    close = load_csv(path)
    assert close.tolist() == [100.5, 101.0]


def test_load_csv_first_column_is_date_fallback(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.csv", "when,close\n2024-01-03,101.0\n2024-01-02,100.5\n")
    close = load_csv(path)
    assert close.index[0] == pd.Timestamp("2024-01-02")


def test_load_csv_ambiguous_numeric_columns_raise(tmp_path: Path) -> None:
    path = _write(tmp_path, "amb.csv", "date,a,b\n2024-01-02,1.0,2.0\n")
    with pytest.raises(ValueError, match=r"close column in .*amb\.csv"):
        load_csv(path)


def test_load_csv_empty_file_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.csv", "")
    with pytest.raises(ValueError, match=r"empty\.csv is empty"):
        load_csv(path)


def test_load_csv_header_only_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "hdr.csv", "date,close\n")
    with pytest.raises(ValueError, match=r"hdr\.csv is empty"):
        load_csv(path)


def test_load_csv_no_usable_rows_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.csv", "date,close\nnot-a-date,1.0\n")
    with pytest.raises(ValueError, match=r"No usable rows after parsing .*bad\.csv"):
        load_csv(path)


def test_load_csv_sorts_unsorted_dates(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "x.csv",
        "date,close\n2024-01-04,3.0\n2024-01-02,1.0\n2024-01-03,2.0\n",
    )
    close = load_csv(path)
    assert close.index.is_monotonic_increasing
    assert close.tolist() == [1.0, 2.0, 3.0]


def test_load_csv_drops_unparsable_dates(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "x.csv",
        "date,close\n2024-01-02,1.0\ngarbage,2.0\n2024-01-03,3.0\n",
    )
    close = load_csv(path)
    assert close.tolist() == [1.0, 3.0]


def test_load_csv_drops_non_numeric_closes(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.csv", "date,close\n2024-01-02,1.0\n2024-01-03,n/a\n")
    close = load_csv(path)
    assert close.tolist() == [1.0]


def test_load_csv_round_trips_spy_fixture(spy_csv: Path, spy: pd.DataFrame) -> None:
    close = load_csv(spy_csv)
    assert close.name == "spy_10y"
    assert close.index.equals(pd.DatetimeIndex(spy.index, name=close.index.name))
    assert np.array_equal(close.to_numpy(), spy["close"].to_numpy())


# --------------------------------------------------------------------------- #
# CsvSource
# --------------------------------------------------------------------------- #
def test_csv_source_get_daily_is_valid_frame(spy_csv: Path) -> None:
    frame = CsvSource().get_daily(str(spy_csv))
    assert validate_price_frame(frame) is frame
    assert list(frame.columns) == ["close"]
    close = close_series(frame)
    assert close.dtype == "float64"
    assert close.name == "close"
    assert close_series(frame, "SPY").name == "SPY"


def test_csv_source_round_trips_spy_fixture(spy_csv: Path, spy: pd.DataFrame) -> None:
    frame = CsvSource().get_daily(str(spy_csv))
    assert np.array_equal(frame["close"].to_numpy(), spy["close"].to_numpy())
    assert frame.index.equals(spy.index)


def test_csv_source_slices_inclusive(spy_csv: Path) -> None:
    frame = CsvSource().get_daily(
        str(spy_csv), start=pd.Timestamp("2020-01-02"), end=pd.Timestamp("2020-01-10")
    )
    assert frame.index[0] == pd.Timestamp("2020-01-02")
    assert frame.index[-1] == pd.Timestamp("2020-01-10")
    assert len(frame) == 7  # trading days 2, 3, 6, 7, 8, 9, 10 January 2020


def test_csv_source_keeps_optional_ohlcv_columns(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "ohlcv.csv",
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,1.0,2.0,0.5,1.5,100\n"
        "2024-01-03,1.5,2.5,1.0,2.0,200\n",
    )
    frame = CsvSource().get_daily(str(path))
    assert list(frame.columns) == PRICE_COLUMNS
    assert frame["volume"].dtype == "int64"
    assert frame["open"].dtype == "float64"
    assert validate_price_frame(frame) is frame


def test_csv_source_drops_duplicate_dates_keeping_first(tmp_path: Path) -> None:
    path = _write(tmp_path, "dup.csv", "date,close\n2024-01-02,1.0\n2024-01-02,9.0\n")
    frame = CsvSource().get_daily(str(path))
    assert frame["close"].tolist() == [1.0]


def test_csv_source_empty_slice_raises(spy_csv: Path) -> None:
    with pytest.raises(ValueError, match="no rows between"):
        CsvSource().get_daily(str(spy_csv), start=pd.Timestamp("2030-01-01"))


# --------------------------------------------------------------------------- #
# fetch_ticker / YFinanceSource
# --------------------------------------------------------------------------- #
def test_fetch_ticker_flat_frame(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake = fake_yf([_yf_flat_frame()])
    sleep = RecordingSleep()
    close = fetch_ticker("SPY", sleep=sleep)
    assert close.name == "SPY"
    assert close.tolist() == pytest.approx(np.linspace(100.0, 104.0, 5).tolist())
    assert sleep.calls == []
    assert len(fake.calls) == 1


def test_fetch_ticker_multiindex_frame(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake_yf([_yf_multiindex_frame("SPY")])
    close = fetch_ticker("SPY", sleep=RecordingSleep())
    assert isinstance(close, pd.Series)
    assert close.name == "SPY"
    assert close.tolist() == pytest.approx(np.linspace(100.0, 104.0, 5).tolist())


def test_fetch_ticker_download_kwargs_are_companion_faithful(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    fake = fake_yf([_yf_flat_frame()])
    fetch_ticker("SPY", years=10, sleep=RecordingSleep())
    call = fake.calls[0]
    assert call["ticker"] == "SPY"
    assert call["auto_adjust"] is True
    assert call["progress"] is False
    assert ISO_DATE.match(call["start"])
    assert ISO_DATE.match(call["end"])
    end = pd.Timestamp(call["end"])
    start = pd.Timestamp(call["start"])
    assert end == pd.Timestamp.now("UTC").tz_localize(None).normalize()
    assert start == end - pd.DateOffset(years=10)


def test_fetch_ticker_retries_once_after_empty(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake = fake_yf([pd.DataFrame(), _yf_flat_frame()])
    sleep = RecordingSleep()
    close = fetch_ticker("SPY", retries=2, backoff_seconds=7.5, sleep=sleep)
    assert len(close) == 5
    assert len(fake.calls) == 2
    assert sleep.calls == [7.5]


def test_fetch_ticker_retries_after_exception(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake = fake_yf([RuntimeError("boom"), _yf_flat_frame()])
    sleep = RecordingSleep()
    close = fetch_ticker("SPY", retries=2, backoff_seconds=1.0, sleep=sleep)
    assert len(close) == 5
    assert len(fake.calls) == 2
    assert sleep.calls == [1.0]


def test_fetch_ticker_always_empty_raises_after_retries(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    fake = fake_yf([])
    sleep = RecordingSleep()
    with pytest.raises(ValueError, match=r"empty data for SPY after 3 attempt"):
        fetch_ticker("SPY", retries=3, backoff_seconds=2.0, sleep=sleep)
    assert len(fake.calls) == 3
    assert sleep.calls == [2.0, 2.0]  # no sleep after the final attempt


def test_fetch_ticker_rejects_zero_retries(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake_yf([_yf_flat_frame()])
    with pytest.raises(ValueError, match="retries"):
        fetch_ticker("SPY", retries=0, sleep=RecordingSleep())


def test_yfinance_source_get_daily_matches_fixture_schema(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    fake_yf([_yf_multiindex_frame("SPY")])
    frame = YFinanceSource(sleep=RecordingSleep()).get_daily("SPY")
    assert list(frame.columns) == PRICE_COLUMNS
    assert frame.index.name == "date"
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame["close"].dtype == "float64"
    assert frame["volume"].dtype == "int64"
    assert validate_price_frame(frame) is frame


def test_yfinance_source_strips_timezone_and_duplicates(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    raw = _yf_flat_frame(4)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("America/New_York")
    raw = pd.concat([raw, raw.iloc[[1]]]).iloc[::-1]  # duplicate one row, reverse order
    fake_yf([raw])
    frame = YFinanceSource(sleep=RecordingSleep()).get_daily("SPY")
    assert frame.index.tz is None
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 4


def test_yfinance_source_explicit_window_is_inclusive(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    fake = fake_yf([_yf_flat_frame(5)])  # 2024-01-01 .. 2024-01-05
    frame = YFinanceSource(sleep=RecordingSleep()).get_daily(
        "SPY", start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-04")
    )
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    assert frame.index[-1] == pd.Timestamp("2024-01-04")
    # yfinance's end is exclusive, so the request asks for one extra day.
    assert fake.calls[0]["start"] == "2024-01-02"
    assert fake.calls[0]["end"] == "2024-01-05"


def test_yfinance_source_default_window_is_years_apart(
    fake_yf: Callable[..., FakeYFinance],
) -> None:
    fake = fake_yf([_yf_flat_frame()])
    YFinanceSource(years=3, sleep=RecordingSleep()).get_daily("SPY")
    call = fake.calls[0]
    assert ISO_DATE.match(call["start"]) and ISO_DATE.match(call["end"])
    assert pd.Timestamp(call["start"]) == pd.Timestamp(call["end"]) - pd.DateOffset(years=3)


def test_yfinance_source_always_empty_raises(fake_yf: Callable[..., FakeYFinance]) -> None:
    fake_yf([])
    sleep = RecordingSleep()
    with pytest.raises(ValueError, match="empty data for QQQ after 2 attempt"):
        YFinanceSource(retries=2, backoff_seconds=0.5, sleep=sleep).get_daily("QQQ")
    assert sleep.calls == [0.5]


# --------------------------------------------------------------------------- #
# validate_price_frame / close_series / protocol
# --------------------------------------------------------------------------- #
def test_validate_price_frame_accepts_fixture(spy: pd.DataFrame) -> None:
    assert validate_price_frame(spy) is spy


def test_validate_price_frame_accepts_good_frame() -> None:
    frame = _good_frame()
    assert validate_price_frame(frame) is frame


def test_validate_price_frame_rejects_tz_aware_index() -> None:
    frame = _good_frame()
    frame.index = frame.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-naive"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_unsorted_index() -> None:
    frame = _good_frame().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_duplicate_dates() -> None:
    frame = _good_frame()
    frame = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_nan_close() -> None:
    frame = _good_frame()
    frame.iloc[2, frame.columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_non_float_close() -> None:
    frame = _good_frame().astype({"close": "int64"})
    with pytest.raises(ValueError, match="float64"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_wrong_index_name() -> None:
    frame = _good_frame()
    frame.index.name = "Date"
    with pytest.raises(ValueError, match="named 'date'"):
        validate_price_frame(frame)


def test_validate_price_frame_rejects_missing_close() -> None:
    frame = _good_frame().rename(columns={"close": "px"})
    with pytest.raises(ValueError, match="'close' column"):
        validate_price_frame(frame)


def test_close_series_requires_close_column() -> None:
    with pytest.raises(ValueError, match="'close' column"):
        close_series(pd.DataFrame({"px": [1.0]}))


def test_price_source_protocol_runtime_check() -> None:
    assert isinstance(CsvSource(), PriceSource)
    assert isinstance(YFinanceSource(), PriceSource)
    assert CsvSource().name == "csv"
    assert YFinanceSource().name == "yfinance"
    assert not isinstance(object(), PriceSource)
