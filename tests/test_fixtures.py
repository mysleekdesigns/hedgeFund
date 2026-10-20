"""Schema checks for the frozen fixtures under tests/fixtures/."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SPY_PARQUET = FIXTURES / "spy_10y.parquet"
SPY_PROVENANCE = FIXTURES / "spy_10y.provenance.json"


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return pd.read_parquet(SPY_PARQUET)


def test_spy_fixture_has_price_source_schema(spy: pd.DataFrame) -> None:
    assert list(spy.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(spy.index, pd.DatetimeIndex)
    assert spy.index.name == "date"
    assert spy.index.tz is None
    assert spy.index.is_monotonic_increasing
    assert spy.index.is_unique
    assert spy["close"].dtype == "float64"
    assert not spy["close"].isna().any()
    assert (spy["close"] > 0).all()


def test_spy_fixture_spans_ten_years_of_trading_days(spy: pd.DataFrame) -> None:
    # ~252 trading days per year; 10 years lands between 2450 and 2550 rows.
    assert 2450 <= len(spy) <= 2550
    assert spy.index[0] == pd.Timestamp("2016-10-20")
    assert spy.index[-1] == pd.Timestamp("2026-10-16")


def test_spy_fixture_matches_recorded_provenance(spy: pd.DataFrame) -> None:
    provenance = json.loads(SPY_PROVENANCE.read_text())
    assert provenance["ticker"] == "SPY"
    assert provenance["download_kwargs"]["auto_adjust"] is True
    assert provenance["rows"] == len(spy)
    assert provenance["first_date"] == str(spy.index[0].date())
    assert provenance["last_date"] == str(spy.index[-1].date())
    assert provenance["sha256"] == hashlib.sha256(SPY_PARQUET.read_bytes()).hexdigest()
