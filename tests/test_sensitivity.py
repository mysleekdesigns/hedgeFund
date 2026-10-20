"""Tests for hedgefund.backtest.sensitivity: holdout split and the parameter grid."""

import math

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.metrics import TRADING_DAYS_PER_YEAR
from hedgefund.backtest.sensitivity import (
    DEFAULT_HOLDOUT_YEARS,
    DEFAULT_SENSITIVITY_THRESHOLDS,
    DEFAULT_SENSITIVITY_WINDOWS,
    holdout_cutoff,
    sensitivity_grid,
    split_holdout,
)

SEED = 20251019  # explicit seed for every random path in this file


def _close(n: int, start: str = "2015-01-01", seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n)),
        index=pd.bdate_range(start, periods=n),
    )


def test_holdout_cutoff_is_exactly_two_years_before_last_date():
    index = pd.bdate_range("2014-03-05", "2024-03-05")
    cutoff = holdout_cutoff(index)
    assert DEFAULT_HOLDOUT_YEARS == 2
    assert cutoff == pd.Timestamp("2022-03-05")
    assert holdout_cutoff(index, holdout_years=1) == pd.Timestamp("2023-03-05")


def test_holdout_cutoff_raises_on_empty_index():
    with pytest.raises(ValueError, match="empty"):
        holdout_cutoff(pd.DatetimeIndex([]))


def test_holdout_cutoff_raises_on_non_datetime_index_and_bad_years():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        holdout_cutoff(pd.RangeIndex(10))
    with pytest.raises(ValueError, match="holdout_years"):
        holdout_cutoff(pd.bdate_range("2020-01-01", periods=5), holdout_years=0)


def test_split_holdout_partitions_without_overlap_or_loss():
    close = _close(2520)  # ten trading years
    in_sample, holdout = split_holdout(close)
    cutoff = holdout_cutoff(close.index)
    assert len(in_sample) + len(holdout) == len(close)
    assert in_sample.index.intersection(holdout.index).empty
    assert in_sample.index.union(holdout.index).equals(close.index)
    assert (in_sample.index < cutoff).all()
    assert (holdout.index >= cutoff).all()
    pd.testing.assert_series_equal(pd.concat([in_sample, holdout]), close)
    assert len(holdout) > 0 and len(in_sample) > 0


def test_split_holdout_short_history_puts_everything_in_holdout():
    close = _close(300)  # just over one trading year, shorter than the two-year holdout
    in_sample, holdout = split_holdout(close)
    assert in_sample.empty
    pd.testing.assert_series_equal(holdout, close)


def test_sensitivity_grid_shape_and_call_pattern():
    close = _close(500)
    windows = (10, 20, 60)
    thresholds = (0.02, 0.05)
    calls: list[tuple[int, float]] = []

    def score(c: pd.Series, window: int, threshold: float) -> float:
        assert c is close
        calls.append((window, threshold))
        return window * threshold

    result = sensitivity_grid(close, score, windows=windows, thresholds=thresholds)
    assert result["n_trials"] == len(windows) * len(thresholds)
    assert len(calls) == result["n_trials"]
    # Row-major order: every threshold for the first window, then the next window.
    assert calls == [(w, t) for w in windows for t in thresholds]
    grid = result["net_sharpe"]
    assert len(grid) == len(windows)
    assert all(len(row) == len(thresholds) for row in grid)
    assert result["windows"] == list(windows)
    assert result["thresholds"] == list(thresholds)
    assert grid == [[w * t for t in thresholds] for w in windows]


def test_sensitivity_grid_best_is_argmax_over_finite_cells():
    close = _close(50)
    table = {
        (10, 0.02): 0.3,
        (10, 0.05): math.nan,
        (20, 0.02): 1.7,
        (20, 0.05): -0.4,
    }

    def score(c: pd.Series, window: int, threshold: float) -> float:
        return table[(window, threshold)]

    result = sensitivity_grid(close, score, windows=(10, 20), thresholds=(0.02, 0.05))
    assert result["best"] == {"window": 20, "threshold": 0.02, "net_sharpe": 1.7}
    assert result["share_positive"] == pytest.approx(2 / 3)  # 0.3 and 1.7 of three finite cells


def test_sensitivity_grid_all_nan_gives_none_best_and_nan_summaries():
    close = _close(50)
    result = sensitivity_grid(
        close, lambda c, w, t: math.nan, windows=(10, 20), thresholds=(0.02, 0.05)
    )
    assert result["best"] == {"window": None, "threshold": None, "net_sharpe": None}
    assert math.isnan(result["share_positive"])
    assert math.isnan(result["sharpe_variance"])
    assert result["n_trials"] == 4


def test_sensitivity_grid_sharpe_variance_hand_checked_on_tiny_grid():
    close = _close(50)
    annualised = {(10, 0.05): 1.0, (20, 0.05): 2.0, (60, 0.05): math.nan}

    def score(c: pd.Series, window: int, threshold: float) -> float:
        return annualised[(window, threshold)]

    result = sensitivity_grid(close, score, windows=(10, 20, 60), thresholds=(0.05,))
    # Per-period Sharpes are 1/sqrt(252) and 2/sqrt(252); ddof=1 variance of two
    # points is (difference^2) / 2 = (1/sqrt(252))^2 / 2 = 1 / 504.
    assert result["sharpe_variance"] == pytest.approx(1.0 / (2 * TRADING_DAYS_PER_YEAR))
    # A single finite cell has no sample variance.
    single = sensitivity_grid(close, score, windows=(10, 60), thresholds=(0.05,))
    assert math.isnan(single["sharpe_variance"])
    assert single["best"]["window"] == 10


def test_sensitivity_grid_defaults_cover_plan_grid():
    assert DEFAULT_SENSITIVITY_WINDOWS == (10, 20, 60)
    assert DEFAULT_SENSITIVITY_THRESHOLDS == (0.02, 0.03, 0.05, 0.07, 0.10)
    close = _close(50)
    result = sensitivity_grid(close, lambda c, w, t: 0.0)
    assert result["n_trials"] == 15
    assert result["share_positive"] == 0.0  # zeros are finite but not positive


def test_sensitivity_grid_rejects_empty_windows_or_thresholds():
    close = _close(50)
    with pytest.raises(ValueError, match="windows"):
        sensitivity_grid(close, lambda c, w, t: 0.0, windows=(), thresholds=(0.05,))
    with pytest.raises(ValueError, match="thresholds"):
        sensitivity_grid(close, lambda c, w, t: 0.0, windows=(20,), thresholds=())
