"""Tests for hedgefund.backtest.lag: turning points and detection-lag distributions."""

import math
from functools import partial

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.lag import (
    DEFAULT_LAG_ORDER,
    detection_lag,
    lag_by_labeller,
    local_extrema,
)
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, label_regimes

SEED = 20251019  # explicit seed for every random path in this file
LAG_KEYS = {
    "n",
    "n_undetected",
    "mean",
    "median",
    "p25",
    "p75",
    "max",
    "lags",
    # Review fix W4: the false-positive side of the ledger.
    "n_flips",
    "false_flips",
    "false_flip_rate",
}


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _up_down_up(leg: int = 100, dip_start: int = 40, dip_len: int = 5) -> pd.Series:
    """Rise ``leg`` bars, fall ``leg`` bars, rise ``leg`` bars, one bar per unit.

    A short ``dip_len``-bar dip is carved into the first leg: a wiggle that is a
    local extreme only at a scale smaller than ``DEFAULT_LAG_ORDER``.
    """
    up = np.arange(leg, dtype=float)
    down = up[-1] - np.arange(1, leg + 1, dtype=float)
    up2 = down[-1] + np.arange(1, leg + 1, dtype=float)
    path = np.concatenate([up, down, up2]) + 100.0
    dip = np.arange(1, dip_len + 1, dtype=float) * 1.5  # steeper than the 1-per-bar ramp
    path[dip_start : dip_start + dip_len] -= dip
    return pd.Series(path, index=_dates(len(path)))


def _labels_like(close: pd.Series, fill: int = SIDEWAYS) -> pd.Series:
    return pd.Series(fill, index=close.index, dtype=int)


def test_local_extrema_finds_main_peak_and_trough_and_ignores_short_dip():
    close = _up_down_up()
    peaks, troughs = local_extrema(close, order=DEFAULT_LAG_ORDER)
    # The ramp changes direction at bar 99 (top) and bar 199 (bottom).
    assert peaks.tolist() == [99]
    assert troughs.tolist() == [199]
    # The 5-bar dip is an extreme only at a smaller order.
    small_peaks, small_troughs = local_extrema(close, order=2)
    assert len(small_peaks) > 1
    assert len(small_troughs) > 1
    assert 99 in small_peaks and 199 in small_troughs


def test_local_extrema_excludes_edges_without_a_full_window():
    close = _up_down_up()
    peaks, troughs = local_extrema(close, order=DEFAULT_LAG_ORDER)
    n = len(close)
    for pos in np.concatenate([peaks, troughs]):
        assert DEFAULT_LAG_ORDER <= pos < n - DEFAULT_LAG_ORDER
    # A series shorter than one full window has no interior point at all.
    short = close.iloc[: 2 * DEFAULT_LAG_ORDER]
    empty_peaks, empty_troughs = local_extrema(short, order=DEFAULT_LAG_ORDER)
    assert len(empty_peaks) == 0 and len(empty_troughs) == 0


def test_local_extrema_rejects_order_below_one():
    with pytest.raises(ValueError, match="order"):
        local_extrema(_up_down_up(), order=0)


def test_detection_lag_crash_lag_equals_bars_from_peak_to_first_bear():
    close = _up_down_up()
    k = 7
    labels = _labels_like(close)
    labels.iloc[99 + k :] = BEAR  # flips to Bear exactly k bars after the peak at 99
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    assert result["crash"]["lags"] == [k]
    assert result["crash"]["n"] == 1
    assert result["crash"]["n_undetected"] == 0
    assert result["crash"]["mean"] == result["crash"]["median"] == float(k)
    assert result["crash"]["p25"] == result["crash"]["p75"] == result["crash"]["max"] == float(k)


def test_detection_lag_recovery_lag_equals_bars_from_trough_to_first_bull():
    close = _up_down_up()
    k = 4
    labels = _labels_like(close)
    labels.iloc[199 + k :] = BULL  # flips to Bull exactly k bars after the trough at 199
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    assert result["recovery"]["lags"] == [k]
    assert result["recovery"]["n"] == 1
    assert result["recovery"]["n_undetected"] == 0


def test_detection_lag_ignores_bear_label_before_the_peak():
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[90:95] = BEAR  # noise before the peak must not count
    labels.iloc[110:] = BEAR
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    assert result["crash"]["lags"] == [110 - 99]


def test_detection_lag_censors_peak_when_trough_arrives_before_bear():
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[250:] = BEAR  # Bear only after the trough at 199: too late to count
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    assert result["crash"]["n"] == 0
    assert result["crash"]["n_undetected"] == 1
    assert result["crash"]["lags"] == []


def test_detection_lag_never_flipping_labeller_gives_nan_stats_and_all_undetected():
    close = _up_down_up()
    labels = _labels_like(close)  # Sideways forever
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    for direction in ("crash", "recovery"):
        stats = result[direction]
        assert stats["n"] == 0
        assert stats["n_undetected"] == 1
        assert stats["lags"] == []
        for key in ("mean", "median", "p25", "p75", "max"):
            assert math.isnan(stats[key])
    assert result["order"] == DEFAULT_LAG_ORDER
    assert "hindsight" in result["note"]


# ---- W4: false flips ------------------------------------------------------- #
def test_detection_lag_counts_a_bear_flip_with_no_peak_behind_it_as_false():
    """Two Bear entries: one 6 bars after the peak at 99 (matched), one at 250 with no
    peak within the preceding ``order`` bars (false). The lag stats are unchanged."""
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[105:115] = BEAR  # matched: peak at 99 is 6 bars back
    labels.iloc[250:260] = BEAR  # false: the only peak is 151 bars back
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    crash = result["crash"]
    assert crash["n_flips"] == 2
    assert crash["false_flips"] == 1
    assert crash["false_flip_rate"] == 0.5
    assert crash["lags"] == [6] and crash["n"] == 1  # existing keys untouched
    # No Bull label anywhere: the recovery side has nothing to count.
    recovery = result["recovery"]
    assert recovery["n_flips"] == 0 and recovery["false_flips"] == 0
    assert math.isnan(recovery["false_flip_rate"])


def test_detection_lag_counts_a_bull_flip_with_no_trough_behind_it_as_false():
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[50:60] = BULL  # false: no trough before bar 50 (the only one is at 199)
    labels.iloc[203:] = BULL  # matched: trough at 199 is 4 bars back
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    recovery = result["recovery"]
    assert recovery["n_flips"] == 2
    assert recovery["false_flips"] == 1
    assert recovery["false_flip_rate"] == 0.5
    assert recovery["lags"] == [4]


def test_detection_lag_never_flipping_labeller_has_no_flips_and_nan_rate():
    close = _up_down_up()
    labels = _labels_like(close)  # Sideways forever
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    for direction in ("crash", "recovery"):
        stats = result[direction]
        assert stats["n_flips"] == 0
        assert stats["false_flips"] == 0
        assert math.isnan(stats["false_flip_rate"])
        assert isinstance(stats["n_flips"], int) and isinstance(stats["false_flips"], int)


def test_detection_lag_flip_exactly_order_bars_after_peak_is_matched_one_more_is_false():
    """The match window is inclusive: a flip ``order`` bars after the peak is the fastest
    honest detection (the peak is only known then); ``order + 1`` bars is too late."""
    close = _up_down_up()
    at_order = _labels_like(close)
    at_order.iloc[99 + DEFAULT_LAG_ORDER :] = BEAR
    assert detection_lag(close, at_order, order=DEFAULT_LAG_ORDER)["crash"]["false_flips"] == 0
    one_late = _labels_like(close)
    one_late.iloc[99 + DEFAULT_LAG_ORDER + 1 :] = BEAR
    late = detection_lag(close, one_late, order=DEFAULT_LAG_ORDER)["crash"]
    assert late["n_flips"] == 1 and late["false_flips"] == 1 and late["false_flip_rate"] == 1.0


def test_detection_lag_flip_on_the_peak_bar_itself_is_matched():
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[99:110] = BEAR  # enters Bear on the peak bar: 0 bars after, inclusive window
    crash = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)["crash"]
    assert crash["n_flips"] == 1 and crash["false_flips"] == 0


def test_detection_lag_initial_regime_is_not_a_flip():
    """The first aligned bar has no predecessor, so starting in Bear is not an entry."""
    close = _up_down_up()
    labels = _labels_like(close)
    labels.iloc[:10] = BEAR
    crash = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)["crash"]
    assert crash["n_flips"] == 0 and crash["false_flips"] == 0
    assert math.isnan(crash["false_flip_rate"])


def test_detection_lag_counts_flips_on_the_aligned_labels():
    """A labeller that drops its first rows is counted on the common index: a Bear run
    that starts at the labels' first bar is the initial regime there, not a flip."""
    close = _up_down_up()
    window = 20
    labels = _labels_like(close).iloc[window:]
    labels.iloc[:5] = BEAR  # initial regime of the aligned frame
    labels.iloc[99 - window + 3 :] = BEAR  # a real entry, 3 bars after the peak
    crash = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)["crash"]
    assert crash["n_flips"] == 1 and crash["false_flips"] == 0


def test_detection_lag_aligns_labels_that_start_after_close():
    close = _up_down_up()
    window = 20
    k = 3
    labels = _labels_like(close).iloc[window:]  # a labeller drops its first ``window`` rows
    labels.iloc[99 - window + k :] = BEAR
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)
    # Positions shift by ``window`` in the aligned frame but the lag is unchanged.
    assert result["crash"]["lags"] == [k]


def test_detection_lag_raises_when_series_share_no_dates():
    close = _up_down_up()
    labels = pd.Series(BULL, index=pd.bdate_range("1990-01-01", periods=10), dtype=int)
    with pytest.raises(ValueError, match="share no dates"):
        detection_lag(close, labels)


def _next_opposite_distance(start: int, opposite: np.ndarray, n: int) -> int:
    later = opposite[opposite > start]
    return int(later[0] - start) if len(later) else n - start


def test_detection_lag_on_random_walk_with_real_labeller_is_bounded_by_next_extreme():
    rng = np.random.default_rng(SEED)
    n = 3000
    daily = rng.normal(0.0003, 0.012, size=n)  # drift and vol of a liquid equity index
    close = pd.Series(100.0 * np.cumprod(1.0 + daily), index=_dates(n))
    labels = label_regimes(close, window=20, threshold=0.05)
    result = detection_lag(close, labels, order=DEFAULT_LAG_ORDER)

    common = close.index.intersection(labels.index)
    peaks, troughs = local_extrema(close.loc[common], order=DEFAULT_LAG_ORDER)
    lab = labels.loc[common].to_numpy()
    m = len(common)

    for direction, starts, opposite, flip_state in (
        ("crash", peaks, troughs, BEAR),
        ("recovery", troughs, peaks, BULL),
    ):
        stats = result[direction]
        assert stats["n"] > 0, f"{direction} lag distribution is empty on a 3000-bar walk"
        assert stats["n"] + stats["n_undetected"] == len(starts)
        assert len(stats["lags"]) == stats["n"]
        # Reference: loop over each start and reproduce the rule by hand.
        expected = []
        for s in starts:
            flips = np.flatnonzero(lab[s + 1 :] == flip_state)
            bound = _next_opposite_distance(int(s), opposite, m)
            if len(flips) and flips[0] + 1 <= bound:
                expected.append(int(flips[0] + 1))
        assert stats["lags"] == expected
        for lag in stats["lags"]:
            assert isinstance(lag, int) and lag > 0
        assert stats["p25"] <= stats["median"] <= stats["p75"] <= stats["max"]
        # W4 false flips, reproduced by hand: an entry is matched when some start lies in
        # [flip - order, flip].
        entries = [i for i in range(1, m) if lab[i] == flip_state and lab[i - 1] != flip_state]
        false = sum(1 for i in entries if not any(0 <= i - s <= DEFAULT_LAG_ORDER for s in starts))
        assert stats["n_flips"] == len(entries)
        assert stats["false_flips"] == false
        assert stats["n_flips"] > 0, f"{direction}: a 3000-bar walk must flip at least once"
        assert stats["false_flip_rate"] == false / len(entries)
        assert 0.0 <= stats["false_flip_rate"] <= 1.0


def test_lag_by_labeller_returns_one_entry_per_labeller_with_detection_lag_keys():
    rng = np.random.default_rng(SEED + 1)
    n = 1500
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n)), index=_dates(n))
    labellers = {
        "trailing": partial(label_regimes, window=20, threshold=0.05),
        "fast": partial(label_regimes, window=10, threshold=0.03),
    }
    result = lag_by_labeller(close, labellers, order=DEFAULT_LAG_ORDER)
    assert set(result) == set(labellers)
    for name, entry in result.items():
        assert set(entry) == {"crash", "recovery", "order", "note"}, name
        assert set(entry["crash"]) == LAG_KEYS
        assert set(entry["recovery"]) == LAG_KEYS
        assert entry == detection_lag(close, labellers[name](close), order=DEFAULT_LAG_ORDER)
