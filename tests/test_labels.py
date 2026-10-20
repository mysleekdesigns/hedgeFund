"""Tests for the rolling-return regime labeller in ``hedgefund.regime.labels``."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedgefund.regime.labels import (
    BEAR,
    BULL,
    LABELLERS,
    SIDEWAYS,
    STATES,
    ensemble_label_regimes,
    ew_label_regimes,
    hysteresis_label_regimes,
    label_regimes,
    zscore_label_regimes,
)


def _close(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=index, dtype=float, name="close")


def test_state_constants_match_canonical_order() -> None:
    assert STATES == ["Bear", "Sideways", "Bull"]
    assert (BEAR, SIDEWAYS, BULL) == (0, 1, 2)
    assert STATES[BEAR] == "Bear" and STATES[BULL] == "Bull"


def test_known_rolling_returns_give_expected_labels() -> None:
    # window=1 so each label is the one-day return: +10%, 0%, -9.1%, +4%, -8.7%.
    close = _close([100.0, 110.0, 110.0, 100.0, 104.0, 95.0])
    labels = label_regimes(close, window=1, threshold=0.05)
    assert labels.tolist() == [BULL, SIDEWAYS, BEAR, SIDEWAYS, BEAR]


def test_multi_day_window_uses_trailing_return() -> None:
    # Prices rise 2% a day for 10 days; the 3-day return is 1.02**3 - 1 = 6.1% > 5%.
    close = _close([100.0 * 1.02**k for k in range(10)])
    labels = label_regimes(close, window=3, threshold=0.05)
    assert (labels == BULL).all()
    # The 2-day return is 4.04% < 5%, so the same prices are Sideways on a shorter window.
    assert (label_regimes(close, window=2, threshold=0.05) == SIDEWAYS).all()


def test_exactly_threshold_is_sideways_strict_inequality() -> None:
    # One-day returns: 125/100 - 1 == +0.25, 100/125 - 1 == -0.2, 75/100 - 1 == -0.25.
    # The +/-0.25 cases are exact in binary floating point, so they sit on the boundary.
    close = _close([100.0, 125.0, 100.0, 75.0])
    at_boundary = label_regimes(close, window=1, threshold=0.25)
    assert at_boundary.tolist() == [SIDEWAYS, SIDEWAYS, SIDEWAYS]
    # Nudge the threshold below the boundary and the same returns become Bull / Bear.
    inside = label_regimes(close, window=1, threshold=0.24)
    assert inside.tolist() == [BULL, SIDEWAYS, BEAR]


def test_first_window_rows_are_dropped() -> None:
    close = _close(list(np.linspace(100.0, 120.0, 30)))
    window = 7
    labels = label_regimes(close, window=window)
    assert len(labels) == len(close) - window
    assert labels.index.equals(close.index[window:])


def test_labels_are_int_dtype() -> None:
    labels = label_regimes(_close([100.0, 101.0, 102.0, 103.0]), window=2)
    assert labels.dtype.kind == "i"


def test_log_and_simple_returns_differ_near_threshold() -> None:
    # A doubling: simple return 1.0, log return ln 2 = 0.693. A 0.8 threshold separates them.
    close = _close([100.0, 200.0])
    simple = label_regimes(close, window=1, threshold=0.8, return_type="simple")
    log = label_regimes(close, window=1, threshold=0.8, return_type="log")
    assert simple.tolist() == [BULL]
    assert log.tolist() == [SIDEWAYS]


def test_log_return_is_symmetric_where_simple_is_not() -> None:
    # Up 25% then back down to the start: log returns are +/-0.223 (symmetric),
    # simple returns are +0.25 and -0.20 (not symmetric).
    close = _close([100.0, 125.0, 100.0])
    log = label_regimes(close, window=1, threshold=0.21, return_type="log")
    simple = label_regimes(close, window=1, threshold=0.21, return_type="simple")
    assert log.tolist() == [BULL, BEAR]
    assert simple.tolist() == [BULL, SIDEWAYS]


def test_default_parameters_match_config() -> None:
    # config/default.yaml: window 20, threshold 0.05, return_type simple.
    close = _close([100.0 + k for k in range(40)])
    assert label_regimes(close).equals(
        label_regimes(close, window=20, threshold=0.05, return_type="simple")
    )


def test_no_lookahead_future_prices_do_not_change_past_labels() -> None:
    rng = np.random.default_rng(7)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=120))
    close = _close(list(prices))
    cutoff = 80
    perturbed = close.copy()
    perturbed.iloc[cutoff + 1 :] *= 3.0
    base = label_regimes(close, window=20)
    mutated = label_regimes(perturbed, window=20)
    assert base.loc[: close.index[cutoff]].equals(mutated.loc[: close.index[cutoff]])


@pytest.mark.parametrize("bad_type", ["", "SIMPLE", "pct", "logarithmic"])
def test_unknown_return_type_raises(bad_type: str) -> None:
    with pytest.raises(ValueError, match="return_type"):
        label_regimes(_close([100.0, 101.0]), window=1, return_type=bad_type)


@pytest.mark.parametrize("window", [0, -1])
def test_window_below_one_raises(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        label_regimes(_close([100.0, 101.0]), window=window)


@pytest.mark.parametrize("threshold", [0.0, -0.05])
def test_non_positive_threshold_raises(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        label_regimes(_close([100.0, 101.0]), window=1, threshold=threshold)


def test_non_datetime_index_raises() -> None:
    close = pd.Series([100.0, 101.0, 102.0], index=[0, 1, 2])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        label_regimes(close, window=1)


def test_empty_series_raises() -> None:
    close = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    with pytest.raises(ValueError, match="empty"):
        label_regimes(close, window=1)


# --------------------------------------------------------------------------- #
# Helpers for the alternative labellers (Phase 2, plan.md Fix 3)
# --------------------------------------------------------------------------- #
def _close_from_log_returns(log_returns: np.ndarray, start: float = 100.0) -> pd.Series:
    """Price path whose daily log returns are exactly ``log_returns`` (first bar is flat)."""
    levels = start * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return _close(list(levels))


def _close_from_simple_returns(returns: list[float], start: float = 100.0) -> pd.Series:
    levels = start * np.cumprod([1.0] + [1.0 + r for r in returns])
    return _close(list(levels))


def _n_flips(labels: pd.Series) -> int:
    return int((labels.to_numpy()[1:] != labels.to_numpy()[:-1]).sum())


def _run_lengths(labels: pd.Series) -> list[int]:
    values = labels.to_numpy()
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    return [int(n) for n in np.diff(np.concatenate([[0], boundaries, [len(values)]]))]


def _random_close(seed: int, n: int, daily_vol: float = 0.02) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _close_from_log_returns(rng.normal(0.0, daily_vol, size=n))


# --------------------------------------------------------------------------- #
# label_regimes is unchanged by the refactor
# --------------------------------------------------------------------------- #
def test_label_regimes_equals_direct_pct_change_recomputation() -> None:
    # The refactored helper path must give exactly the companion formula: Bull where
    # pct_change(window) > threshold, Bear where < -threshold, else Sideways, NaNs dropped.
    close = _random_close(seed=11, n=150)
    window, threshold = 20, 0.05
    rolling = close.pct_change(window)
    expected = pd.Series(SIDEWAYS, index=close.index, dtype=int)
    expected[rolling > threshold] = BULL
    expected[rolling < -threshold] = BEAR
    expected = expected.loc[rolling.notna()]
    assert label_regimes(close, window=window, threshold=threshold).equals(expected)


def test_label_regimes_log_equals_direct_log_recomputation() -> None:
    close = _random_close(seed=12, n=150)
    window, threshold = 10, 0.03
    rolling = np.log(close / close.shift(window))
    expected = pd.Series(SIDEWAYS, index=close.index, dtype=int)
    expected[rolling > threshold] = BULL
    expected[rolling < -threshold] = BEAR
    expected = expected.loc[rolling.notna()]
    got = label_regimes(close, window=window, threshold=threshold, return_type="log")
    assert got.equals(expected)


# --------------------------------------------------------------------------- #
# zscore_label_regimes
# --------------------------------------------------------------------------- #
# Daily log returns cycle through this pattern. Over any window whose length is a
# multiple of 6 the sample standard deviation is the same, so the volatility scale is
# constant and the z-score labeller reduces to a fixed-threshold labeller.
_PATTERN_AMPLITUDE = 0.003
_PATTERN = [_PATTERN_AMPLITUDE] * 3 + [-_PATTERN_AMPLITUDE] * 3
_VOL_WINDOW = 60  # multiple of len(_PATTERN) so every rolling std is identical
_Z_WINDOW = 4  # not a multiple of 6, so the 4-day return takes values 2a, 0, -2a


def _constant_vol_close() -> pd.Series:
    return _close_from_log_returns(np.array(_PATTERN * 20))


def test_zscore_matches_fixed_threshold_on_constant_vol_series() -> None:
    close = _constant_vol_close()
    # Sample std (ddof=1) of a zero-mean pattern of +/-a over n points is a*sqrt(n/(n-1)).
    sigma = _PATTERN_AMPLITUDE * np.sqrt(_VOL_WINDOW / (_VOL_WINDOW - 1))
    z = 0.5  # puts the fixed threshold at ~1.0a, between the 4-day returns 0 and 2a
    fixed_threshold = z * sigma * np.sqrt(_Z_WINDOW)
    zscore = zscore_label_regimes(
        close, window=_Z_WINDOW, z_threshold=z, vol_window=_VOL_WINDOW, return_type="log"
    )
    fixed = label_regimes(close, window=_Z_WINDOW, threshold=fixed_threshold, return_type="log")
    # Exact equality: returns sit at 2a, 0, -2a and the threshold at ~1.0a, far from both,
    # so float noise in the rolling std cannot move any label.
    assert zscore.equals(fixed.loc[zscore.index])
    assert set(zscore.tolist()) == {BEAR, SIDEWAYS, BULL}


def test_zscore_three_sigma_shock_flips_label() -> None:
    daily_vol = 0.005
    window, vol_window = 20, 60
    rng = np.random.default_rng(1)
    log_returns = rng.normal(0.0, daily_vol, size=100)
    shock_day = 90
    # A one-day jump equal to three sigma of a window-length move.
    log_returns[shock_day] = 3.0 * daily_vol * np.sqrt(window)
    close = _close_from_log_returns(log_returns)
    labels = zscore_label_regimes(close, window=window, vol_window=vol_window)
    before = labels.loc[: close.index[shock_day]]
    # Noise alone stays inside one sigma on this seed (checked), the shock does not.
    assert (before == SIDEWAYS).all()
    assert labels.loc[close.index[shock_day + 1]] == BULL


def test_zscore_drops_rows_with_undefined_or_zero_vol() -> None:
    # 70 flat days then a move: the first 60-day vol windows are exactly zero and dropped.
    close = _close([100.0] * 70 + [101.0, 102.0, 103.0])
    labels = zscore_label_regimes(close, window=5, vol_window=60)
    assert labels.index.min() > close.index[69]
    assert labels.index.isin(close.index).all()


def test_zscore_output_is_int_in_state_range() -> None:
    labels = zscore_label_regimes(_random_close(seed=3, n=200))
    assert labels.dtype.kind == "i"
    assert set(labels.unique()).issubset({BEAR, SIDEWAYS, BULL})
    assert labels.index.isin(_random_close(seed=3, n=200).index).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window": 0}, "window"),
        ({"vol_window": 1}, "vol_window"),
        ({"z_threshold": 0.0}, "z_threshold"),
        ({"z_threshold": -1.0}, "z_threshold"),
        ({"return_type": "pct"}, "return_type"),
    ],
)
def test_zscore_invalid_inputs_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        zscore_label_regimes(_random_close(seed=0, n=100), **kwargs)


# --------------------------------------------------------------------------- #
# ew_label_regimes
# --------------------------------------------------------------------------- #
def test_ew_monotone_up_is_bull() -> None:
    # +1% a day: EW daily mean ~0.01, times window 20 = 0.2 > 0.05.
    close = _close_from_simple_returns([0.01] * 60)
    assert (ew_label_regimes(close) == BULL).all()


def test_ew_monotone_down_is_bear() -> None:
    close = _close_from_simple_returns([-0.01] * 60)
    assert (ew_label_regimes(close) == BEAR).all()


def test_ew_flat_is_sideways() -> None:
    close = _close([100.0] * 60)
    assert (ew_label_regimes(close) == SIDEWAYS).all()


def test_ew_first_defined_date_matches_trailing_labeller() -> None:
    close = _random_close(seed=5, n=80)
    assert ew_label_regimes(close, window=20).index.equals(label_regimes(close, window=20).index)


def test_ew_flips_to_bear_no_later_than_trailing_after_reversal() -> None:
    # 60 flat days then -1% a day for 10 days. The trailing 20-day return needs six
    # down days to pass -5% (0.99**6 = 0.9415); the EW return (half-life 10) needs five.
    close = _close([100.0] * 60 + [100.0 * 0.99**k for k in range(1, 11)])
    ew = ew_label_regimes(close)
    trailing = label_regimes(close)
    ew_flip = ew.index[ew == BEAR].min()
    trailing_flip = trailing.index[trailing == BEAR].min()
    assert ew_flip <= trailing_flip


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"half_life": 0.0}, "half_life"),
        ({"half_life": -2.0}, "half_life"),
        ({"window": 0}, "window"),
        ({"threshold": 0.0}, "threshold"),
        ({"return_type": "LOG"}, "return_type"),
    ],
)
def test_ew_invalid_inputs_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ew_label_regimes(_random_close(seed=0, n=100), **kwargs)


# --------------------------------------------------------------------------- #
# ensemble_label_regimes
# --------------------------------------------------------------------------- #
def test_ensemble_unanimous_windows_give_that_label() -> None:
    # +1% a day: even the 5-day return (5.1%) clears 5%, so every window votes Bull.
    close = _close_from_simple_returns([0.01] * 100)
    assert (ensemble_label_regimes(close) == BULL).all()
    down = _close_from_simple_returns([-0.0105] * 100)
    assert (ensemble_label_regimes(down) == BEAR).all()


def test_ensemble_two_two_tie_is_sideways() -> None:
    # On the last day: 1- and 2-day returns are +11% (Bull), 3- and 4-day are -17% (Bear).
    close = _close([120.0, 120.0, 90.0, 90.0, 100.0])
    labels = ensemble_label_regimes(close, windows=(1, 2, 3, 4), threshold=0.05)
    assert labels.tolist() == [SIDEWAYS]


def test_ensemble_plurality_without_strict_majority_is_sideways() -> None:
    # Last day: 1-day +11% Bull, 2-day +11% Bull, 3-day 0% Sideways, 4-day -17% Bear: 2-1-1.
    close = _close([120.0, 100.0, 90.0, 90.0, 100.0])
    labels = ensemble_label_regimes(close, windows=(1, 2, 3, 4), threshold=0.05)
    assert labels.tolist() == [SIDEWAYS]


def test_ensemble_strict_majority_wins() -> None:
    # Last day: 1-day +11%, 2-day +11%, 3-day +11% Bull; 4-day -17% Bear: 3 of 4.
    close = _close([120.0, 90.0, 90.0, 90.0, 100.0])
    labels = ensemble_label_regimes(close, windows=(1, 2, 3, 4), threshold=0.05)
    assert labels.tolist() == [BULL]


def test_ensemble_index_is_longest_window_defined_dates() -> None:
    close = _random_close(seed=8, n=150)
    windows = (5, 10, 20, 60)
    labels = ensemble_label_regimes(close, windows=windows)
    assert labels.index.equals(label_regimes(close, window=max(windows)).index)
    assert labels.dtype.kind == "i"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"windows": ()}, "windows"),
        ({"windows": (5, 0)}, "window"),
        ({"threshold": 0.0}, "threshold"),
        ({"return_type": "pct"}, "return_type"),
    ],
)
def test_ensemble_invalid_inputs_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ensemble_label_regimes(_random_close(seed=0, n=100), **kwargs)


# --------------------------------------------------------------------------- #
# hysteresis_label_regimes
# --------------------------------------------------------------------------- #
def test_hysteresis_oscillation_around_entry_threshold_flips_less() -> None:
    # window=1: daily returns alternate +6% / +4%. The plain labeller flips Bull /
    # Sideways every bar; hysteresis enters Bull once and stays (4% >= 2% exit).
    close = _close_from_simple_returns([0.06, 0.04] * 15)
    plain = label_regimes(close, window=1)
    hyst = hysteresis_label_regimes(close, window=1)
    assert _n_flips(hyst) == 0
    assert _n_flips(plain) == len(plain) - 1
    assert _n_flips(hyst) < _n_flips(plain)


def test_hysteresis_non_initial_runs_hold_at_least_min_hold_bars() -> None:
    min_hold = 3
    # High daily vol with window=1 so the plain rule would flip almost every bar.
    close = _random_close(seed=21, n=400, daily_vol=0.06)
    hyst = hysteresis_label_regimes(close, window=1, min_hold=min_hold)
    runs = _run_lengths(hyst)
    assert len(runs) > 5, "need several regimes for the check to have teeth"
    # The first run may be shorter (it starts the machine); the last may be cut off
    # by the end of the series. Every run in between is a completed hold.
    assert all(run >= min_hold for run in runs[1:-1])


def test_hysteresis_exit_threshold_keeps_bull_between_exit_and_enter() -> None:
    # +6% enters Bull; +3% three times sits between exit (2%) and enter (5%) so Bull
    # holds; +1% is below exit and drops to Sideways.
    close = _close_from_simple_returns([0.06, 0.03, 0.03, 0.03, 0.01])
    hyst = hysteresis_label_regimes(close, window=1, min_hold=1)
    assert hyst.tolist() == [BULL, BULL, BULL, BULL, SIDEWAYS]
    # The plain labeller already reads those +3% bars as Sideways.
    assert label_regimes(close, window=1).tolist() == [BULL, SIDEWAYS, SIDEWAYS, SIDEWAYS, SIDEWAYS]


def test_hysteresis_crash_goes_straight_from_bull_to_bear() -> None:
    close = _close_from_simple_returns([0.06, 0.03, 0.03, -0.10])
    hyst = hysteresis_label_regimes(close, window=1, min_hold=1)
    assert hyst.tolist() == [BULL, BULL, BULL, BEAR]


def test_hysteresis_bear_mirror_exits_above_minus_exit() -> None:
    # -6% enters Bear; -3% holds (between -5% and -2%); -1% is above -2% so Sideways;
    # +6% from Sideways enters Bull.
    close = _close_from_simple_returns([-0.06, -0.03, -0.01, 0.06])
    hyst = hysteresis_label_regimes(close, window=1, min_hold=1)
    assert hyst.tolist() == [BEAR, BEAR, SIDEWAYS, BULL]


def test_hysteresis_min_hold_blocks_early_flip() -> None:
    # Bar 0 Bull (initial). Bar 1 +1% flips to Sideways. Bars 2 and 3 at +6% would flip
    # back but the 3-bar hold blocks them; bar 4 is the first allowed flip.
    close = _close_from_simple_returns([0.06, 0.01, 0.06, 0.06, 0.06])
    hyst = hysteresis_label_regimes(close, window=1, min_hold=3)
    assert hyst.tolist() == [BULL, SIDEWAYS, SIDEWAYS, SIDEWAYS, BULL]


def test_hysteresis_first_bar_matches_trailing_labeller() -> None:
    close = _random_close(seed=4, n=60)
    hyst = hysteresis_label_regimes(close, window=20)
    plain = label_regimes(close, window=20)
    assert hyst.index.equals(plain.index)
    assert hyst.iloc[0] == plain.iloc[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"exit_threshold": 0.0}, "exit_threshold"),
        ({"exit_threshold": 0.05, "enter_threshold": 0.05}, "exit_threshold"),
        ({"exit_threshold": 0.06, "enter_threshold": 0.05}, "exit_threshold"),
        ({"min_hold": 0}, "min_hold"),
        ({"window": 0}, "window"),
        ({"return_type": "pct"}, "return_type"),
    ],
)
def test_hysteresis_invalid_inputs_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hysteresis_label_regimes(_random_close(seed=0, n=100), **kwargs)


# --------------------------------------------------------------------------- #
# Registry invariants over every labeller
# --------------------------------------------------------------------------- #
def test_registry_names_and_targets() -> None:
    assert {
        "trailing": label_regimes,
        "zscore": zscore_label_regimes,
        "ew": ew_label_regimes,
        "ensemble": ensemble_label_regimes,
        "hysteresis": hysteresis_label_regimes,
    } == LABELLERS


@pytest.mark.parametrize("name", sorted(LABELLERS))
@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1), n=st.integers(min_value=70, max_value=250)
)
def test_every_labeller_returns_states_on_a_datetime_subset(name: str, seed: int, n: int) -> None:
    # n >= 70 so the 60-day ensemble and zscore vol windows are defined on some rows.
    close = _random_close(seed=seed, n=n)
    labels = LABELLERS[name](close)
    assert labels.dtype.kind == "i"
    assert set(labels.unique()).issubset({BEAR, SIDEWAYS, BULL})
    assert isinstance(labels.index, pd.DatetimeIndex)
    assert labels.index.isin(close.index).all()
    assert labels.index.is_monotonic_increasing and labels.index.is_unique


@pytest.mark.parametrize("name", sorted(LABELLERS))
def test_every_labeller_rejects_non_datetime_index(name: str) -> None:
    close = pd.Series(np.linspace(100.0, 110.0, 100), index=range(100))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        LABELLERS[name](close)
