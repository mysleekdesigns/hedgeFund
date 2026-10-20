"""Tests for the rolling-return regime labeller in ``hedgefund.regime.labels``."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, STATES, label_regimes


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
