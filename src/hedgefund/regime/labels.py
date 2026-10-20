"""Regime labellers (Bull / Bear / Sideways) from a close series.

``label_regimes`` is ported from the companion script so that the simple-return
variant is bit-identical to it on the same input. The alternative labellers
(plan.md Fix 3) attack the built-in lag of a fixed-threshold trailing return:

- ``zscore_label_regimes``     : threshold adapts to realised volatility
- ``ew_label_regimes``         : exponentially weighted return, recent days count more
- ``ensemble_label_regimes``   : majority vote over several windows
- ``hysteresis_label_regimes`` : separate entry / exit thresholds and a minimum hold

All of them are pure (a pandas ``Series`` in, a pandas ``Series`` out, no I/O) and
strictly causal: the label on day ``t`` uses closes at or before ``t`` only, so
perturbing prices after ``t`` cannot change labels at or before ``t``. Every
labeller returns an int ``Series`` of state indices in the ``STATES`` order, indexed
by the dates where its label is defined (undefined leading rows are dropped).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# Canonical state order used by every matrix and vector in the repo (config/default.yaml).
STATES: list[str] = ["Bear", "Sideways", "Bull"]
BEAR, SIDEWAYS, BULL = 0, 1, 2

RETURN_TYPES: tuple[str, ...] = ("simple", "log")


def _validate_return_type(return_type: str) -> None:
    if return_type not in RETURN_TYPES:
        raise ValueError(f"return_type must be one of {RETURN_TYPES}, got {return_type!r}")


def _validate_close(close: pd.Series) -> None:
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(f"close must have a DatetimeIndex, got {type(close.index).__name__}")
    if close.empty:
        raise ValueError("close is empty; need at least window + 1 rows to label any day")


def _rolling_return(close: pd.Series, window: int, return_type: str) -> pd.Series:
    """Trailing ``window``-bar return, NaN for the first ``window`` rows.

    ``"simple"`` is ``close.pct_change(window)`` (the companion script), ``"log"`` is
    ``log(close / close.shift(window))`` (the Pine indicator). Both use ``close[t]``
    and ``close[t - window]`` only.
    """
    if return_type == "simple":
        return close.pct_change(window)
    return np.log(close / close.shift(window))


def _threshold_labels(signal: pd.Series, upper: float, lower: float) -> pd.Series:
    """Bull where ``signal > upper``, Bear where ``signal < lower``, else Sideways.

    NaNs compare false on both sides and so come out Sideways; callers drop them.
    """
    labels = pd.Series(SIDEWAYS, index=signal.index, dtype=int)
    labels[signal > upper] = BULL
    labels[signal < lower] = BEAR
    return labels


def label_regimes(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window: 20 trading days of trailing return
    threshold: float = 0.05,  # config/default.yaml labelling.threshold: +/-5% trailing return
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Label each day from its trailing ``window``-day return.

    Bull (2)     : rolling return >  +threshold   (strict)
    Bear (0)     : rolling return <  -threshold   (strict)
    Sideways (1) : otherwise, including a return of exactly +/-threshold

    ``return_type`` selects the rolling return: ``"simple"`` is
    ``close.pct_change(window)`` (the companion script), ``"log"`` is
    ``log(close / close.shift(window))`` (the Pine indicator).

    Returns an int ``Series`` indexed like ``close`` minus the first ``window`` rows,
    which are dropped because their rolling return is undefined.

    Raises ``ValueError`` for an unknown ``return_type``, ``window < 1``,
    ``threshold <= 0``, a non-``DatetimeIndex``, or an empty series.
    """
    _validate_return_type(return_type)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    _validate_close(close)

    rolling_return = _rolling_return(close, window, return_type)
    labels = _threshold_labels(rolling_return, threshold, -threshold)
    return labels.loc[rolling_return.notna()]


def zscore_label_regimes(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window: 20 trading days of trailing return
    z_threshold: float = 1.0,  # one sigma of a window-length move; ~+/-5% at 16% annualised vol
    vol_window: int = 60,  # three months: vol moves too slowly to flip a label by itself
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Label on a volatility-normalised trailing return (plan.md Fix 3).

    The z-score on day ``t`` is the trailing ``window``-bar return divided by
    ``sigma_daily * sqrt(window)``, where ``sigma_daily`` is the sample standard
    deviation (ddof=1) of daily log returns over the trailing ``vol_window`` bars
    ending at ``t``. Both numerator and denominator use data through ``t`` only.

    Bull (2)     : z >  +z_threshold   (strict)
    Bear (0)     : z <  -z_threshold   (strict)
    Sideways (1) : otherwise

    With ``z_threshold = 1`` the effective price threshold is one standard deviation
    of a ``window``-day move, which at 16% annualised volatility and ``window = 20``
    is ``0.16 * sqrt(20 / 252) ~= 4.5%``, close to the fixed +/-5% of ``label_regimes``.
    In calmer markets the threshold tightens, in volatile ones it loosens.

    Rows where the return or the volatility is undefined, or the volatility is not
    strictly positive (a flat window), are dropped.

    Raises ``ValueError`` for an unknown ``return_type``, ``window < 1``,
    ``vol_window < 2`` (a sample standard deviation needs two points),
    ``z_threshold <= 0``, a non-``DatetimeIndex``, or an empty series.
    """
    _validate_return_type(return_type)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if vol_window < 2:
        raise ValueError(
            f"vol_window must be >= 2 for a sample standard deviation, got {vol_window}"
        )
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be > 0, got {z_threshold}")
    _validate_close(close)

    rolling_return = _rolling_return(close, window, return_type)
    # Volatility is always measured on log returns so the scale does not depend on
    # return_type; ddof=1 is the pandas default and the unbiased sample estimate.
    sigma_daily = np.log(close).diff().rolling(vol_window).std()
    scale = sigma_daily * np.sqrt(window)
    z = rolling_return / scale
    labels = _threshold_labels(z, z_threshold, -z_threshold)
    defined = rolling_return.notna() & (scale > 0)
    return labels.loc[defined]


def ew_label_regimes(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window: sets the scale of the EW return
    half_life: float = 10.0,  # plan.md Fix 3: half-life about 10 days so recent days count more
    threshold: float = 0.05,  # config/default.yaml labelling.threshold: +/-5% trailing return
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Label on an exponentially weighted trailing return (plan.md Fix 3).

    Daily returns ``r_t`` are ``close.pct_change()`` for ``"simple"`` or
    ``log(close).diff()`` for ``"log"``. The exponentially weighted mean uses
    ``r.ewm(halflife=half_life, adjust=True, min_periods=window).mean()``, a trailing
    recursion that only ever sees returns at or before ``t``.

    Scaling: the EW mean is an average *daily* return, so it is multiplied by
    ``window`` to put it on the same scale as a ``window``-day trailing return (a
    20-day window of average daily return r-bar moves about 20 r-bar). The result is
    then compared to +/-``threshold`` exactly like ``label_regimes``, so the same
    config threshold applies to both. Because weights decay with ``half_life``, a
    reversal shows up in the EW return sooner than in the plain trailing return.

    ``min_periods=window`` makes the first defined label fall on the same date as
    ``label_regimes`` with the same ``window``; rows before it are dropped.

    Raises ``ValueError`` for an unknown ``return_type``, ``window < 1``,
    ``half_life <= 0``, ``threshold <= 0``, a non-``DatetimeIndex``, or an empty
    series.
    """
    _validate_return_type(return_type)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    _validate_close(close)

    daily = _rolling_return(close, 1, return_type)
    ew_mean = daily.ewm(halflife=half_life, adjust=True, min_periods=window).mean()
    ew_return = ew_mean * window
    labels = _threshold_labels(ew_return, threshold, -threshold)
    return labels.loc[ew_return.notna()]


def ensemble_label_regimes(
    close: pd.Series,
    # plan.md Fix 3: 5, 10, 20 and 60 days; short windows flip early, long ones filter noise.
    windows: tuple[int, ...] = (5, 10, 20, 60),
    threshold: float = 0.05,  # config/default.yaml labelling.threshold: +/-5% trailing return
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Majority vote of ``label_regimes`` over several windows (plan.md Fix 3).

    Each window votes with its own ``label_regimes`` label on each day. A state wins a
    day only with a strict majority (more than half of the windows); any day with no
    strict majority, including a 2-2 tie, is Sideways because the windows disagree
    and Sideways is the no-position state.

    The output is indexed by the dates where every window is defined, which is the
    defined index of the longest window.

    Raises ``ValueError`` for an empty ``windows``, any window ``< 1``, an unknown
    ``return_type``, ``threshold <= 0``, a non-``DatetimeIndex``, or an empty series.
    """
    if len(windows) == 0:
        raise ValueError("windows must contain at least one window, got an empty tuple")
    for w in windows:
        if w < 1:
            raise ValueError(f"every window must be >= 1, got {w} in windows={windows}")
    _validate_return_type(return_type)
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    _validate_close(close)

    votes = [
        label_regimes(close, window=w, threshold=threshold, return_type=return_type)
        for w in windows
    ]
    common = votes[0].index
    for v in votes[1:]:
        common = common.intersection(v.index)
    common = common.sort_values()

    ballots = np.stack([v.reindex(common).to_numpy() for v in votes], axis=1)  # (days, windows)
    counts = np.stack([(ballots == s).sum(axis=1) for s in (BEAR, SIDEWAYS, BULL)], axis=1)
    winner = counts.argmax(axis=1)
    strict_majority = counts.max(axis=1) * 2 > len(windows)
    labels = np.where(strict_majority, winner, SIDEWAYS)
    return pd.Series(labels, index=common, dtype=int)


def hysteresis_label_regimes(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window: 20 trading days of trailing return
    enter_threshold: float = 0.05,  # plan.md Fix 3: enter Bear below -5% (config threshold)
    exit_threshold: float = 0.02,  # plan.md Fix 3: leave Bear above -2%
    min_hold: int = 3,  # plan.md Fix 3: minimum hold of 2 to 3 bars; 3 cuts more whipsaw
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Trailing-return state machine with entry / exit thresholds and a minimum hold.

    Let ``r`` be the trailing ``window``-bar return (as in ``label_regimes``).

    From Sideways : Bull if ``r > enter``, Bear if ``r < -enter``, else stay.
    From Bull     : Bear if ``r < -enter``; Sideways if ``r < exit``; else stay.
    From Bear     : Bull if ``r > enter``; Sideways if ``r > -exit``; else stay.

    After any flip no further flip is allowed for the next ``min_hold - 1`` bars, so
    every regime after the first holds at least ``min_hold`` bars. The first defined
    bar starts in the state ``label_regimes`` would give it (thresholds +/-``enter``).

    The machine is sequential by construction (each state depends on the previous
    one) so it cannot be expressed with rolling operations; this is why the loop over
    a numpy array below is acceptable under ``.claude/rules/python.md``. The state at
    ``t`` depends on returns at or before ``t`` only, so there is no lookahead.

    Raises ``ValueError`` for an unknown ``return_type``, ``window < 1``,
    ``exit_threshold <= 0``, ``exit_threshold >= enter_threshold``, ``min_hold < 1``,
    a non-``DatetimeIndex``, or an empty series.
    """
    _validate_return_type(return_type)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if exit_threshold <= 0:
        raise ValueError(f"exit_threshold must be > 0, got {exit_threshold}")
    if exit_threshold >= enter_threshold:
        raise ValueError(
            "exit_threshold must be < enter_threshold, "
            f"got exit={exit_threshold}, enter={enter_threshold}"
        )
    if min_hold < 1:
        raise ValueError(f"min_hold must be >= 1, got {min_hold}")
    _validate_close(close)

    rolling_return = _rolling_return(close, window, return_type)
    defined = rolling_return.notna()
    r = rolling_return.loc[defined].to_numpy()
    n = len(r)
    labels = np.empty(n, dtype=int)
    if n == 0:
        return pd.Series(labels, index=rolling_return.index[defined], dtype=int)

    # Sequential state machine: a Python loop over a numpy array (see docstring and
    # .claude/rules/python.md on the one intentional loop class).
    state = _initial_state(r[0], enter_threshold)
    labels[0] = state
    last_flip = -min_hold  # far enough back that the first bar is never under a hold
    for i in range(1, n):
        if i - last_flip >= min_hold:
            new_state = _next_state(state, r[i], enter_threshold, exit_threshold)
            if new_state != state:
                last_flip = i
                state = new_state
        labels[i] = state
    return pd.Series(labels, index=rolling_return.index[defined], dtype=int)


def _initial_state(r: float, enter: float) -> int:
    if r > enter:
        return BULL
    if r < -enter:
        return BEAR
    return SIDEWAYS


def _next_state(state: int, r: float, enter: float, exit_: float) -> int:
    if state == SIDEWAYS:
        return _initial_state(r, enter)
    if state == BULL:
        if r < -enter:
            return BEAR
        if r < exit_:
            return SIDEWAYS
        return BULL
    # state == BEAR, mirror of Bull
    if r > enter:
        return BULL
    if r > -exit_:
        return SIDEWAYS
    return BEAR


# Registry so later phases can pick a labeller by name from config.
LABELLERS: dict[str, Callable[..., pd.Series]] = {
    "trailing": label_regimes,
    "zscore": zscore_label_regimes,
    "ew": ew_label_regimes,
    "ensemble": ensemble_label_regimes,
    "hysteresis": hysteresis_label_regimes,
}
