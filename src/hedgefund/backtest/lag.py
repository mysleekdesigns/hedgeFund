"""Detection-lag diagnostic: bars from a price turning point to the label flip.

plan.md Fix 3: a trailing-return labeller needs enough adverse days to drag its
window past the threshold, so it flips *after* the market has turned. This module
measures that lag as a distribution, split by direction (crash: peak -> first Bear
label; recovery: trough -> first Bull label), per labeller variant.

HINDSIGHT DIAGNOSTIC. A local peak or trough is only known ``order`` bars after it
happens, because the definition looks at the ``order`` bars on both sides. Nothing
here is causal and nothing here may feed a signal, a position, or a label. It
exists only to score labellers after the fact.

All functions are pure: pandas in, dict / numpy out, no I/O.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from hedgefund.regime.labels import BEAR, BULL

# config backtest.lag_order: a turning point is the extreme within +/-20 bars, the
# same horizon as the default 20-day labeller window.
DEFAULT_LAG_ORDER = 20

LAG_NOTE = (
    "hindsight diagnostic: turning points use +/-order future bars and are only known "
    "order bars after they occur; never use this to build a signal"
)


def _validate_order(order: int) -> None:
    if order < 1:
        raise ValueError(f"order must be >= 1 bars, got {order}")


def local_extrema(
    close: pd.Series, order: int = DEFAULT_LAG_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    """Integer positions of local peaks and troughs of ``close``.

    A peak at position ``i`` is a close strictly greater than every other close in
    ``close[i - order : i + order + 1]``; a trough is strictly smaller. Only interior
    positions ``order <= i < len(close) - order`` qualify, because a turning point
    nearer the edge does not have its full ``order``-bar window on both sides.

    HINDSIGHT DIAGNOSTIC: deciding that ``i`` is a peak needs the ``order`` closes
    *after* ``i``, so a peak is only known ``order`` bars late. Use the result to
    score labellers after the fact, never to build a signal.

    Returns ``(peaks, troughs)`` as sorted int arrays of positions into ``close``.
    Raises ``ValueError`` for ``order < 1``.
    """
    _validate_order(order)
    values = np.asarray(close.to_numpy(), dtype=float)
    n = len(values)
    if n < 2 * order + 1:
        empty = np.empty(0, dtype=int)
        return empty, empty
    # argrelextrema compares each point with every neighbour up to ``order`` away
    # on both sides (strict), vectorised over the series; "clip" mode lets edge
    # points pass with a truncated window, so interior filtering follows.
    peaks = argrelextrema(values, np.greater, order=order, mode="clip")[0]
    troughs = argrelextrema(values, np.less, order=order, mode="clip")[0]
    interior = slice(order, n - order)
    peaks = peaks[(peaks >= interior.start) & (peaks < interior.stop)]
    troughs = troughs[(troughs >= interior.start) & (troughs < interior.stop)]
    return peaks.astype(int), troughs.astype(int)


def _first_after(positions: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each position, the first event strictly after it.

    Returns ``(found, first)``: a boolean mask and the event position (undefined
    where ``found`` is False).
    """
    if len(events) == 0:
        return np.zeros(len(positions), dtype=bool), np.full(len(positions), -1, dtype=int)
    idx = np.searchsorted(events, positions, side="right")
    found = idx < len(events)
    first = np.where(found, events[np.minimum(idx, len(events) - 1)], -1)
    return found, first


def _lags_to_flip(starts: np.ndarray, flips: np.ndarray, censors: np.ndarray) -> np.ndarray:
    """Bars from each start to the first flip after it, dropping censored starts.

    A start is censored when a censoring event (the opposite extreme) comes strictly
    before the first flip, or when no flip follows it at all.
    """
    if len(starts) == 0:
        return np.empty(0, dtype=int)
    flip_found, first_flip = _first_after(starts, flips)
    censor_found, first_censor = _first_after(starts, censors)
    detected = flip_found & (~censor_found | (first_flip <= first_censor))
    return (first_flip - starts)[detected].astype(int)


def _regime_entries(lab: np.ndarray, regime: int) -> np.ndarray:
    """Positions ``i >= 1`` where ``lab[i] == regime`` and ``lab[i - 1] != regime``.

    The first bar has no predecessor, so it is never an entry: a labeller is not
    penalised for the regime it happens to start in.
    """
    if len(lab) < 2:
        return np.empty(0, dtype=int)
    is_regime = lab == regime
    return np.flatnonzero(is_regime[1:] & ~is_regime[:-1]) + 1


def _false_flips(flips: np.ndarray, extrema: np.ndarray, order: int) -> int:
    """Flips with no extremum at ``p`` such that ``0 <= flip - p <= order``.

    A turning point is only known ``order`` bars after it happens, so a flip up to
    ``order`` bars after one (inclusive) is the fastest honest detection and counts as
    matched; anything else is a flip with no turning point behind it.
    """
    if len(flips) == 0:
        return 0
    if len(extrema) == 0:
        return len(flips)
    idx = np.searchsorted(extrema, flips, side="right") - 1
    has_prior = idx >= 0
    last = extrema[np.maximum(idx, 0)]
    matched = has_prior & ((flips - last) <= order)
    return int(np.count_nonzero(~matched))


def _flip_counts(lab: np.ndarray, regime: int, extrema: np.ndarray, order: int) -> dict:
    flips = _regime_entries(lab, regime)
    n_flips = len(flips)
    false = _false_flips(flips, extrema, order)
    return {
        "n_flips": n_flips,
        "false_flips": false,
        "false_flip_rate": false / n_flips if n_flips else np.nan,
    }


def _summarise(lags: np.ndarray, n_events: int) -> dict:
    n = len(lags)
    if n == 0:
        stats = {"mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "max": np.nan}
    else:
        stats = {
            "mean": float(np.mean(lags)),
            "median": float(np.median(lags)),
            "p25": float(np.percentile(lags, 25)),
            "p75": float(np.percentile(lags, 75)),
            "max": float(np.max(lags)),
        }
    return {
        "n": n,
        "n_undetected": int(n_events - n),
        **stats,
        "lags": [int(x) for x in lags],
    }


def detection_lag(close: pd.Series, labels: pd.Series, order: int = DEFAULT_LAG_ORDER) -> dict:
    """Lag distribution from each turning point of ``close`` to the label flip.

    ``close`` and ``labels`` are first aligned on their common index (a labeller
    drops its first ``window`` rows, so the labels start later than the closes).
    Turning points come from ``local_extrema`` on the aligned close.

    Crash lag    : bars from a peak to the first Bear label strictly after it. The
                   peak is *undetected* (censored) if the next trough comes strictly
                   before any Bear label, or no Bear label follows at all.
    Recovery lag : bars from a trough to the first Bull label strictly after it,
                   censored symmetrically by an intervening peak.

    Returns ``{"crash": {...}, "recovery": {...}, "order": int, "note": str}`` where
    each direction holds ``n`` (detected), ``n_undetected``, ``mean``, ``median``,
    ``p25``, ``p75``, ``max`` (NaN when ``n == 0``) and the raw ``lags`` list, plus the
    false-positive side of the ledger so a trigger-happy labeller cannot look best just
    by flipping often:

    ``n_flips``        : entries into the regime (Bear for crash, Bull for recovery):
                         bars ``i >= 1`` with ``label[i] == regime != label[i - 1]``.
    ``false_flips``    : entries with no matching extremum (peak for Bear, trough for
                         Bull) in the ``order`` bars before the flip, inclusive of the
                         flip bar itself (``0 <= flip - extremum <= order``).
    ``false_flip_rate``: ``false_flips / n_flips``, NaN when ``n_flips == 0``.

    HINDSIGHT DIAGNOSTIC: the turning points need ``order`` future bars; the note in
    the result says so. Never feed this into a signal.

    Raises ``ValueError`` for ``order < 1`` or when the two series share no dates.
    """
    _validate_order(order)
    common = close.index.intersection(labels.index).sort_values()
    if len(common) == 0:
        raise ValueError(
            "close and labels share no dates; expected labels indexed by a subset of close's index"
        )
    close_c = close.loc[common]
    lab = np.asarray(labels.loc[common].to_numpy(), dtype=int)

    peaks, troughs = local_extrema(close_c, order)
    bear = np.flatnonzero(lab == BEAR)
    bull = np.flatnonzero(lab == BULL)

    crash_lags = _lags_to_flip(peaks, bear, troughs)
    recovery_lags = _lags_to_flip(troughs, bull, peaks)
    return {
        "crash": {**_summarise(crash_lags, len(peaks)), **_flip_counts(lab, BEAR, peaks, order)},
        "recovery": {
            **_summarise(recovery_lags, len(troughs)),
            **_flip_counts(lab, BULL, troughs, order),
        },
        "order": int(order),
        "note": LAG_NOTE,
    }


def lag_by_labeller(
    close: pd.Series,
    labellers: dict[str, Callable[[pd.Series], pd.Series]],
    order: int = DEFAULT_LAG_ORDER,
) -> dict[str, dict]:
    """``detection_lag`` for each labeller, keyed by the labeller's name.

    Each callable maps the close series to a label series (for example
    ``functools.partial(label_regimes, window=20, threshold=0.05)``). The result is
    a hindsight diagnostic, see ``detection_lag``.
    """
    return {
        name: detection_lag(close, labeller(close), order) for name, labeller in labellers.items()
    }
