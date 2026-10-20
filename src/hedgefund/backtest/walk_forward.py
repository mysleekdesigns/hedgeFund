"""Incremental, no-lookahead walk-forward backtest of the Markov regime signal.

At each decision day ``t`` the transition matrix is the MLE estimate from
``labels[:t]`` only (indices ``0..t-1``), the signal is read from the row of the
regime observed at ``t``, the position is the sign of that signal, and the position
earns the close-to-close return of day ``t+1``. Nothing after ``t`` touches the
decision at ``t``; ``tests/test_no_lookahead.py`` enforces that by perturbation.

The running count matrix is updated with one transition per step instead of being
rebuilt from scratch. MLE counting is a plain sum, so the per-step matrix is
bit-for-bit identical to ``build_transition_matrix(labels[:t])``; the loop below is
the one intentional Python loop over a price series in this repo.

Ported from the companion script. ``walk_forward_backtest`` returns its exact dict
shape and numbers; ``run_walk_forward`` additionally exposes the per-day series.

Pure: pandas and numpy in, pandas and numpy out, no I/O. Backtests are historical,
not forward-looking.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedgefund.backtest.metrics import max_drawdown, sharpe_ratio
from hedgefund.regime.labels import STATES
from hedgefund.regime.markov import normalise_rows
from hedgefund.regime.signal import sign_position, signal_from_matrix

DEFAULT_MIN_TRAIN = 252  # config/default.yaml backtest.min_train: one trading year before trading
MIN_TEST_DAYS = 30  # companion rule: refuse to backtest with fewer than 30 out-of-sample days


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-day output of the walk-forward.

    ``strategy_returns`` is indexed by the date whose return was earned (``t+1``).
    ``positions`` and ``signals`` are indexed by the decision date ``t``. All three
    are empty when the history is too short to backtest.
    """

    strategy_returns: pd.Series
    positions: pd.Series
    signals: pd.Series

    def summary(self) -> dict:
        """The companion-shaped dict: ``{"sharpe", "max_drawdown", "n_trades"}``."""
        sr = self.strategy_returns.to_numpy(dtype=float)
        return {
            "sharpe": sharpe_ratio(sr),
            "max_drawdown": max_drawdown(sr),
            "n_trades": len(sr),
        }


def _empty_result() -> WalkForwardResult:
    empty = pd.Series(np.empty(0, dtype=float), index=pd.DatetimeIndex([]), dtype=float)
    return WalkForwardResult(
        strategy_returns=empty.copy(), positions=empty.copy(), signals=empty.copy()
    )


def _align(close: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily simple returns and labels on their common index, companion order."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    return labels.loc[common_index], daily_returns.loc[common_index]


def run_walk_forward(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> WalkForwardResult:
    """Walk the regime signal forward one day at a time with no lookahead.

    ``close`` is the price series; ``labels`` the integer regime per day (order
    ``STATES``), typically from ``label_regimes``. The first ``min_train`` labelled
    days are training only. Returns empty series when
    ``len(labels) < min_train + MIN_TEST_DAYS`` after alignment.
    """
    if min_train < 1:
        raise ValueError(f"min_train must be a positive number of training days, got {min_train}")
    labels, daily_returns = _align(close, labels)

    if len(labels) < min_train + MIN_TEST_DAYS:
        return _empty_result()

    lab = np.asarray(labels, dtype=int)
    rets = daily_returns.to_numpy(dtype=float)
    n_states = len(STATES)
    if lab.min() < 0 or lab.max() >= n_states:
        raise ValueError(f"labels must be integers in 0..{n_states - 1} (order {STATES})")

    # Seed running counts with all transitions strictly inside [0, min_train).
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    n_steps = len(lab) - 1 - min_train
    strategy_returns = np.empty(n_steps, dtype=float)
    positions = np.empty(n_steps, dtype=float)
    signals = np.empty(n_steps, dtype=float)
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        # counts now holds exactly the transitions among labels[:t] (indices 0..t-1).
        matrix_t = normalise_rows(counts)

        signal = signal_from_matrix(matrix_t, int(lab[t]))
        position = sign_position(signal)
        signals[k] = signal
        positions[k] = position
        strategy_returns[k] = position * rets[t + 1]

        # Slide forward by one: add transition t-1 -> t so counts next covers labels[:t+1].
        counts[lab[t - 1], lab[t]] += 1.0

    decision_dates = labels.index[min_train : len(lab) - 1]
    earned_dates = labels.index[min_train + 1 :]
    return WalkForwardResult(
        strategy_returns=pd.Series(strategy_returns, index=earned_dates, dtype=float),
        positions=pd.Series(positions, index=decision_dates, dtype=float),
        signals=pd.Series(signals, index=decision_dates, dtype=float),
    )


def walk_forward_backtest(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> dict:
    """The companion script's ``walk_forward_backtest``: same inputs, same dict.

    Returns ``{"sharpe": nan, "max_drawdown": nan, "n_trades": 0}`` when the
    aligned history is shorter than ``min_train + MIN_TEST_DAYS``.
    """
    return run_walk_forward(close, labels, min_train=min_train).summary()
