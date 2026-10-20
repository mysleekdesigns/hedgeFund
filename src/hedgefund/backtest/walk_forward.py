"""Incremental, no-lookahead walk-forward backtest of the Markov regime signal.

At each decision day ``t`` the transition matrix is the MLE estimate from labels
strictly before ``t``: all of ``labels[:t]`` in the default *expanding* mode, or the
trailing ``labels[max(0, t - rolling_bars):t]`` in *rolling* mode. The signal is read
from the row of the regime observed at ``t``, the position follows one of the rules in
``hedgefund.regime.signal.POSITION_RULES`` (sign by default), an optional dead band
suppresses small changes (hysteresis on the signal for the sign rules, a band on the
position for the continuous rules; see ``banded_position``), and the held position
earns the close-to-close return of day ``t+1``. Nothing after ``t`` touches the
decision at ``t``;
``tests/test_no_lookahead.py`` enforces that by perturbation.

The running count matrix is updated with one transition per step instead of being
rebuilt from scratch: add the transition ``t-1 -> t`` and, in rolling mode, subtract
the one that leaves the window. MLE counting is a plain sum, so the per-step matrix is
bit-for-bit identical to ``build_transition_matrix`` on the same slice; the loop below
is the one intentional Python loop over a price series in this repo.

Ported from the companion script. ``walk_forward_backtest`` returns its exact dict
shape and numbers; ``run_walk_forward`` additionally exposes the per-day series and,
with its default keyword arguments, is bit-identical to the Phase 1 port.

Pure: pandas and numpy in, pandas and numpy out, no I/O. Backtests are historical,
not forward-looking.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedgefund.backtest.metrics import max_drawdown, sharpe_ratio
from hedgefund.regime.labels import STATES
from hedgefund.regime.markov import normalise_rows
from hedgefund.regime.signal import POSITION_RULES, position_from_signal, signal_from_matrix

DEFAULT_MIN_TRAIN = 252  # config/default.yaml backtest.min_train: one trading year before trading
MIN_TEST_DAYS = 30  # companion rule: refuse to backtest with fewer than 30 out-of-sample days

REFIT_MODES: tuple[str, ...] = ("expanding", "rolling")
DEFAULT_REFIT = "expanding"  # config backtest.refit: all history, the companion's behaviour
DEFAULT_ROLLING_BARS = (
    756  # config backtest.rolling_bars: 3 trading years (3 * 252), plan section 5
)
DEFAULT_DEAD_BAND = (
    0.0  # config backtest.dead_band: 0 = rebalance on every change (Phase 1 behaviour)
)
MIN_ROLLING_BARS = 2  # fewer labels cannot form a single transition
FLAT = 0.0  # the walk-forward starts out of the market


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-day output of the walk-forward.

    ``strategy_returns`` is the gross return ``position * asset return`` indexed by the
    date whose return was earned (``t+1``); ``asset_returns`` is the close-to-close
    simple return of that same day on the same index. ``positions`` (the held position
    after the dead band) and ``signals`` (bull minus bear) are indexed by the decision
    date ``t``. Entry ``k`` of every series refers to the same step. All four are empty
    when the history is too short to backtest.
    """

    strategy_returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    asset_returns: pd.Series

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
        strategy_returns=empty.copy(),
        positions=empty.copy(),
        signals=empty.copy(),
        asset_returns=empty.copy(),
    )


def _align(close: pd.Series, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Daily simple returns and labels on their common index, companion order."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    return labels.loc[common_index], daily_returns.loc[common_index]


def _validate_options(
    min_train: int, refit: str, rolling_bars: int, rule: str, dead_band: float
) -> None:
    if min_train < 1:
        raise ValueError(f"min_train must be a positive number of training days, got {min_train}")
    if refit not in REFIT_MODES:
        raise ValueError(f"refit must be one of {REFIT_MODES}, got {refit!r}")
    if rolling_bars < MIN_ROLLING_BARS:
        raise ValueError(f"rolling_bars must be >= {MIN_ROLLING_BARS}, got {rolling_bars}")
    if rule not in POSITION_RULES:
        raise ValueError(f"rule must be one of {POSITION_RULES}, got {rule!r}")
    if not np.isfinite(dead_band) or dead_band < 0:
        raise ValueError(f"dead_band must be a finite number >= 0, got {dead_band}")


LONG = 1.0
SHORT = -1.0


def banded_position(
    held: float, target: float, signal: float, rule: str, dead_band: float
) -> float:
    """The position to hold after seeing ``signal`` under ``rule`` with a dead band.

    Plan Fix 6 threshold rebalancing: "only change position when the signal moves by
    more than a band". The band means something different per rule because the sign
    rules emit targets in ``{-1, 0, +1}`` (a position-space band below 1 would be a
    no-op and one at or above 2 would never trade), so for them the band is applied
    as hysteresis on the *signal* around zero:

    - ``dead_band == 0``: every target change is applied; bit-identical to Phase 1.
    - ``sign``: from flat, enter ``sign(signal)`` only when ``|signal| > dead_band``;
      from long, flip short only when ``signal < -dead_band``, otherwise stay long;
      from short, flip long only when ``signal > dead_band``, otherwise stay short.
    - ``long_only``: from flat, go long when ``signal > dead_band``; from long, go flat
      when ``signal < -dead_band``.
    - ``conviction``, ``persistence``: move to ``target`` when
      ``|target - held| > dead_band`` (position-space band on a continuous target).

    Pure; ``rule`` must be one of ``POSITION_RULES``.
    """
    if dead_band == 0.0 or rule in ("conviction", "persistence"):
        return target if abs(target - held) > dead_band else held
    if rule == "sign":
        if held == FLAT:
            if signal > dead_band:
                return LONG
            if signal < -dead_band:
                return SHORT
            return held
        if held == LONG:
            return SHORT if signal < -dead_band else held
        return LONG if signal > dead_band else held
    if rule == "long_only":
        if held == FLAT:
            return LONG if signal > dead_band else held
        return FLAT if signal < -dead_band else held
    raise ValueError(f"rule must be one of {POSITION_RULES}, got {rule!r}")


def run_walk_forward(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = DEFAULT_MIN_TRAIN,
    *,
    refit: str = DEFAULT_REFIT,
    rolling_bars: int = DEFAULT_ROLLING_BARS,
    rule: str = "sign",
    dead_band: float = DEFAULT_DEAD_BAND,
) -> WalkForwardResult:
    """Walk the regime signal forward one day at a time with no lookahead.

    ``close`` is the price series; ``labels`` the integer regime per day (order
    ``STATES``), typically from ``label_regimes``. The first ``min_train`` labelled
    days are training only. Returns empty series when
    ``len(labels) < min_train + MIN_TEST_DAYS`` after alignment.

    ``refit="expanding"`` (default) fits the matrix at ``t`` on ``labels[:t]``;
    ``refit="rolling"`` fits it on ``labels[max(0, t - rolling_bars):t]``, so the matrix
    forgets transitions older than ``rolling_bars`` labels. ``rule`` is one of
    ``POSITION_RULES``. ``dead_band`` is a threshold rebalance (plan.md Fix 6) that
    cuts turnover; the held position starts flat and ``banded_position`` decides each
    step. Per rule:

    - ``dead_band == 0`` (default): every target change is applied, Phase 1 behaviour.
    - ``sign``: hysteresis on the signal; enter from flat when ``|signal| > dead_band``,
      flip long->short only when ``signal < -dead_band``, short->long only when
      ``signal > dead_band``.
    - ``long_only``: enter when ``signal > dead_band``, exit when ``signal < -dead_band``.
    - ``conviction`` / ``persistence``: move to the target when
      ``|target - held| > dead_band``.

    With the default keyword arguments the output is bit-identical to the Phase 1 port
    and to the companion script.
    """
    _validate_options(min_train, refit, rolling_bars, rule, dead_band)
    labels, daily_returns = _align(close, labels)

    if len(labels) < min_train + MIN_TEST_DAYS:
        return _empty_result()

    lab = np.asarray(labels, dtype=int)
    rets = daily_returns.to_numpy(dtype=float)
    n_states = len(STATES)
    if lab.min() < 0 or lab.max() >= n_states:
        raise ValueError(f"labels must be integers in 0..{n_states - 1} (order {STATES})")

    rolling = refit == "rolling"
    # Seed running counts with all transitions strictly inside the first window
    # [window_start, min_train): everything from 0 when expanding, the trailing
    # ``rolling_bars`` labels when rolling.
    window_start = max(0, min_train - rolling_bars) if rolling else 0
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(window_start, min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    n_steps = len(lab) - 1 - min_train
    strategy_returns = np.empty(n_steps, dtype=float)
    positions = np.empty(n_steps, dtype=float)
    signals = np.empty(n_steps, dtype=float)
    held = FLAT
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        # counts now holds exactly the transitions among labels[max(0, t - rolling_bars):t]
        # (rolling) or labels[:t] (expanding), i.e. indices strictly before t.
        matrix_t = normalise_rows(counts)

        state = int(lab[t])
        signal = signal_from_matrix(matrix_t, state)
        target = position_from_signal(signal, float(matrix_t[state, state]), rule)
        held = banded_position(held, target, signal, rule, dead_band)
        signals[k] = signal
        positions[k] = held
        strategy_returns[k] = held * rets[t + 1]

        # Slide forward by one: add transition t-1 -> t so counts next covers labels
        # through index t. In rolling mode also drop the transition that leaves the
        # window, (t - rolling_bars) -> (t - rolling_bars + 1), once the window is full.
        counts[lab[t - 1], lab[t]] += 1.0
        if rolling and t >= rolling_bars:
            counts[lab[t - rolling_bars], lab[t - rolling_bars + 1]] -= 1.0

    decision_dates = labels.index[min_train : len(lab) - 1]
    earned_dates = labels.index[min_train + 1 :]
    return WalkForwardResult(
        strategy_returns=pd.Series(strategy_returns, index=earned_dates, dtype=float),
        positions=pd.Series(positions, index=decision_dates, dtype=float),
        signals=pd.Series(signals, index=decision_dates, dtype=float),
        asset_returns=pd.Series(rets[min_train + 1 :], index=earned_dates, dtype=float),
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
