"""Out-of-sample holdout split and the window x threshold sensitivity grid.

plan.md Fix 6: the last ``holdout_years`` of history are never used for parameter
selection and are reported separately. The sensitivity grid scores every
(window, threshold) cell on IN-SAMPLE data only; the caller is responsible for
passing the in-sample close. The number of cells is the trial count the Deflated
Sharpe needs, and the variance of the per-period Sharpes across cells is its
``sharpe_variance`` input.

All functions are pure: pandas in, pandas / dict out, no I/O.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from hedgefund.backtest.metrics import TRADING_DAYS_PER_YEAR

# config backtest.holdout_years: plan Fix 6 keeps the last two years out of sample.
DEFAULT_HOLDOUT_YEARS = 2
# config backtest.sensitivity.windows: plan section 5 exposes 10 / 20 / 60 day windows.
DEFAULT_SENSITIVITY_WINDOWS: tuple[int, ...] = (10, 20, 60)
# config backtest.sensitivity.thresholds: 0.02 is the on-camera value, 0.05 the default.
DEFAULT_SENSITIVITY_THRESHOLDS: tuple[float, ...] = (0.02, 0.03, 0.05, 0.07, 0.10)


def _validate_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if len(index) == 0:
        raise ValueError("index is empty; need at least one date to place a holdout cutoff")
    return index


def holdout_cutoff(
    index: pd.DatetimeIndex, holdout_years: int = DEFAULT_HOLDOUT_YEARS
) -> pd.Timestamp:
    """First date of the holdout: ``holdout_years`` calendar years before the last date.

    ``index.max() - pd.DateOffset(years=holdout_years)``. Dates at or after the cutoff
    are out of sample. Raises ``ValueError`` on an empty or non-datetime index or
    ``holdout_years < 1``.
    """
    if holdout_years < 1:
        raise ValueError(f"holdout_years must be >= 1, got {holdout_years}")
    index = _validate_index(index)
    return pd.Timestamp(index.max()) - pd.DateOffset(years=holdout_years)


def split_holdout(
    series: pd.Series, holdout_years: int = DEFAULT_HOLDOUT_YEARS
) -> tuple[pd.Series, pd.Series]:
    """Split into ``(in_sample, holdout)``: dates before the cutoff, dates at or after.

    The two parts partition ``series`` with no overlap and no loss. When the history
    is shorter than ``holdout_years`` the cutoff falls before the first date, so the
    in-sample part is empty and everything is holdout; callers must check for an
    empty in-sample before selecting parameters.
    """
    cutoff = holdout_cutoff(series.index, holdout_years)
    before = series.index < cutoff
    return series.loc[before], series.loc[~before]


def sensitivity_grid(
    close: pd.Series,
    score: Callable[[pd.Series, int, float], float],
    windows: Sequence[int] = DEFAULT_SENSITIVITY_WINDOWS,
    thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
) -> dict:
    """Net Sharpe for every (window, threshold) cell of the parameter grid.

    ``score(close, window, threshold)`` returns the annualised net Sharpe of one
    cell (NaN allowed when the cell cannot be scored). Cells are scored in row-major
    order: rows are windows, columns are thresholds. Pass IN-SAMPLE close only; the
    holdout must never see the selection (plan Fix 6).

    Returns a dict with:

    - ``windows`` / ``thresholds``: the grid axes as lists
    - ``net_sharpe``: ``list[list[float]]``, rows = windows, cols = thresholds
    - ``n_trials``: ``len(windows) * len(thresholds)``
    - ``best``: ``{"window", "threshold", "net_sharpe"}`` at the argmax over finite
      cells; all three values are ``None`` when no cell is finite
    - ``share_positive``: share of finite cells with Sharpe > 0; NaN if none finite
    - ``sharpe_variance``: ddof=1 variance of the finite cells on the per-period scale

    ``sharpe_variance`` is on the per-period scale the Deflated Sharpe expects, so
    each annualised cell is divided by ``sqrt(TRADING_DAYS_PER_YEAR)`` first. It is
    NaN with fewer than two finite cells.

    Raises ``ValueError`` on empty ``windows`` or ``thresholds``.
    """
    if len(windows) == 0:
        raise ValueError("windows must contain at least one window, got an empty sequence")
    if len(thresholds) == 0:
        raise ValueError("thresholds must contain at least one threshold, got an empty sequence")

    grid = np.array(
        [[float(score(close, int(w), float(t))) for t in thresholds] for w in windows],
        dtype=float,
    )
    finite = np.isfinite(grid)
    n_finite = int(finite.sum())

    if n_finite == 0:
        best = {"window": None, "threshold": None, "net_sharpe": None}
        share_positive = np.nan
    else:
        masked = np.where(finite, grid, -np.inf)
        row, col = np.unravel_index(int(np.argmax(masked)), grid.shape)
        best = {
            "window": int(windows[row]),
            "threshold": float(thresholds[col]),
            "net_sharpe": float(grid[row, col]),
        }
        share_positive = float((grid[finite] > 0).mean())

    if n_finite < 2:
        sharpe_variance = np.nan
    else:
        per_period = grid[finite] / np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_variance = float(np.var(per_period, ddof=1))

    return {
        "windows": [int(w) for w in windows],
        "thresholds": [float(t) for t in thresholds],
        "net_sharpe": grid.tolist(),
        "n_trials": int(len(windows) * len(thresholds)),
        "best": best,
        "share_positive": share_positive,
        "sharpe_variance": sharpe_variance,
    }
