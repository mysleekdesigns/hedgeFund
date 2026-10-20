"""Performance metrics for a strategy return series.

Ported from the companion script so that Sharpe and max drawdown are bit-identical
to its ``walk_forward_backtest``. Pure numpy: arrays in, floats out, no I/O.

Backtests are historical, not forward-looking.
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252  # annualisation factor for daily returns (US equity calendar)


def _as_returns(returns: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array; raise on anything else."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"returns must be a 1-D array of period returns, got shape {arr.shape}")
    return arr


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative growth of one unit: ``cumprod(1 + r)``. Empty in, empty out."""
    return (1.0 + _as_returns(returns)).cumprod()


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate.

    ``mean / std(ddof=1) * sqrt(periods_per_year)``. Returns NaN when there are
    fewer than two observations, when the sample standard deviation is zero
    (constant returns) or when it is not finite, exactly as the companion does.
    """
    arr = _as_returns(returns)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(arr.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Deepest peak-to-trough fall of the equity curve, as a non-positive fraction.

    ``min((equity - running_max) / running_max)`` over the cumprod equity curve.
    Returns NaN on an empty series.

    The curve starts at the first period's equity, not at a leading 1.0, exactly as
    the companion computes it. A loss on the very first period therefore sets the
    initial peak and is not counted as a drawdown.
    """
    equity = equity_curve(returns)
    if len(equity) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())
