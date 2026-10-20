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


# --------------------------------------------------------------------------- #
# Phase 3: full metric set (plan Fix 6). Everything above this line is the Phase 1
# port and stays byte-identical; the reproduction tests depend on it.
# --------------------------------------------------------------------------- #
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)  # config backtest.horizons: 1d, 1w, 2w, 1m holds


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``values`` to a 1-D float array; raise naming the offending argument."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    return arr


def _one_way_turnover(positions: np.ndarray) -> np.ndarray:
    """|Δposition| per step. The first position is entered from flat, so it costs |p0|."""
    pos = _as_1d(positions, "positions")
    if len(pos) == 0:
        return pos
    return np.abs(np.diff(pos, prepend=0.0))


def total_return(returns: np.ndarray) -> float:
    """Compound return over the whole series: ``prod(1 + r) - 1``. NaN if empty."""
    arr = _as_returns(returns)
    if len(arr) == 0:
        return float("nan")
    return float(np.prod(1.0 + arr) - 1.0)


def annualised_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualised return ``(1 + total)^(periods_per_year / n) - 1``.

    NaN if empty, and NaN when the strategy lost more than everything (a non-positive
    terminal equity has no real geometric growth rate).
    """
    arr = _as_returns(returns)
    if len(arr) == 0:
        return float("nan")
    growth = 1.0 + total_return(arr)
    if growth <= 0.0:
        return float("nan")
    return float(growth ** (periods_per_year / len(arr)) - 1.0)


def sortino_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sortino ratio with a zero target: mean over downside deviation.

    Downside deviation is ``sqrt(mean(min(r, 0)^2))`` over every period (losses only
    contribute, gains count as zero). NaN when fewer than two observations or when the
    downside deviation is zero (no losing periods), mirroring ``sharpe_ratio``.
    """
    arr = _as_returns(returns)
    if len(arr) < 2:
        return float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(arr, 0.0) ** 2)))
    if downside == 0.0 or not np.isfinite(downside):
        return float("nan")
    return float(arr.mean() / downside * np.sqrt(periods_per_year))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised return divided by the absolute max drawdown. NaN if empty or no drawdown."""
    arr = _as_returns(returns)
    if len(arr) == 0:
        return float("nan")
    drawdown = max_drawdown(arr)
    if drawdown == 0.0 or not np.isfinite(drawdown):
        return float("nan")
    return float(annualised_return(arr, periods_per_year) / abs(drawdown))


def hit_rate(returns: np.ndarray) -> float:
    """Share of periods with ``r > 0`` among periods with ``r != 0``. NaN if none are active.

    Flat days (position zero, return exactly zero) are excluded so sitting in cash
    does not count as a loss.
    """
    arr = _as_returns(returns)
    active = arr[arr != 0.0]
    if len(active) == 0:
        return float("nan")
    return float(np.mean(active > 0.0))


def average_win(returns: np.ndarray) -> float:
    """Mean of the strictly positive period returns. NaN when there are none."""
    arr = _as_returns(returns)
    wins = arr[arr > 0.0]
    return float(wins.mean()) if len(wins) else float("nan")


def average_loss(returns: np.ndarray) -> float:
    """Mean of the strictly negative period returns (a negative number). NaN when none."""
    arr = _as_returns(returns)
    losses = arr[arr < 0.0]
    return float(losses.mean()) if len(losses) else float("nan")


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit over gross loss: ``sum(r > 0) / |sum(r < 0)|``. NaN when there are no losses."""
    arr = _as_returns(returns)
    gross_loss = float(arr[arr < 0.0].sum())
    if gross_loss == 0.0:
        return float("nan")
    return float(arr[arr > 0.0].sum() / abs(gross_loss))


def time_in_market(positions: np.ndarray) -> float:
    """Share of steps with a non-zero position (exposure of any sign). NaN if empty."""
    pos = _as_1d(positions, "positions")
    if len(pos) == 0:
        return float("nan")
    return float(np.mean(pos != 0.0))


def annual_turnover(positions: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """One-way turnover per year in units of notional: ``sum(|Δposition|) / n * periods_per_year``.

    The first position is entered from flat and counts in full. A sign-only rule that
    flips once a week turns over roughly 100 units of notional a year. NaN if empty.
    """
    pos = _as_1d(positions, "positions")
    if len(pos) == 0:
        return float("nan")
    return float(_one_way_turnover(pos).sum() / len(pos) * periods_per_year)


def horizon_pnl(
    positions: np.ndarray,
    asset_returns: np.ndarray,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[int, dict]:
    """P&L of holding each decision for ``h`` steps instead of one.

    For horizon ``h`` the k-th sample is ``positions[k] * (prod(1 + asset_returns[k:k+h]) - 1)``
    for ``k = 0 .. n - h``, so there are ``n - h + 1`` overlapping samples. The arrays are
    aligned as the walk-forward emits them: ``positions[k]`` is the decision that earns
    ``asset_returns[k]``. At ``h = 1`` this reproduces the gross per-step strategy returns.

    Returns ``{h: {"mean", "median", "hit_rate", "n"}}``; every statistic is NaN and
    ``n`` is 0 when the series is shorter than ``h``.
    """
    pos = _as_1d(positions, "positions")
    rets = _as_1d(asset_returns, "asset_returns")
    if len(pos) != len(rets):
        raise ValueError(
            f"positions and asset_returns must have equal length, got {len(pos)} and {len(rets)}"
        )
    out: dict[int, dict] = {}
    for h in horizons:
        if int(h) != h or h < 1:
            raise ValueError(f"horizons must be positive integers, got {h!r}")
        h = int(h)
        n_samples = len(pos) - h + 1
        if n_samples <= 0:
            nan = float("nan")
            out[h] = {"mean": nan, "median": nan, "hit_rate": nan, "n": 0}
            continue
        windows = np.lib.stride_tricks.sliding_window_view(1.0 + rets, h)
        pnl = pos[:n_samples] * (windows.prod(axis=1) - 1.0)
        out[h] = {
            "mean": float(pnl.mean()),
            "median": float(np.median(pnl)),
            "hit_rate": hit_rate(pnl),
            "n": int(n_samples),
        }
    return out


def metric_set(
    returns: np.ndarray,
    positions: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """The full Fix 6 metric set for one return series and its aligned positions.

    ``returns`` may be gross or net; call it twice to report both, never gross alone
    (``.claude/rules/no-lookahead.md``). Floats may be NaN; the report layer maps NaN
    to JSON null.
    """
    arr = _as_returns(returns)
    pos = _as_1d(positions, "positions")
    if len(arr) != len(pos):
        raise ValueError(
            f"returns and positions must have equal length, got {len(arr)} and {len(pos)}"
        )
    return {
        "sharpe": sharpe_ratio(arr, periods_per_year),
        "sortino": sortino_ratio(arr, periods_per_year),
        "calmar": calmar_ratio(arr, periods_per_year),
        "max_drawdown": max_drawdown(arr),
        "total_return": total_return(arr),
        "annual_return": annualised_return(arr, periods_per_year),
        "hit_rate": hit_rate(arr),
        "avg_win": average_win(arr),
        "avg_loss": average_loss(arr),
        "profit_factor": profit_factor(arr),
        "turnover": annual_turnover(pos, periods_per_year),
        "time_in_market": time_in_market(pos),
        "n_days": len(arr),
    }
