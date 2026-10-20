"""Benchmarks and null models for a strategy return series (plan.md Phase 3, Fix 6).

Benchmarks answer "what would a trivial rule have earned on the same asset?":
buy-and-hold, a constant fraction of exposure, and the signal's own long-only version.
Null models answer "what does luck look like?": random +1/-1 positions and label
permutations pushed through the caller's pipeline. ``percentile_rank`` and
``permutation_p_value`` turn a real statistic plus a null draw into the numbers the
report shows next to the raw Sharpe.

Pure numpy and pandas: arrays in, arrays or floats out, no I/O. Seeds are explicit
arguments. Backtests are historical, not forward-looking.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from hedgefund.backtest.metrics import sharpe_ratio

DEFAULT_EXPOSURE = 0.5  # plan Fix 6: a constant 50% exposure benchmark
DEFAULT_NULL_PATHS = 1000  # config/default.yaml statistics.null_paths; plan Fix 6 asks for 1,000


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    return arr


def _check_n_paths(n_paths: int) -> None:
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")


def buy_and_hold(asset_returns: np.ndarray) -> np.ndarray:
    """Return series of a +1 position held every step: the asset returns themselves."""
    return _as_1d(asset_returns, "asset_returns").copy()


def constant_exposure(
    asset_returns: np.ndarray,
    exposure: float = DEFAULT_EXPOSURE,
) -> np.ndarray:
    """Return series of a fixed ``exposure`` fraction of the asset: ``exposure * r``.

    The remainder is held in cash earning nothing, matching the zero risk-free rate
    used by ``sharpe_ratio``.
    """
    return exposure * _as_1d(asset_returns, "asset_returns")


def long_only_positions(positions: np.ndarray) -> np.ndarray:
    """The signal's own long-only version: short positions become flat (clip at 0)."""
    return np.clip(_as_1d(positions, "positions"), 0.0, None)


def random_sign_null(
    asset_returns: np.ndarray,
    n_paths: int = DEFAULT_NULL_PATHS,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = sharpe_ratio,
) -> np.ndarray:
    """Random-sign null: an iid +1/-1 position each step, ``statistic`` per path.

    A coin-flip trader has no information, so the spread of this draw is what the
    statistic looks like for a strategy that is always fully invested but never right
    for a reason. Returns an array of shape ``(n_paths,)``.
    """
    _check_n_paths(n_paths)
    rets = _as_1d(asset_returns, "asset_returns")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_paths, rets.size))
    path_returns = signs * rets
    return np.array([float(statistic(path)) for path in path_returns], dtype=float)


def label_shuffle_null(
    labels: pd.Series,
    pipeline: Callable[[pd.Series], np.ndarray],
    n_paths: int = DEFAULT_NULL_PATHS,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = sharpe_ratio,
) -> np.ndarray:
    """Label-shuffle null: permute the label values, keep the index, rerun the pipeline.

    ``pipeline(shuffled_labels)`` must return the strategy return series as a 1-D array.
    Permuting keeps the exact regime frequencies and destroys the temporal order, so the
    draw shows what the statistic looks like when the labels carry no information about
    what comes next. Returns an array of shape ``(n_paths,)``.
    """
    _check_n_paths(n_paths)
    if labels.size < 2:
        raise ValueError(f"need at least 2 labels for a null model, got {labels.size}")
    values = labels.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n_paths, dtype=float)
    for p in range(n_paths):  # one full pipeline run per permutation
        shuffled = pd.Series(rng.permutation(values), index=labels.index, name=labels.name)
        out[p] = float(statistic(np.asarray(pipeline(shuffled), dtype=float)))
    return out


def _finite_null(null: np.ndarray) -> np.ndarray:
    arr = _as_1d(null, "null")
    return arr[np.isfinite(arr)]


def percentile_rank(real: float, null: np.ndarray) -> float:
    """Percentage of finite null values strictly below ``real``.

    NaN when ``real`` is not finite or when no null value is finite, so a strategy
    whose own statistic is undefined is never ranked.
    """
    finite = _finite_null(null)
    if not np.isfinite(real) or finite.size == 0:
        return float("nan")
    return float(100.0 * np.mean(finite < real))


def permutation_p_value(real: float, null: np.ndarray) -> float:
    """One-sided permutation p-value ``(count(null >= real) + 1) / (N + 1)``.

    ``N`` counts the finite null values only. The ``+1`` counts the real statistic as one
    more draw from the null, which keeps the p-value in ``(0, 1]`` and makes it exactly
    uniform on ``{1/(N+1), ..., 1}`` when the strategy has no edge (plan Fix 6). NaN when
    ``real`` is not finite or no null value is finite.
    """
    finite = _finite_null(null)
    if not np.isfinite(real) or finite.size == 0:
        return float("nan")
    count = int(np.sum(finite >= real))
    return float((count + 1) / (finite.size + 1))
