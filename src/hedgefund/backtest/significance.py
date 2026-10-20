"""Statistical significance of a backtest (plan.md Phase 3, Fix 6).

Three questions, three tools:

* Is the result better than luck? ``permutation_test`` shuffles the asset's daily
  returns, reruns the caller's entire pipeline (labeller, walk-forward, costs) on each
  shuffled path and reports ``p = (count + 1) / (N + 1)`` with and without costs.
  Running the whole pipeline on every shuffle keeps the selection effect inside the null.
* Is the Sharpe ratio distinguishable from a benchmark given the sample size, skew and
  fat tails? ``probabilistic_sharpe_ratio`` and ``min_track_record_length`` (Bailey and
  Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2)).
* Does it survive the number of parameter combinations tried? ``expected_max_sharpe``
  and ``deflated_sharpe_ratio`` (Bailey and Lopez de Prado 2014, "The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal
  of Portfolio Management 40(5)).

All Sharpe ratios here are PER PERIOD (daily for daily returns), not annualised: the
PSR and DSR formulas are written for the per-period statistic. Pure numpy, pandas and
scipy: arrays in, floats or dicts out, no I/O. Seeds are explicit arguments.
Backtests are historical, not forward-looking.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from hedgefund.backtest.benchmarks import (
    DEFAULT_NULL_PATHS,
    percentile_rank,
    permutation_p_value,
)
from hedgefund.backtest.metrics import sharpe_ratio

DEFAULT_CONFIDENCE = 0.95  # plan Fix 6 minimum track record length at the usual 95% level
GAUSSIAN_KURTOSIS = 3.0  # non-excess kurtosis of a normal distribution

Pipeline = Callable[[pd.Series], tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------- permutation


def _daily_simple_returns(close: pd.Series) -> np.ndarray:
    if close.size < 2:
        raise ValueError(f"close needs at least 2 prices to permute returns, got {close.size}")
    rets = close.pct_change().dropna().to_numpy(dtype=float)
    if rets.size != close.size - 1:
        raise ValueError("close must not contain NaN: the permuted path keeps close.index")
    return rets


def _rebuild_close(first_price: float, returns: np.ndarray, index: pd.Index) -> pd.Series:
    """Price path ``first_price * cumprod(1 + r)`` with ``first_price`` in front."""
    path = np.empty(returns.size + 1, dtype=float)
    path[0] = first_price
    path[1:] = first_price * np.cumprod(1.0 + returns)
    return pd.Series(path, index=index, dtype=float)


def _finite_summary(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    return float(finite.mean()), std


def permutation_test(
    close: pd.Series,
    pipeline: Pipeline,
    *,
    n_paths: int = DEFAULT_NULL_PATHS,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = sharpe_ratio,
) -> dict:
    """Return-permutation test through the full pipeline, with and without costs.

    ``pipeline(close) -> (gross_returns, net_returns)`` is the caller's whole backtest
    (labeller, walk-forward, cost model). Each of the ``n_paths`` null paths permutes the
    daily simple returns of ``close``, rebuilds a price series from ``close.iloc[0]`` on
    the same index and reruns the pipeline on it. The p-value is
    ``(count(null >= real) + 1) / (N + 1)`` (plan Fix 6) and the percentile is the share
    of the null strictly below the real statistic; both ignore non-finite null draws.

    Shuffling destroys every temporal pattern in the returns while keeping their
    marginal distribution, so a strategy that only exploits the return distribution
    (not its ordering) scores a uniform p-value. Reporting the null with and without
    costs matters because random signals that trade more pay more, which can make the
    cost-inclusive null look too easy to beat.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    rets = _daily_simple_returns(close)
    first_price = float(close.iloc[0])

    real_gross_returns, real_net_returns = pipeline(close)
    real_gross = float(statistic(np.asarray(real_gross_returns, dtype=float)))
    real_net = float(statistic(np.asarray(real_net_returns, dtype=float)))

    rng = np.random.default_rng(seed)
    null_gross = np.empty(n_paths, dtype=float)
    null_net = np.empty(n_paths, dtype=float)
    for p in range(n_paths):  # one full pipeline run per permutation
        path = _rebuild_close(first_price, rng.permutation(rets), close.index)
        gross, net = pipeline(path)
        null_gross[p] = float(statistic(np.asarray(gross, dtype=float)))
        null_net[p] = float(statistic(np.asarray(net, dtype=float)))

    mean_gross, std_gross = _finite_summary(null_gross)
    mean_net, std_net = _finite_summary(null_net)
    return {
        "p_value_gross": permutation_p_value(real_gross, null_gross),
        "p_value_net": permutation_p_value(real_net, null_net),
        "percentile_gross": percentile_rank(real_gross, null_gross),
        "percentile_net": percentile_rank(real_net, null_net),
        "real_gross": real_gross,
        "real_net": real_net,
        "null_mean_gross": mean_gross,
        "null_mean_net": mean_net,
        "null_std_gross": std_gross,
        "null_std_net": std_net,
        "n_paths": int(n_paths),
    }


# ------------------------------------------------------------- probabilistic Sharpe


def _sharpe_moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    """Per-period ``(SR, skew g3, non-excess kurtosis g4, n)``; SR is NaN when undefined.

    Skew and kurtosis are the biased sample moments (``scipy.stats`` defaults), as in
    Bailey and Lopez de Prado (2012) eq. 8. SR uses the sample standard deviation
    (``ddof=1``) so the per-period figure is ``sharpe_ratio(returns, 1)``.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"returns must be a 1-D array of period returns, got shape {arr.shape}")
    n = int(arr.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    std = arr.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan"), float("nan"), float("nan"), n
    sr = float(arr.mean() / std)
    g3 = float(sp_stats.skew(arr))
    g4 = float(sp_stats.kurtosis(arr, fisher=False))
    return sr, g3, g4, n


def _sharpe_variance_scale(sr: float, g3: float, g4: float) -> float:
    """``1 - g3*SR + (g4 - 1)/4 * SR^2``: ``(n-1) * Var(SR_hat)`` for non-normal returns.

    Bailey and Lopez de Prado (2012) eq. 8, from Mertens (2002). Equals 1 + SR^2/2 for
    Gaussian returns (g3 = 0, g4 = 3).
    """
    return 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sharpe: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds ``benchmark_sharpe``.

    Bailey and Lopez de Prado (2012) eq. 11::

        PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with ``SR`` the per-period sample Sharpe, ``g3`` the skew, ``g4`` the (non-excess)
    kurtosis and ``n`` the number of periods. ``benchmark_sharpe`` is also per period.
    Returns NaN when ``n < 2``, when the standard deviation is zero or when the
    variance term under the root is not positive (the asymptotic formula breaks down).
    PSR of a zero-edge strategy is a probability integral transform, so it is
    uniform on (0, 1) with mean 0.5 (plan Fix 6 acceptance test).
    """
    sr, g3, g4, n = _sharpe_moments(returns)
    if not np.isfinite(sr):
        return float("nan")
    scale = _sharpe_variance_scale(sr, g3, g4)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    z = (sr - benchmark_sharpe) * np.sqrt(n - 1) / np.sqrt(scale)
    return float(sp_stats.norm.cdf(z))


def min_track_record_length(
    returns: np.ndarray,
    target_sharpe: float = 0.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """Periods of history needed before PSR(``target_sharpe``) reaches ``confidence``.

    Bailey and Lopez de Prado (2012) eq. 13::

        MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2) * ( Z_conf / (SR - SR*) )^2

    in periods, with ``Z_conf = Phi^-1(confidence)``. Infinite when the sample Sharpe does
    not exceed the target (no amount of history would make the evidence conclusive);
    NaN when the Sharpe is undefined (``n < 2`` or zero standard deviation) or the
    variance term is not positive. ``confidence`` must lie strictly inside (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    sr, g3, g4, _n = _sharpe_moments(returns)
    if not np.isfinite(sr):
        return float("nan")
    scale = _sharpe_variance_scale(sr, g3, g4)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    if sr <= target_sharpe:
        return float("inf")
    z_conf = sp_stats.norm.ppf(confidence)
    return float(1.0 + scale * (z_conf / (sr - target_sharpe)) ** 2)


# ------------------------------------------------------------------- deflated Sharpe


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` zero-skill trials.

    Bailey and Lopez de Prado (2014) eq. 5, the Euler-Mascheroni approximation to the
    expected maximum of ``N`` iid normals with variance ``V``::

        SR0 = sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)) )

    with ``gamma`` the Euler-Mascheroni constant (``numpy.euler_gamma``). ``sharpe_variance``
    is the variance of the per-period Sharpe ratios across the trials of the parameter
    search. Returns 0.0 for a single trial (nothing was selected). Raises ``ValueError``
    for ``n_trials < 1`` or a negative variance.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sharpe_variance < 0 or not np.isfinite(sharpe_variance):
        raise ValueError(
            f"sharpe_variance must be a finite non-negative number, got {sharpe_variance}"
        )
    if n_trials == 1:
        return 0.0
    gamma = np.euler_gamma
    z_max = (1.0 - gamma) * sp_stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_max += gamma * sp_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * z_max)


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, sharpe_variance: float) -> dict:
    """Deflated Sharpe ratio: PSR against the Sharpe the best of ``n_trials`` would reach by luck.

    Bailey and Lopez de Prado (2014) eq. 6: ``DSR = PSR(SR0)`` with ``SR0`` from
    ``expected_max_sharpe``. ``n_trials`` and ``sharpe_variance`` come from the
    window x threshold (x rule) parameter search; ``sharpe_variance`` is the variance of
    the PER-PERIOD Sharpe across those trials. Returns
    ``{"dsr", "sr0", "n_trials", "sharpe_variance"}``; ``dsr <= psr`` whenever ``sr0 >= 0``.
    """
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    return {
        "dsr": probabilistic_sharpe_ratio(returns, benchmark_sharpe=sr0),
        "sr0": sr0,
        "n_trials": int(n_trials),
        "sharpe_variance": float(sharpe_variance),
    }
