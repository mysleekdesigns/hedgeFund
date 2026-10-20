"""Tests for ``hedgefund.backtest.significance`` (plan.md Phase 3, Fix 6).

The headline test is the plan's acceptance check: on a no-signal synthetic series the
permutation p-value is uniform and the Probabilistic Sharpe Ratio is near 0.5.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from hedgefund.backtest.metrics import sharpe_ratio
from hedgefund.backtest.significance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    permutation_test,
    probabilistic_sharpe_ratio,
)
from hedgefund.backtest.walk_forward import run_walk_forward
from hedgefund.regime.labels import label_regimes

# Real pipeline used by the uniformity test: the on-camera labeller settings on a short
# training window so a 300-day series still leaves ~220 traded days.
LABEL_WINDOW = 20
LABEL_THRESHOLD = 0.02
MIN_TRAIN = 60
N_DAYS = 300
DAILY_VOL = 0.01  # ~16% annualised, an ordinary equity index

PERMUTATION_KEYS = {
    "p_value_gross",
    "p_value_net",
    "percentile_gross",
    "percentile_net",
    "real_gross",
    "real_net",
    "null_mean_gross",
    "null_mean_net",
    "null_std_gross",
    "null_std_net",
    "n_paths",
}


def _no_signal_close(seed: int, n: int = N_DAYS) -> pd.Series:
    """Iid Gaussian returns with zero drift: no regime, no momentum, nothing to learn."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, DAILY_VOL, size=n - 1)
    close = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    return pd.Series(close, index=pd.bdate_range("2015-01-01", periods=n), name="close")


def _regime_pipeline(close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """label_regimes -> run_walk_forward, costs as the identity (gross == net)."""
    labels = label_regimes(close, window=LABEL_WINDOW, threshold=LABEL_THRESHOLD)
    gross = run_walk_forward(close, labels, MIN_TRAIN).strategy_returns.to_numpy(dtype=float)
    return gross, gross


def _halving_pipeline(close: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold gross; net is half of it, so gross and net results must differ."""
    gross = close.pct_change().dropna().to_numpy(dtype=float)
    return gross, 0.5 * gross


# ------------------------------------------------------------- permutation_test


def test_permutation_test_returns_the_exact_keys_and_n_paths() -> None:
    out = permutation_test(_no_signal_close(1), _halving_pipeline, n_paths=5, seed=0)
    assert set(out) == PERMUTATION_KEYS
    assert out["n_paths"] == 5


def test_permutation_test_rebuilds_each_path_on_the_same_index_from_the_first_price() -> None:
    close = _no_signal_close(2, n=40)
    seen: list[pd.Series] = []

    def recording(path: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        seen.append(path)
        rets = path.pct_change().dropna().to_numpy(dtype=float)
        return rets, rets

    permutation_test(close, recording, n_paths=3, seed=0)

    assert len(seen) == 4  # the real series, then three null paths
    assert seen[0] is close
    real_rets = np.sort(close.pct_change().dropna().to_numpy())
    for path in seen[1:]:
        assert path.index.equals(close.index)
        assert path.iloc[0] == close.iloc[0]
        # Permuting the returns keeps their multiset, so the final price is preserved too.
        np.testing.assert_allclose(np.sort(path.pct_change().dropna().to_numpy()), real_rets)
        assert path.iloc[-1] == pytest.approx(close.iloc[-1])
        assert not np.array_equal(path.to_numpy(), close.to_numpy())


def test_permutation_test_is_deterministic_under_a_seed() -> None:
    close = _no_signal_close(3, n=60)
    a = permutation_test(close, _halving_pipeline, n_paths=4, seed=7)
    b = permutation_test(close, _halving_pipeline, n_paths=4, seed=7)
    assert a == b


def test_permutation_test_gross_and_net_differ_when_costs_bite() -> None:
    out = permutation_test(_no_signal_close(4, n=60), _halving_pipeline, n_paths=4, seed=0)
    # Halving every return halves the mean and the std, so Sharpe is unchanged ...
    assert out["real_net"] == pytest.approx(out["real_gross"])
    # ... but a statistic that is not scale-free sees it.
    out_sum = permutation_test(
        _no_signal_close(4, n=60),
        _halving_pipeline,
        n_paths=4,
        seed=0,
        statistic=lambda r: float(np.sum(r)),
    )
    assert out_sum["real_net"] == pytest.approx(0.5 * out_sum["real_gross"])
    assert out_sum["null_mean_net"] == pytest.approx(0.5 * out_sum["null_mean_gross"])


def test_permutation_test_buy_and_hold_pipeline_is_invariant_to_shuffling_by_sum() -> None:
    # The sum of buy-and-hold returns is a permutation invariant, so every null draw equals
    # the real statistic and p = (N + 1) / (N + 1) = 1 while the percentile is 0. Rebuilding
    # prices from permuted returns adds ~1e-16 of rounding noise, so round the statistic.
    out = permutation_test(
        _no_signal_close(5, n=50),
        _halving_pipeline,
        n_paths=6,
        seed=0,
        statistic=lambda r: float(np.round(np.sum(r), 10)),
    )
    assert out["p_value_gross"] == pytest.approx(1.0)
    assert out["percentile_gross"] == pytest.approx(0.0)
    assert out["null_std_gross"] == pytest.approx(0.0, abs=1e-12)


def test_permutation_test_rejects_bad_inputs() -> None:
    close = _no_signal_close(6, n=30)
    with pytest.raises(ValueError, match="n_paths"):
        permutation_test(close, _halving_pipeline, n_paths=0)
    with pytest.raises(ValueError, match="at least 2 prices"):
        permutation_test(close.iloc[:1], _halving_pipeline, n_paths=2)
    with_nan = close.copy()
    with_nan.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        permutation_test(with_nan, _halving_pipeline, n_paths=2)


def _uniformity_check(n_series: int, n_paths: int) -> None:
    """Plan Fix 6 acceptance: no-signal series -> uniform p-value and PSR near 0.5.

    Each independent series gives one permutation p-value and one PSR of the real
    strategy. Under exchangeable (iid) returns the real series is one more draw from
    the permutation null, so ``p = (count + 1) / (N + 1)`` is exactly uniform on the
    lattice ``{1/(N+1), ..., 1}``. We compare against the continuous U(0, 1) with a
    Kolmogorov-Smirnov test at the 1% level: the lattice moves the KS statistic by at most
    ``1/(N+1)`` (0.05 for N = 19), well inside the 1% critical value ``1.63/sqrt(M)`` (0.26 for
    M = 40), and 1% rather than 5% keeps a seeded test from failing on an ordinary draw while
    still catching a p-value that piles up near 0 or 1. PSR of a zero-edge strategy is a
    probability integral transform, so it is uniform with mean 0.5 and standard deviation
    ``0.29 / sqrt(M)`` (0.046 for M = 40); the +/-0.1 band is more than two sigma.
    """
    p_values = np.empty(n_series)
    psr_values = np.empty(n_series)
    for i in range(n_series):
        close = _no_signal_close(seed=1000 + i)
        out = permutation_test(close, _regime_pipeline, n_paths=n_paths, seed=i)
        p_values[i] = out["p_value_gross"]
        psr_values[i] = probabilistic_sharpe_ratio(_regime_pipeline(close)[0])

    assert np.all(np.isfinite(p_values)), "every no-signal series must yield a p-value"
    ks = sp_stats.kstest(p_values, "uniform")
    assert ks.pvalue > 0.01, f"p-values not uniform: KS D={ks.statistic:.3f}, p={ks.pvalue:.4f}"
    assert np.all((p_values > 0.0) & (p_values <= 1.0))
    assert abs(np.nanmean(psr_values) - 0.5) < 0.1, f"mean PSR {np.nanmean(psr_values):.3f}"


def test_no_signal_series_gives_uniform_p_value_and_psr_near_half() -> None:
    _uniformity_check(n_series=40, n_paths=19)


@pytest.mark.slow
def test_no_signal_series_gives_uniform_p_value_and_psr_near_half_slow() -> None:
    _uniformity_check(n_series=200, n_paths=99)


# ------------------------------------------------------- probabilistic Sharpe


def _returns_with_sharpe(seed: int, per_period_sharpe: float, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(per_period_sharpe * DAILY_VOL, DAILY_VOL, size=n)


def test_psr_near_one_for_a_long_strongly_positive_series() -> None:
    assert probabilistic_sharpe_ratio(_returns_with_sharpe(1, 1.0)) > 0.999


def test_psr_near_zero_for_a_strongly_negative_series() -> None:
    assert probabilistic_sharpe_ratio(_returns_with_sharpe(2, -1.0)) < 0.001


def test_psr_is_half_when_the_benchmark_equals_the_sample_sharpe() -> None:
    rets = _returns_with_sharpe(3, 0.3, n=500)
    sr = rets.mean() / rets.std(ddof=1)  # the per-period SR the function computes
    assert probabilistic_sharpe_ratio(rets, benchmark_sharpe=sr) == 0.5  # Phi(0) exactly


def test_psr_uses_per_period_not_annualised_sharpe() -> None:
    rets = _returns_with_sharpe(4, 0.1, n=800)
    sr = rets.mean() / rets.std(ddof=1)
    annualised = sharpe_ratio(rets)
    assert annualised == pytest.approx(sr * np.sqrt(252))
    # Benchmarking at the per-period SR gives 0.5; at the annualised figure it would not.
    assert probabilistic_sharpe_ratio(rets, benchmark_sharpe=sr) == 0.5
    assert probabilistic_sharpe_ratio(rets, benchmark_sharpe=annualised) < 0.5


def test_psr_decreases_with_a_higher_benchmark() -> None:
    rets = _returns_with_sharpe(5, 0.2, n=500)
    assert probabilistic_sharpe_ratio(rets, 0.0) > probabilistic_sharpe_ratio(rets, 0.1)


def test_psr_is_nan_for_too_short_or_constant_returns() -> None:
    assert np.isnan(probabilistic_sharpe_ratio(np.array([0.01])))
    assert np.isnan(probabilistic_sharpe_ratio(np.array([])))
    assert np.isnan(probabilistic_sharpe_ratio(np.full(50, 0.5)))  # exactly constant


def test_psr_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        probabilistic_sharpe_ratio(np.zeros((4, 2)))


# ------------------------------------------------------ min_track_record_length


def _symmetric_three_point_returns(mean: float, spread: float, k: int = 50) -> np.ndarray:
    """``k`` returns at ``mean - spread``, ``4k`` at ``mean`` and ``k`` at ``mean + spread``.

    Symmetric about the mean, so skew g3 = 0. A share q = 1/3 of the mass sits at
    ``mean +/- spread`` and the rest at the mean, so the biased non-excess kurtosis is
    m4 / m2^2 = (q a^4) / (q a^2)^2 = 1 / q = 3 exactly: Gaussian moments with no
    sampling noise.
    """
    return np.concatenate(
        [np.full(k, mean - spread), np.full(4 * k, mean), np.full(k, mean + spread)]
    )


def test_min_track_record_length_hand_value_for_gaussian_moments() -> None:
    rets = _symmetric_three_point_returns(mean=0.002, spread=0.01)
    sr = rets.mean() / rets.std(ddof=1)
    z = sp_stats.norm.ppf(0.95)
    expected = 1.0 + (1.0 + sr**2 / 2.0) * (z / sr) ** 2  # g3 = 0, g4 = 3
    assert min_track_record_length(rets, 0.0, 0.95) == pytest.approx(expected, rel=1e-9)


def test_min_track_record_length_is_infinite_when_sharpe_does_not_beat_target() -> None:
    rets = _symmetric_three_point_returns(mean=0.002, spread=0.01)
    sr = rets.mean() / rets.std(ddof=1)
    assert min_track_record_length(rets, target_sharpe=sr) == float("inf")  # equal: not above
    assert min_track_record_length(rets, target_sharpe=sr + 0.1) == float("inf")
    assert min_track_record_length(-rets) == float("inf")  # negative Sharpe vs target 0


def test_min_track_record_length_increases_with_confidence() -> None:
    rets = _returns_with_sharpe(6, 0.1, n=500)
    low = min_track_record_length(rets, confidence=0.90)
    mid = min_track_record_length(rets, confidence=0.95)
    high = min_track_record_length(rets, confidence=0.99)
    assert low < mid < high


def test_min_track_record_length_increases_as_target_approaches_sharpe() -> None:
    rets = _returns_with_sharpe(7, 0.2, n=500)
    sr = rets.mean() / rets.std(ddof=1)
    assert min_track_record_length(rets, 0.0) < min_track_record_length(rets, sr / 2)


def test_min_track_record_length_is_nan_when_undefined() -> None:
    assert np.isnan(min_track_record_length(np.array([0.01])))
    assert np.isnan(min_track_record_length(np.full(20, 0.5)))  # exactly constant


def test_min_track_record_length_rejects_confidence_outside_unit_interval() -> None:
    rets = _returns_with_sharpe(8, 0.1, n=100)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="confidence"):
            min_track_record_length(rets, confidence=bad)


# ----------------------------------------------------------- expected_max_sharpe


def test_expected_max_sharpe_is_zero_for_a_single_trial() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_expected_max_sharpe_hand_value_for_two_trials() -> None:
    # Phi^-1(1 - 1/2) = 0, so only the Euler-Mascheroni term survives for N = 2.
    gamma = np.euler_gamma
    expected = np.sqrt(0.25) * gamma * sp_stats.norm.ppf(1.0 - 1.0 / (2.0 * np.e))
    assert expected_max_sharpe(2, 0.25) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_increases_with_trials_and_variance() -> None:
    assert (
        expected_max_sharpe(2, 1.0) < expected_max_sharpe(10, 1.0) < expected_max_sharpe(100, 1.0)
    )
    assert (
        expected_max_sharpe(10, 0.5) < expected_max_sharpe(10, 1.0) < expected_max_sharpe(10, 2.0)
    )


def test_expected_max_sharpe_is_zero_for_zero_variance() -> None:
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_expected_max_sharpe_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError, match="sharpe_variance"):
        expected_max_sharpe(5, -0.1)
    with pytest.raises(ValueError, match="sharpe_variance"):
        expected_max_sharpe(5, float("nan"))


# ---------------------------------------------------------- deflated_sharpe_ratio


def test_deflated_sharpe_ratio_has_exact_keys() -> None:
    out = deflated_sharpe_ratio(
        _returns_with_sharpe(9, 0.1, n=500), n_trials=15, sharpe_variance=0.01
    )
    assert set(out) == {"dsr", "sr0", "n_trials", "sharpe_variance"}
    assert out["n_trials"] == 15
    assert out["sharpe_variance"] == 0.01


def test_deflated_sharpe_ratio_is_at_most_the_psr_when_several_trials_were_run() -> None:
    rets = _returns_with_sharpe(10, 0.1, n=500)
    out = deflated_sharpe_ratio(rets, n_trials=15, sharpe_variance=0.01)
    psr = probabilistic_sharpe_ratio(rets)
    assert out["sr0"] > 0.0
    assert out["dsr"] < psr
    assert out["dsr"] == pytest.approx(
        probabilistic_sharpe_ratio(rets, benchmark_sharpe=out["sr0"])
    )


def test_deflated_sharpe_ratio_equals_psr_for_a_single_trial() -> None:
    rets = _returns_with_sharpe(11, 0.1, n=500)
    out = deflated_sharpe_ratio(rets, n_trials=1, sharpe_variance=0.01)
    assert out["sr0"] == 0.0
    assert out["dsr"] == probabilistic_sharpe_ratio(rets)


def test_deflated_sharpe_ratio_falls_as_more_trials_are_recorded() -> None:
    rets = _returns_with_sharpe(12, 0.1, n=500)
    few = deflated_sharpe_ratio(rets, n_trials=5, sharpe_variance=0.01)["dsr"]
    many = deflated_sharpe_ratio(rets, n_trials=50, sharpe_variance=0.01)["dsr"]
    assert many < few
