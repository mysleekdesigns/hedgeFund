"""Tests for ``hedgefund.backtest.benchmarks`` (plan.md Phase 3, Fix 6)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hedgefund.backtest.benchmarks import (
    buy_and_hold,
    constant_exposure,
    label_shuffle_null,
    long_only_positions,
    percentile_rank,
    permutation_p_value,
    random_sign_null,
)
from hedgefund.backtest.metrics import sharpe_ratio

N_DAYS = 500  # long enough that a null Sharpe is well defined per path, short enough to be fast


def _driftless_returns(seed: int, n: int = N_DAYS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


# ------------------------------------------------------------------- benchmarks


def test_buy_and_hold_returns_the_asset_returns() -> None:
    rets = _driftless_returns(1)
    out = buy_and_hold(rets)
    np.testing.assert_array_equal(out, rets)
    assert out is not rets  # a copy, so callers cannot alias the input


def test_constant_exposure_half_halves_the_returns() -> None:
    rets = _driftless_returns(2)
    np.testing.assert_allclose(constant_exposure(rets, 0.5), 0.5 * rets)
    np.testing.assert_allclose(constant_exposure(rets), 0.5 * rets)  # default is 50%


def test_long_only_positions_clips_shorts_to_flat() -> None:
    positions = np.array([-1.0, 0.0, 1.0, -0.5, 0.25])
    np.testing.assert_array_equal(long_only_positions(positions), [0.0, 0.0, 1.0, 0.0, 0.25])


def test_benchmarks_reject_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        buy_and_hold(np.zeros((3, 2)))


# ------------------------------------------------------------- random-sign null


def test_random_sign_null_has_one_statistic_per_path() -> None:
    null = random_sign_null(_driftless_returns(3), n_paths=7, seed=0)
    assert null.shape == (7,)
    assert np.all(np.isfinite(null))


def test_random_sign_null_is_deterministic_under_a_seed() -> None:
    rets = _driftless_returns(4)
    np.testing.assert_array_equal(
        random_sign_null(rets, n_paths=5, seed=11), random_sign_null(rets, n_paths=5, seed=11)
    )


def test_random_sign_null_differs_across_seeds() -> None:
    rets = _driftless_returns(5)
    a = random_sign_null(rets, n_paths=5, seed=1)
    b = random_sign_null(rets, n_paths=5, seed=2)
    assert not np.array_equal(a, b)


def test_random_sign_null_mean_sharpe_near_zero_on_driftless_series() -> None:
    # Each path's annualised Sharpe has sd ~ sqrt(252 / N_DAYS) ~ 0.71; the mean of 200
    # paths has sd ~ 0.05, so 0.15 is a 3-sigma tolerance on a seeded draw.
    null = random_sign_null(_driftless_returns(6), n_paths=200, seed=0)
    assert abs(null.mean()) < 0.15


def test_random_sign_null_accepts_a_custom_statistic() -> None:
    rets = _driftless_returns(7)
    null = random_sign_null(rets, n_paths=3, seed=0, statistic=lambda r: float(np.sum(np.abs(r))))
    # A +1/-1 position never changes |r|, so the sum of absolute returns is the same per path.
    np.testing.assert_allclose(null, np.sum(np.abs(rets)))


def test_random_sign_null_rejects_zero_paths() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        random_sign_null(_driftless_returns(8), n_paths=0)


@pytest.mark.slow
def test_random_sign_null_thousand_paths_slow() -> None:
    null = random_sign_null(_driftless_returns(9), n_paths=1000, seed=0)
    assert null.shape == (1000,)
    assert abs(null.mean()) < 0.07  # 3 sigma of 0.71 / sqrt(1000)


# ----------------------------------------------------------- label-shuffle null


def _labels(seed: int, n: int = 60) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.integers(0, 3, size=n), index=index, name="label")


def test_label_shuffle_null_calls_pipeline_once_per_path_with_permuted_values() -> None:
    labels = _labels(10)
    seen: list[pd.Series] = []

    def recording_pipeline(shuffled: pd.Series) -> np.ndarray:
        seen.append(shuffled)
        return _driftless_returns(len(seen))

    null = label_shuffle_null(labels, recording_pipeline, n_paths=6, seed=0)

    assert null.shape == (6,)
    assert len(seen) == 6
    for shuffled in seen:
        assert shuffled.index.equals(labels.index)  # the index is kept
        assert sorted(shuffled.to_numpy()) == sorted(labels.to_numpy())  # values permuted
    assert any(not shuffled.equals(labels) for shuffled in seen)  # the order actually moved


def test_label_shuffle_null_applies_the_statistic_to_the_pipeline_output() -> None:
    labels = _labels(11)
    fixed = _driftless_returns(12)
    null = label_shuffle_null(labels, lambda _s: fixed, n_paths=3, seed=0)
    np.testing.assert_allclose(null, sharpe_ratio(fixed))


def test_label_shuffle_null_is_deterministic_under_a_seed() -> None:
    labels = _labels(13)

    def pipeline(shuffled: pd.Series) -> np.ndarray:
        # A pipeline whose output depends on the permutation: label as a position.
        return (shuffled.to_numpy() - 1.0) * _driftless_returns(0, n=len(shuffled))

    a = label_shuffle_null(labels, pipeline, n_paths=4, seed=5)
    b = label_shuffle_null(labels, pipeline, n_paths=4, seed=5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, label_shuffle_null(labels, pipeline, n_paths=4, seed=6))


def test_label_shuffle_null_rejects_too_few_labels() -> None:
    with pytest.raises(ValueError, match="at least 2 labels"):
        label_shuffle_null(_labels(14, n=1), lambda s: np.zeros(1), n_paths=2)


# -------------------------------------------------------- p-value and percentile


def test_permutation_p_value_hand_values() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4])  # N = 4
    assert permutation_p_value(0.25, null) == pytest.approx((2 + 1) / (4 + 1))  # two >= 0.25
    assert permutation_p_value(0.3, null) == pytest.approx((2 + 1) / 5)  # ties count as >=
    assert permutation_p_value(1.0, null) == pytest.approx(1 / 5)  # beats every null draw
    assert permutation_p_value(0.0, null) == pytest.approx(1.0)  # beats none


def test_permutation_p_value_ignores_non_finite_null_draws() -> None:
    null = np.array([0.1, np.nan, 0.3, np.inf, -np.inf])
    # Finite draws are [0.1, 0.3] (N = 2); one of them is >= 0.2.
    assert permutation_p_value(0.2, null) == pytest.approx((1 + 1) / (2 + 1))


def test_permutation_p_value_is_nan_when_real_is_nan_or_null_is_empty() -> None:
    assert np.isnan(permutation_p_value(float("nan"), np.array([0.1, 0.2])))
    assert np.isnan(permutation_p_value(0.5, np.array([])))
    assert np.isnan(permutation_p_value(0.5, np.array([np.nan, np.nan])))


def test_percentile_rank_hand_values() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert percentile_rank(0.25, null) == pytest.approx(50.0)
    assert percentile_rank(0.3, null) == pytest.approx(50.0)  # strictly below: ties do not count
    assert percentile_rank(1.0, null) == pytest.approx(100.0)
    assert percentile_rank(0.0, null) == pytest.approx(0.0)


def test_percentile_rank_ignores_non_finite_null_draws() -> None:
    null = np.array([0.1, np.nan, 0.3, np.inf])
    assert percentile_rank(0.2, null) == pytest.approx(50.0)


def test_percentile_rank_is_nan_when_real_is_nan_or_null_is_empty() -> None:
    assert np.isnan(percentile_rank(float("nan"), np.array([0.1])))
    assert np.isnan(percentile_rank(0.5, np.array([])))


@given(
    real=st.floats(-10, 10),
    null=st.lists(st.floats(-10, 10), min_size=1, max_size=50),
)
def test_percentile_and_p_value_stay_in_range(real: float, null: list[float]) -> None:
    arr = np.array(null)
    pct = percentile_rank(real, arr)
    p = permutation_p_value(real, arr)
    assert 0.0 <= pct <= 100.0
    assert 0.0 < p <= 1.0
    # Both count the same ordering: percentile + 100 * (count >= real) / N == 100.
    assert pct + 100.0 * (p * (arr.size + 1) - 1) / arr.size == pytest.approx(100.0)
