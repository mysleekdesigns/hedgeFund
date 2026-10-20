"""Tests for transition-matrix estimation and chain algebra in ``hedgefund.regime.markov``."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedgefund.regime.labels import label_regimes
from hedgefund.regime.markov import (
    build_transition_matrix,
    count_transitions,
    dirichlet_shrinkage,
    drift_tvd,
    effective_sample_size,
    ew_transition_counts,
    ew_transition_matrix,
    mixing_time,
    normalise_rows,
    nstep_forecast,
    rolling_transition_counts,
    rolling_transition_matrix,
    run_length_persistence,
    stationary_distribution,
    validate_transition_matrix,
)

TRUE_MATRIX = np.array(
    [
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ]
)


def _simulate_chain(matrix: np.ndarray, n: int, seed: int, start: int = 1) -> np.ndarray:
    """Sample ``n`` states from ``matrix`` with an explicit ``seed``, one uniform draw per step.

    ``start`` is the state the chain is in before the first draw (default Sideways).
    """
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(n)
    path = np.empty(n, dtype=int)
    state = start
    for t in range(n):
        state = int(np.searchsorted(cumulative[state], draws[t]))
        path[t] = state
    return path


def _loop_counts(labels: np.ndarray, n_states: int = 3) -> np.ndarray:
    """The companion script's Python-loop count, kept as the reference for the vectorised one."""
    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(labels) - 1):
        counts[labels[i], labels[i + 1]] += 1.0
    return counts


# Random row-stochastic 3x3 matrices with every entry > 0.01 (hence ergodic). Entries are
# drawn in [0.05, 1] and normalised, so the smallest possible entry is 0.05 / 2.05 > 0.02.
_positive_entry = st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False)
ergodic_matrices = st.lists(_positive_entry, min_size=9, max_size=9).map(
    lambda cells: normalise_rows(np.array(cells).reshape(3, 3))
)

# Fast-mixing matrices for the 50-step convergence test. Entries are drawn in [0.3, 1] and
# normalised, so every entry is at least 0.3 / 2.3 = 0.13. The Dobrushin bound gives a
# contraction rate of at most 1 - 3 * 0.13 = 0.61 per step, and 0.61 ** 50 = 2e-11 < 1e-6,
# so the assertion below is guaranteed, not luck. (The wider strategy above admits chains
# with a second eigenvalue near 0.76, which need more than 50 steps to reach 1e-6.)
_fast_entry = st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False)
fast_mixing_matrices = st.lists(_fast_entry, min_size=9, max_size=9).map(
    lambda cells: normalise_rows(np.array(cells).reshape(3, 3))
)

label_sequences = st.lists(st.integers(min_value=0, max_value=2), min_size=0, max_size=200).map(
    lambda seq: np.array(seq, dtype=int)
)


# --------------------------------------------------------------------------- #
# count_transitions / normalise_rows / build_transition_matrix
# --------------------------------------------------------------------------- #
def test_count_transitions_hand_example() -> None:
    labels = [0, 0, 1, 2, 2, 1, 0]
    expected = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(count_transitions(labels), expected)


@given(labels=label_sequences)
def test_count_transitions_vectorised_equals_loop(labels: np.ndarray) -> None:
    np.testing.assert_array_equal(count_transitions(labels), _loop_counts(labels))


def test_count_transitions_short_sequences_are_all_zero() -> None:
    np.testing.assert_array_equal(count_transitions([]), np.zeros((3, 3)))
    np.testing.assert_array_equal(count_transitions([2]), np.zeros((3, 3)))


def test_count_transitions_total_is_length_minus_one() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=500)
    assert count_transitions(labels).sum() == len(labels) - 1


@pytest.mark.parametrize("bad", [[0, 3, 1], [-1, 0], [[0, 1], [1, 2]], [0.5, 1.0]])
def test_count_transitions_rejects_bad_labels(bad: list) -> None:
    with pytest.raises(ValueError, match="labels"):
        count_transitions(bad)


def test_normalise_rows_leaves_zero_rows_zero() -> None:
    # Companion behaviour: an empty row is divided by 1.0 and stays all zeros, not uniform.
    counts = np.array([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 3.0]])
    result = normalise_rows(counts)
    np.testing.assert_allclose(result[0], [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result[2], [0.25, 0.0, 0.75])


def test_normalise_rows_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalise_rows(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_build_transition_matrix_matches_companion_arithmetic() -> None:
    # Reproduce the companion's loop + divide exactly to confirm the port is bit-identical.
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 3, size=1000)
    counts = _loop_counts(labels)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    np.testing.assert_array_equal(build_transition_matrix(labels), counts / row_sums)


@given(labels=label_sequences)
def test_rows_of_every_estimated_matrix_sum_to_one(labels: np.ndarray) -> None:
    matrix = build_transition_matrix(labels)
    sources = set(labels[:-1].tolist())
    for state in range(3):
        if state in sources:
            assert abs(matrix[state].sum() - 1.0) < 1e-12
        else:
            np.testing.assert_array_equal(matrix[state], np.zeros(3))
    assert (matrix >= 0).all() and (matrix <= 1).all()
    validate_transition_matrix(matrix)


def test_synthetic_chain_estimate_converges_to_true_matrix() -> None:
    # 20000 steps gives roughly 6000 visits per state. The binomial standard error of a
    # cell near 0.8 is sqrt(0.8 * 0.2 / 6000) = 0.005, so 0.02 is about 4 sigma.
    labels = _simulate_chain(TRUE_MATRIX, n=20_000, seed=12345)
    estimate = build_transition_matrix(labels)
    np.testing.assert_allclose(estimate, TRUE_MATRIX, atol=0.02)
    validate_transition_matrix(estimate)


def test_synthetic_chain_is_reproducible_with_explicit_seed() -> None:
    first = _simulate_chain(TRUE_MATRIX, 2000, seed=99)
    second = _simulate_chain(TRUE_MATRIX, 2000, seed=99)
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# nstep_forecast
# --------------------------------------------------------------------------- #
def test_nstep_zero_is_identity_and_one_is_matrix() -> None:
    np.testing.assert_array_equal(nstep_forecast(TRUE_MATRIX, 0), np.eye(3))
    np.testing.assert_array_equal(nstep_forecast(TRUE_MATRIX, 1), TRUE_MATRIX)


def test_nstep_two_equals_matrix_product() -> None:
    np.testing.assert_allclose(nstep_forecast(TRUE_MATRIX, 2), TRUE_MATRIX @ TRUE_MATRIX)


def test_nstep_negative_raises() -> None:
    with pytest.raises(ValueError, match="n must be >= 0"):
        nstep_forecast(TRUE_MATRIX, -1)


@given(matrix=ergodic_matrices)
def test_nstep_rows_stay_stochastic(matrix: np.ndarray) -> None:
    for n in (1, 5, 50):
        validate_transition_matrix(nstep_forecast(matrix, n))


# --------------------------------------------------------------------------- #
# stationary_distribution
# --------------------------------------------------------------------------- #
def test_stationary_distribution_known_answer() -> None:
    # A symmetric birth-death chain whose stationary vector is (1/4, 1/2, 1/4).
    matrix = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
    np.testing.assert_allclose(stationary_distribution(matrix), [0.25, 0.5, 0.25], atol=1e-12)


def test_stationary_distribution_matches_companion_algorithm() -> None:
    eigvals, eigvecs = np.linalg.eig(TRUE_MATRIX.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    np.testing.assert_array_equal(stationary_distribution(TRUE_MATRIX), vec / vec.sum())


def test_stationary_distribution_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        stationary_distribution(np.ones((2, 3)))


@given(matrix=ergodic_matrices)
def test_stationary_is_left_eigenvector_and_sums_to_one(matrix: np.ndarray) -> None:
    pi = stationary_distribution(matrix)
    assert abs(pi.sum() - 1.0) < 1e-10
    assert (pi >= 0).all()
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-10)


def test_rows_of_p_to_the_fifty_equal_stationary_for_known_matrix() -> None:
    pi = stationary_distribution(TRUE_MATRIX)
    for row in nstep_forecast(TRUE_MATRIX, 50):
        np.testing.assert_allclose(row, pi, atol=1e-6)


@settings(deadline=None)
@given(matrix=fast_mixing_matrices)
def test_rows_of_p_to_the_fifty_equal_stationary(matrix: np.ndarray) -> None:
    pi = stationary_distribution(matrix)
    limit = nstep_forecast(matrix, 50)
    for row in limit:
        np.testing.assert_allclose(row, pi, atol=1e-6)


@settings(deadline=None)
@given(matrix=ergodic_matrices)
def test_p_to_the_n_converges_for_any_ergodic_matrix(matrix: np.ndarray) -> None:
    # For a slower chain choose n from the Dobrushin bound: (1 - 3 * min_entry) ** n <= 1e-9.
    # A rate of 0 (every row identical) converges in one step, so n is at least 1.
    rate = 1.0 - 3.0 * matrix.min()
    n = 1 if rate <= 0 else max(1, int(np.ceil(np.log(1e-9) / np.log(rate))))
    pi = stationary_distribution(matrix)
    for row in nstep_forecast(matrix, n):
        np.testing.assert_allclose(row, pi, atol=1e-9)


# --------------------------------------------------------------------------- #
# validate_transition_matrix
# --------------------------------------------------------------------------- #
def test_validate_accepts_matrix_with_all_zero_row() -> None:
    validate_transition_matrix(np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_validate_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_transition_matrix(np.eye(2))


def test_validate_rejects_entries_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_transition_matrix(np.array([[1.5, -0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_validate_rejects_row_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="do not sum to 1"):
        validate_transition_matrix(np.array([[0.5, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_validate_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_transition_matrix(np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


# --------------------------------------------------------------------------- #
# Phase 2: stride counting and effective sample size (Fix 1)
# --------------------------------------------------------------------------- #
IID_WINDOW = 20  # the labeller window; stride must equal it for non-overlapping windows
IID_THRESHOLD = 0.02  # the on-camera +/-2% threshold, so all three states occur often
IID_DAILY_VOL = 0.01  # 1% daily vol: 20-day vol of 4.5%, so |return| > 2% about 65% of the time
IID_N_DAYS = 20_000  # 1000 stride-20 labels, about 330 per state; see the test comment


def _iid_labels() -> np.ndarray:
    """Daily labels of a 20-day labeller run on iid Gaussian log returns (no true regimes)."""
    rng = np.random.default_rng(2024)
    log_returns = rng.normal(0.0, IID_DAILY_VOL, size=IID_N_DAYS)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(log_returns)),
        index=pd.bdate_range("2000-01-03", periods=IID_N_DAYS),
    )
    return label_regimes(close, window=IID_WINDOW, threshold=IID_THRESHOLD).to_numpy()


def _marginal_frequencies(labels: np.ndarray, n_states: int = 3) -> np.ndarray:
    return np.bincount(labels, minlength=n_states) / labels.size


def test_stride_one_is_bit_identical_to_existing_counts() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 3, size=1000)
    assert np.array_equal(count_transitions(labels, stride=1), count_transitions(labels))
    assert np.array_equal(
        build_transition_matrix(labels, stride=1), build_transition_matrix(labels)
    )


def test_stride_w_equals_fit_on_subsampled_labels() -> None:
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 3, size=1000)
    for stride in (2, 5, 20):
        np.testing.assert_array_equal(
            count_transitions(labels, stride=stride), count_transitions(labels[::stride])
        )
        np.testing.assert_array_equal(
            build_transition_matrix(labels, stride=stride),
            build_transition_matrix(labels[::stride]),
        )


def test_stride_zero_raises() -> None:
    with pytest.raises(ValueError, match="stride must be >= 1"):
        count_transitions([0, 1, 2], stride=0)
    with pytest.raises(ValueError, match="stride must be >= 1"):
        build_transition_matrix([0, 1, 2], stride=0)


def test_iid_series_inflates_overlapping_diagonal_but_not_stride_diagonal() -> None:
    # The returns are iid, so the true next-day label is independent of today's label
    # given nothing, and the honest diagonal is the marginal frequency of each state. The
    # 20-day labeller reuses 19 of 20 returns from one day to the next, so the overlapping
    # (stride-1) diagonal is mostly the labeller's own persistence (plan.md Fix 1): we
    # require it to beat the marginal by more than 0.3 absolute, far above any sampling
    # noise on ~6000 observations per state. Stride-20 labels use disjoint return windows
    # and are genuinely independent, so their diagonal should sit at the marginal. With
    # 20000 days there are 1000 stride labels, about 330 per state: the binomial standard
    # error of a cell near 1/3 is sqrt(0.33 * 0.67 / 330) = 0.026, so 0.1 is about 4 sigma.
    # (5000 days would leave fewer than 100 stride observations per state, which is why the
    # series is longer than that.)
    labels = _iid_labels()
    marginal = _marginal_frequencies(labels)
    overlapping = build_transition_matrix(labels, stride=1)
    stride_counts = count_transitions(labels, stride=IID_WINDOW)
    stride = normalise_rows(stride_counts)
    tested = 0
    for state in range(3):
        assert overlapping[state, state] - marginal[state] > 0.3
        if stride_counts[state].sum() >= 100:
            assert abs(stride[state, state] - marginal[state]) < 0.1
            tested += 1
    assert tested == 3, "every state should have at least 100 stride observations"


def test_effective_sample_size_divides_row_totals_by_window() -> None:
    counts = np.array([[30.0, 10.0, 0.0], [5.0, 5.0, 10.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(effective_sample_size(counts, window=20), [2.0, 1.0, 0.0])
    np.testing.assert_allclose(effective_sample_size(counts, window=1), [40.0, 20.0, 0.0])


def test_effective_sample_size_rejects_window_zero() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        effective_sample_size(np.ones((3, 3)), window=0)


# --------------------------------------------------------------------------- #
# Phase 2: run-length persistence (Fix 1)
# --------------------------------------------------------------------------- #
PERSISTENT_MATRIX = np.array(
    [
        [0.90, 0.07, 0.03],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85],
    ]
)


def test_run_length_persistence_hand_example() -> None:
    # Episodes: [0,0], [1], [2,2,2], [1,1] (censored, still counted).
    # State 0: one episode of length 2 -> 1 - 1/2 = 0.5. State 1: lengths 1 and 2, mean
    # 1.5 -> 1/3. State 2: one episode of length 3 -> 2/3.
    result = run_length_persistence([0, 0, 1, 2, 2, 2, 1, 1])
    np.testing.assert_allclose(result, [0.5, 1.0 / 3.0, 2.0 / 3.0])


def test_run_length_persistence_unseen_state_is_nan_and_single_run_is_zero() -> None:
    result = run_length_persistence([1])
    assert np.isnan(result[0]) and np.isnan(result[2])
    assert result[1] == 0.0  # a single bar is a length-1 episode: 1 - 1/1
    assert np.isnan(run_length_persistence([])).all()


def test_run_length_persistence_recovers_diagonal_of_first_order_chain() -> None:
    # 20000 steps gives roughly 6000 visits per state and, with mean episode lengths of
    # 10 / 5 / 6.7 bars, about 600 to 1200 episodes per state. The geometric mean length
    # L has standard error L * sqrt(p / n), so with p = 0.9, L = 10 and n = 600 it is
    # about 0.39 bars; the delta method maps that to 0.39 / L**2 = 0.004 on 1 - 1/L.
    # 0.03 is therefore more than 7 sigma.
    labels = _simulate_chain(PERSISTENT_MATRIX, n=20_000, seed=4242)
    estimate = run_length_persistence(labels)
    np.testing.assert_allclose(estimate, np.diag(PERSISTENT_MATRIX), atol=0.03)


def test_run_length_persistence_on_stride_labels_matches_stride_not_overlap() -> None:
    # On a daily sequence, mean episode length and the overlapping diagonal are two views
    # of the same exits-per-visit ratio, so run-length persistence on the daily labels of
    # the iid series is just as inflated as the overlapping diagonal (within 0.05). Applied
    # to the stride-20 subsample, whose labels come from disjoint windows, it recovers the
    # honest persistence: within 0.1 of the stride diagonal and far (> 0.3) below the
    # overlapping one. The 0.1 tolerance is ~4 sigma for ~330 stride observations per state.
    labels = _iid_labels()
    overlapping = np.diag(build_transition_matrix(labels, stride=1))
    stride = np.diag(build_transition_matrix(labels, stride=IID_WINDOW))
    on_daily = run_length_persistence(labels)
    on_stride = run_length_persistence(labels[::IID_WINDOW])
    np.testing.assert_allclose(on_daily, overlapping, atol=0.05)
    np.testing.assert_allclose(on_stride, stride, atol=0.1)
    assert (overlapping - on_stride > 0.3).all()


# --------------------------------------------------------------------------- #
# Phase 2: exponentially weighted counts (Fix 4)
# --------------------------------------------------------------------------- #
def test_ew_counts_weight_last_pair_one_and_decay_by_half_life() -> None:
    # Pairs in order: (0,2) two steps before the end, (2,0) one step before, (0,1) last.
    counts = ew_transition_counts([0, 2, 0, 1], half_life=1.0)
    assert counts[0, 1] == 1.0
    assert counts[2, 0] == 0.5
    assert counts[0, 2] == 0.25


def test_ew_matrix_with_huge_half_life_equals_full_history_fit() -> None:
    # With half_life = 1e12 and 500 pairs the smallest weight is 0.5 ** (499 / 1e12), which
    # differs from 1 by 499 * ln(2) / 1e12 = 3.5e-10 < 1e-9, so every normalised entry is
    # within 1e-9 of the unweighted estimate.
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 3, size=500)
    np.testing.assert_allclose(
        ew_transition_matrix(labels, half_life=1e12),
        build_transition_matrix(labels),
        atol=1e-9,
    )


def test_ew_weights_are_monotone_in_recency() -> None:
    # One 0->1 transition at the very start and one 0->2 at the very end; the recent one
    # must count more for any finite half-life, and by exactly 0.5 ** (k / half_life).
    middle = [1] * 100
    labels = np.array([0, *middle, 0, 2])
    counts = ew_transition_counts(labels, half_life=50.0)
    assert counts[0, 2] == 1.0
    assert counts[0, 1] < counts[0, 2]
    steps_apart = len(labels) - 2  # the first pair is this many steps before the last
    np.testing.assert_allclose(counts[0, 1], 0.5 ** (steps_apart / 50.0))


def test_ew_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError, match="half_life must be > 0"):
        ew_transition_counts([0, 1], half_life=0.0)
    with pytest.raises(ValueError, match="half_life must be > 0"):
        ew_transition_matrix([0, 1], half_life=-1.0)


# --------------------------------------------------------------------------- #
# Phase 2: rolling-window fits (Fix 4)
# --------------------------------------------------------------------------- #
def test_rolling_matrix_on_short_sequence_equals_full_fit() -> None:
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 3, size=300)
    np.testing.assert_array_equal(
        rolling_transition_counts(labels, window_bars=756), count_transitions(labels)
    )
    np.testing.assert_array_equal(
        rolling_transition_matrix(labels, window_bars=756), build_transition_matrix(labels)
    )


def test_rolling_matrix_on_long_sequence_equals_fit_on_tail() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 3, size=2000)
    np.testing.assert_array_equal(
        rolling_transition_counts(labels, window_bars=756), count_transitions(labels[-756:])
    )
    np.testing.assert_array_equal(
        rolling_transition_matrix(labels, window_bars=756),
        build_transition_matrix(labels[-756:]),
    )
    assert rolling_transition_counts(labels, window_bars=756).sum() == 755


def test_rolling_rejects_window_below_two() -> None:
    with pytest.raises(ValueError, match="window_bars must be >= 2"):
        rolling_transition_counts([0, 1, 2], window_bars=1)


# --------------------------------------------------------------------------- #
# Phase 2: Dirichlet shrinkage (Fix 4)
# --------------------------------------------------------------------------- #
RECENT_COUNTS = np.array([[8.0, 2.0, 0.0], [1.0, 1.0, 2.0], [0.0, 0.0, 0.0]])


def test_dirichlet_alpha_zero_returns_normalised_recent_counts() -> None:
    np.testing.assert_array_equal(
        dirichlet_shrinkage(RECENT_COUNTS, TRUE_MATRIX, alpha=0.0), normalise_rows(RECENT_COUNTS)
    )


def test_dirichlet_large_alpha_returns_prior() -> None:
    # With alpha = 1e9 the recent counts (at most 10 per row) move each entry by less than
    # 10 / 1e9 = 1e-8.
    np.testing.assert_allclose(
        dirichlet_shrinkage(RECENT_COUNTS, TRUE_MATRIX, alpha=1e9), TRUE_MATRIX, atol=1e-7
    )


def test_dirichlet_hand_example_and_rows_sum_to_one() -> None:
    posterior = dirichlet_shrinkage(RECENT_COUNTS, TRUE_MATRIX, alpha=20.0)
    # Row 0: (8 + 16, 2 + 3, 0 + 1) / (10 + 20).
    np.testing.assert_allclose(posterior[0], [24.0 / 30.0, 5.0 / 30.0, 1.0 / 30.0])
    # Row 2 has no recent counts, so it is exactly the prior row.
    np.testing.assert_allclose(posterior[2], TRUE_MATRIX[2])
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    validate_transition_matrix(posterior)


def test_dirichlet_zero_prior_row_with_zero_counts_stays_zero() -> None:
    prior = TRUE_MATRIX.copy()
    prior[2] = 0.0
    posterior = dirichlet_shrinkage(RECENT_COUNTS, prior, alpha=20.0)
    np.testing.assert_array_equal(posterior[2], np.zeros(3))
    assert np.isfinite(posterior).all()


def test_dirichlet_rejects_shape_mismatch_and_negative_alpha() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dirichlet_shrinkage(np.ones((2, 2)), TRUE_MATRIX)
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        dirichlet_shrinkage(RECENT_COUNTS, TRUE_MATRIX, alpha=-1.0)


# --------------------------------------------------------------------------- #
# Phase 2: drift metric (Fix 4)
# --------------------------------------------------------------------------- #
def test_drift_tvd_is_zero_for_identical_matrices() -> None:
    assert drift_tvd(TRUE_MATRIX, TRUE_MATRIX) == 0.0


def test_drift_tvd_is_one_for_disjoint_support_rows() -> None:
    a = np.eye(3)
    b = np.roll(np.eye(3), 1, axis=1)
    assert drift_tvd(a, b) == 1.0


def test_drift_tvd_is_worst_row_and_symmetric() -> None:
    b = TRUE_MATRIX.copy()
    b[1] = [0.30, 0.40, 0.30]  # row 1 moves by 0.5 * (0.2 + 0.4 + 0.2) = 0.4; others by 0
    assert drift_tvd(TRUE_MATRIX, b) == pytest.approx(0.4)
    assert drift_tvd(b, TRUE_MATRIX) == drift_tvd(TRUE_MATRIX, b)


def test_drift_tvd_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        drift_tvd(TRUE_MATRIX, np.eye(2))


def test_drift_tvd_skips_a_row_that_is_all_zero_in_one_matrix() -> None:
    # Row 0 unseen in the rolling window: a populated row against zeros would score
    # exactly 0.5, which is "no recent data", not drift. The remaining rows are equal.
    rolling = TRUE_MATRIX.copy()
    rolling[0] = 0.0
    assert drift_tvd(TRUE_MATRIX, rolling) == 0.0
    assert drift_tvd(rolling, TRUE_MATRIX) == 0.0
    # With real drift in a populated row the skipped row still does not dominate.
    rolling[1] = [0.30, 0.40, 0.30]  # TVD 0.4 against TRUE_MATRIX row 1
    assert drift_tvd(TRUE_MATRIX, rolling) == pytest.approx(0.4)


def test_drift_tvd_is_nan_when_every_row_is_skipped() -> None:
    assert math.isnan(drift_tvd(TRUE_MATRIX, np.zeros((3, 3))))
    assert math.isnan(drift_tvd(np.zeros((3, 3)), np.zeros((3, 3))))


# --------------------------------------------------------------------------- #
# Phase 2: mixing time
# --------------------------------------------------------------------------- #
def test_mixing_time_of_rank_one_matrix_is_one() -> None:
    # P ** 0 is the identity, which is not within 0.01 of pi, so the first step that
    # qualifies is n = 1, where every row already equals pi exactly.
    pi = np.array([0.2, 0.5, 0.3])
    rank_one = np.tile(pi, (3, 1))
    assert mixing_time(rank_one) == 1


def test_mixing_time_of_identity_is_none() -> None:
    assert mixing_time(np.eye(3), max_steps=500) is None


def test_mixing_time_of_well_mixing_chain_is_small_and_minimal() -> None:
    epsilon = 0.01
    n = mixing_time(TRUE_MATRIX, epsilon=epsilon)
    assert isinstance(n, int) and 0 < n < 100
    pi = stationary_distribution(TRUE_MATRIX)
    within = 0.5 * np.abs(nstep_forecast(TRUE_MATRIX, n) - pi).sum(axis=1)
    assert (within <= epsilon).all()
    before = 0.5 * np.abs(nstep_forecast(TRUE_MATRIX, n - 1) - pi).sum(axis=1)
    assert (before > epsilon).any()


def test_mixing_time_rejects_bad_epsilon_and_shape() -> None:
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        mixing_time(TRUE_MATRIX, epsilon=0.0)
    with pytest.raises(ValueError, match="square"):
        mixing_time(np.ones((2, 3)))


# --------------------------------------------------------------------------- #
# Phase 2: the Fix 4 verification test (matrix changes halfway)
# --------------------------------------------------------------------------- #
MATRIX_BEFORE = np.array(
    [
        [0.95, 0.03, 0.02],
        [0.025, 0.95, 0.025],
        [0.02, 0.03, 0.95],
    ]
)
MATRIX_AFTER = np.array(
    [
        [0.60, 0.25, 0.15],
        [0.20, 0.60, 0.20],
        [0.15, 0.25, 0.60],
    ]
)


def test_ew_and_rolling_track_a_regime_change_but_full_history_does_not() -> None:
    # 3000 steps from MATRIX_BEFORE then 3000 from MATRIX_AFTER. The rolling window (756
    # bars) lies entirely in the second half: about 250 transitions per row, binomial
    # standard error sqrt(0.6 * 0.4 / 250) = 0.031, so 0.08 is about 2.5 sigma. The EW
    # estimate with half-life 250 has an effective count of 250 / ln 2 = 360 transitions,
    # about 120 per row (standard error 0.045); the first half carries total weight
    # 0.5 ** 12 = 2e-4 and cannot pull it back toward 0.95. The full-history diagonal
    # averages the two halves (both contribute about 3000 transitions) and lands near
    # 0.78, which is 0.18 from 0.6, so the 0.1 separation is a comfortable margin.
    first_half = _simulate_chain(MATRIX_BEFORE, n=3000, seed=31)
    second_half = _simulate_chain(MATRIX_AFTER, n=3000, seed=32, start=int(first_half[-1]))
    labels = np.concatenate([first_half, second_half])
    target = np.diag(MATRIX_AFTER)

    ew = np.diag(ew_transition_matrix(labels, half_life=250.0))
    rolling = np.diag(rolling_transition_matrix(labels, window_bars=756))
    full = np.diag(build_transition_matrix(labels))

    np.testing.assert_allclose(ew, target, atol=0.08)
    np.testing.assert_allclose(rolling, target, atol=0.08)
    assert (np.abs(full - target) >= 0.1).all()


# --------------------------------------------------------------------------- #
# Phase 2: hypothesis invariants for every new estimator
# --------------------------------------------------------------------------- #
@settings(deadline=None)
@given(labels=label_sequences, stride=st.integers(min_value=1, max_value=5))
def test_new_estimators_return_valid_transition_matrices(labels: np.ndarray, stride: int) -> None:
    full = build_transition_matrix(labels)
    for matrix in (
        build_transition_matrix(labels, stride=stride),
        ew_transition_matrix(labels),
        ew_transition_matrix(labels, half_life=3.0),
        rolling_transition_matrix(labels),
        rolling_transition_matrix(labels, window_bars=10),
        dirichlet_shrinkage(rolling_transition_counts(labels, window_bars=10), full),
        dirichlet_shrinkage(count_transitions(labels), full, alpha=0.0),
    ):
        validate_transition_matrix(matrix)
        row_sums = matrix.sum(axis=1)
        assert ((np.abs(row_sums - 1.0) < 1e-9) | (row_sums == 0)).all()
        assert (matrix >= 0).all() and (matrix <= 1).all()


@given(labels=label_sequences)
def test_run_length_persistence_is_in_unit_interval_or_nan(labels: np.ndarray) -> None:
    result = run_length_persistence(labels)
    finite = result[np.isfinite(result)]
    assert (finite >= 0).all() and (finite < 1).all()
    assert np.isfinite(result).sum() == len(set(labels.tolist()))
