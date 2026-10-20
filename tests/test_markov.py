"""Tests for transition-matrix estimation and chain algebra in ``hedgefund.regime.markov``."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedgefund.regime.markov import (
    build_transition_matrix,
    count_transitions,
    normalise_rows,
    nstep_forecast,
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


def _simulate_chain(matrix: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a state path from ``matrix`` starting in state 0, one uniform draw per step."""
    cumulative = np.cumsum(matrix, axis=1)
    draws = rng.random(n_steps)
    path = np.empty(n_steps, dtype=int)
    state = 0
    for t in range(n_steps):
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
    rng = np.random.default_rng(12345)
    labels = _simulate_chain(TRUE_MATRIX, n_steps=20_000, rng=rng)
    estimate = build_transition_matrix(labels)
    np.testing.assert_allclose(estimate, TRUE_MATRIX, atol=0.02)
    validate_transition_matrix(estimate)


def test_synthetic_chain_is_reproducible_with_explicit_seed() -> None:
    first = _simulate_chain(TRUE_MATRIX, 2000, np.random.default_rng(99))
    second = _simulate_chain(TRUE_MATRIX, 2000, np.random.default_rng(99))
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
