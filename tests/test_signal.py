"""Tests for hedgefund.regime.signal: the bull-minus-bear signal and the sign rule."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS
from hedgefund.regime.signal import (
    POSITION_RULES,
    next_state_probabilities,
    position_from_matrix,
    position_from_signal,
    sign_position,
    signal_from_matrix,
)

# A hand-built row-stochastic matrix with distinct, easy-to-check entries.
HAND_MATRIX = np.array(
    [
        [0.7, 0.2, 0.1],  # from Bear
        [0.1, 0.6, 0.3],  # from Sideways
        [0.05, 0.25, 0.7],  # from Bull
    ]
)


def test_next_state_probabilities_is_row_of_matrix() -> None:
    for state in (BEAR, SIDEWAYS, BULL):
        assert np.array_equal(next_state_probabilities(HAND_MATRIX, state), HAND_MATRIX[state])


def test_signal_equals_bull_minus_bear_on_hand_matrix() -> None:
    assert signal_from_matrix(HAND_MATRIX, BEAR) == 0.1 - 0.7
    assert signal_from_matrix(HAND_MATRIX, SIDEWAYS) == 0.3 - 0.1
    assert signal_from_matrix(HAND_MATRIX, BULL) == 0.7 - 0.05


def test_signal_matches_companion_expression_bitwise() -> None:
    """The companion computes matrix[s, 2] - matrix[s, 0]; we must return that exact float."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        raw = rng.random((3, 3))
        matrix = raw / raw.sum(axis=1, keepdims=True)
        for state in range(3):
            assert signal_from_matrix(matrix, state) == float(matrix[state, 2] - matrix[state, 0])


def test_signal_is_a_python_float() -> None:
    assert type(signal_from_matrix(HAND_MATRIX, BULL)) is float


@st.composite
def stochastic_matrices(draw: st.DrawFn) -> np.ndarray:
    rows = []
    for _ in range(3):
        raw = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=3,
                max_size=3,
            )
        )
        total = sum(raw)
        rows.append([x / total for x in raw] if total > 0 else [0.0, 0.0, 0.0])
    return np.array(rows)


@given(stochastic_matrices(), st.integers(min_value=0, max_value=2))
def test_signal_in_unit_interval(matrix: np.ndarray, state: int) -> None:
    signal = signal_from_matrix(matrix, state)
    assert -1.0 <= signal <= 1.0


def test_signal_is_zero_for_all_zero_row() -> None:
    """A state never seen as a source has an all-zero row; its signal is flat, not NaN."""
    matrix = np.zeros((3, 3))
    matrix[SIDEWAYS] = [0.2, 0.6, 0.2]
    assert signal_from_matrix(matrix, BEAR) == 0.0


@pytest.mark.parametrize("bad_state", [-1, 3, 10])
def test_signal_raises_on_bad_state(bad_state: int) -> None:
    with pytest.raises(ValueError, match="current_state"):
        signal_from_matrix(HAND_MATRIX, bad_state)


@pytest.mark.parametrize("bad_state", [1.0, "1", None, True])
def test_signal_raises_on_non_integer_state(bad_state: object) -> None:
    with pytest.raises(ValueError, match="current_state"):
        signal_from_matrix(HAND_MATRIX, bad_state)  # type: ignore[arg-type]


def test_numpy_integer_state_is_accepted() -> None:
    assert signal_from_matrix(HAND_MATRIX, np.int64(BULL)) == signal_from_matrix(HAND_MATRIX, BULL)


def test_signal_raises_on_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        signal_from_matrix(np.ones((3, 2)), BEAR)


def test_signal_raises_on_too_small_matrix() -> None:
    with pytest.raises(ValueError, match="states"):
        signal_from_matrix(np.eye(2), 0)


def test_sign_position_of_zero_is_zero() -> None:
    assert sign_position(0.0) == 0.0


def test_sign_position_is_plus_minus_one() -> None:
    assert sign_position(0.3) == 1.0
    assert sign_position(-0.0001) == -1.0
    assert type(sign_position(0.3)) is float


def test_sign_position_propagates_nan() -> None:
    assert math.isnan(sign_position(float("nan")))


# --------------------------------------------------------------------------- #
# Phase 3 position rules
# --------------------------------------------------------------------------- #
def test_position_rules_are_the_four_plan_rules_with_sign_first() -> None:
    assert POSITION_RULES == ("sign", "conviction", "persistence", "long_only")


def test_sign_rule_is_the_default_and_equals_sign_position() -> None:
    for state in (BEAR, SIDEWAYS, BULL):
        signal = signal_from_matrix(HAND_MATRIX, state)
        assert position_from_matrix(HAND_MATRIX, state) == sign_position(signal)
        assert position_from_matrix(HAND_MATRIX, state, rule="sign") == sign_position(signal)
    assert position_from_matrix(HAND_MATRIX, BEAR) == -1.0
    assert position_from_matrix(HAND_MATRIX, SIDEWAYS) == 1.0
    assert position_from_matrix(HAND_MATRIX, BULL) == 1.0


def test_conviction_rule_returns_the_signal_itself() -> None:
    for state in (BEAR, SIDEWAYS, BULL):
        assert position_from_matrix(HAND_MATRIX, state, rule="conviction") == signal_from_matrix(
            HAND_MATRIX, state
        )
    assert position_from_matrix(HAND_MATRIX, BEAR, rule="conviction") == 0.1 - 0.7


def test_persistence_rule_is_signal_times_stay_probability() -> None:
    # Bear row: signal 0.1 - 0.7 = -0.6, P(stay) 0.7 -> -0.42; Bull row: 0.65 * 0.7.
    assert position_from_matrix(HAND_MATRIX, BEAR, rule="persistence") == (0.1 - 0.7) * 0.7
    assert position_from_matrix(HAND_MATRIX, SIDEWAYS, rule="persistence") == (0.3 - 0.1) * 0.6
    assert position_from_matrix(HAND_MATRIX, BULL, rule="persistence") == (0.7 - 0.05) * 0.7


def test_long_only_rule_floors_the_sign_at_zero() -> None:
    assert position_from_matrix(HAND_MATRIX, BEAR, rule="long_only") == 0.0
    assert position_from_matrix(HAND_MATRIX, SIDEWAYS, rule="long_only") == 1.0
    assert position_from_matrix(HAND_MATRIX, BULL, rule="long_only") == 1.0


def test_long_only_of_zero_signal_is_flat() -> None:
    matrix = np.array([[0.2, 0.6, 0.2], [0.3, 0.4, 0.3], [0.1, 0.8, 0.1]])
    for state in (BEAR, SIDEWAYS, BULL):
        assert position_from_matrix(matrix, state, rule="long_only") == 0.0
        assert position_from_matrix(matrix, state, rule="sign") == 0.0


def test_position_is_a_python_float_for_every_rule() -> None:
    for rule in POSITION_RULES:
        assert type(position_from_matrix(HAND_MATRIX, BULL, rule=rule)) is float


@pytest.mark.parametrize("bad_rule", ["signs", "Sign", "", "momentum", None])
def test_position_raises_on_unknown_rule(bad_rule: object) -> None:
    with pytest.raises(ValueError, match="rule"):
        position_from_matrix(HAND_MATRIX, BULL, rule=bad_rule)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rule"):
        position_from_signal(0.5, 0.5, rule=bad_rule)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_state", [-1, 3])
def test_position_raises_on_bad_state_for_every_rule(bad_state: int) -> None:
    for rule in POSITION_RULES:
        with pytest.raises(ValueError, match="current_state"):
            position_from_matrix(HAND_MATRIX, bad_state, rule=rule)


def test_position_from_signal_matches_position_from_matrix() -> None:
    """The walk-forward uses the split form; both paths must give the same float."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        raw = rng.random((3, 3))
        matrix = raw / raw.sum(axis=1, keepdims=True)
        for state in range(3):
            signal = signal_from_matrix(matrix, state)
            for rule in POSITION_RULES:
                assert position_from_signal(signal, matrix[state, state], rule) == (
                    position_from_matrix(matrix, state, rule=rule)
                )


@given(stochastic_matrices(), st.integers(min_value=0, max_value=2))
def test_position_rule_ranges(matrix: np.ndarray, state: int) -> None:
    signal = signal_from_matrix(matrix, state)
    assert position_from_matrix(matrix, state, rule="sign") in {-1.0, 0.0, 1.0}
    assert position_from_matrix(matrix, state, rule="long_only") in {0.0, 1.0}
    conviction = position_from_matrix(matrix, state, rule="conviction")
    assert -1.0 <= conviction <= 1.0
    persistence = position_from_matrix(matrix, state, rule="persistence")
    # P(stay) is in [0, 1], so persistence never exceeds conviction in size or flips it.
    assert abs(persistence) <= abs(conviction)
    assert np.sign(persistence) in {np.sign(signal), 0.0}
