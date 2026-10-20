"""Trading signal read off a transition matrix.

The signal is the companion script's ``signal_from_matrix``: the probability of
moving to Bull next minus the probability of moving to Bear, read from the row of
the current state. Positive means long bias, negative means short bias, and the
magnitude is conviction. ``sign_position`` is the Phase 1 position rule: take the
sign of the signal and nothing else (+1, 0 or -1).

Phase 3 adds ``position_from_matrix`` with the rules in ``POSITION_RULES``: the sign
rule, conviction scaling (the raw signal), persistence weighting (signal times the
probability of staying in the current regime) and the signal's own long-only version.

Pure functions: numpy in, floats out, no I/O.
"""

import numpy as np

from hedgefund.regime.labels import BEAR, BULL, STATES


def _validate_state(current_state: int, n_states: int) -> None:
    """Raise ValueError unless ``current_state`` is a valid row index."""
    if isinstance(current_state, bool) or not isinstance(current_state, int | np.integer):
        raise ValueError(
            f"current_state must be an integer in 0..{n_states - 1}, got {current_state!r}"
        )
    if not 0 <= int(current_state) < n_states:
        raise ValueError(
            f"current_state must be in 0..{n_states - 1} (order {STATES}), got {current_state}"
        )


def next_state_probabilities(matrix: np.ndarray, current_state: int) -> np.ndarray:
    """Return P(next state | current state): row ``current_state`` of ``matrix``.

    The row is ordered as ``STATES`` ([Bear, Sideways, Bull]). Raises ``ValueError``
    if the matrix is not square ``n x n`` or the state is outside ``0..n-1``.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square (n, n), got shape {matrix.shape}")
    _validate_state(current_state, matrix.shape[0])
    return matrix[int(current_state)]


def signal_from_matrix(matrix: np.ndarray, current_state: int) -> float:
    """P(next = Bull | current) - P(next = Bear | current), in [-1, 1].

    Identical to the companion script: ``matrix[s, 2] - matrix[s, 0]`` with the
    canonical state order, so the result is bit-for-bit the same number.
    """
    row = next_state_probabilities(matrix, current_state)
    if row.shape[0] <= max(BULL, BEAR):
        raise ValueError(
            f"matrix needs at least {len(STATES)} states (order {STATES}), got {row.shape[0]}"
        )
    return float(row[BULL] - row[BEAR])


def sign_position(signal: float) -> float:
    """The Phase 1 sign-only position rule: +1 long, -1 short, 0 flat.

    ``float(np.sign(signal))`` exactly as the companion does it, so a zero signal
    is a flat position and NaN propagates rather than being coerced.
    """
    return float(np.sign(signal))


# Position rules the walk-forward understands (plan.md Phase 3). "sign" is Phase 1.
POSITION_RULES: tuple[str, ...] = ("sign", "conviction", "persistence", "long_only")
FLAT_POSITION = 0.0  # the long-only floor: never short, never leveraged


def _validate_rule(rule: str) -> None:
    if rule not in POSITION_RULES:
        raise ValueError(f"rule must be one of {POSITION_RULES}, got {rule!r}")


def position_from_signal(signal: float, stay_probability: float, rule: str = "sign") -> float:
    """Turn a signal into a position under ``rule`` (see ``position_from_matrix``).

    ``stay_probability`` is ``matrix[s, s]`` for the current state ``s``; only the
    persistence rule reads it. Split out so the walk-forward can compute the signal
    once and derive the position from it without re-reading the matrix.
    """
    _validate_rule(rule)
    if rule == "sign":
        return sign_position(signal)
    if rule == "conviction":
        return float(signal)
    if rule == "persistence":
        return float(signal * stay_probability)
    return max(sign_position(signal), FLAT_POSITION)


def position_from_matrix(matrix: np.ndarray, current_state: int, rule: str = "sign") -> float:
    """Position in ``[-1, 1]`` for the current state under one of ``POSITION_RULES``.

    With ``signal = signal_from_matrix(matrix, s)`` and ``s = current_state``:

    - ``sign``: ``sign_position(signal)``, +1 / 0 / -1 (the Phase 1 rule).
    - ``conviction``: ``signal`` itself, so a confident row sizes up.
    - ``persistence``: ``signal * matrix[s, s]``, conviction times the probability of
      staying in the current regime; a sticky regime sizes up, a flighty one down.
    - ``long_only``: ``max(sign_position(signal), 0.0)``, the signal's long-only version.

    Raises ``ValueError`` on an unknown rule, a non-square matrix or a bad state.
    """
    _validate_rule(rule)
    signal = signal_from_matrix(matrix, current_state)
    stay = float(np.asarray(matrix, dtype=float)[int(current_state), int(current_state)])
    return position_from_signal(signal, stay, rule)
