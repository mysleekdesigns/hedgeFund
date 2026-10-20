"""Trading signal read off a transition matrix.

The signal is the companion script's ``signal_from_matrix``: the probability of
moving to Bull next minus the probability of moving to Bear, read from the row of
the current state. Positive means long bias, negative means short bias, and the
magnitude is conviction. ``sign_position`` is the Phase 1 position rule: take the
sign of the signal and nothing else (+1, 0 or -1).

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
