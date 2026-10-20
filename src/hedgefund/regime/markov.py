"""Transition-matrix estimation and Markov-chain algebra.

Ported from the companion script's ``build_transition_matrix``, ``nstep_forecast`` and
``stationary_distribution`` so the numbers are bit-identical on the same label
sequence. Pure: numpy in, numpy out, no I/O.

Index order everywhere is ``0 = Bear, 1 = Sideways, 2 = Bull`` (see
``hedgefund.regime.labels.STATES``).
"""

import numpy as np
from numpy.typing import ArrayLike

N_STATES = 3  # Bear, Sideways, Bull; the only chain size the companion script supports
ROW_SUM_TOLERANCE = 1e-9  # a non-zero row of a stochastic matrix must sum to 1 within this
UNIT_EIGENVALUE = 1.0  # a row-stochastic matrix always has a left eigenvector for eigenvalue 1


def _as_label_array(labels: ArrayLike, n_states: int) -> np.ndarray:
    """Coerce ``labels`` to a 1-D int array and reject anything outside ``[0, n_states)``."""
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels must be integer state indices, got dtype {arr.dtype}")
    arr = arr.astype(int, copy=False)
    if arr.size and (arr.min() < 0 or arr.max() >= n_states):
        raise ValueError(
            f"labels must lie in [0, {n_states}), got min {arr.min()} and max {arr.max()}"
        )
    return arr


def count_transitions(labels: ArrayLike, n_states: int = N_STATES) -> np.ndarray:
    """Count consecutive label pairs into an ``(n_states, n_states)`` float matrix.

    ``counts[i, j]`` is the number of times state ``i`` at ``t`` is followed by state
    ``j`` at ``t + 1``. Vectorised with ``np.add.at``; equals the companion's Python
    loop exactly because both are plain integer sums. A sequence shorter than two
    labels yields all zeros.
    """
    arr = _as_label_array(labels, n_states)
    counts = np.zeros((n_states, n_states), dtype=float)
    if arr.size >= 2:
        np.add.at(counts, (arr[:-1], arr[1:]), 1.0)
    return counts


def normalise_rows(counts: np.ndarray) -> np.ndarray:
    """Divide each row by its sum to get a row-stochastic matrix.

    Rows with zero count stay all zeros (the companion divides them by 1.0): a state
    that never appears as a source carries no information, and the caller decides
    what to do with it. ``validate_transition_matrix`` accepts such rows.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D, got shape {counts.shape}")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    row_sums = counts.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return counts / safe


def build_transition_matrix(labels: ArrayLike, n_states: int = N_STATES) -> np.ndarray:
    """Maximum-likelihood transition matrix from consecutive label pairs.

    Equivalent to ``normalise_rows(count_transitions(labels, n_states))``. Rows for
    states that never occur as a source are all zeros, not uniform.
    """
    return normalise_rows(count_transitions(labels, n_states))


def nstep_forecast(matrix: np.ndarray, n: int) -> np.ndarray:
    """Chapman-Kolmogorov: the ``n``-step transition matrix is ``P ** n``.

    ``n = 0`` gives the identity. Raises ``ValueError`` for ``n < 0``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return np.linalg.matrix_power(np.asarray(matrix, dtype=float), n)


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Left eigenvector of ``P`` for eigenvalue 1, normalised to sum to 1.

    This is the long-run regime mix implied by the current matrix, not a forecast:
    it is where the chain settles if ``P`` held forever, regardless of the start.

    The algorithm is deliberately identical to the companion script (eigen-decompose
    ``P.T``, pick the eigenvalue closest to 1, take ``abs(real(v))``, normalise) so
    the result matches it to the last digit.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    idx = np.argmin(np.abs(eigvals - UNIT_EIGENVALUE))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def validate_transition_matrix(matrix: np.ndarray, n_states: int = N_STATES) -> None:
    """Raise ``ValueError`` unless ``matrix`` is a valid ``n_states`` transition matrix.

    Checks: square with side ``n_states``; every entry finite and in ``[0, 1]``; every
    non-zero row sums to 1 within ``ROW_SUM_TOLERANCE``. All-zero rows are allowed
    because ``normalise_rows`` produces them for states never seen as a source.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (n_states, n_states):
        raise ValueError(
            f"transition matrix must have shape ({n_states}, {n_states}), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("transition matrix has non-finite entries")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError(
            f"transition matrix entries must lie in [0, 1], got min {matrix.min()} "
            f"and max {matrix.max()}"
        )
    row_sums = matrix.sum(axis=1)
    non_zero = row_sums != 0
    bad = non_zero & (np.abs(row_sums - 1.0) > ROW_SUM_TOLERANCE)
    if bad.any():
        raise ValueError(
            f"rows {np.flatnonzero(bad).tolist()} of the transition matrix do not sum to 1: "
            f"{row_sums[bad].tolist()}"
        )
