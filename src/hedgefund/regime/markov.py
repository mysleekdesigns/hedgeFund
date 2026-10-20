"""Transition-matrix estimation and Markov-chain algebra.

Ported from the companion script's ``build_transition_matrix``, ``nstep_forecast`` and
``stationary_distribution`` so the numbers are bit-identical on the same label
sequence. Pure: numpy in, numpy out, no I/O.

Phase 2 adds the honest-statistics estimators from plan.md Fix 1 (stride counting,
run-length persistence, effective sample size) and Fix 4 (exponentially weighted
counts, rolling-window fits, Dirichlet shrinkage, drift, mixing time). Every matrix
returned here is row-stochastic with all-zero rows allowed for unseen source states,
exactly like ``normalise_rows``.

Index order everywhere is ``0 = Bear, 1 = Sideways, 2 = Bull`` (see
``hedgefund.regime.labels.STATES``).
"""

import numpy as np
from numpy.typing import ArrayLike

N_STATES = 3  # Bear, Sideways, Bull; the only chain size the companion script supports
ROW_SUM_TOLERANCE = 1e-9  # a non-zero row of a stochastic matrix must sum to 1 within this
UNIT_EIGENVALUE = 1.0  # a row-stochastic matrix always has a left eigenvector for eigenvalue 1

# Fix 4: decay each transition count with a half-life of about one trading year (250 bars)
# so the matrix tracks recent behaviour while still using all history.
DEFAULT_HALF_LIFE_BARS = 250.0
# plan.md section 5 ``rolling:756``: a 3-trading-year window (3 * 252) for the rolling fit.
DEFAULT_ROLLING_WINDOW_BARS = 756
# Fix 4 Dirichlet prior strength: 20 pseudo-counts per row is about one month of daily
# transitions, enough to hold a row with few recent observations near the full-history
# prior without swamping a row that has hundreds of recent observations.
DEFAULT_SHRINKAGE_ALPHA = 20.0
# Mixing time stops when every row of P**n is within 1 percentage point of total variation
# from the stationary vector.
DEFAULT_MIXING_EPSILON = 0.01
# Upper bound on the iterative search; a chain that has not mixed by then is treated as
# periodic or reducible and ``mixing_time`` returns ``None``.
DEFAULT_MIXING_MAX_STEPS = 10_000


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


def _check_stride(stride: int) -> None:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")


def count_transitions(labels: ArrayLike, n_states: int = N_STATES, stride: int = 1) -> np.ndarray:
    """Count consecutive label pairs into an ``(n_states, n_states)`` float matrix.

    ``counts[i, j]`` is the number of times state ``i`` at ``t`` is followed by state
    ``j`` at ``t + stride``. With ``stride=1`` (the default) this is the overlapping
    count: vectorised with ``np.add.at``, it equals the companion's Python loop exactly
    because both are plain integer sums. A sequence shorter than two labels yields all
    zeros.

    With ``stride=w`` the label sequence is subsampled as ``labels[::w]`` (offset 0) and
    consecutive pairs of that subsampled sequence are counted, so each transition uses
    non-overlapping ``w``-bar windows (plan.md Fix 1). The result is the ``w``-step
    matrix of the subsampled chain, which is why its diagonal is not directly comparable
    to the 1-step diagonal: if the daily chain were first-order Markov the comparable
    quantity would be ``nstep_forecast(P_overlap, w)``.

    Raises ``ValueError`` for ``stride < 1``.
    """
    _check_stride(stride)
    arr = _as_label_array(labels, n_states)
    if stride > 1:
        arr = arr[::stride]
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


def build_transition_matrix(
    labels: ArrayLike, n_states: int = N_STATES, stride: int = 1
) -> np.ndarray:
    """Maximum-likelihood transition matrix from consecutive label pairs.

    Equivalent to ``normalise_rows(count_transitions(labels, n_states, stride))``. Rows
    for states that never occur as a source are all zeros, not uniform.

    ``stride=1`` is the overlapping (companion) matrix. ``stride=w`` fits the ``w``-step
    matrix of the subsampled chain ``labels[::w]``, so consecutive transitions use
    non-overlapping windows; its diagonal is not directly comparable to the 1-step
    diagonal. ``nstep_forecast(P_overlap, w)`` is the comparable quantity if the daily
    chain were first-order Markov. See ``count_transitions``.
    """
    return normalise_rows(count_transitions(labels, n_states, stride))


def effective_sample_size(counts: np.ndarray, window: int) -> np.ndarray:
    """Per-row effective number of independent transitions, ``counts.sum(axis=1) / window``.

    Consecutive overlapping-window transitions share ``window - 1`` of ``window`` bars
    (the label at ``t`` and ``t + 1`` are computed from almost the same returns), so
    they are far from independent draws. Treating ``n / window`` of them as independent
    is the conservative count that rules/no-lookahead.md prescribes for confidence
    intervals on the overlapping matrix (plan.md Fix 1). Returns shape ``(n_states,)``.

    Raises ``ValueError`` for ``window < 1`` or a non-2-D ``counts``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D, got shape {counts.shape}")
    return counts.sum(axis=1) / window


def run_length_persistence(labels: ArrayLike, n_states: int = N_STATES) -> np.ndarray:
    """Persistence per state from episode lengths: ``1 - 1 / mean_episode_length``.

    An episode is a maximal run of identical consecutive labels. For a true first-order
    chain the episode length of state ``i`` is geometric with mean ``1 / (1 - p_ii)``,
    so this recovers ``p_ii`` without the inflation that overlapping windows put into
    the diagonal of ``build_transition_matrix`` (plan.md Fix 1).

    The final run is censored (the series ends before the regime does) and is included
    anyway; dropping it would bias the estimate the other way on short series. States
    with no episodes get ``np.nan``. Returns shape ``(n_states,)``.

    Caveat: on a daily sequence produced by an overlapping-window labeller the episode
    lengths carry the same overlap inflation as the 1-step diagonal (both are the
    exits-per-visit ratio of the same sequence), so for an overlap-free estimate pass
    the stride-subsampled labels ``labels[::window]``. ``tests/test_markov.py`` shows
    both behaviours on an iid series.
    """
    arr = _as_label_array(labels, n_states)
    result = np.full(n_states, np.nan)
    if arr.size == 0:
        return result
    boundaries = np.flatnonzero(np.diff(arr) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [arr.size]))
    lengths = (ends - starts).astype(float)
    episode_states = arr[starts]
    for state in range(n_states):
        mask = episode_states == state
        if mask.any():
            result[state] = 1.0 - 1.0 / lengths[mask].mean()
    return result


def ew_transition_counts(
    labels: ArrayLike,
    half_life: float = DEFAULT_HALF_LIFE_BARS,
    n_states: int = N_STATES,
) -> np.ndarray:
    """Exponentially weighted transition counts (plan.md Fix 4).

    The transition pair ending at the last label gets weight 1; the pair ``k`` steps
    earlier gets ``0.5 ** (k / half_life)``. Weights are all strictly positive so the
    full history still contributes, but the matrix tracks recent behaviour. A sequence
    shorter than two labels yields all zeros.

    Raises ``ValueError`` for ``half_life <= 0``.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    arr = _as_label_array(labels, n_states)
    counts = np.zeros((n_states, n_states), dtype=float)
    if arr.size >= 2:
        n_pairs = arr.size - 1
        steps_before_end = np.arange(n_pairs - 1, -1, -1, dtype=float)
        weights = 0.5 ** (steps_before_end / half_life)
        np.add.at(counts, (arr[:-1], arr[1:]), weights)
    return counts


def ew_transition_matrix(
    labels: ArrayLike,
    half_life: float = DEFAULT_HALF_LIFE_BARS,
    n_states: int = N_STATES,
) -> np.ndarray:
    """Row-normalised ``ew_transition_counts``; unseen source states stay all zeros."""
    return normalise_rows(ew_transition_counts(labels, half_life, n_states))


def rolling_transition_counts(
    labels: ArrayLike,
    window_bars: int = DEFAULT_ROLLING_WINDOW_BARS,
    n_states: int = N_STATES,
) -> np.ndarray:
    """Transition counts from the trailing ``window_bars`` labels only (plan.md Fix 4).

    Uses the last ``window_bars`` labels (default 756, three trading years). If the
    sequence is shorter than ``window_bars`` all of it is used, so the result then
    equals ``count_transitions``.

    Raises ``ValueError`` for ``window_bars < 2`` (fewer labels cannot form a pair).
    """
    if window_bars < 2:
        raise ValueError(f"window_bars must be >= 2, got {window_bars}")
    arr = _as_label_array(labels, n_states)
    return count_transitions(arr[-window_bars:], n_states)


def rolling_transition_matrix(
    labels: ArrayLike,
    window_bars: int = DEFAULT_ROLLING_WINDOW_BARS,
    n_states: int = N_STATES,
) -> np.ndarray:
    """Row-normalised ``rolling_transition_counts``; unseen source states stay all zeros."""
    return normalise_rows(rolling_transition_counts(labels, window_bars, n_states))


def dirichlet_shrinkage(
    recent_counts: np.ndarray,
    prior_matrix: np.ndarray,
    alpha: float = DEFAULT_SHRINKAGE_ALPHA,
) -> np.ndarray:
    """Shrink recent transition counts toward a prior matrix (plan.md Fix 4).

    Posterior row ``i`` is ``(recent_counts[i] + alpha * prior_matrix[i])
    / (recent_counts[i].sum() + alpha)``: a Dirichlet prior with ``alpha`` pseudo-counts
    spread along the full-history row. ``alpha = 0`` returns the normalised recent
    counts; a large ``alpha`` returns the prior. A row whose prior is all zeros (a state
    never seen in the full history) and whose recent counts are also zero stays all
    zeros rather than dividing by zero.

    Raises ``ValueError`` for a shape mismatch, negative counts, or ``alpha < 0``.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    recent = np.asarray(recent_counts, dtype=float)
    prior = np.asarray(prior_matrix, dtype=float)
    if recent.ndim != 2 or recent.shape != prior.shape:
        raise ValueError(
            f"recent_counts and prior_matrix must be 2-D with the same shape, "
            f"got {recent.shape} and {prior.shape}"
        )
    if (recent < 0).any():
        raise ValueError("recent_counts must be non-negative")
    return normalise_rows(recent + alpha * prior)


def drift_tvd(full_matrix: np.ndarray, rolling_matrix: np.ndarray) -> float:
    """Total variation distance between two transition matrices, worst populated row.

    Defined as ``max_i 0.5 * sum_j |a_ij - b_ij|``: the per-row total variation distance
    between the two conditional next-state distributions, then the maximum over source
    states. It lies in ``[0, 1]`` and reads as "how far the most-drifted regime's
    transition probabilities have moved", which is the number a report should flag
    (plan.md Fix 4). Averaging over rows would let one badly drifted but rarely visited
    state hide behind the others.

    A row that is all zeros in either matrix (``normalise_rows`` leaves a state never
    seen as a source that way) is skipped: comparing a populated full-history row with
    an empty rolling row would always give exactly 0.5, which reads as massive drift
    when it only means "no recent observations of that state". If every row is skipped
    the drift is undefined and ``np.nan`` is returned (the JSON contract emits ``null``).

    Raises ``ValueError`` for a shape mismatch or non-2-D input.
    """
    a = np.asarray(full_matrix, dtype=float)
    b = np.asarray(rolling_matrix, dtype=float)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"full_matrix and rolling_matrix must be 2-D with the same shape, "
            f"got {a.shape} and {b.shape}"
        )
    populated = (a.sum(axis=1) != 0) & (b.sum(axis=1) != 0)
    if not populated.any():
        return float("nan")
    row_tvd = 0.5 * np.abs(a - b).sum(axis=1)
    return float(row_tvd[populated].max())


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


def mixing_time(
    matrix: np.ndarray,
    epsilon: float = DEFAULT_MIXING_EPSILON,
    max_steps: int = DEFAULT_MIXING_MAX_STEPS,
) -> int | None:
    """Smallest ``n >= 0`` such that every row of ``P**n`` is within ``epsilon`` of ``pi``.

    Distance is total variation, ``0.5 * sum_j |P**n[i, j] - pi[j]|``, taken over every
    start state ``i``; ``pi`` is ``stationary_distribution(matrix)``. ``n = 0`` is
    checked first (``P**0`` is the identity), so a matrix is only reported as mixing at
    step 0 if the identity itself is within ``epsilon`` of ``pi``, which never happens
    for ``epsilon < 1 - max(pi)``. A rank-one matrix (every row equal to ``pi``)
    therefore has mixing time 1.

    Returns ``None`` if the bound is not reached within ``max_steps`` (periodic or
    reducible chains such as the identity never converge). The powers are computed by
    repeated multiplication rather than ``matrix_power`` at each step, so the cost is
    linear in the number of steps.

    Raises ``ValueError`` for a non-square matrix, ``epsilon <= 0`` or ``max_steps < 0``.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    pi = stationary_distribution(matrix)
    power = np.eye(matrix.shape[0])
    for n in range(max_steps + 1):
        tvd = 0.5 * np.abs(power - pi).sum(axis=1)
        if (tvd <= epsilon).all():
            return n
        power = power @ matrix
    return None


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
