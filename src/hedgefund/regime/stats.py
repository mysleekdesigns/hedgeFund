"""Honest statistics for the regime transition matrix (plan.md Phase 2).

Null models, block-bootstrap confidence intervals, and the Anderson-Goodman chi-square
tests that decide whether the measured persistence is real or manufactured by the
labeller. Pure: numpy or pandas in, numpy or plain dicts out, no I/O, no module state.
Every random draw takes an explicit ``seed`` and uses ``np.random.default_rng(seed)``.

Index order everywhere is ``0 = Bear, 1 = Sideways, 2 = Bull`` (see
``hedgefund.regime.labels.STATES``).

References
----------
Anderson, T. W. and Goodman, L. A. (1957). Statistical inference about Markov chains.
Annals of Mathematical Statistics 28(1), 89-110.
Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations.
Annals of Statistics 17(3), 1217-1241. (moving block bootstrap)
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats as sp_stats

from hedgefund.regime.labels import label_regimes
from hedgefund.regime.markov import (
    N_STATES,
    build_transition_matrix,
    count_transitions,
    normalise_rows,
    stationary_distribution,
    validate_transition_matrix,
)

# Two-sided 95% band for null-diagonal reporting; the conventional level, paired with
# ``HOMOGENEITY_ALPHA`` below so every "significant" in the report means the same thing.
NULL_BAND_LOWER_PCT = 2.5
NULL_BAND_UPPER_PCT = 97.5
PERCENT = 100.0  # percentile scale for ``diagonal_percentile``
# Conventional 5% level for the homogeneity "stable" flag. A rejection is reported as
# "matrix is not stable", which is information about the data, not a failure (Fix 4).
HOMOGENEITY_ALPHA = 0.05
MIN_LABELS_PER_PERIOD = 2  # a period needs at least one transition to contribute counts
MIN_LABELS_FOR_TEST = 2  # a chi-square test on transitions needs at least one transition


def _labels_1d(labels: ArrayLike, n_states: int) -> np.ndarray:
    """Coerce to a 1-D int array in ``[0, n_states)`` (delegates validation to markov)."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    count_transitions(arr, n_states)  # raises ValueError on dtype / range problems
    return arr.astype(int, copy=False)


def _diagonal_or_nan(matrix: np.ndarray) -> np.ndarray:
    """Diagonal of an estimated matrix with all-zero rows (never-visited states) as NaN.

    ``normalise_rows`` leaves a never-visited source state as an all-zero row. Its
    diagonal of 0 is not an estimate of persistence, it is missing data, so it is
    reported as NaN and the null summaries use nan-aware reductions.
    """
    diag = np.diag(matrix).astype(float).copy()
    diag[matrix.sum(axis=1) == 0] = np.nan
    return diag


def _summarise_null_diagonals(real_diag: np.ndarray, null_diags: np.ndarray) -> dict:
    """Percentile of the real diagonal within the null draws plus null mean and 95% band.

    ``null_diags`` has shape ``(n_paths, n_states)`` and may contain NaN for states a
    null path never visited; those entries are ignored in every reduction. The
    percentile is ``100 * P(null <= real)`` among the non-NaN draws for that state, and
    is NaN when the real diagonal itself is NaN or no null path visited the state.
    """
    valid = ~np.isnan(null_diags)
    n_valid = valid.sum(axis=0).astype(float)
    at_or_below = np.where(valid, null_diags <= real_diag[np.newaxis, :], False).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        percentile = PERCENT * at_or_below / n_valid
    percentile = np.where(np.isnan(real_diag) | (n_valid == 0), np.nan, percentile)
    # Reduce state by state over the non-NaN draws; a state no path visited stays NaN.
    # (Done by hand rather than with nanmean / nanpercentile so an all-NaN column does
    # not emit RuntimeWarnings for a case that is expected and documented.)
    n_states = null_diags.shape[1]
    null_mean = np.full(n_states, np.nan)
    null_lower = np.full(n_states, np.nan)
    null_upper = np.full(n_states, np.nan)
    for j in range(n_states):
        visited = null_diags[valid[:, j], j]
        if visited.size:
            null_mean[j] = visited.mean()
            null_lower[j] = np.percentile(visited, NULL_BAND_LOWER_PCT)
            null_upper[j] = np.percentile(visited, NULL_BAND_UPPER_PCT)
    return {
        "diagonal_percentile": percentile,
        "null_diagonal_mean": null_mean,
        "null_diagonal_lower": null_lower,
        "null_diagonal_upper": null_upper,
    }


def _nan_aware_band(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column 2.5% / 97.5% quantiles over the non-NaN rows of ``draws``.

    ``draws`` has shape ``(n_draws, n_states)``. A column with no finite draw yields
    NaN for both bounds. Done by hand rather than with ``nanpercentile`` so an all-NaN
    column does not emit a RuntimeWarning for a documented, expected case.
    """
    n_states = draws.shape[1]
    lower = np.full(n_states, np.nan)
    upper = np.full(n_states, np.nan)
    for j in range(n_states):
        finite = draws[~np.isnan(draws[:, j]), j]
        if finite.size:
            lower[j] = np.percentile(finite, NULL_BAND_LOWER_PCT)
            upper[j] = np.percentile(finite, NULL_BAND_UPPER_PCT)
    return lower, upper


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate_markov_chain(
    matrix: np.ndarray,
    n_steps: int,
    seed: int,
    start_state: int | None = None,
) -> np.ndarray:
    """Sample ``n_steps`` integer labels from a row-stochastic ``matrix``.

    ``start_state=None`` draws the first state from the stationary distribution of
    ``matrix`` so the path is stationary from step 0. Each step is an inverse-CDF draw
    against the precomputed cumulative row of the current state, with one uniform per
    step drawn up front. The loop over steps is the one unavoidable sequential pass:
    state ``t + 1`` depends on the realised state ``t``, so it cannot be vectorised.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_states = matrix.shape[0] if matrix.ndim == 2 else 0
    validate_transition_matrix(matrix, n_states)
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if start_state is not None and not 0 <= start_state < n_states:
        raise ValueError(f"start_state must lie in [0, {n_states}), got {start_state}")

    rng = np.random.default_rng(seed)
    if start_state is None:
        start_state = int(rng.choice(n_states, p=stationary_distribution(matrix)))

    cumulative = np.cumsum(matrix, axis=1)
    # Guard against rows summing to 1 - 1e-12 sending a uniform draw past the last column.
    cumulative[:, -1] = np.where(cumulative[:, -1] > 0, 1.0, cumulative[:, -1])
    uniforms = rng.random(n_steps)
    path = np.empty(n_steps, dtype=int)
    state = start_state
    path[0] = state
    for t in range(1, n_steps):  # sequential by construction, see docstring
        if cumulative[state, -1] == 0:
            raise ValueError(f"state {state} has an all-zero row; the chain cannot leave it")
        state = int(np.searchsorted(cumulative[state], uniforms[t], side="right"))
        state = min(state, n_states - 1)
        path[t] = state
    return path


# --------------------------------------------------------------------------- #
# Block bootstrap
# --------------------------------------------------------------------------- #
def _block_bootstrap_matrices(
    arr: np.ndarray,
    n_resamples: int,
    block_days: int,
    rng: np.random.Generator,
    n_states: int,
) -> np.ndarray:
    """Circular moving-block-bootstrap refits of the transition matrix.

    ``arr`` is an already validated 1-D int label array with at least two entries.
    Returns the resampled matrices, shape ``(n_resamples, n_states, n_states)``, one
    ``normalise_rows`` refit per resample. Transitions are counted within blocks only;
    see ``block_bootstrap_ci`` for why. The only random draw is the block-start array,
    so the stream consumed from ``rng`` is exactly ``n_resamples * n_blocks`` integers.
    """
    n = arr.size
    n_blocks = -(-n // block_days)  # ceiling division
    # All block starts for all resamples at once: (n_resamples, n_blocks).
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    offsets = np.arange(block_days)
    # (n_resamples, n_blocks * block_days) -> wrap indices, cut to length n.
    index = ((starts[:, :, np.newaxis] + offsets) % n).reshape(n_resamples, -1)[:, :n]
    resampled = arr[index]
    # Keep only the pairs (t, t + 1) with both positions inside the same block.
    positions = np.arange(n - 1)
    within_block = (positions + 1) % block_days != 0
    src = resampled[:, :-1][:, within_block]
    dst = resampled[:, 1:][:, within_block]
    counts = np.zeros((n_resamples, n_states, n_states))
    resample_id = np.broadcast_to(np.arange(n_resamples)[:, np.newaxis], src.shape)
    np.add.at(counts, (resample_id, src, dst), 1.0)
    return np.stack([normalise_rows(counts[b]) for b in range(n_resamples)])


def block_bootstrap_ci(
    labels: ArrayLike,
    n_resamples: int = 500,  # config/default.yaml statistics.bootstrap.resamples
    block_days: int = 20,  # config/default.yaml statistics.bootstrap.block_days = labeller window
    ci: float = 0.95,  # conventional two-sided 95% interval
    seed: int = 0,
    n_states: int = N_STATES,
) -> dict:
    """Moving (circular) block bootstrap CI for every matrix cell and the stationary vector.

    Each resample draws ``ceil(n / block_days)`` blocks of ``block_days`` consecutive
    labels with uniformly random start positions, wrapping around the end of the
    sequence (circular block bootstrap, Politis and Romano 1992), concatenates them and
    truncates to the original length. Transitions are counted *within* blocks only: the
    pair that straddles two blocks joins labels that were never adjacent in the data, and
    with ``n / block_days`` such joins per resample it would pull every diagonal towards
    the marginal frequency by roughly ``(p_ii - pi_i) / block_days``, enough to push a
    0.9 diagonal out of its own interval. Each resample therefore holds
    ``n - n_blocks`` transitions instead of ``n - 1``; the interval is marginally wider,
    never biased. The matrix is refitted per resample and the stationary vector derived
    from it; the ``(1 - ci) / 2`` and ``1 - (1 - ci) / 2`` quantiles are taken per cell.

    The default block length equals the labeller window (20 days) so each block carries
    a full window of the dependence the overlapping labeller induces; shorter blocks
    would break that dependence and understate the interval width.

    Returns a dict with numpy values: ``matrix_lower``, ``matrix_upper`` of shape
    ``(n_states, n_states)``; ``stationary_lower``, ``stationary_upper`` of shape
    ``(n_states,)``; plus ``n_resamples``, ``block_days`` and ``ci`` echoed back.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must lie strictly in (0, 1), got {ci}")
    arr = _labels_1d(labels, n_states)
    n = arr.size
    if n < 2:
        raise ValueError(f"need at least 2 labels to bootstrap a transition matrix, got {n}")

    rng = np.random.default_rng(seed)
    matrices = _block_bootstrap_matrices(arr, n_resamples, block_days, rng, n_states)
    stationaries = np.stack([stationary_distribution(m) for m in matrices])

    lower_q = PERCENT * (1.0 - ci) / 2.0
    upper_q = PERCENT - lower_q
    return {
        "matrix_lower": np.percentile(matrices, lower_q, axis=0),
        "matrix_upper": np.percentile(matrices, upper_q, axis=0),
        "stationary_lower": np.percentile(stationaries, lower_q, axis=0),
        "stationary_upper": np.percentile(stationaries, upper_q, axis=0),
        "n_resamples": int(n_resamples),
        "block_days": int(block_days),
        "ci": float(ci),
    }


# --------------------------------------------------------------------------- #
# Null models on the label sequence
# --------------------------------------------------------------------------- #
def label_shuffle_null(
    labels: ArrayLike,
    n_paths: int = 1000,  # config/default.yaml statistics.null_paths
    seed: int = 0,
    n_states: int = N_STATES,
) -> dict:
    """Label-shuffle null: permute the labels ``n_paths`` times and refit the matrix.

    A permutation keeps the exact regime frequencies and destroys all temporal order,
    so the null diagonal is what persistence looks like with zero dependence. Returns
    ``diagonal_percentile`` (``100 * P(null diagonal <= real diagonal)`` per state),
    ``null_diagonal_mean``, ``null_diagonal_lower`` / ``null_diagonal_upper`` (2.5% and
    97.5% quantiles) and ``n_paths``. States never visited by a path contribute NaN,
    which the nan-aware summaries ignore.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    arr = _labels_1d(labels, n_states)
    if arr.size < 2:
        raise ValueError(f"need at least 2 labels for a null model, got {arr.size}")
    rng = np.random.default_rng(seed)
    real_diag = _diagonal_or_nan(build_transition_matrix(arr, n_states))
    null_diags = np.empty((n_paths, n_states))
    for p in range(n_paths):  # one refit per permutation
        shuffled = rng.permutation(arr)
        null_diags[p] = _diagonal_or_nan(build_transition_matrix(shuffled, n_states))
    out = _summarise_null_diagonals(real_diag, null_diags)
    out["n_paths"] = int(n_paths)
    return out


def iid_null(
    labels: ArrayLike,
    n_paths: int = 1000,  # config/default.yaml statistics.null_paths
    seed: int = 0,
    n_states: int = N_STATES,
) -> dict:
    """IID null with the same marginal regime frequencies as ``labels``.

    Each null path draws labels independently from the empirical marginal
    distribution, same length as the real sequence, then refits the matrix. Unlike the
    permutation null the frequencies vary from path to path, which is the sampling
    variation an iid model actually implies. Same return keys as
    ``label_shuffle_null``.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    arr = _labels_1d(labels, n_states)
    if arr.size < 2:
        raise ValueError(f"need at least 2 labels for a null model, got {arr.size}")
    rng = np.random.default_rng(seed)
    marginals = np.bincount(arr, minlength=n_states) / arr.size
    real_diag = _diagonal_or_nan(build_transition_matrix(arr, n_states))
    # All draws at once: (n_paths, n), then one refit per path.
    draws = rng.choice(n_states, size=(n_paths, arr.size), p=marginals)
    null_diags = np.empty((n_paths, n_states))
    for p in range(n_paths):
        null_diags[p] = _diagonal_or_nan(build_transition_matrix(draws[p], n_states))
    out = _summarise_null_diagonals(real_diag, null_diags)
    out["n_paths"] = int(n_paths)
    return out


# --------------------------------------------------------------------------- #
# Labeller-induced null (Fix 1)
# --------------------------------------------------------------------------- #
def labeller_null(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window
    threshold: float = 0.05,  # config/default.yaml labelling.threshold
    return_type: str = "simple",  # config/default.yaml labelling.return_type
    n_paths: int = 1000,  # config/default.yaml statistics.null_paths
    seed: int = 0,
    n_states: int = N_STATES,
    bootstrap_resamples: int = 500,  # config/default.yaml statistics.bootstrap.resamples
    block_days: int = 20,  # config/default.yaml statistics.bootstrap.block_days = labeller window
) -> dict:
    """Labeller-induced null (Fix 1): how much persistence the labeller manufactures.

    Simulates ``n_paths`` iid Gaussian log-return series with the mean and standard
    deviation of the real daily log returns, builds a price path
    ``close.iloc[0] * exp(cumsum)`` on the same ``DatetimeIndex``, runs the identical
    labeller, and fits the transition matrix. Because consecutive labels share
    ``window - 1`` returns, even iid prices produce a fat diagonal; that diagonal is the
    labeller's own contribution and is subtracted from the real one.

    Returns ``real_diagonal``, ``null_diagonal_mean``, ``null_diagonal_lower`` /
    ``null_diagonal_upper`` (2.5% / 97.5% of the null draws), ``diagonal_percentile``,
    ``excess_persistence = real - null_mean`` (point estimate), ``excess_lower`` /
    ``excess_upper`` (a 95% interval for the excess, below) and ``n_paths``.

    The excess interval combines two sources of error treated as independent: the
    sampling error of the real diagonal, from a circular block bootstrap of the real
    labels (``bootstrap_resamples`` refits with ``block_days`` blocks, as in
    ``block_bootstrap_ci``), and the labeller-null spread across the ``n_paths`` null
    diagonals. ``max(bootstrap_resamples, n_paths)`` pairs ``(i, j)`` are drawn with
    replacement from the two sets of draws, the excess draws ``real_b[i] - null_p[j]``
    are formed, and the 2.5% / 97.5% quantiles of those are reported. Subtracting the
    null band from the point estimate alone would ignore the real diagonal's own
    sampling error, which on a decade of daily data is several percentage points for
    the rarer regimes.

    A null path (or bootstrap resample) that never visits a state has an all-zero row
    for it; that state's diagonal on that draw is recorded as NaN rather than 0 and all
    summaries are nan-aware, so a rarely visited regime is summarised from the draws
    that did visit it rather than dragged towards zero.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if bootstrap_resamples < 1:
        raise ValueError(f"bootstrap_resamples must be >= 1, got {bootstrap_resamples}")
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")
    if not isinstance(close, pd.Series):
        raise ValueError(f"close must be a pandas Series, got {type(close).__name__}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(f"close must have a DatetimeIndex, got {type(close.index).__name__}")
    if close.size < window + MIN_LABELS_FOR_TEST:
        raise ValueError(
            f"close needs at least window + 2 = {window + MIN_LABELS_FOR_TEST} rows to fit a "
            f"matrix, got {close.size}"
        )
    prices = close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("close must be finite and strictly positive to take log returns")

    real_labels = label_regimes(close, window, threshold, return_type).to_numpy()
    real_diag = _diagonal_or_nan(build_transition_matrix(real_labels, n_states))

    log_returns = np.diff(np.log(prices))
    mu = log_returns.mean()
    sigma = log_returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    # All null log returns at once: (n_paths, n - 1); the price level starts at close[0].
    null_returns = rng.normal(mu, sigma, size=(n_paths, log_returns.size))
    null_log_prices = np.log(prices[0]) + np.cumsum(null_returns, axis=1)
    null_prices = np.exp(
        np.concatenate([np.full((n_paths, 1), np.log(prices[0])), null_log_prices], axis=1)
    )

    null_diags = np.empty((n_paths, n_states))
    for p in range(n_paths):  # one labeller pass and one refit per path
        path = pd.Series(null_prices[p], index=close.index)
        null_labels = label_regimes(path, window, threshold, return_type).to_numpy()
        null_diags[p] = _diagonal_or_nan(build_transition_matrix(null_labels, n_states))

    out = _summarise_null_diagonals(real_diag, null_diags)

    # Sampling error of the real diagonal: block-bootstrap refits of the real labels,
    # drawn after the null returns so the null stream for a given seed is unchanged.
    real_matrices = _block_bootstrap_matrices(
        real_labels, bootstrap_resamples, block_days, rng, n_states
    )
    real_b = np.stack([_diagonal_or_nan(m) for m in real_matrices])
    # Pair bootstrap draws with null draws by independent indexing with replacement;
    # unequal lengths are fine because each index is drawn from its own range.
    n_pairs = max(bootstrap_resamples, n_paths)
    i = rng.integers(0, bootstrap_resamples, size=n_pairs)
    j = rng.integers(0, n_paths, size=n_pairs)
    excess_draws = real_b[i] - null_diags[j]
    excess_lower, excess_upper = _nan_aware_band(excess_draws)
    return {
        "real_diagonal": real_diag,
        "null_diagonal_mean": out["null_diagonal_mean"],
        "null_diagonal_lower": out["null_diagonal_lower"],
        "null_diagonal_upper": out["null_diagonal_upper"],
        "diagonal_percentile": out["diagonal_percentile"],
        "excess_persistence": real_diag - out["null_diagonal_mean"],
        "excess_lower": excess_lower,
        "excess_upper": excess_upper,
        "n_paths": int(n_paths),
        "bootstrap_resamples": int(bootstrap_resamples),
        "block_days": int(block_days),
    }


# --------------------------------------------------------------------------- #
# Anderson-Goodman chi-square tests
# --------------------------------------------------------------------------- #
def anderson_goodman_test(labels: ArrayLike, n_states: int = N_STATES) -> dict:
    """Anderson-Goodman (1957) chi-square test of first-order dependence vs independence.

    Apply this to the stride sequence (``labels[::window]``), not the overlapping one:
    consecutive overlapping labels share ``window - 1`` returns, so they reject the
    independence null trivially and the p-value would say nothing about the market.

    With transition counts ``n_ij``, row totals ``n_i.`` and destination marginals
    ``p_j = n_.j / n``, the Pearson statistic is the sum over cells with positive
    expected count of ``(n_ij - n_i. p_j)^2 / (n_i. p_j)``. Degrees of freedom are
    ``(k - 1)^2`` where ``k`` is the number of states that actually occur in the
    sequence, not ``n_states``: a state with no observations contributes neither rows
    nor columns, and counting it would inflate ``df`` and understate the evidence.

    Returns ``statistic``, ``p_value`` (``scipy.stats.chi2.sf``), ``df`` and
    ``n_transitions``.
    """
    arr = _labels_1d(labels, n_states)
    if arr.size < MIN_LABELS_FOR_TEST:
        raise ValueError(f"need at least {MIN_LABELS_FOR_TEST} labels to test, got {arr.size}")
    counts = count_transitions(arr, n_states)
    n = counts.sum()
    row_totals = counts.sum(axis=1, keepdims=True)
    col_marginals = counts.sum(axis=0) / n
    expected = row_totals * col_marginals[np.newaxis, :]
    mask = expected > 0
    statistic = float(((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    k = int(np.unique(arr).size)
    df = (k - 1) ** 2
    p_value = float(sp_stats.chi2.sf(statistic, df)) if df > 0 else 1.0
    return {
        "statistic": statistic,
        "p_value": p_value,
        "df": int(df),
        "n_transitions": int(n),
    }


def homogeneity_test(
    labels: ArrayLike,
    n_periods: int = 2,  # two halves of the ~125-label stride sequence on 10 years of daily data
    n_states: int = N_STATES,
) -> dict:
    """Anderson-Goodman homogeneity test: one transition matrix across sub-periods (Fix 4).

    Apply this to the stride sequence (``labels[::window]``), not the overlapping one,
    for the same reason as ``anderson_goodman_test``: consecutive overlapping labels
    share ``window - 1`` returns, which violates the independent-transitions assumption
    behind the chi-square statistic and inflates it. On iid Gaussian prices with the
    20-day labeller the daily sequence is rejected at the 5% level roughly 30% of the
    time (four periods); the stride sequence is rejected at about the nominal rate.

    Splits the sequence into ``n_periods`` contiguous, equal-length sub-periods (the
    remainder goes to the last one). For period ``s`` with counts ``n_ij^(s)`` and row
    totals ``n_i.^(s)``, and pooled MLE ``p_ij`` over all periods, the statistic is the
    sum over periods and cells with positive expected count of
    ``(n_ij^(s) - n_i.^(s) p_ij)^2 / (n_i.^(s) p_ij)``, with
    ``df = (S - 1) * k * (k - 1)`` where ``k`` is the number of occurring states.

    ``stable`` is ``p_value >= 0.05`` (conventional 5% level). A rejection is reported as
    "matrix is not stable": useful information about drift, not a failure.

    The default of two sub-periods suits the stride sequence: ten years of daily data
    give only about 125 stride labels, and two halves keep enough transitions per row
    for the chi-square approximation while still detecting a change in regime
    probabilities between the halves. More periods on that few labels thin the
    per-period counts and lose power.
    Raises ``ValueError`` for ``n_periods < 2`` or a sequence too short to give each
    period at least two labels.
    """
    if n_periods < 2:
        raise ValueError(f"n_periods must be >= 2, got {n_periods}")
    arr = _labels_1d(labels, n_states)
    if arr.size < n_periods * MIN_LABELS_PER_PERIOD:
        raise ValueError(
            f"need at least {n_periods * MIN_LABELS_PER_PERIOD} labels for {n_periods} periods "
            f"of {MIN_LABELS_PER_PERIOD}, got {arr.size}"
        )
    period_len = arr.size // n_periods
    bounds = [s * period_len for s in range(n_periods)] + [arr.size]
    period_counts = np.stack(
        [count_transitions(arr[bounds[s] : bounds[s + 1]], n_states) for s in range(n_periods)]
    )
    pooled = period_counts.sum(axis=0)
    pooled_rows = pooled.sum(axis=1, keepdims=True)
    pooled_p = pooled / np.where(pooled_rows == 0, 1.0, pooled_rows)
    expected = period_counts.sum(axis=2, keepdims=True) * pooled_p[np.newaxis, :, :]
    mask = expected > 0
    statistic = float(((period_counts[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    k = int(np.unique(arr).size)
    df = (n_periods - 1) * k * (k - 1)
    p_value = float(sp_stats.chi2.sf(statistic, df)) if df > 0 else 1.0
    return {
        "statistic": statistic,
        "p_value": p_value,
        "df": int(df),
        "n_periods": int(n_periods),
        "stable": bool(p_value >= HOMOGENEITY_ALPHA),
    }
