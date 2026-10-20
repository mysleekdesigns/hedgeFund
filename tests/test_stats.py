"""Tests for the honest statistics in ``hedgefund.regime.stats`` (plan.md Phase 2)."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, label_regimes
from hedgefund.regime.markov import (
    build_transition_matrix,
    stationary_distribution,
)
from hedgefund.regime.stats import (
    anderson_goodman_test,
    block_bootstrap_ci,
    homogeneity_test,
    iid_null,
    label_shuffle_null,
    labeller_null,
    simulate_markov_chain,
)

# Known persistent chain; diagonal 0.9 / 0.8 / 0.85 as the plan's coverage test specifies.
TRUE_MATRIX = np.array(
    [
        [0.90, 0.07, 0.03],
        [0.10, 0.80, 0.10],
        [0.03, 0.12, 0.85],
    ]
)
N_STATES = 3

label_sequences = st.lists(st.integers(min_value=0, max_value=2), min_size=50, max_size=300).map(
    lambda seq: np.array(seq, dtype=int)
)


def _iid_labels(n: int, seed: int, p: tuple[float, ...] = (0.2, 0.6, 0.2)) -> np.ndarray:
    return np.random.default_rng(seed).choice(N_STATES, size=n, p=p)


def _gaussian_close(n: int, seed: int, daily_vol: float = 0.01, mu: float = 0.0) -> pd.Series:
    """IID Gaussian log-return price path with a business-day index."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu, daily_vol, size=n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return pd.Series(prices, index=pd.bdate_range("2010-01-01", periods=n))


# --------------------------------------------------------------------------- #
# simulate_markov_chain
# --------------------------------------------------------------------------- #
def test_simulate_markov_chain_outputs_valid_states_and_length() -> None:
    path = simulate_markov_chain(TRUE_MATRIX, n_steps=500, seed=1)
    assert path.shape == (500,)
    assert path.dtype.kind == "i"
    assert path.min() >= 0 and path.max() < N_STATES


def test_simulate_markov_chain_empirical_matrix_converges_to_truth() -> None:
    # 20000 steps gives at least ~4000 visits to the least frequent state. The binomial
    # standard error of a cell near 0.8 is sqrt(0.8 * 0.2 / 4000) = 0.006, so 0.03 is 5 sigma.
    path = simulate_markov_chain(TRUE_MATRIX, n_steps=20_000, seed=7)
    np.testing.assert_allclose(build_transition_matrix(path), TRUE_MATRIX, atol=0.03)


def test_simulate_markov_chain_is_deterministic_for_same_seed() -> None:
    first = simulate_markov_chain(TRUE_MATRIX, 1000, seed=42)
    second = simulate_markov_chain(TRUE_MATRIX, 1000, seed=42)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, simulate_markov_chain(TRUE_MATRIX, 1000, seed=43))


def test_simulate_markov_chain_honours_start_state() -> None:
    assert simulate_markov_chain(TRUE_MATRIX, 10, seed=0, start_state=BULL)[0] == BULL


def test_simulate_markov_chain_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        simulate_markov_chain(TRUE_MATRIX, 0, seed=0)
    with pytest.raises(ValueError, match="start_state"):
        simulate_markov_chain(TRUE_MATRIX, 10, seed=0, start_state=3)
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_markov_chain(np.full((3, 3), 0.5), 10, seed=0)


# --------------------------------------------------------------------------- #
# block_bootstrap_ci
# --------------------------------------------------------------------------- #
def test_block_bootstrap_ci_shapes_and_echoed_parameters() -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 500, seed=3)
    out = block_bootstrap_ci(labels, n_resamples=50, block_days=10, ci=0.9, seed=1)
    assert out["matrix_lower"].shape == (N_STATES, N_STATES)
    assert out["matrix_upper"].shape == (N_STATES, N_STATES)
    assert out["stationary_lower"].shape == (N_STATES,)
    assert out["stationary_upper"].shape == (N_STATES,)
    assert out["n_resamples"] == 50 and out["block_days"] == 10 and out["ci"] == 0.9


def test_block_bootstrap_ci_brackets_point_estimate_on_long_chain() -> None:
    # On a 5000-step chain the point estimate sits inside its own 95% interval: the
    # bootstrap distribution is centred on it, so lower <= point <= upper cell by cell.
    labels = simulate_markov_chain(TRUE_MATRIX, 5000, seed=5)
    out = block_bootstrap_ci(labels, n_resamples=100, seed=2)
    point = build_transition_matrix(labels)
    pi = stationary_distribution(point)
    assert (out["matrix_lower"] <= point).all() and (point <= out["matrix_upper"]).all()
    assert (out["stationary_lower"] <= pi).all() and (pi <= out["stationary_upper"]).all()


def test_block_bootstrap_ci_bounds_are_valid_probabilities() -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 2000, seed=8)
    out = block_bootstrap_ci(labels, n_resamples=100, seed=0)
    for key in ("matrix_lower", "matrix_upper", "stationary_lower", "stationary_upper"):
        assert (out[key] >= 0).all() and (out[key] <= 1).all()
    # Each row of a bound is a quantile per cell, not a resampled matrix, so it need not
    # sum to 1; but the lower row sum cannot exceed 1 and the upper row sum cannot be below 1.
    assert (out["matrix_lower"].sum(axis=1) <= 1 + 1e-12).all()
    assert (out["matrix_upper"].sum(axis=1) >= 1 - 1e-12).all()


def test_block_bootstrap_ci_is_deterministic_for_same_seed() -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 500, seed=3)
    a = block_bootstrap_ci(labels, n_resamples=30, seed=11)
    b = block_bootstrap_ci(labels, n_resamples=30, seed=11)
    np.testing.assert_array_equal(a["matrix_lower"], b["matrix_lower"])
    np.testing.assert_array_equal(a["stationary_upper"], b["stationary_upper"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_resamples": 0}, "n_resamples"),
        ({"block_days": 0}, "block_days"),
        ({"ci": 0.0}, "ci"),
        ({"ci": 1.0}, "ci"),
    ],
)
def test_block_bootstrap_ci_rejects_bad_parameters(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        block_bootstrap_ci(np.array([0, 1, 2, 1]), **kwargs)


def test_block_bootstrap_ci_rejects_fewer_than_two_labels() -> None:
    with pytest.raises(ValueError, match="at least 2 labels"):
        block_bootstrap_ci(np.array([1]))


def _ci_coverage(n_chains: int, n_steps: int, n_resamples: int) -> float:
    """Fraction of (chain, cell) pairs whose bootstrap CI contains the true matrix entry."""
    hits = 0
    for c in range(n_chains):
        labels = simulate_markov_chain(TRUE_MATRIX, n_steps, seed=1000 + c)
        out = block_bootstrap_ci(labels, n_resamples=n_resamples, seed=c)
        inside = (out["matrix_lower"] <= TRUE_MATRIX) & (out["matrix_upper"] >= TRUE_MATRIX)
        hits += int(inside.sum())
    return hits / (n_chains * TRUE_MATRIX.size)


def test_bootstrap_ci_contains_true_matrix_at_nominal_rate() -> None:
    # 40 chains x 9 cells = 360 checks of a nominal 95% interval. The binomial standard
    # error is sqrt(0.95 * 0.05 / 360) = 0.011, so 0.85 is about 9 sigma below nominal and
    # leaves room for the block bootstrap's small downward bias: joins between blocks
    # inject spurious transitions and the chain's 2000 steps hold only 100 blocks.
    assert 0.85 <= _ci_coverage(n_chains=40, n_steps=2000, n_resamples=200) <= 1.0


@pytest.mark.slow
def test_bootstrap_ci_contains_true_matrix_at_nominal_rate_slow() -> None:
    # 200 chains x 9 cells = 1800 checks; binomial standard error 0.005, so [0.90, 0.99]
    # is roughly 10 sigma on each side of the nominal 0.95 after the block-join bias.
    assert 0.90 <= _ci_coverage(n_chains=200, n_steps=2000, n_resamples=500) <= 0.99


@settings(deadline=None, max_examples=25)
@given(labels=label_sequences)
def test_block_bootstrap_ci_bounds_lie_in_unit_interval_and_are_ordered(
    labels: np.ndarray,
) -> None:
    out = block_bootstrap_ci(labels, n_resamples=20, block_days=5, seed=0)
    for key in ("matrix_lower", "matrix_upper", "stationary_lower", "stationary_upper"):
        assert (out[key] >= 0).all() and (out[key] <= 1).all()
    assert (out["matrix_lower"] <= out["matrix_upper"]).all()
    assert (out["stationary_lower"] <= out["stationary_upper"]).all()


# --------------------------------------------------------------------------- #
# label_shuffle_null / iid_null
# --------------------------------------------------------------------------- #
NULL_KEYS = (
    "diagonal_percentile",
    "null_diagonal_mean",
    "null_diagonal_lower",
    "null_diagonal_upper",
)


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_rank_persistent_chain_diagonal_at_top(null_fn) -> None:
    # A 0.8+ diagonal on 2000 steps is far outside anything a shuffle can produce, so the
    # real diagonal beats every one of the 200 null draws: percentile 100 >= 99.
    labels = simulate_markov_chain(TRUE_MATRIX, 2000, seed=21)
    out = null_fn(labels, n_paths=200, seed=1)
    assert (out["diagonal_percentile"] >= 99).all()


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_do_not_flag_iid_labels(null_fn) -> None:
    # Under the null the percentile is uniform on [0, 100], so each state lands in the
    # extreme 2% with probability 0.02. The chance that two or more of three states do so
    # is about 0.001; with a fixed seed the test is deterministic, so that is the risk
    # accepted once at authoring time, not per run.
    labels = _iid_labels(2000, seed=22)
    out = null_fn(labels, n_paths=200, seed=2)
    not_extreme = (out["diagonal_percentile"] > 1) & (out["diagonal_percentile"] < 99)
    assert not_extreme.sum() >= 2


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_diagonal_mean_matches_marginal_frequencies(null_fn) -> None:
    # Without dependence P(j | i) = P(j), so the null diagonal is the marginal frequency.
    # Each null diagonal is a binomial proportion on ~400+ transitions (se <= 0.025) and
    # the mean over 200 paths shrinks that to ~0.002; 0.05 is a generous bound.
    labels = _iid_labels(2000, seed=23)
    marginals = np.bincount(labels, minlength=N_STATES) / labels.size
    out = null_fn(labels, n_paths=200, seed=3)
    np.testing.assert_allclose(out["null_diagonal_mean"], marginals, atol=0.05)


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_shapes_and_bounds(null_fn) -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 300, seed=24)
    out = null_fn(labels, n_paths=50, seed=4)
    for key in NULL_KEYS:
        assert out[key].shape == (N_STATES,)
    assert out["n_paths"] == 50
    assert (out["diagonal_percentile"] >= 0).all() and (out["diagonal_percentile"] <= 100).all()
    assert (out["null_diagonal_lower"] <= out["null_diagonal_mean"]).all()
    assert (out["null_diagonal_mean"] <= out["null_diagonal_upper"]).all()


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_are_deterministic_for_same_seed(null_fn) -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 300, seed=24)
    a = null_fn(labels, n_paths=30, seed=9)
    b = null_fn(labels, n_paths=30, seed=9)
    for key in NULL_KEYS:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_reject_bad_inputs(null_fn) -> None:
    with pytest.raises(ValueError, match="n_paths"):
        null_fn(np.array([0, 1, 2]), n_paths=0)
    with pytest.raises(ValueError, match="at least 2 labels"):
        null_fn(np.array([1]), n_paths=5)


@pytest.mark.slow
@pytest.mark.parametrize("null_fn", [label_shuffle_null, iid_null])
def test_null_models_rank_persistent_chain_diagonal_at_top_slow(null_fn) -> None:
    labels = simulate_markov_chain(TRUE_MATRIX, 2000, seed=21)
    out = null_fn(labels, n_paths=1000, seed=1)
    assert (out["diagonal_percentile"] >= 99).all()


# --------------------------------------------------------------------------- #
# labeller_null (Fix 1)
# --------------------------------------------------------------------------- #
LABELLER_KEYS = (
    "real_diagonal",
    "null_diagonal_mean",
    "null_diagonal_lower",
    "null_diagonal_upper",
    "diagonal_percentile",
    "excess_persistence",
    "excess_lower",
    "excess_upper",
)


def test_labeller_null_finds_no_excess_persistence_on_iid_prices() -> None:
    # The real series is itself an iid Gaussian path, so it is one more draw from the
    # null. The null diagonals on 3000 bars have a spread of a few percent (the
    # Sideways row has ~2000 transitions, Bull / Bear ~500 each), so |excess| < 0.1 is
    # a 3+ sigma bound; the 95% band of the null brackets zero excess with probability
    # 0.95 per state, so at least two of three bracket it with probability > 0.99.
    close = _gaussian_close(3000, seed=31)
    out = labeller_null(close, n_paths=100, seed=5)
    assert (np.abs(out["excess_persistence"]) < 0.1).all()
    brackets_zero = (out["excess_lower"] <= 0) & (out["excess_upper"] >= 0)
    assert brackets_zero.sum() >= 2


def test_labeller_null_shows_inflated_diagonal_even_on_iid_prices() -> None:
    # The whole point of Fix 1: 20-day overlapping windows manufacture persistence, so
    # the null diagonal on iid prices is far above the marginal frequency of each state.
    close = _gaussian_close(3000, seed=32)
    out = labeller_null(close, n_paths=50, seed=6)
    # Bull and Bear are each hit on well under half the days, yet their null persistence
    # sits above 0.7 because a 5% move takes many days to unwind. Sideways is above 0.8.
    assert (out["null_diagonal_mean"][[BEAR, BULL]] > 0.7).all()
    assert out["null_diagonal_mean"][SIDEWAYS] > 0.8


def _regime_switching_close(n: int, seed: int) -> pd.Series:
    """Prices from a sticky Bull / Bear / Sideways drift process so persistence is real.

    Regime drift of +/-0.5% per day against 1% daily vol is strong: a 20-day Bull run
    returns ~10% on average against a 4.5% window standard deviation, so the trailing
    return clears the 5% threshold on most days and labels follow the latent regime.
    """
    drift = np.array([-0.005, 0.0, 0.005])  # Bear, Sideways, Bull mean log return per day
    sticky = np.array(
        [
            [0.98, 0.01, 0.01],
            [0.02, 0.96, 0.02],
            [0.01, 0.01, 0.98],
        ]
    )
    regimes = simulate_markov_chain(sticky, n - 1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    log_returns = drift[regimes] + rng.normal(0.0, 0.01, size=n - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return pd.Series(prices, index=pd.bdate_range("2010-01-01", periods=n))


def test_labeller_null_detects_real_persistence_on_regime_switching_prices() -> None:
    # Latent regimes with mean episode length 50 days keep the trailing return on one side
    # of the threshold for long stretches, so the Bull and Bear diagonals exceed what the
    # labeller alone produces by a clear margin: 0.05 is the smallest excess that the
    # 95% null band (width ~0.05 on 3000 bars) could not explain.
    close = _regime_switching_close(3000, seed=33)
    out = labeller_null(close, n_paths=100, seed=7)
    assert out["excess_persistence"][BULL] > 0.05
    assert out["excess_persistence"][BEAR] > 0.05
    assert out["excess_lower"][BULL] > 0 and out["excess_lower"][BEAR] > 0
    assert (out["diagonal_percentile"][[BEAR, BULL]] >= 99).all()


def test_labeller_null_shapes_and_excess_arithmetic() -> None:
    close = _gaussian_close(600, seed=34)
    out = labeller_null(close, n_paths=20, seed=8, bootstrap_resamples=100)
    for key in LABELLER_KEYS:
        assert out[key].shape == (N_STATES,)
    assert out["n_paths"] == 20
    assert out["bootstrap_resamples"] == 100 and out["block_days"] == 20
    np.testing.assert_allclose(
        out["excess_persistence"], out["real_diagonal"] - out["null_diagonal_mean"]
    )
    finite = ~np.isnan(out["excess_lower"])
    assert finite.all()  # 600 bars at 1% vol visit every state on every draw
    # The excess interval adds the real diagonal's block-bootstrap sampling error to the
    # labeller-null spread, so it cannot be narrower than the null-only band
    # (null_upper - null_lower) that the old "real - null band" shortcut reported.
    excess_width = out["excess_upper"] - out["excess_lower"]
    null_width = out["null_diagonal_upper"] - out["null_diagonal_lower"]
    assert (excess_width[finite] >= null_width[finite]).all()
    assert (out["excess_lower"] <= out["excess_persistence"]).all()
    assert (out["excess_persistence"] <= out["excess_upper"]).all()


def test_labeller_null_excess_interval_brackets_zero_for_every_state_on_iid_prices() -> None:
    # The real series is one more iid draw, so the true excess is zero in every state.
    # The 95% excess interval brackets zero with probability >= 0.95 per state; all three
    # do so with probability about 0.86, a risk taken once when the seed was fixed.
    close = _gaussian_close(3000, seed=38)
    out = labeller_null(close, n_paths=100, seed=14, bootstrap_resamples=200)
    assert (out["excess_lower"] <= 0).all()
    assert (out["excess_upper"] >= 0).all()


def test_labeller_null_is_deterministic_for_same_seed() -> None:
    close = _gaussian_close(400, seed=35)
    a = labeller_null(close, n_paths=10, seed=12)
    b = labeller_null(close, n_paths=10, seed=12)
    for key in LABELLER_KEYS:
        np.testing.assert_array_equal(a[key], b[key])


def test_labeller_null_handles_null_paths_that_miss_a_state() -> None:
    # With a 30% threshold and 0.1% daily vol no 20-day window ever moves 30%, so every
    # null path is all Sideways and Bull / Bear are never visited: their null summaries
    # must be NaN (missing), not 0, while Sideways is still summarised normally.
    close = _gaussian_close(300, seed=36, daily_vol=0.001)
    out = labeller_null(close, threshold=0.30, n_paths=10, seed=13)
    assert np.isnan(out["null_diagonal_mean"][[BEAR, BULL]]).all()
    assert np.isnan(out["diagonal_percentile"][[BEAR, BULL]]).all()
    assert out["null_diagonal_mean"][SIDEWAYS] == 1.0
    assert out["real_diagonal"][SIDEWAYS] == 1.0
    assert out["excess_persistence"][SIDEWAYS] == 0.0


def test_labeller_null_rejects_bad_inputs() -> None:
    close = _gaussian_close(100, seed=37)
    with pytest.raises(ValueError, match="n_paths"):
        labeller_null(close, n_paths=0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        labeller_null(close, n_paths=2, bootstrap_resamples=0)
    with pytest.raises(ValueError, match="block_days"):
        labeller_null(close, n_paths=2, block_days=0)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        labeller_null(pd.Series(close.to_numpy()), n_paths=2)
    with pytest.raises(ValueError, match="at least window"):
        labeller_null(close.iloc[:21], n_paths=2)
    with pytest.raises(ValueError, match="positive"):
        labeller_null(close.where(close.index != close.index[5], -1.0), n_paths=2)


@pytest.mark.slow
def test_labeller_null_finds_no_excess_persistence_on_iid_prices_slow() -> None:
    close = _gaussian_close(3000, seed=31)
    out = labeller_null(close, n_paths=1000, seed=5)
    assert (np.abs(out["excess_persistence"]) < 0.1).all()


# --------------------------------------------------------------------------- #
# anderson_goodman_test
# --------------------------------------------------------------------------- #
def test_anderson_goodman_rejects_independence_on_persistent_chain() -> None:
    # A 0.9 diagonal against marginals near 1/3 on 5000 transitions gives a chi-square
    # statistic in the thousands on 4 df; p < 1e-6 is a conservative floor.
    labels = simulate_markov_chain(TRUE_MATRIX, 5000, seed=41)
    out = anderson_goodman_test(labels)
    assert out["p_value"] < 1e-6
    assert out["df"] == 4
    assert out["n_transitions"] == 4999


def test_anderson_goodman_does_not_reject_on_iid_labels() -> None:
    # A single seeded draw: under the null p is uniform, so p > 0.01 fails with 1%
    # probability for a fresh seed. The seed is fixed, so the risk was taken once.
    labels = _iid_labels(3000, seed=42)
    out = anderson_goodman_test(labels)
    assert out["p_value"] > 0.01
    assert out["statistic"] >= 0


def test_anderson_goodman_df_shrinks_with_occurring_states() -> None:
    # Only two states occur, so k = 2 and df = (2 - 1) ** 2 = 1, not 4.
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    out = anderson_goodman_test(labels)
    assert out["df"] == 1
    assert out["n_transitions"] == 9


def test_anderson_goodman_matches_hand_computation() -> None:
    labels = np.array([0, 0, 1, 2, 2, 1, 0, 1, 1, 2])
    counts = np.zeros((3, 3))
    for a, b in pairwise(labels):
        counts[a, b] += 1
    n = counts.sum()
    expected = counts.sum(axis=1, keepdims=True) * (counts.sum(axis=0) / n)
    mask = expected > 0
    stat = ((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum()
    out = anderson_goodman_test(labels)
    np.testing.assert_allclose(out["statistic"], stat)


def test_anderson_goodman_rejects_too_few_labels() -> None:
    with pytest.raises(ValueError, match="at least 2 labels"):
        anderson_goodman_test(np.array([1]))


# --------------------------------------------------------------------------- #
# homogeneity_test (Fix 4)
# --------------------------------------------------------------------------- #
def test_homogeneity_does_not_reject_on_stationary_chain() -> None:
    # One seeded draw from a stationary chain: p is uniform under the null, so p > 0.01
    # has a 1% false-rejection risk that was accepted when the seed was fixed.
    labels = simulate_markov_chain(TRUE_MATRIX, 6000, seed=51)
    out = homogeneity_test(labels, n_periods=4)
    assert out["p_value"] > 0.01
    assert out["stable"] is True
    assert out["n_periods"] == 4


def test_homogeneity_default_is_two_periods() -> None:
    # Ten years of daily data give ~125 stride labels; two halves keep enough transitions
    # per row for the chi-square approximation, so the default must be 2 (C1 fix).
    labels = simulate_markov_chain(TRUE_MATRIX, 200, seed=55)
    assert homogeneity_test(labels)["n_periods"] == 2


def test_homogeneity_does_not_reject_on_iid_prices_with_stride_sequence() -> None:
    # The valid input is the stride sequence labels[::window]: on iid prices it has no
    # regime drift, so a single seeded draw gives p > 0.01 with a 1% false-rejection
    # risk accepted when the seed was fixed. 2520 bars is ten years of daily data.
    close = _gaussian_close(2520, seed=56)
    labels = label_regimes(close, window=20, threshold=0.05).to_numpy()
    out = homogeneity_test(labels[::20], n_periods=2)
    assert out["p_value"] > 0.01
    assert out["stable"] is True


def test_homogeneity_over_rejects_on_daily_sequence_versus_stride() -> None:
    # The overlapping daily sequence is not a valid input: consecutive labels share 19
    # returns, so the independent-transitions chi-square over-rejects. The reviewer
    # measured ~30% false rejections at alpha 0.05 with four periods on the daily
    # sequence and ~1% on the stride sequence. Over 30 seeds that is ~9 daily rejections
    # against ~0.3 stride ones. Bounds: stride <= 3 (P(Bin(30, 0.02) > 3) < 0.004),
    # daily >= 3 (P(Bin(30, 0.3) < 3) < 0.003) and daily >= 3x stride. Seeds are fixed,
    # so the risk was taken once at authoring time.
    daily_rejections = 0
    stride_rejections = 0
    for seed in range(30):
        close = _gaussian_close(2520, seed=100 + seed)
        labels = label_regimes(close, window=20, threshold=0.05).to_numpy()
        daily_rejections += homogeneity_test(labels, n_periods=4)["p_value"] < 0.05
        stride_rejections += homogeneity_test(labels[::20], n_periods=4)["p_value"] < 0.05
    assert stride_rejections <= 3
    assert daily_rejections >= 3
    assert daily_rejections >= 3 * stride_rejections


def test_homogeneity_rejects_when_matrix_changes_halfway() -> None:
    # Diagonal 0.95 in the first half and 0.6 in the second: with 3000 transitions per
    # half the per-cell discrepancy of ~0.35 gives a statistic in the hundreds on 12 df.
    p1 = np.array([[0.95, 0.03, 0.02], [0.025, 0.95, 0.025], [0.02, 0.03, 0.95]])
    p2 = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    first = simulate_markov_chain(p1, 3000, seed=52)
    second = simulate_markov_chain(p2, 3000, seed=53, start_state=int(first[-1]))
    out = homogeneity_test(np.concatenate([first, second]), n_periods=4)
    assert out["p_value"] < 1e-6
    assert out["stable"] is False


def test_homogeneity_df_formula() -> None:
    # (S - 1) * k * (k - 1) with S = 4 periods and k = 3 occurring states gives 18.
    labels = simulate_markov_chain(TRUE_MATRIX, 1000, seed=54)
    assert homogeneity_test(labels, n_periods=4)["df"] == 18
    # Only two states occur: k = 2, df = (3 - 1) * 2 * 1 = 4.
    two_state = np.array([0, 1] * 30)
    assert homogeneity_test(two_state, n_periods=3)["df"] == 4


def test_homogeneity_statistic_is_zero_when_periods_are_identical() -> None:
    # Four copies of the same block give identical per-period counts, so every cell's
    # observed count equals its expectation and the statistic is exactly 0.
    block = np.array([0, 0, 1, 2, 2, 1, 0, 1, 1, 2, 0])
    # Make the last label of each block equal the first so the joins add no new cells.
    labels = np.tile(block, 4)
    out = homogeneity_test(labels, n_periods=4)
    assert out["statistic"] == 0.0
    assert out["p_value"] == 1.0


def test_homogeneity_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_periods"):
        homogeneity_test(np.arange(3) % 3, n_periods=1)
    with pytest.raises(ValueError, match="at least 8 labels"):
        homogeneity_test(np.array([0, 1, 2, 0, 1]), n_periods=4)
