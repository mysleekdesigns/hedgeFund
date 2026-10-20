"""Tests for the pure-numpy HMM filter, smoother, Viterbi and hysteresis rule.

Everything is synthetic: a known 3-state HMM with a sticky transition matrix and
2-D Gaussian emissions, simulated with an explicit seed. hmmlearn is used only as an
independent cross-check and is skipped when absent.
"""

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.stats import multivariate_normal, multivariate_t

from hedgefund.regime.hmm_filter import (
    DEFAULT_CONFIRM_BARS,
    DEFAULT_ENTER,
    DEFAULT_EXIT,
    HMMParams,
    HysteresisState,
    filter_step,
    forward_filter,
    hysteresis_decision,
    hysteresis_step,
    initial_filtered,
    initial_hysteresis_state,
    log_emission,
    smooth,
    viterbi,
)
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS

SEED = 20241020  # explicit, arbitrary; every simulation below derives from it
N_BARS = 400  # long enough for every regime to appear several times

# Sticky chain: 0.9 on the diagonal so regimes last ~10 bars, Bear and Bull never touch directly.
TRANSMAT = np.array(
    [
        [0.90, 0.08, 0.02],
        [0.05, 0.90, 0.05],
        [0.02, 0.08, 0.90],
    ]
)
STARTPROB = np.array([0.2, 0.6, 0.2])
# Means ordered by return (Bear < Sideways < Bull); the second feature is "volatility".
MEANS = np.array([[-1.5, 1.0], [0.0, 0.0], [1.5, -0.5]])
# Full covariances with off-diagonal terms so the Cholesky path is really exercised.
COVARS = np.array(
    [
        [[1.0, 0.4], [0.4, 0.8]],
        [[0.6, -0.2], [-0.2, 0.5]],
        [[0.9, 0.3], [0.3, 0.7]],
    ]
)


def make_params(emission: str = "gaussian", df: float | None = None) -> HMMParams:
    return HMMParams(
        startprob=STARTPROB,
        transmat=TRANSMAT,
        means=MEANS,
        covars=COVARS,
        emission=emission,
        df=df,
    )


def simulate(params: HMMParams, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate (states, observations) from a Gaussian HMM with an explicit seed."""
    rng = np.random.default_rng(seed)
    k = params.n_states
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(k, p=params.startprob)
    for t in range(1, n):
        states[t] = rng.choice(k, p=params.transmat[states[t - 1]])
    x = np.empty((n, params.n_features))
    for t in range(n):
        x[t] = rng.multivariate_normal(params.means[states[t]], params.covars[states[t]])
    return states, x


@pytest.fixture(scope="module")
def params() -> HMMParams:
    return make_params()


@pytest.fixture(scope="module")
def sample(params: HMMParams) -> np.ndarray:
    _, x = simulate(params, N_BARS, SEED)
    return x


# --------------------------------------------------------------------------- emissions


def test_gaussian_log_emission_matches_scipy(params: HMMParams, sample: np.ndarray) -> None:
    ours = log_emission(params, sample)
    for i in range(params.n_states):
        expected = multivariate_normal(params.means[i], params.covars[i]).logpdf(sample)
        np.testing.assert_allclose(ours[:, i], expected, atol=1e-9, rtol=0)


def test_log_emission_single_row_matches_matrix_row(params: HMMParams, sample: np.ndarray) -> None:
    full = log_emission(params, sample)
    single = log_emission(params, sample[7])
    assert single.shape == (params.n_states,)
    np.testing.assert_array_equal(single, full[7])


@pytest.mark.parametrize("df", [3.0, 5.0, 30.0])
def test_student_t_log_emission_matches_scipy(df: float, sample: np.ndarray) -> None:
    params = make_params("student_t", df)
    ours = log_emission(params, sample)
    for i in range(params.n_states):
        expected = multivariate_t(params.means[i], params.covars[i], df=df).logpdf(sample)
        np.testing.assert_allclose(ours[:, i], expected, atol=1e-9, rtol=0)


def test_student_t_converges_to_gaussian_for_large_df(sample: np.ndarray) -> None:
    gaussian = log_emission(make_params(), sample)
    # The gap log t - log N is ~ (m^4 / 4 - m^2 / 2) / df for Mahalanobis distance m. The
    # sample has rows with m^2 ~ 40 under the wrong state, so the gap at df = 1e6 is ~1e-3
    # on those rows; the rate is what proves convergence, and 1e-4 holds once df = 1e8.
    gaps = [
        np.max(np.abs(log_emission(make_params("student_t", df), sample) - gaussian))
        for df in (1e4, 1e5, 1e6)
    ]
    for coarse, fine in pairwise(gaps):
        assert 8.0 < coarse / fine < 12.0  # a tenfold df step shrinks the gap ~tenfold
    heavy = log_emission(make_params("student_t", 1e8), sample)
    np.testing.assert_allclose(heavy, gaussian, atol=1e-4, rtol=0)


@pytest.mark.parametrize("df", [2.0, 1.5, 0.0, -3.0, None, float("nan"), float("inf")])
def test_student_t_df_must_exceed_two(df: float | None) -> None:
    with pytest.raises(ValueError, match="df"):
        make_params("student_t", df)


def test_gaussian_rejects_df() -> None:
    with pytest.raises(ValueError, match="df must be None"):
        make_params("gaussian", 5.0)


def test_log_emission_rejects_wrong_feature_count(params: HMMParams) -> None:
    with pytest.raises(ValueError, match="shape"):
        log_emission(params, np.zeros((10, 3)))


# --------------------------------------------------------------------------- params validation


def test_params_rejects_transmat_rows_not_summing_to_one() -> None:
    bad = TRANSMAT.copy()
    bad[0, 0] += 0.1
    with pytest.raises(ValueError, match="row must sum to 1"):
        HMMParams(STARTPROB, bad, MEANS, COVARS)


def test_params_rejects_startprob_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="startprob must sum to 1"):
        HMMParams(np.array([0.5, 0.5, 0.5]), TRANSMAT, MEANS, COVARS)


def test_params_rejects_non_positive_definite_covariance() -> None:
    bad = COVARS.copy()
    bad[1] = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric but indefinite
    with pytest.raises(ValueError, match="positive definite"):
        HMMParams(STARTPROB, TRANSMAT, MEANS, bad)


def test_params_rejects_asymmetric_covariance() -> None:
    bad = COVARS.copy()
    bad[2, 0, 1] = 0.0
    with pytest.raises(ValueError, match="symmetric"):
        HMMParams(STARTPROB, TRANSMAT, MEANS, bad)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("startprob", np.ones((3, 1)) / 3, "startprob must have shape"),
        ("transmat", np.eye(2), "transmat must have shape"),
        ("means", np.zeros((2, 2)), "means must have shape"),
        ("covars", np.stack([np.eye(3)] * 3), "covars must have shape"),
    ],
)
def test_params_rejects_wrong_shapes(field: str, value: np.ndarray, message: str) -> None:
    kwargs = {"startprob": STARTPROB, "transmat": TRANSMAT, "means": MEANS, "covars": COVARS}
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        HMMParams(**kwargs)


def test_params_rejects_unknown_emission() -> None:
    with pytest.raises(ValueError, match="emission must be one of"):
        make_params("laplace")


def test_params_properties(params: HMMParams) -> None:
    assert params.n_states == 3
    assert params.n_features == 2


# --------------------------------------------------------------------------- forward filter


def test_filtered_rows_are_probabilities(params: HMMParams, sample: np.ndarray) -> None:
    filtered, log_likelihood = forward_filter(params, sample)
    assert filtered.shape == (N_BARS, 3)
    assert np.all(filtered >= 0.0) and np.all(filtered <= 1.0)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    assert np.isfinite(log_likelihood) and log_likelihood < 0.0


def test_forward_filter_equals_online_steps_bit_for_bit(
    params: HMMParams, sample: np.ndarray
) -> None:
    filtered, log_likelihood = forward_filter(params, sample)
    online = np.empty_like(filtered)
    online[0], total = initial_filtered(params, sample[0])
    increments = [total]
    for t in range(1, N_BARS):
        online[t], increment = filter_step(params, online[t - 1], sample[t])
        increments.append(increment)
    assert np.array_equal(filtered, online)
    # Summed in the same order as forward_filter sums them, so this is tighter than 1e-9.
    assert abs(sum(increments) - log_likelihood) < 1e-9


def test_filter_recovers_the_simulated_states_most_of_the_time(params: HMMParams) -> None:
    states, x = simulate(params, N_BARS, SEED)
    filtered, _ = forward_filter(params, x)
    accuracy = np.mean(filtered.argmax(axis=1) == states)
    # Means are 1.5 sigma apart on the return axis and the chain is sticky, so the
    # causal filter should get well over two thirds right; 0.7 leaves room for the
    # inevitable lag at every regime change.
    assert accuracy > 0.7


def test_filter_step_rejects_bad_previous_row(params: HMMParams, sample: np.ndarray) -> None:
    with pytest.raises(ValueError, match="prev_filtered must have shape"):
        filter_step(params, np.array([0.5, 0.5]), sample[0])
    with pytest.raises(ValueError, match="finite"):
        filter_step(params, np.array([np.nan, 0.5, 0.5]), sample[0])


def test_forward_filter_rejects_bad_sequences(params: HMMParams) -> None:
    with pytest.raises(ValueError, match="shape"):
        forward_filter(params, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="at least one row"):
        forward_filter(params, np.zeros((0, 2)))
    bad = np.zeros((5, 2))
    bad[3, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        forward_filter(params, bad)


# --------------------------------------------------------------------------- hmmlearn cross-check


def test_matches_hmmlearn_score_posteriors_viterbi_and_forward(
    params: HMMParams, sample: np.ndarray
) -> None:
    hmm = pytest.importorskip("hmmlearn.hmm")
    hmmc = pytest.importorskip("hmmlearn._hmmc")
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", init_params="", params="")
    model.startprob_ = params.startprob
    model.transmat_ = params.transmat
    model.means_ = params.means
    model.covars_ = params.covars

    filtered, log_likelihood = forward_filter(params, sample)
    assert abs(model.score(sample) - log_likelihood) < 1e-8
    np.testing.assert_allclose(model.predict_proba(sample), smooth(params, sample), atol=1e-8)
    np.testing.assert_array_equal(model.predict(sample), viterbi(params, sample))

    # hmmlearn 0.3.3 exposes the forward pass as _hmmc.forward_log(startprob, transmat,
    # log_frameprob) -> (logprob, fwdlattice); earlier versions had _do_forward_log_pass.
    log_frameprob = model._compute_log_likelihood(sample)
    forward_logprob, fwdlattice = hmmc.forward_log(model.startprob_, model.transmat_, log_frameprob)
    assert abs(forward_logprob - log_likelihood) < 1e-8
    lattice = np.exp(fwdlattice - fwdlattice.max(axis=1, keepdims=True))
    lattice /= lattice.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(lattice, filtered, atol=1e-8, rtol=0)


# --------------------------------------------------------------------------- smoothed vs filtered


def test_smoothed_and_filtered_paths_differ_on_the_same_data(
    params: HMMParams, sample: np.ndarray
) -> None:
    filtered, _ = forward_filter(params, sample)
    smoothed = smooth(params, sample)
    assert smoothed.shape == filtered.shape
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    # Hindsight changes the picture somewhere in the middle of the sample...
    assert np.max(np.abs(smoothed - filtered)) > 0.01
    # ...but on the last bar there is no future to use, so the two coincide.
    np.testing.assert_allclose(smoothed[-1], filtered[-1], atol=1e-8, rtol=0)


def test_smooth_and_viterbi_docstrings_warn_about_the_future() -> None:
    for fn in (smooth, viterbi):
        first_line = fn.__doc__.strip().splitlines()[0]
        assert "DIAGNOSTIC ONLY" in first_line
        assert "future observations" in first_line


def test_viterbi_returns_valid_state_path(params: HMMParams, sample: np.ndarray) -> None:
    path = viterbi(params, sample)
    assert path.shape == (N_BARS,)
    assert path.dtype.kind == "i"
    assert set(np.unique(path)) <= {BEAR, SIDEWAYS, BULL}


def test_filter_has_no_lookahead_but_smoother_does(params: HMMParams, sample: np.ndarray) -> None:
    t = 150
    filtered, _ = forward_filter(params, sample)
    smoothed = smooth(params, sample)

    rng = np.random.default_rng(SEED + 1)
    perturbed = sample.copy()
    perturbed[t + 1 :] = rng.normal(size=perturbed[t + 1 :].shape) * 3.0

    filtered_p, _ = forward_filter(params, perturbed)
    smoothed_p = smooth(params, perturbed)

    assert np.array_equal(filtered[: t + 1], filtered_p[: t + 1])
    assert not np.array_equal(filtered[t + 1 :], filtered_p[t + 1 :])
    # The smoother reads the future, so rows at or before t move when the future moves.
    assert np.max(np.abs(smoothed[: t + 1] - smoothed_p[: t + 1])) > 1e-6


# --------------------------------------------------------------------------- hysteresis


def rows(*bear_probs: float, state: int = BEAR) -> np.ndarray:
    """Probability rows that put ``p`` on ``state`` and the remainder on the other two."""
    out = np.empty((len(bear_probs), 3))
    for i, p in enumerate(bear_probs):
        out[i] = (1.0 - p) / 2.0
        out[i, state] = p
    return out


def test_hysteresis_defaults_match_plan_fix_5() -> None:
    assert DEFAULT_ENTER == 0.6
    assert DEFAULT_EXIT == 0.4
    assert DEFAULT_CONFIRM_BARS in (2, 3)


def test_hysteresis_oscillation_inside_the_band_never_leaves_initial_regime() -> None:
    probs = rows(*([0.45, 0.55] * 20))
    assert np.all(hysteresis_decision(probs) == SIDEWAYS)
    probs_bull = rows(*([0.45, 0.55] * 20), state=BULL)
    assert np.all(hysteresis_decision(probs_bull) == SIDEWAYS)


def test_hysteresis_enters_on_the_confirm_bars_th_bar_not_earlier() -> None:
    confirm = 3
    lead = 4
    probs = np.vstack([rows(*([0.5] * lead)), rows(*([0.9] * confirm)), rows(0.5, 0.5)])
    out = hysteresis_decision(probs, confirm_bars=confirm)
    assert np.all(out[: lead + confirm - 1] == SIDEWAYS)
    assert out[lead + confirm - 1] == BEAR
    # 0.5 sits between exit and enter, so Bear is held afterwards.
    assert np.all(out[lead + confirm :] == BEAR)


def test_hysteresis_one_bar_short_of_confirmation_does_not_enter() -> None:
    confirm = 3
    probs = np.vstack([rows(0.9, 0.9), rows(0.5), rows(0.9, 0.9), rows(0.5)])
    assert np.all(hysteresis_decision(probs, confirm_bars=confirm) == SIDEWAYS)


def test_hysteresis_falls_back_to_sideways_after_confirm_bars_below_exit() -> None:
    confirm = 2
    probs = np.vstack([rows(0.9, 0.9), rows(0.5, 0.45), rows(0.3, 0.3), rows(0.5)])
    out = hysteresis_decision(probs, confirm_bars=confirm)
    expected = [SIDEWAYS, BEAR, BEAR, BEAR, BEAR, SIDEWAYS, SIDEWAYS]
    assert out.tolist() == expected


def test_hysteresis_single_bar_below_exit_does_not_release() -> None:
    probs = np.vstack([rows(0.9, 0.9), rows(0.3), rows(0.5), rows(0.3), rows(0.5)])
    out = hysteresis_decision(probs, confirm_bars=2)
    assert out.tolist() == [SIDEWAYS, BEAR, BEAR, BEAR, BEAR, BEAR]


def test_hysteresis_switches_directly_from_bear_to_bull() -> None:
    probs = np.vstack([rows(0.9, 0.9), rows(0.9, 0.9, state=BULL), rows(0.5, state=BULL)])
    out = hysteresis_decision(probs, confirm_bars=2)
    assert out.tolist() == [SIDEWAYS, BEAR, BEAR, BULL, BULL]


def test_hysteresis_tie_picks_highest_probability_candidate() -> None:
    # Both Bear and Bull above enter for confirm_bars bars with K = 4 and a tiny Sideways;
    # Bull is higher, so Bull wins.
    probs = np.array([[0.62, 0.0, 0.0, 0.38], [0.62, 0.0, 0.0, 0.38]])
    out = hysteresis_decision(
        probs[:, [0, 1, 3, 2]], enter=0.3, exit=0.1, confirm_bars=2, initial_regime=1
    )
    assert out[-1] == 0  # Bear (0.62) beats Bull (0.38) when both are confirmed
    probs_bull_higher = probs[:, [3, 1, 0, 2]]
    out2 = hysteresis_decision(
        probs_bull_higher, enter=0.3, exit=0.1, confirm_bars=2, initial_regime=1
    )
    assert out2[-1] == 2


def test_hysteresis_decision_equals_iterated_steps_bit_for_bit(
    params: HMMParams, sample: np.ndarray
) -> None:
    filtered, _ = forward_filter(params, sample)
    batch = hysteresis_decision(filtered)
    state = initial_hysteresis_state(params.n_states)
    online = np.empty(N_BARS, dtype=int)
    for t in range(N_BARS):
        state = hysteresis_step(state, filtered[t])
        online[t] = state.regime
    assert np.array_equal(batch, online)
    assert len(set(batch.tolist())) == 3  # the simulated path visits every regime


def test_hysteresis_prefix_is_stable_under_future_perturbation(
    params: HMMParams, sample: np.ndarray
) -> None:
    filtered, _ = forward_filter(params, sample)
    t = 123
    base = hysteresis_decision(filtered)
    rng = np.random.default_rng(SEED + 2)
    perturbed = filtered.copy()
    future = rng.dirichlet(np.ones(3), size=N_BARS - t - 1)
    perturbed[t + 1 :] = future
    out = hysteresis_decision(perturbed)
    assert np.array_equal(base[: t + 1], out[: t + 1])


def test_hysteresis_state_is_frozen_and_counters_are_fresh_arrays() -> None:
    state = initial_hysteresis_state(3)
    assert isinstance(state, HysteresisState)
    with pytest.raises(AttributeError):
        state.regime = 2  # type: ignore[misc]
    nxt = hysteresis_step(state, np.array([0.9, 0.05, 0.05]))
    assert nxt.enter_runs[BEAR] == 1 and state.enter_runs[BEAR] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enter": 0.4, "exit": 0.6},  # exit above enter
        {"enter": 0.6, "exit": 0.6},  # equal
        {"enter": 1.0, "exit": 0.4},  # enter not < 1
        {"enter": 0.6, "exit": 0.0},  # exit not > 0
        {"confirm_bars": 0},
        {"initial_regime": 3},
        {"initial_regime": -1},
    ],
)
def test_hysteresis_rejects_bad_parameters(kwargs: dict) -> None:
    probs = rows(0.5, 0.5)
    with pytest.raises(ValueError):
        hysteresis_decision(probs, **kwargs)
    with pytest.raises(ValueError):
        hysteresis_step(initial_hysteresis_state(3), probs[0], **kwargs)


def test_hysteresis_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="finite"):
        hysteresis_decision(np.array([[0.5, np.nan, 0.5]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        hysteresis_decision(np.array([[1.2, -0.2, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        hysteresis_decision(np.array([0.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match="K >= 2"):
        hysteresis_decision(np.ones((4, 1)))
    with pytest.raises(ValueError, match="probs_t must have shape"):
        hysteresis_step(initial_hysteresis_state(3), np.array([0.5, 0.5]))


def test_initial_hysteresis_state_validates() -> None:
    with pytest.raises(ValueError, match="n_states"):
        initial_hysteresis_state(1)
    with pytest.raises(ValueError, match="initial_regime"):
        initial_hysteresis_state(3, 5)


simplex_rows = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(1, 60), st.just(3)),
    elements=st.floats(1e-3, 1.0),
).map(lambda a: a / a.sum(axis=1, keepdims=True))


@settings(max_examples=100, deadline=None)
@given(
    probs=simplex_rows,
    enter=st.floats(0.35, 0.95),
    exit_gap=st.floats(0.05, 0.3),
    confirm_bars=st.integers(1, 4),
    initial_regime=st.integers(0, 2),
)
def test_hysteresis_outputs_are_always_valid_state_indices(
    probs: np.ndarray, enter: float, exit_gap: float, confirm_bars: int, initial_regime: int
) -> None:
    exit = max(enter - exit_gap, 1e-3)
    out = hysteresis_decision(
        probs, enter=enter, exit=exit, confirm_bars=confirm_bars, initial_regime=initial_regime
    )
    assert out.shape == (probs.shape[0],)
    assert out.dtype.kind == "i"
    assert np.all((out >= 0) & (out < 3))


@settings(max_examples=100, deadline=None)
@given(probs=simplex_rows)
def test_hysteresis_with_one_bar_confirmation_follows_the_argmax(probs: np.ndarray) -> None:
    eps = 1e-6  # enter = 0.5 + eps: a row with a probability above it has a unique argmax
    enter = 0.5 + eps
    out = hysteresis_decision(probs, enter=enter, exit=0.25, confirm_bars=1)
    decisive = probs.max(axis=1) > enter
    np.testing.assert_array_equal(out[decisive], probs.argmax(axis=1)[decisive])
