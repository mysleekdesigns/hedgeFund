"""Tests for the optional HMM port in ``hedgefund.regime.hmm``.

All tests are offline. The available-path tests use a synthetic 3-state mixture with
clearly separated means and skip when ``hmmlearn`` is not installed.

The Phase 4 tests (``fit_hmm_features`` and friends) use a two-feature standardised matrix:
returns from the same sticky mixture plus a second feature correlated with their magnitude,
so the full covariance actually carries information.
"""

import math
import sys
import types
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from hedgefund.regime import hmm as hmm_module
from hedgefund.regime.hmm import (
    DEFAULT_T_DFS,
    EMISSION_CHOICES,
    FULL_COVARIANCE,
    HMM_CAVEAT,
    REASON_DISABLED,
    REASON_UNAVAILABLE,
    HMMFit,
    _floor_singular_covars,
    crash_state_diagnostic,
    evaluate_student_t,
    fit_hmm,
    fit_hmm_features,
    fit_summary,
    hmm_summary,
    order_states_by_mean,
    student_t_scale,
)
from hedgefund.regime.hmm_filter import HMMParams, forward_filter

SEED = 7  # arbitrary but fixed so the synthetic mixture is reproducible
N_ROWS = 1500  # ~6 trading years; enough for Baum-Welch to separate three states
# hmmlearn's GaussianHMM adds a covariance floor of min_covar=1e-3 (a variance, so sigma
# ~0.032) that the companion leaves at its default. Real daily returns sit below that floor,
# so the synthetic mixture is scaled up: means 0.1 apart with sigma 0.02 keep the states
# ~2.7 effective sigma apart and the Bear / Sideways / Bull recovery unambiguous.
STATE_MEANS = np.array([-0.1, 0.0, 0.1])
STATE_SIGMA = 0.02
RECOVERING_RANDOM_STATE = 3  # a seed whose hmmlearn Dirichlet init lets EM find the truth


def _synthetic_close(seed: int = SEED, n_rows: int = N_ROWS) -> pd.Series:
    """Close series whose returns come from a persistent 3-state Gaussian mixture."""
    rng = np.random.default_rng(seed)
    # Sticky chain: stay with 0.9, otherwise move to one of the other two states.
    transition = np.full((3, 3), 0.05)
    np.fill_diagonal(transition, 0.9)
    states = np.empty(n_rows, dtype=int)
    states[0] = 1
    for t in range(1, n_rows):
        states[t] = rng.choice(3, p=transition[states[t - 1]])
    returns = STATE_MEANS[states] + STATE_SIGMA * rng.standard_normal(n_rows)
    index = pd.bdate_range("2015-01-01", periods=n_rows + 1)
    close = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    return pd.Series(close, index=index, name="Close")


@pytest.fixture
def quiet_hmmlearn() -> Iterator[None]:
    """Silence warnings that hmmlearn itself emits; hedgefund warnings stay fatal."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="hmmlearn.*")
        yield


# --------------------------------------------------------------------------- #
# Degrade paths: these must work with or without hmmlearn installed.
# --------------------------------------------------------------------------- #


def test_fit_hmm_returns_none_pair_when_hmmlearn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setting sys.modules[name] = None makes ``import name`` raise ImportError.
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    returns = _synthetic_close().pct_change().dropna()
    assert fit_hmm(returns) == (None, None)


def test_hmm_summary_degrades_when_hmmlearn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    result = hmm_summary(_synthetic_close())
    assert result == {"available": False, "reason": REASON_UNAVAILABLE}


class _BrokenExtensionModule(types.ModuleType):
    """Stand-in for hmmlearn whose submodule access fails like a bad compiled extension."""

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(f"simulated compiled-extension failure loading {name}")


def test_fit_hmm_degrades_on_compiled_extension_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _BrokenExtensionModule("hmmlearn")
    monkeypatch.setitem(sys.modules, "hmmlearn", broken)
    # ``from hmmlearn import hmm`` first tries getattr(broken, "hmm"), which raises
    # RuntimeError, not ImportError. Also make sure no stale submodule short-circuits it.
    monkeypatch.delitem(sys.modules, "hmmlearn.hmm", raising=False)
    returns = _synthetic_close().pct_change().dropna()
    assert fit_hmm(returns) == (None, None)
    assert hmm_summary(_synthetic_close()) == {"available": False, "reason": REASON_UNAVAILABLE}


def test_hmm_summary_disabled_reason() -> None:
    result = hmm_summary(_synthetic_close(), enabled=False)
    assert result == {"available": False, "reason": REASON_DISABLED}


def test_hmm_summary_reports_runtime_error_on_short_series(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two closes give one return, fewer than three components, so fit_hmm raises
    # ValueError. This path must not depend on hmmlearn, so stub it as importable.
    monkeypatch.setitem(sys.modules, "hmmlearn", types.ModuleType("hmmlearn"))
    monkeypatch.setitem(sys.modules, "hmmlearn.hmm", types.ModuleType("hmmlearn.hmm"))
    sys.modules["hmmlearn"].hmm = sys.modules["hmmlearn.hmm"]  # type: ignore[attr-defined]
    short = pd.Series([100.0, 101.0], index=pd.bdate_range("2020-01-01", periods=2))
    result = hmm_summary(short)
    assert result["available"] is False
    assert result["reason"].startswith("hmm runtime error:")
    assert "n_components=3" in result["reason"]


def test_fit_hmm_raises_value_error_on_too_few_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", types.ModuleType("hmmlearn"))
    monkeypatch.setitem(sys.modules, "hmmlearn.hmm", types.ModuleType("hmmlearn.hmm"))
    sys.modules["hmmlearn"].hmm = sys.modules["hmmlearn.hmm"]  # type: ignore[attr-defined]
    returns = pd.Series([0.01, np.nan, -0.01])  # only two non-NaN rows
    with pytest.raises(ValueError, match="2 non-NaN rows"):
        fit_hmm(returns, n_components=3)


def test_order_states_by_mean_is_argsort_of_first_column() -> None:
    fake = types.SimpleNamespace(means_=np.array([[0.5, 9.0], [-1.0, 9.0], [0.1, 9.0]]))
    np.testing.assert_array_equal(order_states_by_mean(fake), np.array([1, 2, 0]))


def test_module_constants_match_companion() -> None:
    assert hmm_module.RANK_LABELS == ["Bear", "Sideways", "Bull"]
    assert hmm_module.DEFAULT_N_COMPONENTS == 3
    assert hmm_module.DEFAULT_RANDOM_STATE == 42
    assert hmm_module.DEFAULT_N_ITER == 200
    assert hmm_module.COVARIANCE_TYPE == "diag"
    assert HMM_CAVEAT.startswith("HMM states are labelled by ascending mean return")


# --------------------------------------------------------------------------- #
# Available path: requires hmmlearn.
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_hmm_summary_available_path_recovers_three_ordered_regimes() -> None:
    pytest.importorskip("hmmlearn")
    result = hmm_summary(_synthetic_close())
    assert result["available"] is True
    assert result["caveat"] == HMM_CAVEAT
    regimes = result["regimes"]
    assert [r["label"] for r in regimes] == ["Bear", "Sideways", "Bull"]
    means = [r["mean_daily_return"] for r in regimes]
    assert means == sorted(means)
    assert len(set(means)) == 3, "mean_daily_return must be strictly increasing"
    latent = [r["latent_state"] for r in regimes]
    assert sorted(latent) == [0, 1, 2]
    # hmmlearn 0.3.x draws the initial startprob/transmat from a spiky Dirichlet(1/3), and
    # with the companion's random_state=42 that locks two latent states into alternation
    # so Bear and Sideways merge near their midpoint. Phase 1 ports that behaviour as is
    # (Phase 4 adds multi-restart), so here we only require that the fit separates real
    # structure: the Bear-to-Bull spread must exceed half the true spread.
    true_spread = STATE_MEANS[-1] - STATE_MEANS[0]
    assert means[-1] - means[0] > 0.5 * true_spread


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_hmm_summary_recovers_true_means_for_a_well_initialised_seed() -> None:
    pytest.importorskip("hmmlearn")
    # random_state=3 is one of the seeds whose Dirichlet init does not trap Baum-Welch
    # (13 of seeds 0..29 recover the mixture on this data). Known-answer check: each fitted
    # mean lands within half a true sigma of its generating mean.
    result = hmm_summary(_synthetic_close(), random_state=RECOVERING_RANDOM_STATE)
    means = [r["mean_daily_return"] for r in result["regimes"]]
    np.testing.assert_allclose(means, STATE_MEANS, atol=0.5 * STATE_SIGMA)


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_fit_hmm_returns_model_and_states_of_matching_length() -> None:
    pytest.importorskip("hmmlearn")
    returns = _synthetic_close().pct_change().dropna()
    model, hidden_states = fit_hmm(returns)
    assert model is not None
    assert hidden_states is not None
    assert hidden_states.shape == (len(returns),)
    assert set(np.unique(hidden_states)) <= {0, 1, 2}


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_hmm_summary_is_deterministic_for_same_random_state() -> None:
    pytest.importorskip("hmmlearn")
    close = _synthetic_close()
    first = hmm_summary(close, random_state=42)
    second = hmm_summary(close, random_state=42)

    # Labels, latent-state assignment and the caveat must match exactly. The fitted means
    # agree only to floating-point reduction order: with a multithreaded BLAS the same seed
    # gives last-bit differences (~1e-16) between runs, single-threaded it is bit-identical.
    # Bit-identical repeated fits are a Phase 4 item (it pins the thread pool); here we
    # assert the fit is deterministic up to that reduction order.
    def without_mean(regime: dict) -> dict:
        return {k: v for k, v in regime.items() if k != "mean_daily_return"}

    assert [without_mean(r) for r in first["regimes"]] == [
        without_mean(r) for r in second["regimes"]
    ]
    assert first["caveat"] == second["caveat"] and first["available"] and second["available"]
    np.testing.assert_allclose(
        [r["mean_daily_return"] for r in first["regimes"]],
        [r["mean_daily_return"] for r in second["regimes"]],
        rtol=1e-9,
        atol=0,
    )


# --------------------------------------------------------------------------- #
# Phase 4: multi-restart full-covariance fit on a feature matrix.
# --------------------------------------------------------------------------- #

FEATURE_NAMES = ("log_return", "vol_10")
MANY_SEEDS = tuple(range(6))  # includes seed 1, which Baum-Welch traps on this data
HEAVY_TAIL_DF = 3.0  # t noise with 3 degrees of freedom: about as fat as daily equity returns
# Covariance-matched Student-t gains measured on the synthetic matrices (nats per bar, seed 7,
# 1500 rows): Gaussian data +0.0008 at df=30 (never above +0.005 over five seeds), jointly
# t(3) data +0.136 at df=5. The thresholds below sit well inside those two figures.
GAUSSIAN_T_GAIN_PER_BAR = 0.005  # at df=30 the t is within O(1/df) of the Gaussian: noise level
HEAVY_T_GAIN_PER_BAR = 0.05  # a real heavy tail buys far more than that
SPIKE_SHARE = 0.01  # 1% of bars replaced by a one-bar crash: below the 2% occupancy threshold
SPIKE_RETURN = -1.0  # ten true sigma below Bear: a Gaussian fit has to give it its own state
# The known-answer tolerance for the recovered standardised means: with three unit-variance
# clusters ~1.2 apart and 1500 bars, EM lands within 0.1 of the truth on every seed tried.
MEAN_TOL = 0.1


def _synthetic_feature_matrix(
    seed: int = SEED,
    n_rows: int = N_ROWS,
    *,
    heavy_tails: bool = False,
    spikes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised ``(n_rows, 2)`` matrix: sticky-mixture returns plus a correlated feature.

    Column 0 is the return; column 1 is its absolute value plus noise, a stand-in for
    realised volatility that is correlated with the return within each state. Standardising
    with full-sample statistics is fine here because this exercises the fitter, not the
    no-lookahead pipeline. Returns the matrix and the generating state path.
    """
    rng = np.random.default_rng(seed)
    transition = np.full((3, 3), 0.05)
    np.fill_diagonal(transition, 0.9)
    states = np.empty(n_rows, dtype=int)
    states[0] = 1
    for t in range(1, n_rows):
        states[t] = rng.choice(3, p=transition[states[t - 1]])
    noise = rng.standard_normal(n_rows)
    second_noise = rng.standard_normal(n_rows)
    if heavy_tails:
        # One shared chi-square divisor makes the two noises jointly multivariate t, which
        # is the family the covariance-matched evaluation is looking for.
        radial = np.sqrt(rng.chisquare(HEAVY_TAIL_DF, n_rows) / HEAVY_TAIL_DF)
        noise = noise / radial
        second_noise = second_noise / radial
    if spikes:
        # Only two persistent states (Bear and Sideways) plus rare one-bar crashes.
        states = np.where(states == 2, 1, states)
    returns = STATE_MEANS[states] + STATE_SIGMA * noise
    if spikes:
        spike_rows = rng.choice(n_rows, size=int(SPIKE_SHARE * n_rows), replace=False)
        returns[spike_rows] = SPIKE_RETURN
    second = np.abs(returns) + 0.5 * STATE_SIGMA * second_noise
    x = np.column_stack([returns, second])
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, states


def _gaussian_params(fit: HMMFit) -> HMMParams:
    """The Gaussian emission the fit was switched from: undo the (df - 2) / df scale."""
    covars = fit.params.covars
    if fit.params.emission == "student_t":
        assert fit.params.df is not None
        covars = covars * fit.params.df / (fit.params.df - 2.0)
    return HMMParams(
        startprob=fit.params.startprob,
        transmat=fit.params.transmat,
        means=fit.params.means,
        covars=covars,
        emission="gaussian",
        df=None,
    )


def _argmax_path(params: HMMParams, x: np.ndarray) -> np.ndarray:
    filtered, _ = forward_filter(params, x)
    return np.argmax(filtered, axis=1)


# ---- validation and degrade: no hmmlearn needed ---------------------------------------


def test_fit_hmm_features_rejects_volatility_alone() -> None:
    x = np.abs(np.random.default_rng(SEED).standard_normal((50, 1)))
    with pytest.raises(ValueError, match="rolling volatility alone is not allowed"):
        fit_hmm_features(x, feature_names=("vol_10",))


def test_fit_hmm_features_requires_return_in_column_zero() -> None:
    x, _ = _synthetic_feature_matrix(n_rows=50)
    with pytest.raises(ValueError, match="feature_names\\[0\\] must be 'log_return'"):
        fit_hmm_features(x, feature_names=("vol_10", "log_return"))


@pytest.mark.parametrize(
    ("bad", "names", "kwargs", "match"),
    [
        (np.zeros(10), FEATURE_NAMES, {}, "2-D float ndarray"),
        (np.zeros((10, 2), dtype=int), FEATURE_NAMES, {}, "2-D float ndarray"),
        (np.full((10, 2), np.nan), FEATURE_NAMES, {}, "NaN"),
        (np.zeros((2, 2)), FEATURE_NAMES, {}, "n_components=3"),
        (np.zeros((10, 2)), ("log_return",), {}, "feature_names has 1 names"),
        (np.zeros((10, 2)), FEATURE_NAMES, {"seeds": ()}, "seeds is empty"),
        (np.zeros((10, 2)), FEATURE_NAMES, {"emission": "cauchy"}, "emission must be one of"),
    ],
)
def test_fit_hmm_features_names_each_bad_input(
    bad: np.ndarray, names: tuple[str, ...], kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_hmm_features(bad, feature_names=names, **kwargs)


def test_fit_hmm_features_returns_none_when_hmmlearn_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    x, _ = _synthetic_feature_matrix(n_rows=50)
    assert fit_hmm_features(x, feature_names=FEATURE_NAMES) is None


def test_phase4_constants() -> None:
    assert hmm_module.DEFAULT_SEEDS == (0, 1, 2)
    assert FULL_COVARIANCE == "full"
    assert DEFAULT_T_DFS == (3.0, 5.0, 10.0, 30.0)
    assert hmm_module.DEFAULT_MAX_OCCUPANCY == 0.02
    assert hmm_module.DEFAULT_MAX_MEAN_RUN == 1.5
    assert EMISSION_CHOICES == ("gaussian", "student_t", "auto")


# ---- crash diagnostic and Student-t evaluation are pure: handcrafted params -----------


def _toy_params(emission: str = "gaussian", df: float | None = None) -> HMMParams:
    """Three well-separated 2-D states with tiny spherical covariance."""
    means = np.array([[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    covars = np.array([0.01 * np.eye(2)] * 3)
    transmat = np.full((3, 3), 0.05)
    np.fill_diagonal(transmat, 0.9)
    return HMMParams(
        startprob=np.full(3, 1 / 3),
        transmat=transmat,
        means=means,
        covars=covars,
        emission=emission,
        df=df,
    )


def test_crash_state_diagnostic_flags_one_bar_state() -> None:
    params = _toy_params()
    # 200 bars: states 1 and 2 in long episodes, plus two isolated bars of state 0
    # (occupancy 1%, mean run 1): the one-bar "crash" state the diagnostic must catch.
    path = np.ones(200, dtype=int)
    path[100:140] = 2
    path[[50, 150]] = 0
    x = params.means[path]
    result = crash_state_diagnostic(params, x)
    assert result["flagged"] is True
    assert result["state"] == 0
    assert result["occupancy"] == pytest.approx([0.01, 0.79, 0.20])
    # state 1 runs: bars 0-49, 51-99, 140-149, 151-199 -> lengths 50, 49, 10, 49.
    assert result["mean_run_length"] == pytest.approx([1.0, (50 + 49 + 10 + 49) / 4, 40.0])


def test_crash_state_diagnostic_flags_a_state_that_never_appears() -> None:
    params = _toy_params()
    path = np.repeat([0, 1, 0, 1], 25)  # state 2 is never the argmax
    x = params.means[path]
    result = crash_state_diagnostic(params, x)
    assert result["flagged"] is True
    assert result["state"] == 2
    assert result["occupancy"][2] == 0.0
    assert result["mean_run_length"][2] is None


def test_crash_state_diagnostic_passes_persistent_states() -> None:
    params = _toy_params()
    path = np.repeat([0, 1, 2, 1, 0, 2], 20)  # six episodes of 20 bars each
    x = params.means[path]
    result = crash_state_diagnostic(params, x)
    assert result["flagged"] is False
    assert result["state"] is None
    assert result["occupancy"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result["mean_run_length"] == pytest.approx([20.0, 20.0, 20.0])
    assert result["max_occupancy"] == hmm_module.DEFAULT_MAX_OCCUPANCY
    assert result["max_mean_run"] == hmm_module.DEFAULT_MAX_MEAN_RUN


def test_evaluate_student_t_reports_grid_and_best() -> None:
    params = _toy_params()
    x = params.means[np.repeat([0, 1, 2], 30)]
    result = evaluate_student_t(params, x, dfs=(3.0, 30.0))
    assert set(result["by_df"]) == {"3.0", "30.0"}
    assert result["student_t_log_likelihood"] == max(result["by_df"].values())
    assert result["by_df"][str(result["best_df"])] == result["student_t_log_likelihood"]
    assert result["prefer_student_t"] == (
        result["student_t_log_likelihood"] > result["gaussian_log_likelihood"]
    )
    assert result["note"].startswith("same transition matrix")
    assert result["scale"] == "covariance-matched: scale = (df-2)/df * sigma"
    # Only the emission family changes: the Gaussian figure is the forward-filter likelihood
    # of the input params themselves, and each t figure uses the covariance-matched scale.
    assert result["gaussian_log_likelihood"] == pytest.approx(forward_filter(params, x)[1])
    matched = HMMParams(
        startprob=params.startprob,
        transmat=params.transmat,
        means=params.means,
        covars=student_t_scale(params.covars, 3.0),
        emission="student_t",
        df=3.0,
    )
    assert result["by_df"]["3.0"] == pytest.approx(forward_filter(matched, x)[1])


def test_student_t_scale_matches_gaussian_covariance() -> None:
    covars = _toy_params().covars
    for df in DEFAULT_T_DFS:
        scale = student_t_scale(covars, df)
        # A multivariate t with scale S has covariance df / (df - 2) * S.
        np.testing.assert_allclose(scale * df / (df - 2.0), covars, rtol=1e-12)
    with pytest.raises(ValueError, match=r"must exceed 2\.0"):
        student_t_scale(covars, 2.0)


def test_evaluate_student_t_rejects_already_switched_params() -> None:
    switched = _toy_params(emission="student_t", df=5.0)
    x = switched.means[np.repeat([0, 1, 2], 10)]
    with pytest.raises(ValueError, match="expects the Gaussian fit"):
        evaluate_student_t(switched, x)


# ---- fitting: requires hmmlearn --------------------------------------------------------


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_fit_hmm_features_recovers_ordered_means_with_full_covariance() -> None:
    pytest.importorskip("hmmlearn")
    x, states = _synthetic_feature_matrix()
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert fit is not None
    assert fit.emission == "gaussian" and fit.params.emission == "gaussian"
    assert fit.student_t is None
    assert fit.floored_states == ()
    assert fit.feature_names == FEATURE_NAMES
    assert fit.params.covars.shape == (3, 2, 2)
    # Known answer: the standardised state means of the generating mixture.
    returns_std = x[:, 0]
    truth = np.array([returns_std[states == k].mean() for k in range(3)])
    np.testing.assert_allclose(fit.params.means[:, 0], truth, atol=MEAN_TOL)
    assert np.mean(_argmax_path(fit.params, x) == states) > 0.95


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_multi_restart_keeps_best_log_likelihood() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix()
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert fit is not None
    assert fit.seeds == MANY_SEEDS
    assert set(fit.restart_log_likelihoods) == set(MANY_SEEDS)
    finite = {s: ll for s, ll in fit.restart_log_likelihoods.items() if math.isfinite(ll)}
    assert fit.log_likelihood == max(finite.values())
    assert fit.seed == max(finite, key=finite.__getitem__)
    # Seed 1 is trapped in a local optimum on this data; the restart logic must step over it.
    assert fit.restart_log_likelihoods[1] < fit.log_likelihood - 1.0


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_single_seed_fit_matches_that_restart() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix()
    multi = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert multi is not None
    single = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=[multi.seed])
    assert single is not None
    assert single.seed == multi.seed and single.seeds == (multi.seed,)
    assert single.log_likelihood == multi.log_likelihood
    assert single.restart_log_likelihoods == {multi.seed: multi.log_likelihood}
    for field in ("startprob", "transmat", "means", "covars"):
        np.testing.assert_allclose(
            getattr(single.params, field), getattr(multi.params, field), rtol=1e-9, atol=0
        )


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_states_are_ordered_by_mean_return_and_permuted_consistently() -> None:
    hmm = pytest.importorskip("hmmlearn.hmm")
    x, _ = _synthetic_feature_matrix()
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert fit is not None
    means0 = fit.params.means[:, 0]
    assert np.all(np.diff(means0) > 0), "means of column 0 must be strictly ascending"
    np.testing.assert_allclose(fit.params.transmat.sum(axis=1), 1.0, atol=1e-12)
    assert fit.params.startprob.sum() == pytest.approx(1.0)
    for cov in fit.params.covars:
        np.testing.assert_allclose(cov, cov.T, atol=0)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    # Relabelling must not change the likelihood: the forward filter on the permuted
    # parameters reproduces hmmlearn's score for the winning restart.
    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=hmm_module.DEFAULT_N_ITER,
        random_state=fit.seed,
    )
    model.fit(x)
    assert fit.log_likelihood == pytest.approx(model.score(x), rel=1e-9)
    _, filtered_ll = forward_filter(fit.params, x)
    assert filtered_ll == pytest.approx(model.score(x), rel=1e-9)
    # ... and the permutation is the argsort of the unpermuted means.
    order = order_states_by_mean(model)
    np.testing.assert_allclose(fit.params.means, model.means_[order], rtol=1e-9)
    np.testing.assert_allclose(
        fit.params.transmat, model.transmat_[np.ix_(order, order)], rtol=1e-9
    )


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_fixed_seed_repeated_fits_give_identical_state_sequences() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix()
    first = fit_hmm_features(x, feature_names=FEATURE_NAMES)
    second = fit_hmm_features(x, feature_names=FEATURE_NAMES)
    assert first is not None and second is not None
    assert first.seed == second.seed
    assert np.array_equal(_argmax_path(first.params, x), _argmax_path(second.params, x))
    for field in ("startprob", "transmat", "means", "covars"):
        np.testing.assert_allclose(
            getattr(first.params, field), getattr(second.params, field), rtol=1e-9, atol=0
        )
    assert first.restart_log_likelihoods == second.restart_log_likelihoods


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_gaussian_fit_on_spike_data_is_flagged_as_crash_state() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix(spikes=True)
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert fit is not None
    crash = fit.crash_state
    assert crash["flagged"] is True
    # The spikes are the most negative returns, so they land in the lowest-mean state 0.
    assert crash["state"] == 0
    assert crash["occupancy"][0] == pytest.approx(SPIKE_SHARE, abs=0.005)
    assert crash["mean_run_length"][0] < hmm_module.DEFAULT_MAX_MEAN_RUN


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_gaussian_fit_on_sticky_mixture_is_not_flagged() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix()
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS)
    assert fit is not None
    assert fit.crash_state["flagged"] is False
    assert fit.crash_state["state"] is None
    assert min(fit.crash_state["occupancy"]) > hmm_module.DEFAULT_MAX_OCCUPANCY
    assert min(fit.crash_state["mean_run_length"]) > hmm_module.DEFAULT_MAX_MEAN_RUN


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_student_t_is_preferred_on_heavy_tailed_data() -> None:
    pytest.importorskip("hmmlearn")
    x, _ = _synthetic_feature_matrix(heavy_tails=True)
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, emission="student_t")
    assert fit is not None
    assert fit.student_t is not None
    assert fit.student_t["prefer_student_t"] is True
    # Jointly t(3) noise: at matched covariance a small df wins by a wide margin (measured
    # +0.136 nats per bar at df=5 on this seed; 3 or 5 across seeds).
    assert fit.student_t["best_df"] in (3.0, 5.0)
    gain = fit.student_t["student_t_log_likelihood"] - fit.student_t["gaussian_log_likelihood"]
    assert gain / x.shape[0] > HEAVY_T_GAIN_PER_BAR
    assert fit.emission == "student_t" and fit.params.emission == "student_t"
    assert fit.params.df == fit.student_t["best_df"]
    # log_likelihood stays the Gaussian figure of the winning restart.
    assert fit.log_likelihood == pytest.approx(fit.student_t["gaussian_log_likelihood"])
    # The switched params carry the covariance-matched scale, so undoing it recovers the
    # Gaussian fit exactly (location and transitions untouched).
    _, gaussian_ll = forward_filter(_gaussian_params(fit), x)
    assert gaussian_ll == pytest.approx(fit.log_likelihood)
    # ... and filtering with the switched params reproduces the best-df figure.
    assert forward_filter(fit.params, x)[1] == pytest.approx(
        fit.student_t["student_t_log_likelihood"]
    )


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_student_t_on_gaussian_data_picks_largest_df() -> None:
    pytest.importorskip("hmmlearn")
    # On Gaussian data the covariance-matched Student-t family improves monotonically with
    # df towards its Gaussian limit, so the largest df on the grid wins. The preference flag
    # is not asserted False: at df=30 the t is within O(1/df) of the Gaussian and its
    # marginally heavier tail can absorb the handful of bars the three-state mixture
    # misassigns, which on this data is at most a few nats in 1500 bars (measured
    # +0.0008 per bar on this seed, never above +0.005 over five seeds). What must hold is
    # that any gain stays at that noise level, far below the heavy-tailed case.
    x, _ = _synthetic_feature_matrix()
    fit = fit_hmm_features(x, feature_names=FEATURE_NAMES, emission="student_t")
    assert fit is not None
    assert fit.student_t is not None
    assert fit.student_t["best_df"] == max(DEFAULT_T_DFS)
    by_df = fit.student_t["by_df"]
    grid = [by_df[str(df)] for df in DEFAULT_T_DFS]
    assert grid == sorted(grid), "likelihood must rise monotonically with df on Gaussian data"
    gain = fit.student_t["student_t_log_likelihood"] - fit.student_t["gaussian_log_likelihood"]
    assert gain / x.shape[0] <= GAUSSIAN_T_GAIN_PER_BAR
    # A heavy tail at df=3 is firmly rejected on Gaussian data.
    assert by_df["3.0"] < fit.student_t["gaussian_log_likelihood"] - 100.0


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_auto_emission_evaluates_student_t_only_when_flagged() -> None:
    pytest.importorskip("hmmlearn")
    x_clean, _ = _synthetic_feature_matrix()
    clean = fit_hmm_features(
        x_clean, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS, emission="auto"
    )
    assert clean is not None
    assert clean.crash_state["flagged"] is False
    assert clean.student_t is None
    assert clean.emission == "gaussian" and clean.params.df is None

    x_spiky, _ = _synthetic_feature_matrix(spikes=True)
    spiky = fit_hmm_features(
        x_spiky, feature_names=FEATURE_NAMES, seeds=MANY_SEEDS, emission="auto"
    )
    assert spiky is not None
    assert spiky.crash_state["flagged"] is True
    assert spiky.student_t is not None
    expected = "student_t" if spiky.student_t["prefer_student_t"] else "gaussian"
    assert spiky.emission == expected and spiky.params.emission == expected
    if expected == "student_t":
        assert spiky.params.df == spiky.student_t["best_df"]
    else:
        assert spiky.params.df is None


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_all_restarts_failing_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    hmm = pytest.importorskip("hmmlearn.hmm")
    x, _ = _synthetic_feature_matrix(n_rows=200)

    class _Exploding(hmm.GaussianHMM):
        def fit(self, x: np.ndarray, lengths: object = None) -> object:
            raise ArithmeticError("simulated Baum-Welch failure")

    monkeypatch.setattr(hmm, "GaussianHMM", _Exploding)
    with pytest.raises(RuntimeError, match=r"seeds \(0, 1\)"):
        fit_hmm_features(x, feature_names=FEATURE_NAMES, seeds=(0, 1))


# ---- singular covariance from a one-bar state ------------------------------------------

JUMP_LOG_RETURN = np.log(1.5)  # a +50% close-to-close jump: ~40 sigma on 1% daily vol
JUMP_ROWS = 310  # the walk-forward's first refit window where the failure was found
JUMP_N_ITER = 50  # the walk-forward's iteration cap, which leaves the one-bar state in place
HMMLEARN_COVARS_PRIOR = 1e-2  # hmmlearn's default covars_prior for "full"; fills a one-bar state
HMMLEARN_MIN_COVAR = 1e-3  # hmmlearn's default min_covar, the floor re-applied to a singular state
VOL_WINDOWS = (10, 20)  # same rolling windows as hmm_features


def _jump_feature_matrix(n_rows: int = JUMP_ROWS, jump_row: int = 100) -> np.ndarray:
    """Standardised return / vol_10 / vol_20 matrix with one huge single-bar jump.

    This mirrors what the walk-forward sees after ``perturb_after`` scales the closes: every
    later return is unchanged except the one bar at the cut, which becomes an outlier that
    Baum-Welch parks in its own state.
    """
    rng = np.random.default_rng(SEED)
    warm_up = max(VOL_WINDOWS)
    returns = 0.0003 + 0.01 * rng.standard_normal(n_rows + warm_up)
    returns[jump_row + warm_up] = JUMP_LOG_RETURN
    series = pd.Series(returns)
    columns = [returns] + [series.rolling(w).std().to_numpy() for w in VOL_WINDOWS]
    x = np.column_stack(columns)[warm_up:]
    return (x - x.mean(axis=0)) / x.std(axis=0)


def test_floor_singular_covars_touches_only_rank_deficient_states() -> None:
    healthy = np.array([[1.0, 0.3], [0.3, 2.0]])
    one_bar = np.full((2, 2), HMMLEARN_COVARS_PRIOR)  # rank 1: what hmmlearn's M-step returns
    floored, which = _floor_singular_covars(np.stack([healthy, one_bar]), HMMLEARN_MIN_COVAR)
    assert which == (1,)
    assert np.array_equal(floored[0], healthy), "a positive definite matrix must be untouched"
    np.testing.assert_array_equal(floored[1], one_bar + HMMLEARN_MIN_COVAR * np.eye(2))
    for cov in floored:
        np.linalg.cholesky(cov)  # both now factorise


def test_floor_singular_covars_uses_eigenvalue_tolerance() -> None:
    # Cholesky can succeed on a matrix that is positive definite only by round-off; the
    # eigenvalue tolerance catches it.
    barely = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    np.linalg.cholesky(barely)  # precondition: Cholesky alone would let it through
    floored, which = _floor_singular_covars(barely[np.newaxis], HMMLEARN_MIN_COVAR)
    assert which == (0,)
    assert np.linalg.eigvalsh(floored[0]).min() > hmm_module.MIN_COVAR_EIGENVALUE


def test_params_from_model_floors_only_the_singular_state() -> None:
    # A fitted-model stand-in with latent state 1 holding a single bar (rank-1 covariance).
    # Means are deliberately out of order so the permutation is exercised too.
    healthy_a = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]])
    healthy_b = np.array([[0.5, 0.0, 0.1], [0.0, 0.7, 0.0], [0.1, 0.0, 0.9]])
    one_bar = np.full((3, 3), HMMLEARN_COVARS_PRIOR)
    transmat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    model = types.SimpleNamespace(
        startprob_=np.array([0.2, 0.3, 0.5]),
        transmat_=transmat,
        means_=np.array([[0.5, 0.0, 0.0], [-8.0, 9.0, 9.0], [-0.5, 0.0, 0.0]]),
        covars_=np.stack([healthy_a, one_bar, healthy_b]),
        min_covar=HMMLEARN_MIN_COVAR,
    )
    params, floored = hmm_module._params_from_model(model)
    # Ascending mean return: latent 1 (-8) -> 0, latent 2 (-0.5) -> 1, latent 0 (0.5) -> 2.
    assert floored == (0,)
    np.testing.assert_array_equal(params.means[:, 0], [-8.0, -0.5, 0.5])
    np.testing.assert_array_equal(params.covars[0], one_bar + HMMLEARN_MIN_COVAR * np.eye(3))
    assert np.array_equal(params.covars[1], healthy_b)
    assert np.array_equal(params.covars[2], healthy_a)
    for cov in params.covars:
        np.linalg.cholesky(cov)
    order = [1, 2, 0]
    np.testing.assert_array_equal(params.startprob, model.startprob_[order])
    np.testing.assert_array_equal(params.transmat, transmat[np.ix_(order, order)])


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_one_bar_state_with_singular_covariance_still_fits_and_is_flagged() -> None:
    hmm = pytest.importorskip("hmmlearn.hmm")
    x = _jump_feature_matrix()
    names = ("log_return", "vol_10", "vol_20")
    model = hmm.GaussianHMM(
        n_components=3, covariance_type="full", n_iter=JUMP_N_ITER, random_state=0
    )
    model.fit(x)
    raw_min_eig = min(np.linalg.eigvalsh(cov).min() for cov in model.covars_)
    assert raw_min_eig <= 0, "precondition: hmmlearn left a singular state covariance"

    fit = fit_hmm_features(x, feature_names=names, seeds=(0,), n_iter=JUMP_N_ITER)
    assert fit is not None
    for cov in fit.params.covars:
        np.linalg.cholesky(cov)
    # The reported log-likelihood is hmmlearn's own score of the unfloored model.
    assert fit.log_likelihood == pytest.approx(model.score(x), rel=1e-9)
    # The one-bar state is exactly what the crash diagnostic exists to catch, and it is
    # the state that had to be floored.
    assert fit.crash_state["flagged"] is True
    state = fit.crash_state["state"]
    assert fit.floored_states == (state,)
    assert fit.crash_state["occupancy"][state] == pytest.approx(1 / JUMP_ROWS)
    assert fit.crash_state["mean_run_length"][state] == pytest.approx(1.0)
    # The floored params still filter to a finite likelihood, within the floor's effect on
    # that single bar (the Gaussian log-density of one point changes with the log-det).
    filtered_ll = forward_filter(fit.params, x)[1]
    assert math.isfinite(filtered_ll)
    assert abs(filtered_ll - fit.log_likelihood) < 0.05 * abs(fit.log_likelihood) + 20.0
    assert fit_summary(fit)["floored_states"] == [state]


# ---- fit_summary ----------------------------------------------------------------------


def _walk(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v)


def _toy_fit(emission: str = "gaussian", df: float | None = None) -> HMMFit:
    params = _toy_params(emission=emission, df=df)
    return HMMFit(
        params=params,
        log_likelihood=-123.5,
        seed=np.int64(2),  # numpy ints must be converted
        seeds=(0, 1, 2),
        converged=np.bool_(True),
        n_iter=np.int64(17),
        emission=emission,
        feature_names=FEATURE_NAMES,
        restart_log_likelihoods={0: -130.0, 1: math.nan, 2: -123.5},
        crash_state={
            "flagged": False,
            "state": None,
            "occupancy": [np.float64(0.3), 0.4, 0.3],
            "mean_run_length": [5.0, None, 6.0],
            "max_occupancy": 0.02,
            "max_mean_run": 1.5,
        },
        student_t=None,
        floored_states=(np.int64(2),),
    )


def test_fit_summary_is_json_ready() -> None:
    summary = fit_summary(_toy_fit())
    for node in _walk(summary):
        assert not isinstance(node, np.generic), f"numpy scalar leaked: {node!r}"
        assert not isinstance(node, np.ndarray), "numpy array leaked"
        if isinstance(node, float):
            assert math.isfinite(node), f"non-finite float leaked: {node!r}"
    assert summary["restart_log_likelihoods"] == {"0": -130.0, "1": None, "2": -123.5}
    assert summary["seed"] == 2 and summary["seeds"] == [0, 1, 2]
    assert summary["converged"] is True and summary["n_iter"] == 17
    assert summary["covariance_type"] == "full"
    assert summary["emission"] == "gaussian" and summary["df"] is None
    assert summary["feature_names"] == ["log_return", "vol_10"]
    assert summary["transition_matrix"] == _toy_params().transmat.tolist()
    assert summary["student_t"] is None
    assert summary["floored_states"] == [2]
    assert [r["label"] for r in summary["regimes"]] == ["Bear", "Sideways", "Bull"]
    assert [r["latent_state"] for r in summary["regimes"]] == [0, 1, 2]
    assert [r["mean_daily_return"] for r in summary["regimes"]] == [-5.0, 0.0, 5.0]
    import json

    json.dumps(summary)


def test_fit_summary_destandardises_mean_return() -> None:
    fit = _toy_fit(emission="student_t", df=5.0)
    scaler_mean = np.array([0.0004, 0.01])
    scaler_std = np.array([0.012, 0.005])
    summary = fit_summary(fit, scaler_mean=scaler_mean, scaler_std=scaler_std)
    expected = fit.params.means[:, 0] * scaler_std[0] + scaler_mean[0]
    assert [r["mean_daily_return"] for r in summary["regimes"]] == pytest.approx(expected)
    assert summary["emission"] == "student_t" and summary["df"] == 5.0
    with pytest.raises(ValueError, match="given together"):
        fit_summary(fit, scaler_mean=scaler_mean)
