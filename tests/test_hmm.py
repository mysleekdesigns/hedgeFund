"""Tests for the optional HMM port in ``hedgefund.regime.hmm``.

All tests are offline. The available-path tests use a synthetic 3-state mixture with
clearly separated means and skip when ``hmmlearn`` is not installed.
"""

import sys
import types
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from hedgefund.regime import hmm as hmm_module
from hedgefund.regime.hmm import (
    HMM_CAVEAT,
    REASON_DISABLED,
    REASON_UNAVAILABLE,
    fit_hmm,
    hmm_summary,
    order_states_by_mean,
)

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
