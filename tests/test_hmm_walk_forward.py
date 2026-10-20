"""Tests for the walk-forward HMM (plan.md Phase 4, Fix 5).

Synthetic data only, explicit seeds, no network. Fits are kept small (under a thousand
rows, three seeds, 50 Baum-Welch iterations) so the whole file runs in seconds. Tests that
fit a model call ``pytest.importorskip("hmmlearn")``; the degrade tests run without it.
"""

import json
import sys
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.hmm_walk_forward import (
    DEFAULT_HMM_MIN_TRAIN,
    DEFAULT_REFIT_EVERY,
    DEFAULT_TRAIN_BARS,
    FILTERED_COLUMNS,
    MIN_TEST_DAYS,
    REASON_TOO_SHORT,
    HMMWalkForwardResult,
    hmm_walk_forward_from_close,
    run_hmm_walk_forward,
)
from hedgefund.regime.hmm import REASON_UNAVAILABLE
from hedgefund.regime.hmm_features import build_features
from hedgefund.regime.hmm_filter import (
    DEFAULT_CONFIRM_BARS,
    filter_step,
    forward_filter,
    hysteresis_decision,
    smooth,
)
from hedgefund.regime.labels import STATES

SEED = 7  # fixed so the synthetic mixture is reproducible
N_ROWS = 900  # ~3.6 trading years of returns: ~27 regime runs, 15 small refits, ~3 s to run
# Persistent 3-state mixture in the spirit of tests/test_hmm.py::_synthetic_close, adapted
# to the Phase 4 feature set. The walk-forward fits on (log_return, realised vol), so for
# a known answer the states must differ in volatility as well as in mean (Bear is the
# high-vol state, Bull the calm one, as in real markets); with one flat sigma the HMM has
# nothing to learn from the vol columns except the switch artefact and clusters on that.
# Runs must also be long next to the vol windows (STAY_PROBABILITY 0.97 -> ~33-bar runs,
# windows of 5 and 10 bars), because a trailing vol window describes the *previous*
# regime for its first vol_long bars after every switch. The min_covar caveat in
# test_hmm.py does not bite here: the fitter sees standardised (unit-variance) columns.
STATE_MEANS = np.array([-0.04, 0.0, 0.04])
STATE_SIGMAS = np.array([0.04, 0.02, 0.01])
STAY_PROBABILITY = 0.97  # sticky chain: expected run length ~33 bars
VOL_SHORT = 5  # short realised-vol windows so the features catch up within a run
VOL_LONG = 10  # the first feature row is close position VOL_LONG (build_features warm-up)
# Small, fast walk-forward settings for every fitting test. Three seeds because seed 0 on
# its own lands in a degenerate local optimum on this data (means collapse to zero); the
# multi-restart selection is the fitter's job and costs ~0.1 s per refit here.
MIN_TRAIN = 150
REFIT_EVERY = 50
SEEDS = (0, 1, 2)
N_ITER = 50
FAST = dict(min_train=MIN_TRAIN, refit_every=REFIT_EVERY, seeds=SEEDS, n_iter=N_ITER)
VOL = dict(vol_short=VOL_SHORT, vol_long=VOL_LONG)
PROB_ATOL = 1e-9  # filtered rows must sum to one up to float64 round-off
SMOOTH_MIN_GAP = 0.01  # smoothed and filtered paths must differ by more than this somewhere
# Recovery thresholds; see test_regimes_track_the_true_states for the reasoning. Measured
# over generator seeds 7..12: whole-sample agreement 0.65..0.83, settled-bar 0.73..0.98.
MIN_RECOVERY = 0.55
MIN_SETTLED_RECOVERY = 0.65
# A bar is "settled" once the vol features (VOL_LONG bars of lag after a switch) and the
# hysteresis confirmation window have both caught up with the true regime.
SETTLED_LAG = VOL_LONG + DEFAULT_CONFIRM_BARS


def _synthetic(seed: int = SEED, n_rows: int = N_ROWS) -> tuple[pd.Series, pd.Series]:
    """Close series from a persistent 3-state mixture, plus the true state per return.

    The returned states are indexed by the date of the return they generated (the close
    index minus its first row), so ``states.loc[features.index]`` lines them up with the
    feature rows.
    """
    rng = np.random.default_rng(seed)
    transition = np.full((3, 3), (1.0 - STAY_PROBABILITY) / 2.0)
    np.fill_diagonal(transition, STAY_PROBABILITY)
    states = np.empty(n_rows, dtype=int)
    states[0] = 1
    for t in range(1, n_rows):
        states[t] = rng.choice(3, p=transition[states[t - 1]])
    returns = STATE_MEANS[states] + STATE_SIGMAS[states] * rng.standard_normal(n_rows)
    index = pd.bdate_range("2015-01-01", periods=n_rows + 1)
    close = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    return (
        pd.Series(close, index=index, name="Close"),
        pd.Series(states, index=index[1:], name="state"),
    )


def _features(close: pd.Series) -> pd.DataFrame:
    return build_features(close, **VOL)


def _bars_into_run(truth: np.ndarray) -> np.ndarray:
    """How many bars the true state has already held at each bar (0 on a switch bar)."""
    out = np.zeros(len(truth), dtype=int)
    for i in range(1, len(truth)):
        out[i] = out[i - 1] + 1 if truth[i] == truth[i - 1] else 0
    return out


@pytest.fixture
def quiet_hmmlearn() -> Iterator[None]:
    """Silence warnings hmmlearn itself emits; hedgefund warnings stay fatal."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="hmmlearn.*")
        yield


@pytest.fixture(scope="module")
def fitted() -> tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult]:
    """One walk-forward with several refits, shared by the invariant tests."""
    pytest.importorskip("hmmlearn")
    close, states = _synthetic()
    features = _features(close)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="hmmlearn.*")
        result = run_hmm_walk_forward(features, **FAST)
    return features, states, result


def _assert_json_ready(summary: dict) -> None:
    text = json.dumps(summary, allow_nan=False)  # raises on NaN / inf / numpy types
    assert "NaN" not in text


# --------------------------------------------------------------------------- #
# Degrade paths: must work with or without hmmlearn.
# --------------------------------------------------------------------------- #


def test_degrades_when_hmmlearn_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    close, _ = _synthetic()
    result = hmm_walk_forward_from_close(close, **VOL, **FAST)
    assert result.available is False
    assert result.reason == REASON_UNAVAILABLE
    assert list(result.filtered.columns) == list(FILTERED_COLUMNS)
    assert result.filtered.empty
    assert result.regimes.empty and result.raw_regimes.empty and result.log_likelihood.empty
    assert len(result.refit_dates) == 0 and result.fits == () and result.scalers == ()
    summary = result.summary()
    assert summary["available"] is False and summary["reason"] == REASON_UNAVAILABLE
    assert summary["n_days"] == 0 and summary["n_refits"] == 0
    assert summary["last_regime"] is None and summary["last_filtered"] is None
    assert summary["regime_counts"] == {label: 0 for label in STATES}
    _assert_json_ready(summary)


def test_degrades_when_history_is_too_short() -> None:
    # close has n_rows + 1 entries and build_features drops VOL_LONG warm-up closes, so the
    # feature count is n_rows + 1 - VOL_LONG; pick n_rows to land one row short.
    n_rows = MIN_TRAIN + MIN_TEST_DAYS - 2 + VOL_LONG
    close, _ = _synthetic(n_rows=n_rows)
    features = _features(close)
    assert len(features) == MIN_TRAIN + MIN_TEST_DAYS - 1
    result = run_hmm_walk_forward(features, **FAST)
    assert result.available is False
    assert result.reason is not None and result.reason.startswith(REASON_TOO_SHORT)
    assert f"need {MIN_TRAIN + MIN_TEST_DAYS} feature rows, have {len(features)}" in result.reason
    assert result.filtered.empty and list(result.filtered.columns) == list(FILTERED_COLUMNS)
    summary = result.summary()
    assert summary["params"]["min_train"] == MIN_TRAIN
    assert summary["params"]["seeds"] == list(SEEDS)
    _assert_json_ready(summary)


def test_too_short_check_does_not_need_hmmlearn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    close, _ = _synthetic(n_rows=MIN_TRAIN + VOL_LONG)
    result = hmm_walk_forward_from_close(close, **VOL, **FAST)
    assert result.reason is not None and result.reason.startswith(REASON_TOO_SHORT)


def test_defaults_match_plan_section_5() -> None:
    assert DEFAULT_HMM_MIN_TRAIN == 252  # one trading year
    assert DEFAULT_REFIT_EVERY == 21  # monthly
    assert DEFAULT_TRAIN_BARS == 2520  # ten trading years
    assert MIN_TEST_DAYS == 30
    assert list(FILTERED_COLUMNS) == [s.lower() for s in STATES]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(min_train=1), "min_train"),
        (dict(refit_every=0), "refit_every"),
        (dict(train_bars=1), "train_bars"),
        (dict(emission="laplace"), "emission"),
    ],
)
def test_rejects_bad_options(kwargs: dict, match: str) -> None:
    close, _ = _synthetic(n_rows=200)
    with pytest.raises(ValueError, match=match):
        run_hmm_walk_forward(_features(close), **{**FAST, **kwargs})


def test_rejects_bad_feature_frames() -> None:
    close, _ = _synthetic(n_rows=200)
    features = _features(close)
    with pytest.raises(ValueError, match="DataFrame"):
        run_hmm_walk_forward(features.to_numpy(), **FAST)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        run_hmm_walk_forward(features.reset_index(drop=True), **FAST)
    with pytest.raises(ValueError, match="at least 2 columns"):
        run_hmm_walk_forward(features.iloc[:, :1], **FAST)
    with pytest.raises(ValueError, match="log_return"):
        run_hmm_walk_forward(features.iloc[:, ::-1], **FAST)
    with_nan = features.copy()
    with_nan.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_hmm_walk_forward(with_nan, **FAST)


# --------------------------------------------------------------------------- #
# Available path: requires hmmlearn.
# --------------------------------------------------------------------------- #


def test_shapes_and_invariants(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    features, _, result = fitted
    assert result.available is True and result.reason is None
    expected_index = features.index[MIN_TRAIN:]
    assert result.filtered.index.equals(expected_index)
    assert list(result.filtered.columns) == list(FILTERED_COLUMNS)
    for series in (result.regimes, result.raw_regimes, result.log_likelihood):
        assert series.index.equals(expected_index)
    probs = result.filtered.to_numpy()
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=PROB_ATOL, rtol=0)
    np.testing.assert_array_equal(result.raw_regimes.to_numpy(), probs.argmax(axis=1))
    assert set(np.unique(result.regimes.to_numpy())) <= {0, 1, 2}
    assert np.all(np.isfinite(result.log_likelihood.to_numpy()))
    # Refits land on every REFIT_EVERY-th decision date, starting with the first.
    assert result.refit_dates.equals(pd.DatetimeIndex(expected_index[::REFIT_EVERY]))
    assert len(result.fits) == len(result.refit_dates) == len(result.scalers)
    assert len(result.fits) > 1, "the invariants need several refits to mean anything"
    assert result.params["feature_names"] == list(features.columns)


def test_scalers_are_fitted_on_the_training_block_only(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    features, _, result = fitted
    train_bars = result.params["train_bars"]
    for refit_date, scaler in zip(result.refit_dates, result.scalers, strict=True):
        t = features.index.get_loc(refit_date)
        block = features.iloc[max(0, t - train_bars) : t].to_numpy()
        np.testing.assert_array_equal(scaler.mean, block.mean(axis=0))
        np.testing.assert_array_equal(scaler.std, block.std(axis=0, ddof=0))


def test_decoded_path_is_filtered_not_smoothed(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    """Recompute the last segment with forward_filter: identical. With smooth: different."""
    features, _, result = fitted
    last_refit = features.index.get_loc(result.refit_dates[-1])
    start = max(0, last_refit - result.params["train_bars"])
    x = result.scalers[-1].transform(features.iloc[start:])
    params = result.fits[-1].params
    recomputed, _ = forward_filter(params, x)
    segment = result.filtered.loc[result.refit_dates[-1] :].to_numpy()
    np.testing.assert_array_equal(recomputed[last_refit - start :], segment)

    smoothed = smooth(params, x)[last_refit - start :]
    assert np.abs(smoothed - segment).max() > SMOOTH_MIN_GAP


def test_posterior_is_carried_across_each_refit(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    """First filtered row after a refit = one filter_step from the training block's last row."""
    features, _, result = fitted
    train_bars = result.params["train_bars"]
    for refit_date, fit, scaler in zip(
        result.refit_dates, result.fits, result.scalers, strict=True
    ):
        t = features.index.get_loc(refit_date)
        block = scaler.transform(features.iloc[max(0, t - train_bars) : t])
        block_filtered, _ = forward_filter(fit.params, block)
        x_t = scaler.transform(features.iloc[t].to_numpy())
        expected, expected_ll = filter_step(fit.params, block_filtered[-1], x_t)
        np.testing.assert_array_equal(result.filtered.loc[refit_date].to_numpy(), expected)
        assert result.log_likelihood.loc[refit_date] == expected_ll


def test_online_hysteresis_equals_batch_rule_across_refits(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    """The hysteresis machine reads only the filtered rows, so running the batch rule on
    them must reproduce the online regimes exactly; this also proves the held regime is
    never reset at a refit date."""
    _, _, result = fitted
    p = result.params
    batch = hysteresis_decision(
        result.filtered.to_numpy(),
        enter=p["enter"],
        exit=p["exit"],
        confirm_bars=p["confirm_bars"],
    )
    np.testing.assert_array_equal(result.regimes.to_numpy(), batch)


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_single_fit_online_hysteresis_equals_batch() -> None:
    """With refit_every >= n there is exactly one fit; the online path is then a plain
    forward filter followed by the batch hysteresis rule, bit for bit."""
    pytest.importorskip("hmmlearn")
    close, _ = _synthetic(n_rows=400)
    features = _features(close)
    result = run_hmm_walk_forward(features, **{**FAST, "refit_every": len(features)})
    assert len(result.fits) == 1
    x = result.scalers[0].transform(features)
    full, _ = forward_filter(result.fits[0].params, x)
    np.testing.assert_array_equal(result.filtered.to_numpy(), full[MIN_TRAIN:])
    np.testing.assert_array_equal(
        result.regimes.to_numpy(), hysteresis_decision(result.filtered.to_numpy())
    )


def test_regimes_track_the_true_states(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    """Known answer: the held regime matches the simulated state on most bars.

    Two tolerances, both well above the 1/3 chance level:

    - Whole sample > 0.55. The detector is lagged by construction: the vol features
      describe the previous regime for up to ``VOL_LONG`` bars after each switch and the
      hysteresis rule adds ``confirm_bars`` more, so with ~33-bar runs roughly a third of
      the bars sit inside that lag; a few more go to the early, data-poor fits. A fit that
      clusters on volatility or splits a state scores ~0.4 here, so 0.55 has teeth.
    - Settled bars (at least ``SETTLED_LAG`` bars into a true run) > 0.65: once the
      features and the confirmation window have caught up the held regime should be the
      true one most of the time. Over generator seeds 7..12 this measure is 0.73..0.98.
    """
    _, states, result = fitted
    truth = states.loc[result.regimes.index].to_numpy()
    held = result.regimes.to_numpy()
    agreement = float(np.mean(held == truth))
    assert agreement > MIN_RECOVERY, f"held regime agrees with the truth on only {agreement:.2f}"
    settled = _bars_into_run(truth) >= SETTLED_LAG
    assert settled.sum() > 0
    settled_agreement = float(np.mean(held[settled] == truth[settled]))
    assert settled_agreement > MIN_SETTLED_RECOVERY, (
        f"held regime agrees on only {settled_agreement:.2f} of settled bars"
    )
    assert settled_agreement > agreement, "the lag should be where the misses concentrate"


def test_summary_is_json_ready_and_labelled(
    fitted: tuple[pd.DataFrame, pd.Series, HMMWalkForwardResult],
) -> None:
    _, _, result = fitted
    summary = result.summary()
    _assert_json_ready(summary)
    assert summary["available"] is True and summary["reason"] is None
    assert summary["n_days"] == len(result.filtered)
    assert summary["n_refits"] == len(result.refit_dates)
    assert summary["first_date"] == result.filtered.index[0].isoformat()
    assert summary["last_date"] == result.filtered.index[-1].isoformat()
    assert set(summary["last_filtered"]) == set(FILTERED_COLUMNS)
    assert summary["last_regime"] == STATES[int(result.regimes.iloc[-1])]
    assert summary["last_raw_regime"] == STATES[int(result.raw_regimes.iloc[-1])]
    assert sum(summary["regime_counts"].values()) == len(result.regimes)
    assert set(summary["regime_counts"]) == set(STATES)
    assert summary["mean_log_likelihood"] == pytest.approx(result.log_likelihood.mean())
    assert summary["params"] == result.params


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_from_close_matches_run_on_built_features() -> None:
    pytest.importorskip("hmmlearn")
    close, _ = _synthetic(n_rows=400)
    via_close = hmm_walk_forward_from_close(close, **VOL, **FAST)
    via_features = run_hmm_walk_forward(_features(close), **FAST)
    np.testing.assert_array_equal(via_close.filtered.to_numpy(), via_features.filtered.to_numpy())
    assert via_close.filtered.index.equals(via_features.filtered.index)
    np.testing.assert_array_equal(via_close.regimes.to_numpy(), via_features.regimes.to_numpy())


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_repeated_runs_are_bit_identical() -> None:
    pytest.importorskip("hmmlearn")
    close, _ = _synthetic(n_rows=400)
    a = hmm_walk_forward_from_close(close, **VOL, **FAST)
    b = hmm_walk_forward_from_close(close.copy(), **VOL, **FAST)
    np.testing.assert_array_equal(a.filtered.to_numpy(), b.filtered.to_numpy())
    np.testing.assert_array_equal(a.log_likelihood.to_numpy(), b.log_likelihood.to_numpy())
    np.testing.assert_array_equal(a.regimes.to_numpy(), b.regimes.to_numpy())


@pytest.mark.usefixtures("quiet_hmmlearn")
def test_sliding_train_window_limits_the_training_block() -> None:
    pytest.importorskip("hmmlearn")
    close, _ = _synthetic(n_rows=400)
    features = _features(close)
    train_bars = 120
    result = run_hmm_walk_forward(features, **FAST, train_bars=train_bars)
    for refit_date, scaler in zip(result.refit_dates, result.scalers, strict=True):
        t = features.index.get_loc(refit_date)
        block = features.iloc[max(0, t - train_bars) : t].to_numpy()
        assert len(block) <= train_bars
        np.testing.assert_array_equal(scaler.mean, block.mean(axis=0))
    assert result.params["train_bars"] == train_bars
