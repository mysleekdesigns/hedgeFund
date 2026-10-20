"""Walk-forward HMM: refit on past data only, decode with forward-filtered probabilities.

This is the Phase 4 answer to plan.md Fix 5 ("HMM fitted on the wrong inputs and read
with hindsight"). At every decision date ``t`` the model that scores ``t`` was fitted on
feature rows strictly before some refit date ``t_refit <= t``, standardised with that
training block's own statistics, and the state probability is the causal forward filter
``P(s_t | x_<=t)`` advanced one bar at a time. Smoothed probabilities and Viterbi paths
never enter this module; ``hedgefund.regime.hmm_filter.smooth`` exists for diagnostics
only and ``tests/test_hmm_walk_forward.py`` proves the two paths differ.

Mechanics of ``run_hmm_walk_forward`` (the one intentional sequential loop over a series
in this module, allowed by ``.claude/rules/python.md`` for the incremental refit):

1. Decision dates are ``features.index[min_train:]``: the first ``min_train`` feature rows
   are training only.
2. Every ``refit_every`` decisions, starting with the first, the training block is the
   trailing ``train_bars`` rows strictly before ``t`` (``features.iloc[max(0, t -
   train_bars):t]``). A ``FeatureScaler`` is fitted on that block alone, the block is
   standardised with it and ``fit_hmm_features`` runs the multi-restart Baum-Welch.
3. After a refit the online filter is re-initialised by running ``forward_filter`` over the
   standardised training block and carrying its last row forward: that row is the new
   model's own belief about ``t - 1`` and every observation it read is in the past.
4. Each decision bar standardises row ``t`` with the current scaler, calls
   ``filter_step`` from the carried posterior, records the posterior, its argmax and the
   incremental ``log P(x_t | x_<t)``, advances the hysteresis state machine with the
   posterior, and carries the posterior into the next bar.
5. The hysteresis state (``hysteresis_step``) is a decision-level machine, not a model
   parameter, so it is carried across refits unchanged: the held regime never resets to
   Sideways just because the model was refitted.

No-lookahead property: at decision date ``t`` the scaler and the HMM parameters come from
rows ``< t_refit <= t`` and the filter has read rows ``<= t`` only. Nothing computed at
``t`` sees row ``t + 1``; ``tests/test_no_lookahead.py`` enforces this by perturbation.

Pure: pandas and numpy in, pandas and numpy out, no I/O. ``hmmlearn`` is optional and
imported lazily by the fitter; without it the result degrades to ``available=False``.
Backtests are historical, not forward-looking.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hedgefund.regime.hmm import (
    DEFAULT_N_ITER,
    DEFAULT_SEEDS,
    EMISSION_CHOICES,
    REASON_UNAVAILABLE,
    HMMFit,
    fit_hmm_features,
)
from hedgefund.regime.hmm_features import (
    DEFAULT_VOL_LONG,
    DEFAULT_VOL_SHORT,
    FEATURE_NAMES,
    FeatureScaler,
    build_features,
    fit_scaler,
)
from hedgefund.regime.hmm_filter import (
    DEFAULT_CONFIRM_BARS,
    DEFAULT_ENTER,
    DEFAULT_EXIT,
    filter_step,
    forward_filter,
    hysteresis_step,
    initial_hysteresis_state,
)
from hedgefund.regime.labels import SIDEWAYS, STATES

# config backtest.min_train: one trading year before the first HMM decision (plan section 5).
DEFAULT_HMM_MIN_TRAIN = 252
# config hmm.refit_every: refit monthly, plan Fix 5 "or at least monthly".
DEFAULT_REFIT_EVERY = 21
# config hmm.train_bars: sliding window of ~10 trading years (10 * 252), plan Fix 5.
DEFAULT_TRAIN_BARS = 2520
# Mirror of walk_forward.MIN_TEST_DAYS: fewer scored days than this is not a backtest.
MIN_TEST_DAYS = 30
MIN_MIN_TRAIN = 2  # a scaler needs two rows; anything smaller cannot standardise
MIN_TRAIN_BARS = 2  # same floor for the sliding training window
# Prefix of the too-short reason; ": need N feature rows, have M" is appended.
REASON_TOO_SHORT = "history too short for the walk-forward HMM"

# Column names of the filtered-probability frame: STATES in order, lower-cased for JSON.
FILTERED_COLUMNS: tuple[str, ...] = tuple(state.lower() for state in STATES)
RETURN_FEATURE = FEATURE_NAMES[0]  # column 0 of the feature matrix must be the daily log return


@dataclass(frozen=True)
class HMMWalkForwardResult:
    """Per-decision-date output of the walk-forward HMM.

    ``filtered`` holds ``P(s_t | x_<=t)`` under the model fitted on rows ``< t_refit <= t``,
    one row per decision date, columns ``FILTERED_COLUMNS`` (bear, sideways, bull).
    ``regimes`` is the held regime from the online hysteresis rule, ``raw_regimes`` the
    argmax of each filtered row, ``log_likelihood`` the incremental ``log P(x_t | x_<t)``;
    all share ``filtered``'s index. ``refit_dates`` are the decision dates at which a refit
    happened; ``fits`` and ``scalers`` are aligned with them. ``params`` echoes the run
    options. When ``available`` is False every series is empty and ``reason`` says why.
    """

    available: bool
    reason: str | None
    filtered: pd.DataFrame
    regimes: pd.Series
    raw_regimes: pd.Series
    log_likelihood: pd.Series
    refit_dates: pd.DatetimeIndex
    fits: tuple[HMMFit, ...]
    scalers: tuple[FeatureScaler, ...]
    params: dict

    def summary(self) -> dict:
        """JSON-ready description: builtins only, no numpy scalars, no NaN.

        ``regime_counts`` counts the held regime over every decision date, keyed by the
        ``STATES`` labels. ``last_*`` fields are ``None`` when nothing was scored.
        """
        n_days = len(self.filtered)
        counts = {label: 0 for label in STATES}
        if n_days:
            values, tallies = np.unique(self.regimes.to_numpy(dtype=int), return_counts=True)
            for value, count in zip(values, tallies, strict=True):
                counts[STATES[int(value)]] = int(count)
        last_filtered = None
        last_regime = None
        last_raw_regime = None
        first_date = None
        last_date = None
        mean_ll = None
        if n_days:
            last_row = self.filtered.iloc[-1]
            last_filtered = {col: float(last_row[col]) for col in FILTERED_COLUMNS}
            last_regime = STATES[int(self.regimes.iloc[-1])]
            last_raw_regime = STATES[int(self.raw_regimes.iloc[-1])]
            first_date = self.filtered.index[0].isoformat()
            last_date = self.filtered.index[-1].isoformat()
            mean_value = float(self.log_likelihood.to_numpy(dtype=float).mean())
            mean_ll = mean_value if math.isfinite(mean_value) else None
        return {
            "available": bool(self.available),
            "reason": self.reason,
            "n_days": n_days,
            "n_refits": len(self.refit_dates),
            "first_date": first_date,
            "last_date": last_date,
            "last_filtered": last_filtered,
            "last_regime": last_regime,
            "last_raw_regime": last_raw_regime,
            "regime_counts": counts,
            "mean_log_likelihood": mean_ll,
            "params": _jsonify(self.params),
        }


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy scalars / sequences to builtins; non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _run_params(
    *,
    min_train: int,
    refit_every: int,
    train_bars: int,
    seeds: Sequence[int],
    n_iter: int,
    emission: str,
    enter: float,
    exit: float,
    confirm_bars: int,
    feature_names: Sequence[str],
) -> dict:
    return {
        "min_train": int(min_train),
        "refit_every": int(refit_every),
        "train_bars": int(train_bars),
        "seeds": [int(s) for s in seeds],
        "n_iter": int(n_iter),
        "emission": str(emission),
        "enter": float(enter),
        "exit": float(exit),
        "confirm_bars": int(confirm_bars),
        "feature_names": [str(name) for name in feature_names],
    }


def _empty_result(reason: str, params: dict) -> HMMWalkForwardResult:
    index = pd.DatetimeIndex([])
    return HMMWalkForwardResult(
        available=False,
        reason=reason,
        filtered=pd.DataFrame(
            np.empty((0, len(FILTERED_COLUMNS)), dtype=float),
            index=index,
            columns=list(FILTERED_COLUMNS),
        ),
        regimes=pd.Series(np.empty(0, dtype=int), index=index, dtype=int),
        raw_regimes=pd.Series(np.empty(0, dtype=int), index=index, dtype=int),
        log_likelihood=pd.Series(np.empty(0, dtype=float), index=index, dtype=float),
        refit_dates=pd.DatetimeIndex([]),
        fits=(),
        scalers=(),
        params=params,
    )


def _validate_inputs(
    features: pd.DataFrame,
    *,
    min_train: int,
    refit_every: int,
    train_bars: int,
    emission: str,
) -> None:
    """Raise ``ValueError`` naming the offending input; never coerce."""
    if not isinstance(features, pd.DataFrame):
        raise ValueError(
            f"features must be the DataFrame from build_features, got {type(features).__name__}"
        )
    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError(f"features must have a DatetimeIndex, got {type(features.index).__name__}")
    if features.shape[1] < 2:
        raise ValueError(
            f"features must have at least 2 columns (the return and one volatility), "
            f"got {features.shape[1]}; fitting on volatility alone is forbidden (plan Fix 5)"
        )
    if str(features.columns[0]) != RETURN_FEATURE:
        raise ValueError(
            f"features column 0 must be {RETURN_FEATURE!r} so states can be ordered by mean "
            f"return, got {features.columns[0]!r}"
        )
    values = features.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "features must not contain NaN or inf; build_features drops the warm-up rows"
        )
    if min_train < MIN_MIN_TRAIN:
        raise ValueError(f"min_train must be >= {MIN_MIN_TRAIN} feature rows, got {min_train}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1 decision bar, got {refit_every}")
    if train_bars < MIN_TRAIN_BARS:
        raise ValueError(f"train_bars must be >= {MIN_TRAIN_BARS} feature rows, got {train_bars}")
    if emission not in EMISSION_CHOICES:
        raise ValueError(f"emission must be one of {EMISSION_CHOICES}, got {emission!r}")


def run_hmm_walk_forward(
    features: pd.DataFrame,
    *,
    min_train: int = DEFAULT_HMM_MIN_TRAIN,
    refit_every: int = DEFAULT_REFIT_EVERY,
    train_bars: int = DEFAULT_TRAIN_BARS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_iter: int = DEFAULT_N_ITER,
    emission: str = "gaussian",
    enter: float = DEFAULT_ENTER,
    exit: float = DEFAULT_EXIT,
    confirm_bars: int = DEFAULT_CONFIRM_BARS,
) -> HMMWalkForwardResult:
    """Walk the HMM forward one feature row at a time with no lookahead (plan.md Fix 5).

    ``features`` is the output of ``hmm_features.build_features``: a float DataFrame with a
    ``DatetimeIndex``, no NaN, at least two columns and column 0 named ``"log_return"``.
    Decision dates are ``features.index[min_train:]``. Every ``refit_every`` decisions,
    starting with the first, the model and its scaler are refitted on the ``train_bars``
    rows strictly before the decision row (fewer when the history is shorter); between
    refits the forward filter advances one bar per decision from the posterior carried
    over from the previous bar. After each refit the carried posterior is the last row of
    ``forward_filter`` over the new training block, so the online filter continues from the
    new model's belief about the bar before the refit. The hysteresis state machine is
    carried across refits untouched. See the module docstring for the full mechanics.

    At decision date ``t`` the scaler and parameters come from rows ``< t_refit <= t`` and
    the filter has read rows ``<= t`` only; nothing at ``t`` sees row ``t + 1``.

    Returns ``available=False`` with ``reason`` starting with ``REASON_TOO_SHORT`` when
    ``len(features) < min_train + MIN_TEST_DAYS``, and with ``reason == REASON_UNAVAILABLE``
    when ``hmmlearn`` cannot be imported; both carry empty outputs. Raises ``ValueError`` on
    malformed ``features`` or options (see ``_validate_inputs``); the hysteresis thresholds
    are validated by ``hysteresis_step`` on the first bar.
    """
    _validate_inputs(
        features,
        min_train=min_train,
        refit_every=refit_every,
        train_bars=train_bars,
        emission=emission,
    )
    columns = tuple(str(c) for c in features.columns)
    params = _run_params(
        min_train=min_train,
        refit_every=refit_every,
        train_bars=train_bars,
        seeds=seeds,
        n_iter=n_iter,
        emission=emission,
        enter=enter,
        exit=exit,
        confirm_bars=confirm_bars,
        feature_names=columns,
    )

    n = len(features)
    required = min_train + MIN_TEST_DAYS
    if n < required:
        return _empty_result(f"{REASON_TOO_SHORT}: need {required} feature rows, have {n}", params)

    values = features.to_numpy(dtype=float)
    n_states = len(STATES)
    n_decisions = n - min_train
    filtered = np.empty((n_decisions, n_states), dtype=float)
    raw_regimes = np.empty(n_decisions, dtype=int)
    regimes = np.empty(n_decisions, dtype=int)
    log_likelihood = np.empty(n_decisions, dtype=float)
    refit_positions: list[int] = []
    fits: list[HMMFit] = []
    scalers: list[FeatureScaler] = []

    def refit(t: int) -> tuple[FeatureScaler, HMMFit | None, np.ndarray]:
        """Fit scaler and model on rows [max(0, t - train_bars), t) and return the
        carried posterior: the last row of the forward filter over that block, which is
        the new model's own belief about t - 1 and reads only past data."""
        block = values[max(0, t - train_bars) : t]
        block_scaler = fit_scaler(block, columns)
        block_scaled = block_scaler.transform(block)
        block_fit = fit_hmm_features(
            block_scaled,
            feature_names=columns,
            seeds=seeds,
            n_iter=n_iter,
            emission=emission,
        )
        if block_fit is None:
            return block_scaler, None, np.empty(0)
        block_filtered, _ = forward_filter(block_fit.params, block_scaled)
        return block_scaler, block_fit, block_filtered[-1]

    hysteresis = initial_hysteresis_state(n_states, SIDEWAYS)
    scaler, fit, prev = refit(min_train)
    if fit is None:
        return _empty_result(REASON_UNAVAILABLE, params)
    for k, t in enumerate(range(min_train, n)):
        if k % refit_every == 0:
            if k > 0:
                scaler, fit, prev = refit(t)
                if fit is None:
                    return _empty_result(REASON_UNAVAILABLE, params)
            refit_positions.append(k)
            fits.append(fit)
            scalers.append(scaler)

        x_t = scaler.transform(values[t])
        post, log_inc = filter_step(fit.params, prev, x_t)
        filtered[k] = post
        raw_regimes[k] = int(np.argmax(post))
        log_likelihood[k] = log_inc
        hysteresis = hysteresis_step(
            hysteresis,
            post,
            enter=enter,
            exit=exit,
            confirm_bars=confirm_bars,
            initial_regime=SIDEWAYS,
        )
        regimes[k] = hysteresis.regime
        prev = post

    decision_dates = features.index[min_train:]
    return HMMWalkForwardResult(
        available=True,
        reason=None,
        filtered=pd.DataFrame(filtered, index=decision_dates, columns=list(FILTERED_COLUMNS)),
        regimes=pd.Series(regimes, index=decision_dates, dtype=int),
        raw_regimes=pd.Series(raw_regimes, index=decision_dates, dtype=int),
        log_likelihood=pd.Series(log_likelihood, index=decision_dates, dtype=float),
        refit_dates=pd.DatetimeIndex(decision_dates[refit_positions]),
        fits=tuple(fits),
        scalers=tuple(scalers),
        params=params,
    )


def hmm_walk_forward_from_close(
    close: pd.Series,
    *,
    vol_short: int = DEFAULT_VOL_SHORT,
    vol_long: int = DEFAULT_VOL_LONG,
    **kwargs: Any,
) -> HMMWalkForwardResult:
    """Build the close-only feature matrix and run ``run_hmm_walk_forward`` on it.

    ``kwargs`` are passed through to ``run_hmm_walk_forward`` unchanged. The feature
    matrix is ``log_return``, ``vol_<vol_short>`` and ``vol_<vol_long>``; its first row is
    the ``vol_long``-th return, so decision dates start ``vol_long + min_train`` closes in.
    """
    features = build_features(close, vol_short=vol_short, vol_long=vol_long)
    return run_hmm_walk_forward(features, **kwargs)
