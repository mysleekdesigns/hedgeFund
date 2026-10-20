"""The companion skill's JSON contract: ``analyze()`` returns its dict plus additive fields.

Field names, types, nesting and the ``[Bear, Sideways, Bull]`` index order are frozen
(``.claude/rules/json-contract.md``). The companion's top-level keys come first, in the
companion's order, and every companion value is computed the same way as the companion
script (``float()`` casts, ``str(date)``) so the reproduction test matches to 6 decimals.

Phase 1 added ``params.return_type``. Phase 2 appends the honest-statistics keys listed in
``PHASE2_FIELDS`` after ``disclaimer``: stride and run-length persistence (plan.md Fix 1),
null percentiles, block-bootstrap intervals, rolling / exponentially weighted / shrunk
matrices with a drift metric (Fix 4), mixing time and the Anderson-Goodman tests. Phase 3
appends the single key ``backtest`` (``PHASE3_FIELDS``): the cost-aware walk-forward with
gross and net metric sets, benchmarks, null percentiles, the permutation test, PSR / DSR,
the two-year holdout, the sensitivity grid and the lag diagnostic
(``hedgefund.report.backtest_report``). ``backtest_only`` returns the slim dict behind
``hedgefund backtest``. New floats are emitted as JSON numbers or ``null``, never NaN.
"""

import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hedgefund.backtest.costs import SENSITIVITY_BPS, CostModel
from hedgefund.backtest.lag import DEFAULT_LAG_ORDER
from hedgefund.backtest.metrics import DEFAULT_HORIZONS
from hedgefund.backtest.sensitivity import (
    DEFAULT_HOLDOUT_YEARS,
    DEFAULT_SENSITIVITY_THRESHOLDS,
    DEFAULT_SENSITIVITY_WINDOWS,
)
from hedgefund.backtest.walk_forward import (
    DEFAULT_DEAD_BAND,
    DEFAULT_REFIT,
    DEFAULT_ROLLING_BARS,
    walk_forward_backtest,
)
from hedgefund.regime.hmm import hmm_summary
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, STATES, label_regimes
from hedgefund.regime.markov import (
    DEFAULT_HALF_LIFE_BARS,
    DEFAULT_MIXING_EPSILON,
    DEFAULT_ROLLING_WINDOW_BARS,
    DEFAULT_SHRINKAGE_ALPHA,
    build_transition_matrix,
    count_transitions,
    dirichlet_shrinkage,
    drift_tvd,
    effective_sample_size,
    ew_transition_matrix,
    mixing_time,
    nstep_forecast,
    rolling_transition_counts,
    rolling_transition_matrix,
    run_length_persistence,
    stationary_distribution,
)
from hedgefund.regime.signal import next_state_probabilities, signal_from_matrix
from hedgefund.regime.stats import (
    anderson_goodman_test,
    block_bootstrap_ci,
    homogeneity_test,
    iid_null,
    label_shuffle_null,
    labeller_null,
)
from hedgefund.report.backtest_report import DEFAULT_RULE, DISCLAIMER, backtest_block

FRAMEWORK = "Roan (@RohOnChain)"
# ``DISCLAIMER`` is defined in ``backtest_report`` (the block carries it too) and
# re-exported here so ``from hedgefund.report.json_contract import DISCLAIMER`` keeps working.
__all__ = ["DISCLAIMER", "FRAMEWORK", "analyze", "backtest_only", "to_json"]

# Top-level keys of the companion dict, in the companion's emission order.
COMPANION_FIELDS = [
    "source",
    "rows",
    "date_start",
    "date_end",
    "params",
    "states",
    "current_regime",
    "next_state_probabilities",
    "signal",
    "transition_matrix",
    "persistence_diagonal",
    "stationary_distribution",
    "walk_forward",
    "hmm",
    "framework",
    "disclaimer",
]

# Phase 2 additions, appended after ``disclaimer`` in this order (additive only).
PHASE2_FIELDS = [
    "stride_matrix",
    "stride_diagonal",
    "implied_window_step_diagonal",
    "stride_n_transitions",
    "run_length_persistence",
    "effective_n",
    "null_percentile",
    "confidence_intervals",
    "rolling_matrix",
    "rolling_bars",
    "drift_tvd",
    "ew_matrix",
    "ew_half_life",
    "shrunk_matrix",
    "shrinkage_alpha",
    "mixing_time",
    "mixing_epsilon",
    "tests",
    "statistics_params",
]

# Phase 3 addition, appended after ``PHASE2_FIELDS``: everything about the cost-aware
# backtest nests under this one key (``.claude/rules/json-contract.md``).
PHASE3_FIELDS = ["backtest"]

# Defaults mirror config/default.yaml (plan.md section 5): 20-day window, +/-5% threshold,
# one trading year of training before the walk-forward starts, simple returns.
DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.05
DEFAULT_MIN_TRAIN = 252
DEFAULT_RETURN_TYPE = "simple"

# Statistics defaults, mirroring config/default.yaml ``statistics``.
DEFAULT_NULL_PATHS = 1000  # statistics.null_paths
DEFAULT_BOOTSTRAP_RESAMPLES = 500  # statistics.bootstrap.resamples
DEFAULT_BLOCK_DAYS = 20  # statistics.bootstrap.block_days = labeller window
# statistics.homogeneity_periods: two halves of the ~125-label stride sequence keep enough
# transitions per row for the chi-square approximation
DEFAULT_HOMOGENEITY_PERIODS = 2
DEFAULT_SEED = 0  # statistics.seed
BOOTSTRAP_CI = 0.95  # conventional two-sided 95% interval (block_bootstrap_ci default)

# The companion requires at least one transition, i.e. two labelled days.
MIN_LABELS = 2

# Keys of the labeller-induced null that are carried over as bear/sideways/bull objects.
_LABELLER_VECTOR_KEYS = (
    "null_diagonal_mean",
    "null_diagonal_lower",
    "null_diagonal_upper",
    "diagonal_percentile",
    "excess_persistence",
)


def _finite_or_none(value: object) -> float | None:
    """``float(value)`` for a finite number, ``None`` for NaN / inf (never emit NaN)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _vector(values: np.ndarray) -> dict[str, float | None]:
    """Companion-style ``{"bear", "sideways", "bull"}`` object; NaN entries become ``None``."""
    arr = np.asarray(values, dtype=float)
    return {
        "bear": _finite_or_none(arr[BEAR]),
        "sideways": _finite_or_none(arr[SIDEWAYS]),
        "bull": _finite_or_none(arr[BULL]),
    }


def _matrix(values: np.ndarray) -> list[list[float]]:
    """3x3 nested list of floats, the companion's matrix layout."""
    return [[float(x) for x in row] for row in np.asarray(values, dtype=float)]


def _null_vector() -> dict[str, None]:
    """A bear/sideways/bull object of ``None`` for a statistic that could not be computed."""
    return {"bear": None, "sideways": None, "bull": None}


def _diagonal_or_null(matrix: np.ndarray) -> dict[str, float | None]:
    """Diagonal of a row-stochastic matrix as a bear/sideways/bull object, ``None`` for
    a state whose row is all zeros (never seen as a source: missing data, not 0
    persistence). Same convention as the ``_diagonal_or_nan`` summaries in
    ``hedgefund.regime.stats`` that feed the null blocks."""
    arr = np.asarray(matrix, dtype=float)
    diag = np.diag(arr).astype(float).copy()
    diag[arr.sum(axis=1) == 0] = np.nan
    return _vector(diag)


def _stride_block(labels_arr: np.ndarray, matrix: np.ndarray, window: int) -> dict:
    """Fix 1: stride matrix, its diagonal, the implied ``P**window`` diagonal, run lengths.

    The stride matrix is the ``window``-step matrix of the subsampled chain
    ``labels[::window]``. The comparable quantity from the overlapping fit is the diagonal
    of ``nstep_forecast(matrix, window)``; the gap between the overlapping 1-step diagonal
    and the stride diagonal is the window autocorrelation. Run-length persistence on the
    daily labels equals the overlapping diagonal algebraically, so it is computed on the
    same stride sequence.

    ``stride_diagonal`` is ``null`` for a state whose stride row is all zeros (the state
    never occurs as a source in the stride sequence), matching the null blocks; the
    ``stride_matrix`` itself keeps the all-zero row like every other matrix here.
    """
    stride_labels = labels_arr[::window]
    stride_counts = count_transitions(labels_arr, stride=window)
    stride_matrix = build_transition_matrix(labels_arr, stride=window)
    implied = nstep_forecast(matrix, window)
    has_stride = stride_labels.size >= MIN_LABELS
    return {
        "stride_matrix": _matrix(stride_matrix),
        "stride_diagonal": _diagonal_or_null(stride_matrix) if has_stride else _null_vector(),
        "implied_window_step_diagonal": _vector(np.diag(implied)),
        "stride_n_transitions": int(stride_counts.sum()),
        "run_length_persistence": _vector(run_length_persistence(stride_labels)),
    }


def _null_block(
    close: pd.Series,
    labels_arr: np.ndarray,
    *,
    window: int,
    threshold: float,
    return_type: str,
    n_paths: int,
    seed: int,
    bootstrap_resamples: int,
    block_days: int,
) -> dict:
    """Label-shuffle, iid, and labeller-induced nulls as JSON-ready objects.

    The labeller-induced excess interval uses the same block-bootstrap settings as
    ``confidence_intervals`` so the two sections agree on the real diagonal's error.
    """
    shuffle = label_shuffle_null(labels_arr, n_paths=n_paths, seed=seed)
    iid = iid_null(labels_arr, n_paths=n_paths, seed=seed)
    induced = labeller_null(
        close,
        window=window,
        threshold=threshold,
        return_type=return_type,
        n_paths=n_paths,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        block_days=block_days,
    )
    excess_ci = {
        key: [
            _finite_or_none(induced["excess_lower"][i]),
            _finite_or_none(induced["excess_upper"][i]),
        ]
        for key, i in (("bear", BEAR), ("sideways", SIDEWAYS), ("bull", BULL))
    }
    labeller_induced = {key: _vector(induced[key]) for key in _LABELLER_VECTOR_KEYS}
    labeller_induced["excess_ci"] = excess_ci
    labeller_induced["n_paths"] = int(induced["n_paths"])
    return {
        "label_shuffle": {
            "diagonal_percentile": _vector(shuffle["diagonal_percentile"]),
            "null_diagonal_mean": _vector(shuffle["null_diagonal_mean"]),
            "n_paths": int(shuffle["n_paths"]),
        },
        "iid": {
            "diagonal_percentile": _vector(iid["diagonal_percentile"]),
            "null_diagonal_mean": _vector(iid["null_diagonal_mean"]),
            "n_paths": int(iid["n_paths"]),
        },
        "labeller_induced": labeller_induced,
    }


def _confidence_block(
    labels_arr: np.ndarray, *, n_resamples: int, block_days: int, seed: int
) -> dict:
    """Block-bootstrap intervals for every matrix cell and the stationary vector."""
    ci = block_bootstrap_ci(
        labels_arr, n_resamples=n_resamples, block_days=block_days, ci=BOOTSTRAP_CI, seed=seed
    )
    return {
        "ci": float(ci["ci"]),
        "block_days": int(ci["block_days"]),
        "n_resamples": int(ci["n_resamples"]),
        "transition_matrix": {
            "lower": _matrix(ci["matrix_lower"]),
            "upper": _matrix(ci["matrix_upper"]),
        },
        "stationary_distribution": {
            "lower": _vector(ci["stationary_lower"]),
            "upper": _vector(ci["stationary_upper"]),
        },
    }


def _tests_block(labels_arr: np.ndarray, *, window: int, homogeneity_periods: int) -> dict:
    """Anderson-Goodman dependence and homogeneity tests, both on the stride sequence.

    Both chi-square tests assume the transitions they count are independent draws.
    The overlapping daily labels share ``window - 1`` bars with their neighbours, so
    feeding them to either test miscalibrates the p-value (the homogeneity test then
    rejects on roughly a third of iid price paths, flagging window autocorrelation as
    drift). The stride sequence ``labels[::window]`` removes the overlap.

    Either test is ``None`` when the stride sequence is too short for it (both raise
    ``ValueError`` in that case) so the companion fields still ship.
    """
    stride_labels = labels_arr[::window]
    anderson: dict | None = None
    if stride_labels.size >= MIN_LABELS:
        ag = anderson_goodman_test(stride_labels)
        anderson = {
            "statistic": _finite_or_none(ag["statistic"]),
            "p_value": _finite_or_none(ag["p_value"]),
            "df": int(ag["df"]),
            "n_transitions": int(ag["n_transitions"]),
            "note": "on the stride sequence",
        }
    homogeneity: dict | None = None
    try:
        hg = homogeneity_test(stride_labels, n_periods=homogeneity_periods)
    except ValueError:
        pass
    else:
        homogeneity = {
            "statistic": _finite_or_none(hg["statistic"]),
            "p_value": _finite_or_none(hg["p_value"]),
            "df": int(hg["df"]),
            "n_periods": int(hg["n_periods"]),
            "stable": bool(hg["stable"]),
            "note": "on the stride sequence",
        }
    return {"anderson_goodman": anderson, "homogeneity": homogeneity}


def honest_statistics(
    close: pd.Series,
    labels: pd.Series,
    matrix: np.ndarray,
    *,
    window: int,
    threshold: float,
    return_type: str,
    null_paths: int,
    bootstrap_resamples: int,
    block_days: int,
    rolling_bars: int,
    ew_half_life: float,
    shrinkage_alpha: float,
    mixing_epsilon: float,
    homogeneity_periods: int,
    seed: int,
) -> dict:
    """Compute the Phase 2 fields (``PHASE2_FIELDS``, in order) for a fitted matrix.

    ``labels`` must hold at least two entries (the caller checks). Fields that need a
    stride sequence of fewer than two labels, a stride sequence too short for the
    homogeneity test, or a state never seen as a source (stride diagonal, drift) come
    out as ``null`` instead of failing the whole analysis.
    """
    labels_arr = labels.to_numpy()
    stride = _stride_block(labels_arr, matrix, window)
    rolling_counts = rolling_transition_counts(labels_arr, window_bars=rolling_bars)
    rolling_matrix = rolling_transition_matrix(labels_arr, window_bars=rolling_bars)
    shrunk = dirichlet_shrinkage(rolling_counts, matrix, alpha=shrinkage_alpha)
    out: dict = {
        **stride,
        "effective_n": _vector(effective_sample_size(count_transitions(labels_arr), window)),
        "null_percentile": _null_block(
            close,
            labels_arr,
            window=window,
            threshold=threshold,
            return_type=return_type,
            n_paths=null_paths,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
            block_days=block_days,
        ),
        "confidence_intervals": _confidence_block(
            labels_arr, n_resamples=bootstrap_resamples, block_days=block_days, seed=seed
        ),
        "rolling_matrix": _matrix(rolling_matrix),
        "rolling_bars": int(rolling_bars),
        # NaN when no state is populated in both matrices (see ``drift_tvd``) -> null.
        "drift_tvd": _finite_or_none(drift_tvd(matrix, rolling_matrix)),
        "ew_matrix": _matrix(ew_transition_matrix(labels_arr, half_life=ew_half_life)),
        "ew_half_life": float(ew_half_life),
        "shrunk_matrix": _matrix(shrunk),
        "shrinkage_alpha": float(shrinkage_alpha),
        "mixing_time": mixing_time(matrix, epsilon=mixing_epsilon),
        "mixing_epsilon": float(mixing_epsilon),
        "tests": _tests_block(labels_arr, window=window, homogeneity_periods=homogeneity_periods),
        "statistics_params": {
            "null_paths": int(null_paths),
            "bootstrap_resamples": int(bootstrap_resamples),
            "block_days": int(block_days),
            "rolling_bars": int(rolling_bars),
            "ew_half_life": float(ew_half_life),
            "shrinkage_alpha": float(shrinkage_alpha),
            "mixing_epsilon": float(mixing_epsilon),
            "homogeneity_periods": int(homogeneity_periods),
            "seed": int(seed),
        },
    }
    return {key: out[key] for key in PHASE2_FIELDS}


def _labels_or_raise(
    close: pd.Series, *, window: int, threshold: float, return_type: str
) -> pd.Series:
    """Label the series; raise the companion's error when fewer than two days are labelled."""
    labels = label_regimes(close, window=window, threshold=threshold, return_type=return_type)
    if len(labels) < MIN_LABELS:
        raise ValueError(
            "Not enough data to label regimes — need more rows than the "
            f"rolling window ({window}). Got {len(close)} price rows."
        )
    return labels


def analyze(
    close: pd.Series,
    *,
    source: str,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_train: int = DEFAULT_MIN_TRAIN,
    hmm: bool = True,
    return_type: str = DEFAULT_RETURN_TYPE,
    null_paths: int = DEFAULT_NULL_PATHS,  # config/default.yaml statistics.null_paths
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,  # statistics.bootstrap.resamples
    block_days: int = DEFAULT_BLOCK_DAYS,  # statistics.bootstrap.block_days
    rolling_bars: int = DEFAULT_ROLLING_WINDOW_BARS,  # statistics.rolling_bars
    ew_half_life: float = DEFAULT_HALF_LIFE_BARS,  # statistics.ew_half_life
    shrinkage_alpha: float = DEFAULT_SHRINKAGE_ALPHA,  # statistics.shrinkage_alpha
    mixing_epsilon: float = DEFAULT_MIXING_EPSILON,  # statistics.mixing_epsilon
    # statistics.homogeneity_periods: two halves of the ~125-label stride sequence keep enough
    # transitions per row for the chi-square approximation
    homogeneity_periods: int = DEFAULT_HOMOGENEITY_PERIODS,
    seed: int = DEFAULT_SEED,  # statistics.seed
    refit: str = DEFAULT_REFIT,  # backtest.refit
    backtest_rolling_bars: int = DEFAULT_ROLLING_BARS,  # backtest.rolling_bars
    rule: str = DEFAULT_RULE,  # backtest.rule
    dead_band: float = DEFAULT_DEAD_BAND,  # backtest.dead_band
    cost_model: CostModel | None = None,  # costs.profiles[<universe cost_profile>]; None = us_etfs
    horizons: Sequence[int] = DEFAULT_HORIZONS,  # backtest.horizons
    holdout_years: int = DEFAULT_HOLDOUT_YEARS,  # backtest.holdout_years
    sensitivity_windows: Sequence[int] = DEFAULT_SENSITIVITY_WINDOWS,  # backtest.sensitivity
    sensitivity_thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
    cost_sensitivity_bps: Sequence[int] = SENSITIVITY_BPS,  # backtest.cost_sensitivity_bps
    lag_order: int = DEFAULT_LAG_ORDER,  # backtest.lag_order
) -> dict:
    """Run the whole framework: companion dict, Phase 2 statistics, Phase 3 ``backtest``.

    The companion keys come first and are computed exactly as the companion script does;
    the honest statistics (``PHASE2_FIELDS``) follow, then ``backtest`` (``PHASE3_FIELDS``).
    ``rolling_bars`` is the Phase 2 rolling-matrix window (``statistics.rolling_bars``);
    the walk-forward's rolling refit window is ``backtest_rolling_bars``
    (``backtest.rolling_bars``) because the two are separate config keys. The backtest
    block shares ``null_paths`` and ``seed`` with the statistics. Raises ``ValueError``
    when the series is too short to label at least two days.
    """
    close = close.dropna()
    labels = _labels_or_raise(close, window=window, threshold=threshold, return_type=return_type)

    matrix = build_transition_matrix(labels)
    pi = stationary_distribution(matrix)

    current_state = int(labels.iloc[-1])
    next_probs = next_state_probabilities(matrix, current_state)
    bear_p = float(next_probs[BEAR])
    side_p = float(next_probs[SIDEWAYS])
    bull_p = float(next_probs[BULL])

    bt = walk_forward_backtest(close, labels, min_train=min_train)

    companion = {
        "source": source,
        "rows": len(close),
        "date_start": str(close.index.min().date()),
        "date_end": str(close.index.max().date()),
        "params": {
            "window": window,
            "threshold": threshold,
            "min_train": min_train,
            "return_type": return_type,
        },
        "states": list(STATES),
        "current_regime": STATES[current_state],
        "next_state_probabilities": {
            "bear": bear_p,
            "sideways": side_p,
            "bull": bull_p,
        },
        "signal": signal_from_matrix(matrix, current_state),
        "transition_matrix": [[float(x) for x in row] for row in matrix],
        "persistence_diagonal": {
            "bear": float(matrix[BEAR, BEAR]),
            "sideways": float(matrix[SIDEWAYS, SIDEWAYS]),
            "bull": float(matrix[BULL, BULL]),
        },
        "stationary_distribution": {
            "bear": float(pi[BEAR]),
            "sideways": float(pi[SIDEWAYS]),
            "bull": float(pi[BULL]),
        },
        "walk_forward": {
            "sharpe": bt["sharpe"],
            "max_drawdown": bt["max_drawdown"],
            "n_trades": bt["n_trades"],
        },
        "hmm": hmm_summary(close, enabled=hmm),
        "framework": FRAMEWORK,
        "disclaimer": DISCLAIMER,
    }
    statistics = honest_statistics(
        close,
        labels,
        matrix,
        window=window,
        threshold=threshold,
        return_type=return_type,
        null_paths=null_paths,
        bootstrap_resamples=bootstrap_resamples,
        block_days=block_days,
        rolling_bars=rolling_bars,
        ew_half_life=ew_half_life,
        shrinkage_alpha=shrinkage_alpha,
        mixing_epsilon=mixing_epsilon,
        homogeneity_periods=homogeneity_periods,
        seed=seed,
    )
    backtest = backtest_block(
        close,
        labels,
        window=window,
        threshold=threshold,
        return_type=return_type,
        min_train=min_train,
        refit=refit,
        rolling_bars=backtest_rolling_bars,
        rule=rule,
        dead_band=dead_band,
        cost_model=cost_model,
        horizons=horizons,
        null_paths=null_paths,
        seed=seed,
        holdout_years=holdout_years,
        sensitivity_windows=sensitivity_windows,
        sensitivity_thresholds=sensitivity_thresholds,
        cost_sensitivity_bps=cost_sensitivity_bps,
        lag_order=lag_order,
    )
    return {**companion, **statistics, PHASE3_FIELDS[0]: backtest}


def backtest_only(
    close: pd.Series,
    *,
    source: str,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_train: int = DEFAULT_MIN_TRAIN,
    return_type: str = DEFAULT_RETURN_TYPE,
    null_paths: int = DEFAULT_NULL_PATHS,  # statistics.null_paths
    seed: int = DEFAULT_SEED,  # statistics.seed
    refit: str = DEFAULT_REFIT,  # backtest.refit
    rolling_bars: int = DEFAULT_ROLLING_BARS,  # backtest.rolling_bars
    rule: str = DEFAULT_RULE,  # backtest.rule
    dead_band: float = DEFAULT_DEAD_BAND,  # backtest.dead_band
    cost_model: CostModel | None = None,  # costs.profiles[<universe cost_profile>]; None = us_etfs
    horizons: Sequence[int] = DEFAULT_HORIZONS,  # backtest.horizons
    holdout_years: int = DEFAULT_HOLDOUT_YEARS,  # backtest.holdout_years
    sensitivity_windows: Sequence[int] = DEFAULT_SENSITIVITY_WINDOWS,  # backtest.sensitivity
    sensitivity_thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
    cost_sensitivity_bps: Sequence[int] = SENSITIVITY_BPS,  # backtest.cost_sensitivity_bps
    lag_order: int = DEFAULT_LAG_ORDER,  # backtest.lag_order
) -> dict:
    """The ``hedgefund backtest`` dict: header fields plus the ``backtest`` block only.

    Returns ``{"source", "rows", "date_start", "date_end", "params", "backtest",
    "framework", "disclaimer"}`` with the companion-style ``params`` object. No Phase 2
    statistics and no HMM fit, so it is the fast way to study costs and position rules.
    Raises ``ValueError`` when the series is too short to label at least two days.
    """
    close = close.dropna()
    labels = _labels_or_raise(close, window=window, threshold=threshold, return_type=return_type)
    backtest = backtest_block(
        close,
        labels,
        window=window,
        threshold=threshold,
        return_type=return_type,
        min_train=min_train,
        refit=refit,
        rolling_bars=rolling_bars,
        rule=rule,
        dead_band=dead_band,
        cost_model=cost_model,
        horizons=horizons,
        null_paths=null_paths,
        seed=seed,
        holdout_years=holdout_years,
        sensitivity_windows=sensitivity_windows,
        sensitivity_thresholds=sensitivity_thresholds,
        cost_sensitivity_bps=cost_sensitivity_bps,
        lag_order=lag_order,
    )
    return {
        "source": source,
        "rows": len(close),
        "date_start": str(close.index.min().date()),
        "date_end": str(close.index.max().date()),
        "params": {
            "window": window,
            "threshold": threshold,
            "min_train": min_train,
            "return_type": return_type,
        },
        "backtest": backtest,
        "framework": FRAMEWORK,
        "disclaimer": DISCLAIMER,
    }


def to_json(result: dict) -> str:
    """Serialise the contract dict exactly as the companion does.

    ``json.dumps`` emits ``NaN`` for ``walk_forward.sharpe`` / ``max_drawdown`` on short
    history, which the companion's consumers already accept. No other field may be NaN:
    the Phase 2 fields convert NaN to ``None`` (JSON ``null``) before reaching here.
    """
    return json.dumps(result)
