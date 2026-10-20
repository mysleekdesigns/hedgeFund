"""Optional Gaussian Hidden Markov Model: the Phase 1 port plus the Phase 4 fitter.

Phase 1 (``fit_hmm``, ``hmm_summary``) is a faithful port of ``fit_hmm`` and
``_hmm_summary`` from the companion ``markov_regime.py``: a single Baum-Welch fit on
one-dimensional returns with diagonal covariance. It is kept unchanged so the reproduction
test still matches to six decimals.

Phase 4 (``fit_hmm_features`` and friends) implements plan Fix 5 on an already-standardised
feature matrix (column 0 is the daily log return, the rest realised volatility and friends):

- full covariance, so returns and volatility are modelled jointly and the model cannot
  degenerate into clustering on volatility alone;
- several Baum-Welch restarts with explicit seeds, keeping the best log-likelihood;
- latent states permuted by ascending mean return so index 0 is Bear, 1 Sideways, 2 Bull
  across fits;
- a "crash state" diagnostic that flags a state holding a sliver of bars in one-bar
  episodes, and an evaluation of Student-t emissions at the Gaussian solution when it does.
  The Student-t is compared at matched covariance (scale = (df - 2) / df times the Gaussian
  covariance), so ``prefer_student_t`` reflects tail shape alone, not a wider spread.

The fitted parameters are returned as a ``hedgefund.regime.hmm_filter.HMMParams`` so the
online forward filter, not hmmlearn's smoother, produces the live signal. A state whose
fitted covariance is not positive definite (hmmlearn leaves a rank-deficient matrix behind
when a state holds a single outlier bar) gets hmmlearn's own ``min_covar`` added to its
diagonal, and ``HMMFit.floored_states`` records which states that touched.

``hmmlearn`` is an optional extra. It is imported lazily inside the fitting functions so this
module imports cleanly without it; ``fit_hmm`` returns ``(None, None)`` and
``fit_hmm_features`` returns ``None`` in that case.
"""

import math
import warnings
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from hedgefund.regime.hmm_filter import MIN_STUDENT_T_DF, HMMParams, forward_filter
from hedgefund.regime.labels import STATES

# The companion labels latent states by ascending mean return: worst = Bear, best = Bull.
# STATES is already ordered ["Bear", "Sideways", "Bull"], so rank -> label is a direct index.
RANK_LABELS: list[str] = list(STATES)

# Companion defaults, kept identical so the Phase 1 reproduction test matches.
DEFAULT_N_COMPONENTS = 3  # the three regimes: Bear / Sideways / Bull
DEFAULT_RANDOM_STATE = 42  # companion default seed for Baum-Welch initialisation
DEFAULT_N_ITER = 200  # companion default Baum-Welch iteration cap
COVARIANCE_TYPE = "diag"  # companion choice; Phase 4 moves to "full" on a feature matrix

HMM_CAVEAT = (
    "HMM states are labelled by ascending mean return, so a positive "
    "'Bear' mean just means the worst latent state was still net-"
    "positive over this window. Baum-Welch finds local maxima; for "
    "production fit several random_state values."
)
REASON_DISABLED = "disabled via --no-hmm"
REASON_UNAVAILABLE = "hmmlearn not installed or failed to compile"

# ---------------------------------------------------------------------------
# Phase 4 defaults (config ``hmm`` section).
# ---------------------------------------------------------------------------
# Three Baum-Welch restarts: on the Phase 1 synthetic mixture ~40% of seeds reach the
# global optimum, so three independent seeds reach it with probability ~1 - 0.6**3 = 78%.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
FULL_COVARIANCE = "full"  # plan Fix 5: returns and volatility together need a full covariance
# Degrees-of-freedom grid for the Student-t evaluation: 3 is about as heavy as daily equity
# returns get, 30 is indistinguishable from Gaussian, 5 and 10 bracket the usual estimates.
DEFAULT_T_DFS: tuple[float, ...] = (3.0, 5.0, 10.0, 30.0)
DEFAULT_MAX_OCCUPANCY = 0.02  # a state holding fewer than 2% of bars is suspect ...
DEFAULT_MAX_MEAN_RUN = (
    1.5  # ... and when its episodes average under 1.5 bars it is a one-bar "crash" state
)
EMISSION_CHOICES = ("gaussian", "student_t", "auto")
# hmmlearn's full-covariance M-step does not floor the covariance, so a state that captures
# a single bar comes back rank-deficient (every entry equal to covars_prior). Such a state
# gets ``model.min_covar`` (hmmlearn's own init floor, 1e-3) added to its diagonal so the
# forward filter has a usable matrix. A covariance counts as not positive definite when its
# Cholesky factorisation fails or its smallest eigenvalue is at or below this tolerance,
# which is float64 round-off scaled to unit-variance standardised features.
MIN_COVAR_EIGENVALUE = 1e-10
RETURN_FEATURE = (
    "log_return"  # column 0 must be the return so ordering by mean return is meaningful
)
STUDENT_T_NOTE = (
    "same transition matrix, means and covariance; only the emission family changes "
    "(covariance-matched: scale = (df-2)/df * sigma)"
)


def fit_hmm(
    returns: pd.Series,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any | None, np.ndarray | None]:
    """Fit a Gaussian HMM on daily returns. Returns ``(model, hidden_states)``.

    Returns ``(None, None)`` if ``hmmlearn`` cannot be imported. The import is wrapped in
    a broad ``except Exception`` on purpose: a compiled-extension failure raises something
    other than ``ImportError`` and must degrade the same way.

    Baum-Welch finds local maxima. Phase 4 adds multi-restart selection; Phase 1 keeps the
    single companion fit so outputs reproduce exactly.

    Raises ``ValueError`` if ``returns`` has fewer than ``n_components`` non-NaN rows,
    which is the smallest sample on which a fit is even defined.
    """
    try:
        from hmmlearn import hmm
    except Exception:
        return None, None

    observations = returns.dropna().to_numpy(dtype=float).reshape(-1, 1)
    if observations.shape[0] < n_components:
        raise ValueError(
            f"returns has {observations.shape[0]} non-NaN rows; need at least "
            f"n_components={n_components} rows to fit a {n_components}-state HMM"
        )
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=COVARIANCE_TYPE,
        n_iter=DEFAULT_N_ITER,
        random_state=random_state,
    )
    model.fit(observations)
    hidden_states = model.predict(observations)
    return model, hidden_states


def order_states_by_mean(model: Any) -> np.ndarray:
    """Return latent-state indices sorted by ascending mean of the first feature.

    ``model.means_`` has shape ``(n_components, n_features)``; column 0 is the daily return
    in Phase 1 and stays the return column in the Phase 4 feature matrix, so this helper is
    reused there to map latent states onto Bear / Sideways / Bull.
    """
    means = np.asarray(model.means_, dtype=float)[:, 0]
    return np.argsort(means)


def hmm_summary(
    close: pd.Series,
    enabled: bool = True,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Build the HMM section of the analyze() dict, degrading gracefully.

    This is a full-sample Baum-Welch fit and is diagnostic only: no position in Phase 1
    derives from it, so it is not a no-lookahead estimator. Phase 4 replaces it with a
    walk-forward fit and forward-filtered probabilities before any signal uses it.

    Output matches the companion ``_hmm_summary`` field for field: either
    ``{"available": False, "reason": ...}`` or
    ``{"available": True, "regimes": [...], "caveat": ...}`` with regimes ordered by
    ascending ``mean_daily_return`` and labelled Bear / Sideways / Bull.
    """
    if not enabled:
        return {"available": False, "reason": REASON_DISABLED}
    try:
        model, _ = fit_hmm(
            close.pct_change().dropna(),
            n_components=n_components,
            random_state=random_state,
        )
    except Exception as exc:
        return {"available": False, "reason": f"hmm runtime error: {exc}"}
    if model is None:
        return {"available": False, "reason": REASON_UNAVAILABLE}

    means = np.asarray(model.means_, dtype=float)[:, 0]
    order = order_states_by_mean(model)
    regimes = [
        {
            "label": RANK_LABELS[rank],
            "latent_state": int(k),
            "mean_daily_return": float(means[k]),
        }
        for rank, k in enumerate(order)
    ]
    return {"available": True, "regimes": regimes, "caveat": HMM_CAVEAT}


# ---------------------------------------------------------------------------
# Phase 4: multi-restart full-covariance fit on a feature matrix.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HMMFit:
    """Result of ``fit_hmm_features``.

    ``params`` already has its states permuted so index 0 is the lowest mean of column 0
    (Bear), 1 Sideways, 2 Bull. ``log_likelihood`` is the winning restart's Gaussian
    log-likelihood on the training matrix as hmmlearn scored it, before any Student-t switch
    and before any covariance floor; ``floored_states`` lists the (ordered) states whose
    covariance had to be floored, so when it is non-empty the forward filter on ``params``
    gives a slightly different likelihood. ``n_iter`` is the number of Baum-Welch
    iterations the winner actually ran (``model.monitor_.iter``).
    """

    params: HMMParams
    log_likelihood: float
    seed: int
    seeds: tuple[int, ...]
    converged: bool
    n_iter: int
    emission: str
    feature_names: tuple[str, ...]
    restart_log_likelihoods: dict[int, float]
    crash_state: dict
    student_t: dict | None
    floored_states: tuple[int, ...] = ()  # ordered-state indices whose covariance was floored


def _validate_feature_matrix(
    x: np.ndarray,
    feature_names: Sequence[str],
    n_components: int,
    seeds: Sequence[int],
    emission: str,
) -> None:
    """Raise ``ValueError`` naming the offending input; never coerce."""
    if not isinstance(x, np.ndarray) or x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", type(x).__name__)
        raise ValueError(
            f"x must be a 2-D float ndarray of shape (n_rows, n_features); got shape {shape} "
            f"with dtype {dtype}"
        )
    if np.isnan(x).any():
        raise ValueError(
            f"x contains {int(np.isnan(x).sum())} NaN values; drop the warm-up rows before fitting"
        )
    if x.shape[0] < n_components:
        raise ValueError(
            f"x has {x.shape[0]} rows; need at least n_components={n_components} rows to fit a "
            f"{n_components}-state HMM"
        )
    if x.shape[1] < 2:
        raise ValueError(
            f"x has {x.shape[1]} feature column; fitting on rolling volatility alone is not "
            "allowed (plan Fix 5): pass the return together with at least one volatility feature"
        )
    if len(feature_names) != x.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} names but x has {x.shape[1]} columns"
        )
    if feature_names[0] != RETURN_FEATURE:
        raise ValueError(
            f"feature_names[0] must be {RETURN_FEATURE!r} so states can be ordered by mean "
            f"return; got {feature_names[0]!r}"
        )
    if len(seeds) == 0:
        raise ValueError("seeds is empty; pass at least one Baum-Welch random_state")
    if emission not in EMISSION_CHOICES:
        raise ValueError(f"emission must be one of {EMISSION_CHOICES}; got {emission!r}")


def _single_threaded_blas() -> AbstractContextManager[Any]:
    """Pin BLAS / OpenMP pools to one thread so repeated fits are bit-identical.

    ``threadpoolctl`` ships with scikit-learn, which hmmlearn requires, so it is present
    whenever a fit can run at all; fall back to a no-op if it is somehow missing.
    """
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def _floor_singular_covars(
    covars: np.ndarray, floor: float, *, min_eigenvalue: float = MIN_COVAR_EIGENVALUE
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Add ``floor * I`` to each state covariance that is not positive definite.

    A covariance needs the floor when its Cholesky factorisation fails or its smallest
    eigenvalue is at or below ``min_eigenvalue``. Healthy states are returned bit-identical.
    Returns the covariances and the indices of the states that were floored.
    """
    out = np.array(covars, dtype=float, copy=True)
    n_features = out.shape[-1]
    floored: list[int] = []
    for k in range(out.shape[0]):
        try:
            np.linalg.cholesky(out[k])
            needs_floor = np.linalg.eigvalsh(out[k]).min() <= min_eigenvalue
        except np.linalg.LinAlgError:
            needs_floor = True
        if needs_floor:
            out[k] = out[k] + floor * np.eye(n_features)
            floored.append(k)
    return out, tuple(floored)


def _params_from_model(model: Any) -> tuple[HMMParams, tuple[int, ...]]:
    """Permute a fitted hmmlearn model into an ``HMMParams`` ordered by mean return.

    The permutation is applied to startprob, both axes of transmat, means and covars, so
    the model is unchanged up to relabelling (its log-likelihood is identical). A state
    covariance that is not positive definite gets ``model.min_covar`` times the identity
    added; the returned tuple lists those states in the ordered index space (empty for a
    healthy fit, which stays bit-identical to hmmlearn's parameters).
    """
    order = order_states_by_mean(model)
    covars, floored = _floor_singular_covars(
        np.asarray(model.covars_, dtype=float)[order], float(model.min_covar)
    )
    params = HMMParams(
        startprob=np.asarray(model.startprob_, dtype=float)[order].copy(),
        transmat=np.asarray(model.transmat_, dtype=float)[np.ix_(order, order)].copy(),
        means=np.asarray(model.means_, dtype=float)[order].copy(),
        covars=covars,
        emission="gaussian",
        df=None,
    )
    return params, floored


def crash_state_diagnostic(
    params: HMMParams,
    x: np.ndarray,
    *,
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
    max_mean_run: float = DEFAULT_MAX_MEAN_RUN,
) -> dict:
    """Flag a state that only fires on isolated outlier bars (plan Fix 5).

    Runs the forward filter, takes the argmax path, and for each state computes the share of
    bars it occupies and the mean length of its episodes. A state is a "crash" state when it
    holds fewer than ``max_occupancy`` of the bars and its episodes average fewer than
    ``max_mean_run`` bars; a state that never appears (occupancy 0, run ``None``) is flagged
    too, because a three-state model with an empty state is just as degenerate.

    The argmax of the filtered probabilities is used, not Viterbi, so the diagnostic
    reflects the path the live signal would see.
    """
    filtered, _ = forward_filter(params, x)
    path = np.argmax(filtered, axis=1)
    n_states = params.transmat.shape[0]
    n_rows = path.shape[0]
    occupancy = np.bincount(path, minlength=n_states) / n_rows

    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_rows]])
    run_states = path[starts]
    run_lengths = ends - starts
    mean_runs: list[float | None] = []
    for k in range(n_states):
        lengths = run_lengths[run_states == k]
        mean_runs.append(float(lengths.mean()) if lengths.size else None)

    suspects = [
        k
        for k in range(n_states)
        if occupancy[k] < max_occupancy and (mean_runs[k] is None or mean_runs[k] < max_mean_run)
    ]
    state = min(suspects, key=lambda k: occupancy[k]) if suspects else None
    return {
        "flagged": bool(suspects),
        "state": None if state is None else int(state),
        "occupancy": [float(v) for v in occupancy],
        "mean_run_length": mean_runs,
        "max_occupancy": float(max_occupancy),
        "max_mean_run": float(max_mean_run),
    }


def student_t_scale(covars: np.ndarray, df: float) -> np.ndarray:
    """Scale matrices giving a Student-t with ``df`` the same covariance as ``covars``.

    A multivariate t with scale ``S`` has covariance ``df / (df - 2) * S``, so matching
    the Gaussian covariance ``sigma`` needs ``S = (df - 2) / df * sigma``. Requires
    ``df > 2`` (``MIN_STUDENT_T_DF``), where the covariance is finite.
    """
    if not (df > MIN_STUDENT_T_DF):
        raise ValueError(
            f"Student-t degrees of freedom must exceed {MIN_STUDENT_T_DF} (finite "
            f"covariance); got {df!r}"
        )
    return np.asarray(covars, dtype=float) * ((df - MIN_STUDENT_T_DF) / df)


def _student_t_params(params: HMMParams, df: float) -> HMMParams:
    """The covariance-matched Student-t emission with ``params``' Gaussian covariance."""
    return replace(
        params, emission="student_t", df=float(df), covars=student_t_scale(params.covars, df)
    )


def evaluate_student_t(
    params: HMMParams,
    x: np.ndarray,
    *,
    dfs: Sequence[float] = DEFAULT_T_DFS,
) -> dict:
    """Compare Gaussian and Student-t emissions at the fitted Gaussian solution.

    For each degrees-of-freedom value the same start probabilities, transition matrix and
    means are reused with a Student-t emission whose scale is ``(df - 2) / df`` times the
    Gaussian covariance, so the t has the *same covariance* as the Gaussian and the
    forward-filter log-likelihoods differ by tail shape only. Without that rescaling a t
    built on the raw covariance is ``df / (df - 2)`` times wider, and the comparison mixes
    heavier tails with a wider spread (it then prefers the t on plain Gaussian data).

    This is an evaluation of the emission family at the Gaussian optimum, which is what the
    Phase 4 checklist asks for; a full Student-t EM (re-estimating location and scale under
    the t likelihood) is out of scope. ``params`` must carry Gaussian covariances: pass the
    Gaussian fit, not an already-switched Student-t ``HMMParams``.

    ``prefer_student_t`` is True when the best covariance-matched Student-t log-likelihood
    beats the Gaussian one on the same data.
    """
    if len(dfs) == 0:
        raise ValueError("dfs is empty; pass at least one degrees-of-freedom value")
    if params.emission != "gaussian":
        raise ValueError(
            "evaluate_student_t expects the Gaussian fit (its covars are the covariance); "
            f"got emission={params.emission!r} whose covars are already a t scale matrix"
        )
    _, gaussian_ll = forward_filter(params, x)
    by_df: dict[str, float] = {}
    for df in dfs:
        _, ll = forward_filter(_student_t_params(params, float(df)), x)
        by_df[str(float(df))] = float(ll)
    best_key = max(by_df, key=by_df.__getitem__)
    best_ll = by_df[best_key]
    return {
        "gaussian_log_likelihood": float(gaussian_ll),
        "by_df": by_df,
        "best_df": float(best_key),
        "student_t_log_likelihood": best_ll,
        "prefer_student_t": bool(best_ll > gaussian_ll),
        "scale": "covariance-matched: scale = (df-2)/df * sigma",
        "note": STUDENT_T_NOTE,
    }


def fit_hmm_features(
    x: np.ndarray,
    *,
    feature_names: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITER,
    emission: str = "gaussian",
    t_dfs: Sequence[float] = DEFAULT_T_DFS,
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
    max_mean_run: float = DEFAULT_MAX_MEAN_RUN,
) -> HMMFit | None:
    """Multi-restart, full-covariance Gaussian HMM on a standardised feature matrix.

    ``x`` is ``(n_rows, n_features)`` with column 0 the daily log return, already
    standardised with training-window statistics (see ``hmm_features``). One Baum-Welch fit
    runs per seed with ``covariance_type="full"``; the restart with the highest
    log-likelihood wins. A restart that raises or scores non-finite is recorded as NaN in
    ``restart_log_likelihoods`` and skipped; ``RuntimeError`` is raised if every restart fails.

    States are permuted by ascending mean return so labels are stable across fits. Then:

    - ``emission="gaussian"``: run the crash-state diagnostic only.
    - ``emission="student_t"``: always evaluate the Student-t family and return params with
      the best-df Student-t emission.
    - ``emission="auto"``: evaluate only when the crash diagnostic flags a state, and switch
      to Student-t only when it has the higher log-likelihood.

    The evaluation compares emission families at matched covariance (see
    ``evaluate_student_t``), and a switched ``params`` carries the covariance-matched scale
    ``(df - 2) / df`` times the Gaussian covariance, so the filter keeps the Gaussian
    fit's spread and only the tails change. ``HMMFit.log_likelihood`` stays the Gaussian
    figure.

    Returns ``None`` when hmmlearn cannot be imported (broad ``except`` on purpose, like
    ``fit_hmm``). Raises ``ValueError`` on malformed input; in particular a single-column
    ``x`` is rejected because fitting on rolling volatility alone is forbidden (plan Fix 5).

    Fits run with BLAS pinned to one thread so repeated fits with the same seeds give
    bit-identical parameters. No global random state is touched; every restart is seeded
    explicitly.
    """
    _validate_feature_matrix(x, feature_names, n_components, seeds, emission)
    try:
        from hmmlearn import hmm
    except Exception:
        return None

    seeds_tuple = tuple(int(s) for s in seeds)
    restart_lls: dict[int, float] = {}
    best_model: Any = None
    best_params: HMMParams | None = None
    best_floored: tuple[int, ...] = ()
    best_seed: int | None = None
    best_ll = -math.inf
    with warnings.catch_warnings(), _single_threaded_blas():
        warnings.filterwarnings("ignore", module="hmmlearn.*")
        for seed in seeds_tuple:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=FULL_COVARIANCE,
                    n_iter=n_iter,
                    random_state=seed,
                )
                model.fit(x)
                score = float(model.score(x))
                # Built here so a restart whose covariances HMMParams rejects even after the
                # singular floor counts as a failed restart rather than escaping the loop.
                candidate, floored = _params_from_model(model)
            except Exception:
                score = math.nan
            if not math.isfinite(score):
                restart_lls[seed] = math.nan
                continue
            restart_lls[seed] = score
            if score > best_ll:
                best_ll, best_model, best_params, best_seed = score, model, candidate, seed
                best_floored = floored
    if best_model is None or best_params is None or best_seed is None:
        raise RuntimeError(f"every Baum-Welch restart failed for seeds {seeds_tuple}")

    params = best_params
    crash = crash_state_diagnostic(
        params, x, max_occupancy=max_occupancy, max_mean_run=max_mean_run
    )
    student_t: dict | None = None
    if emission == "student_t" or (emission == "auto" and crash["flagged"]):
        student_t = evaluate_student_t(params, x, dfs=t_dfs)
        if emission == "student_t" or student_t["prefer_student_t"]:
            params = _student_t_params(params, student_t["best_df"])

    return HMMFit(
        params=params,
        log_likelihood=best_ll,
        seed=best_seed,
        seeds=seeds_tuple,
        converged=bool(best_model.monitor_.converged),
        n_iter=int(best_model.monitor_.iter),
        emission=params.emission,
        feature_names=tuple(str(name) for name in feature_names),
        restart_log_likelihoods=restart_lls,
        crash_state=crash,
        student_t=student_t,
        floored_states=best_floored,
    )


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays to builtins and non-finite floats to None."""
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


def fit_summary(
    fit: HMMFit,
    scaler_mean: np.ndarray | None = None,
    scaler_std: np.ndarray | None = None,
) -> dict:
    """JSON-ready description of a fit for the report: builtins only, NaN rendered as None.

    ``regimes`` lists the states in order (Bear, Sideways, Bull) with each state's mean of
    column 0. When ``scaler_mean`` and ``scaler_std`` are given that mean is de-standardised
    back to a daily return (``mean * std[0] + mean[0]``); otherwise the raw standardised
    mean is reported.
    """
    means = np.asarray(fit.params.means, dtype=float)[:, 0]
    if (scaler_mean is None) != (scaler_std is None):
        raise ValueError("scaler_mean and scaler_std must be given together or both omitted")
    if scaler_mean is not None and scaler_std is not None:
        mean0 = float(np.asarray(scaler_mean, dtype=float).reshape(-1)[0])
        std0 = float(np.asarray(scaler_std, dtype=float).reshape(-1)[0])
        means = means * std0 + mean0
    regimes = [
        {
            "label": RANK_LABELS[k] if k < len(RANK_LABELS) else f"state_{k}",
            "latent_state": int(k),
            "mean_daily_return": float(means[k]),
        }
        for k in range(means.shape[0])
    ]
    return _jsonify(
        {
            "seed": fit.seed,
            "seeds": list(fit.seeds),
            "log_likelihood": fit.log_likelihood,
            "restart_log_likelihoods": {str(k): v for k, v in fit.restart_log_likelihoods.items()},
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "emission": fit.emission,
            "df": fit.params.df,
            "covariance_type": FULL_COVARIANCE,
            "feature_names": list(fit.feature_names),
            "regimes": regimes,
            "transition_matrix": np.asarray(fit.params.transmat, dtype=float).tolist(),
            "crash_state": fit.crash_state,
            "student_t": fit.student_t,
            "floored_states": list(fit.floored_states),
        }
    )
