"""Pure-numpy HMM inference: online forward filter, smoother, Viterbi, hysteresis.

This module holds the inference side of the Phase 4 HMM (plan.md Fix 3 and Fix 5). It
depends on numpy and scipy only, never on ``hmmlearn``, so the live signal path works
without the optional extra. Fitting lives in ``hedgefund.regime.hmm``.

Three kinds of state probability are computed here and must never be confused:

- ``forward_filter`` / ``filter_step`` give P(s_t | x_0..x_t), the causal, no-lookahead
  probability. This is the only quantity a signal may use.
- ``smooth`` gives P(s_t | x_0..x_{n-1}) and ``viterbi`` the single most likely path.
  Both read the whole sample, including bars after ``t``. DIAGNOSTIC ONLY.

``hysteresis_decision`` turns the filtered probabilities into a held regime with enter
and exit thresholds and a confirmation count so a single noisy bar cannot flip the
signal. Every function here is pure: numpy in, numpy out, no I/O, no globals.

State order everywhere is ``STATES = ["Bear", "Sideways", "Bull"]`` (index 0, 1, 2).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.special import gammaln, logsumexp

from hedgefund.regime.labels import SIDEWAYS

DEFAULT_ENTER = 0.6  # plan Fix 5: enter a regime when its filtered probability exceeds 0.6
DEFAULT_EXIT = 0.4  # plan Fix 5: leave it when the probability drops below 0.4
DEFAULT_CONFIRM_BARS = 2  # plan Fix 5: 2 to 3 bar confirmation; 2 keeps the lag short
EMISSIONS: tuple[str, ...] = ("gaussian", "student_t")
LOG_EPS = 1e-300  # floor before taking logs of probabilities
SIMPLEX_ATOL = 1e-8  # tolerance for "sums to 1"; float64 round-off is ~1e-16 per term
MIN_STUDENT_T_DF = 2.0  # the multivariate t has a finite covariance only for df > 2


def _as_float_array(value: object, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array, got {type(value).__name__}") from exc


@dataclass(frozen=True)
class HMMParams:
    """Parameters of a K-state HMM with multivariate Gaussian or Student-t emissions.

    ``startprob`` (K,) and ``transmat`` (K, K) are probability simplexes; ``means``
    (K, d); ``covars`` (K, d, d) symmetric positive definite. For ``"student_t"``
    emissions ``covars`` is the *scale* matrix of the multivariate t (its covariance
    is ``df / (df - 2)`` times that) and ``df`` must be ``> 2``. For ``"gaussian"``
    emissions ``df`` must be ``None``.

    ``validate`` runs in ``__post_init__`` so a constructed instance is always valid.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    emission: str = "gaussian"
    df: float | None = None

    def __post_init__(self) -> None:
        # Normalise the array fields to float64 ndarrays so every consumer sees one dtype.
        object.__setattr__(self, "startprob", _as_float_array(self.startprob, "startprob"))
        object.__setattr__(self, "transmat", _as_float_array(self.transmat, "transmat"))
        object.__setattr__(self, "means", _as_float_array(self.means, "means"))
        object.__setattr__(self, "covars", _as_float_array(self.covars, "covars"))
        self.validate()

    @property
    def n_states(self) -> int:
        return int(self.startprob.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def validate(self) -> None:
        """Raise ``ValueError`` on any shape, simplex, positive-definiteness or df violation."""
        if self.startprob.ndim != 1 or self.startprob.shape[0] < 1:
            raise ValueError(f"startprob must have shape (K,), got {self.startprob.shape}")
        k = self.startprob.shape[0]
        if self.transmat.shape != (k, k):
            raise ValueError(f"transmat must have shape ({k}, {k}), got {self.transmat.shape}")
        if self.means.ndim != 2 or self.means.shape[0] != k or self.means.shape[1] < 1:
            raise ValueError(f"means must have shape ({k}, d) with d >= 1, got {self.means.shape}")
        d = self.means.shape[1]
        if self.covars.shape != (k, d, d):
            raise ValueError(f"covars must have shape ({k}, {d}, {d}), got {self.covars.shape}")

        for name, arr in (("startprob", self.startprob), ("transmat", self.transmat)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative, got min {arr.min()}")
        if not np.isclose(self.startprob.sum(), 1.0, atol=SIMPLEX_ATOL, rtol=0):
            raise ValueError(f"startprob must sum to 1, got {self.startprob.sum()}")
        row_sums = self.transmat.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=SIMPLEX_ATOL, rtol=0):
            raise ValueError(f"every transmat row must sum to 1, got row sums {row_sums}")

        if not np.all(np.isfinite(self.means)):
            raise ValueError("means must be finite")
        if not np.all(np.isfinite(self.covars)):
            raise ValueError("covars must be finite")
        for i in range(k):
            cov = self.covars[i]
            if not np.allclose(cov, cov.T, atol=SIMPLEX_ATOL, rtol=0):
                raise ValueError(f"covars[{i}] must be symmetric")
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"covars[{i}] must be positive definite") from exc

        if self.emission not in EMISSIONS:
            raise ValueError(f"emission must be one of {EMISSIONS}, got {self.emission!r}")
        if self.emission == "gaussian":
            if self.df is not None:
                raise ValueError(f"df must be None for gaussian emissions, got {self.df}")
        else:
            if self.df is None or not np.isfinite(self.df) or self.df <= MIN_STUDENT_T_DF:
                raise ValueError(
                    f"df must be a finite number > {MIN_STUDENT_T_DF} for student_t "
                    f"emissions, got {self.df}"
                )


def _mahalanobis_and_logdet(cov: np.ndarray, centred: np.ndarray) -> tuple[np.ndarray, float]:
    """Squared Mahalanobis distance of each row of ``centred`` (n, d) and ``log det cov``."""
    chol, lower = cho_factor(cov, lower=True)
    z = solve_triangular(chol, centred.T, lower=lower)  # (d, n), cov = L L^T so z = L^-1 x
    maha = np.einsum("dn,dn->n", z, z)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return maha, float(log_det)


def log_emission(params: HMMParams, x: np.ndarray) -> np.ndarray:
    """Log emission density of each row of ``x`` under each state.

    ``x`` has shape (n, d) or (d,) for a single observation. Returns shape (n, K), or
    (K,) for a single observation. Gaussian: multivariate normal log pdf with full
    covariance via a Cholesky solve. Student-t: multivariate t log pdf with location
    ``means[k]``, scale matrix ``covars[k]`` and ``df`` degrees of freedom, written out
    with ``gammaln`` so it does not depend on the scipy version.

    Raises ``ValueError`` when the last axis of ``x`` is not ``params.n_features`` or
    ``x`` has more than two dimensions.
    """
    x = _as_float_array(x, "x")
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]
    d = params.n_features
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(f"x must have shape (n, {d}) or ({d},), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")

    n, k = x.shape[0], params.n_states
    out = np.empty((n, k))
    # Loop over the handful of states, not over the series; each state is vectorised over n.
    for i in range(k):
        maha, log_det = _mahalanobis_and_logdet(params.covars[i], x - params.means[i])
        if params.emission == "gaussian":
            out[:, i] = -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)
        else:
            nu = float(params.df)  # validate() guarantees df is a finite number > 2 here
            out[:, i] = (
                gammaln((nu + d) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * log_det
                - 0.5 * (nu + d) * np.log1p(maha / nu)
            )
    return out[0] if single else out


def _log_floor(p: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(p, LOG_EPS))


def _posterior_from_log(log_prior: np.ndarray, log_lik: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise ``prior * likelihood`` in log space; returns (posterior, log normaliser)."""
    log_joint = log_prior + log_lik
    log_norm = float(logsumexp(log_joint))
    return np.exp(log_joint - log_norm), log_norm


def _validate_filtered_row(prev_filtered: np.ndarray, k: int) -> np.ndarray:
    prev_filtered = _as_float_array(prev_filtered, "prev_filtered")
    if prev_filtered.shape != (k,):
        raise ValueError(f"prev_filtered must have shape ({k},), got {prev_filtered.shape}")
    if not np.all(np.isfinite(prev_filtered)):
        raise ValueError("prev_filtered must be finite")
    return prev_filtered


def initial_filtered(params: HMMParams, x_0: np.ndarray) -> tuple[np.ndarray, float]:
    """Filtered probability for the first bar: ``posterior ∝ startprob * emission(x_0)``.

    Returns ``(posterior (K,), log P(x_0))``.
    """
    log_lik = log_emission(params, x_0)
    if log_lik.ndim != 1:
        raise ValueError(f"x_0 must be a single observation of shape ({params.n_features},)")
    return _posterior_from_log(_log_floor(params.startprob), log_lik)


def filter_step(
    params: HMMParams, prev_filtered: np.ndarray, x_t: np.ndarray
) -> tuple[np.ndarray, float]:
    """One online bar of the forward filter.

    ``predicted = prev_filtered @ transmat`` is P(s_t | x_<t); the posterior
    P(s_t | x_<=t) is ``predicted * emission(x_t)`` normalised in log space with the
    log-sum-exp trick. Returns ``(posterior (K,), log P(x_t | x_<t))``. The first bar
    uses ``initial_filtered`` instead, which starts from ``startprob``.
    """
    prev_filtered = _validate_filtered_row(prev_filtered, params.n_states)
    log_lik = log_emission(params, x_t)
    if log_lik.ndim != 1:
        raise ValueError(f"x_t must be a single observation of shape ({params.n_features},)")
    predicted = prev_filtered @ params.transmat
    return _posterior_from_log(_log_floor(predicted), log_lik)


def _validate_sequence(params: HMMParams, x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x, "x")
    if x.ndim != 2 or x.shape[1] != params.n_features:
        raise ValueError(f"x must have shape (n, {params.n_features}), got {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("x must have at least one row")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


def forward_filter(params: HMMParams, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Causal forward filter over a whole sample: ``filtered[t] = P(s_t | x_0..x_t)``.

    Returns ``(filtered (n, K), log_likelihood)`` where the log-likelihood is the sum
    of the incremental ``log P(x_t | x_<t)`` terms. Implemented as ``initial_filtered``
    followed by repeated ``filter_step`` so a batch caller and an online caller agree
    bit for bit. The recursion is sequential by construction, so this is the one
    allowed Python loop over the series (``.claude/rules/python.md``). Row ``t`` depends
    on rows ``<= t`` only: no lookahead.
    """
    x = _validate_sequence(params, x)
    n = x.shape[0]
    filtered = np.empty((n, params.n_states))
    filtered[0], log_likelihood = initial_filtered(params, x[0])
    for t in range(1, n):
        filtered[t], increment = filter_step(params, filtered[t - 1], x[t])
        log_likelihood += increment
    return filtered, float(log_likelihood)


def _log_forward_lattice(params: HMMParams, log_lik: np.ndarray) -> np.ndarray:
    """Unnormalised log forward variables ``log P(s_t, x_0..x_t)`` (n, K)."""
    n, k = log_lik.shape
    log_trans = _log_floor(params.transmat)
    log_alpha = np.empty((n, k))
    log_alpha[0] = _log_floor(params.startprob) + log_lik[0]
    for t in range(1, n):
        # (K_from, 1) + (K_from, K_to) summed over the "from" axis.
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, np.newaxis] + log_trans, axis=0) + log_lik[t]
    return log_alpha


def _log_backward_lattice(params: HMMParams, log_lik: np.ndarray) -> np.ndarray:
    """Log backward variables ``log P(x_{t+1}..x_{n-1} | s_t)`` (n, K)."""
    n, k = log_lik.shape
    log_trans = _log_floor(params.transmat)
    log_beta = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        # (K_from, K_to) + (1, K_to) summed over the "to" axis.
        log_beta[t] = logsumexp(
            log_trans + (log_lik[t + 1] + log_beta[t + 1])[np.newaxis, :], axis=1
        )
    return log_beta


def smooth(params: HMMParams, x: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC ONLY: uses future observations; never use this in a signal.

    Returns the smoothed probabilities ``P(s_t | x_0..x_{n-1})`` with shape (n, K) via
    the forward-backward algorithm in log space. Row ``t`` reads every observation,
    including those after ``t`` (``.claude/rules/no-lookahead.md``). The last row equals
    the last filtered row by definition.
    """
    x = _validate_sequence(params, x)
    log_lik = log_emission(params, x)
    log_alpha = _log_forward_lattice(params, log_lik)
    log_beta = _log_backward_lattice(params, log_lik)
    log_joint = log_alpha + log_beta
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def viterbi(params: HMMParams, x: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC ONLY: uses future observations; never use this in a signal.

    Returns the single most likely state path (n,) as int indices in ``STATES`` order,
    found by the Viterbi algorithm in log space. The whole path is chosen jointly from
    the whole sample, so the state at ``t`` depends on observations after ``t``
    (``.claude/rules/no-lookahead.md``). Ties pick the lowest state index.
    """
    x = _validate_sequence(params, x)
    log_lik = log_emission(params, x)
    n, k = log_lik.shape
    log_trans = _log_floor(params.transmat)
    score = np.empty((n, k))
    back = np.zeros((n, k), dtype=int)
    score[0] = _log_floor(params.startprob) + log_lik[0]
    for t in range(1, n):
        candidates = score[t - 1][:, np.newaxis] + log_trans  # (K_from, K_to)
        back[t] = np.argmax(candidates, axis=0)
        score[t] = candidates[back[t], np.arange(k)] + log_lik[t]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@dataclass(frozen=True)
class HysteresisState:
    """State carried between bars by ``hysteresis_step``.

    ``regime`` is the regime currently held; ``enter_runs[c]`` counts consecutive bars
    with ``P(c) > enter``; ``exit_run`` counts consecutive bars with ``P(regime) < exit``.
    """

    regime: int
    enter_runs: np.ndarray
    exit_run: int


def initial_hysteresis_state(n_states: int, initial_regime: int = SIDEWAYS) -> HysteresisState:
    """Neutral starting state: hold ``initial_regime`` with every counter at zero."""
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    if not 0 <= initial_regime < n_states:
        raise ValueError(f"initial_regime must be in [0, {n_states}), got {initial_regime}")
    return HysteresisState(
        regime=int(initial_regime), enter_runs=np.zeros(n_states, dtype=int), exit_run=0
    )


def _validate_hysteresis_params(
    enter: float, exit: float, confirm_bars: int, initial_regime: int, n_states: int
) -> None:
    if not 0.0 < exit < enter < 1.0:
        raise ValueError(f"need 0 < exit < enter < 1, got exit={exit}, enter={enter}")
    if confirm_bars < 1:
        raise ValueError(f"confirm_bars must be >= 1, got {confirm_bars}")
    if n_states < 2:
        raise ValueError(f"need at least 2 states, got {n_states}")
    if not 0 <= initial_regime < n_states:
        raise ValueError(f"initial_regime must be in [0, {n_states}), got {initial_regime}")


def _validate_probs_row(probs_t: np.ndarray, k: int) -> np.ndarray:
    probs_t = _as_float_array(probs_t, "probs_t")
    if probs_t.shape != (k,):
        raise ValueError(f"probs_t must have shape ({k},), got {probs_t.shape}")
    if not np.all(np.isfinite(probs_t)):
        raise ValueError("probs_t must be finite")
    if np.any(probs_t < 0.0) or np.any(probs_t > 1.0):
        raise ValueError(f"probs_t must lie in [0, 1], got {probs_t}")
    return probs_t


def hysteresis_step(
    state: HysteresisState,
    probs_t: np.ndarray,
    *,
    enter: float = DEFAULT_ENTER,
    exit: float = DEFAULT_EXIT,
    confirm_bars: int = DEFAULT_CONFIRM_BARS,
    initial_regime: int = SIDEWAYS,
) -> HysteresisState:
    """Advance the hysteresis rule by one bar of filtered probabilities ``probs_t`` (K,).

    1. ``enter_runs[c]`` becomes ``enter_runs[c] + 1`` where ``probs_t[c] > enter``, else 0.
    2. If some ``c != regime`` has ``enter_runs[c] >= confirm_bars``, switch to that
       ``c`` (ties: highest ``probs_t[c]``, then lowest index) and reset every counter.
    3. Else, if ``regime != initial_regime`` and ``probs_t[regime] < exit`` has held for
       ``confirm_bars`` consecutive bars, fall back to ``initial_regime`` and reset.
    4. Else hold ``regime`` with the updated counters.

    Only ``probs_t`` and the carried ``state`` are read, so the output at bar ``t``
    depends on bars ``<= t`` only. Raises ``ValueError`` on bad thresholds, counts,
    regime index, or a non-finite / out-of-range probability row.
    """
    k = state.enter_runs.shape[0]
    _validate_hysteresis_params(enter, exit, confirm_bars, initial_regime, k)
    probs_t = _validate_probs_row(probs_t, k)
    if not 0 <= state.regime < k:
        raise ValueError(f"state.regime must be in [0, {k}), got {state.regime}")

    enter_runs = np.where(probs_t > enter, state.enter_runs + 1, 0).astype(int)
    confirmed = enter_runs >= confirm_bars
    confirmed[state.regime] = False
    if confirmed.any():
        # Highest probability among confirmed candidates; np.argmax takes the lowest index on a tie.
        masked = np.where(confirmed, probs_t, -np.inf)
        new_regime = int(np.argmax(masked))
        return HysteresisState(regime=new_regime, enter_runs=np.zeros(k, dtype=int), exit_run=0)

    exit_run = 0
    if state.regime != initial_regime and probs_t[state.regime] < exit:
        exit_run = state.exit_run + 1
    if state.regime != initial_regime and exit_run >= confirm_bars:
        return HysteresisState(
            regime=int(initial_regime), enter_runs=np.zeros(k, dtype=int), exit_run=0
        )
    return HysteresisState(regime=state.regime, enter_runs=enter_runs, exit_run=exit_run)


def hysteresis_decision(
    filtered: np.ndarray,
    *,
    enter: float = DEFAULT_ENTER,
    exit: float = DEFAULT_EXIT,
    confirm_bars: int = DEFAULT_CONFIRM_BARS,
    initial_regime: int = SIDEWAYS,
) -> np.ndarray:
    """Regime held at each bar from filtered probabilities (plan.md Fix 5).

    ``filtered`` has shape (n, K) with rows of probabilities; ``K >= 2``. Returns an int
    array (n,) of state indices in ``STATES`` order. Starts in ``initial_regime``
    (Sideways by default: the neutral state, no hindsight) and iterates
    ``hysteresis_step`` bar by bar, so the batch and the online caller agree bit for
    bit. The sequential state machine is the one allowed loop over the series
    (``.claude/rules/python.md``); the decision at ``t`` depends on rows ``<= t`` only.

    Raises ``ValueError`` unless ``0 < exit < enter < 1``, ``confirm_bars >= 1``,
    ``0 <= initial_regime < K``, ``filtered`` is 2-D with ``K >= 2``, and every row is
    finite and within ``[0, 1]``.
    """
    filtered = _as_float_array(filtered, "filtered")
    if filtered.ndim != 2 or filtered.shape[1] < 2:
        raise ValueError(f"filtered must have shape (n, K) with K >= 2, got {filtered.shape}")
    n, k = filtered.shape
    _validate_hysteresis_params(enter, exit, confirm_bars, initial_regime, k)
    if not np.all(np.isfinite(filtered)):
        raise ValueError("filtered must be finite")
    if np.any(filtered < 0.0) or np.any(filtered > 1.0):
        raise ValueError("filtered must lie in [0, 1]")

    state = initial_hysteresis_state(k, initial_regime)
    regimes = np.empty(n, dtype=int)
    for t in range(n):
        state = hysteresis_step(
            state,
            filtered[t],
            enter=enter,
            exit=exit,
            confirm_bars=confirm_bars,
            initial_regime=initial_regime,
        )
        regimes[t] = state.regime
    return regimes
