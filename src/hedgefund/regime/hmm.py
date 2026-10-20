"""Optional Gaussian Hidden Markov Model fit on daily returns.

Phase 1 is a faithful port of ``fit_hmm`` and ``_hmm_summary`` from the companion
``markov_regime.py``: a single Baum-Welch fit on one-dimensional returns with diagonal
covariance. Phase 4 will switch to full covariance on a feature matrix (returns,
realised volatility, trend) and add multi-restart selection by log-likelihood.

``hmmlearn`` is an optional extra. It is imported lazily inside ``fit_hmm`` so this module
imports cleanly without it, and the summary degrades to ``{"available": False, ...}``.
"""

from typing import Any

import numpy as np
import pandas as pd

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
