"""Rolling-return regime labeller (Bull / Bear / Sideways).

Ported from the companion script's ``label_regimes`` so that the simple-return
variant is bit-identical to it on the same input. Pure: a pandas ``Series`` in, a
pandas ``Series`` out, no I/O.

The label on day ``t`` uses ``close[t]`` and ``close[t - window]`` only, so there is
no lookahead: perturbing prices after ``t`` cannot change labels at or before ``t``.
"""

import numpy as np
import pandas as pd

# Canonical state order used by every matrix and vector in the repo (config/default.yaml).
STATES: list[str] = ["Bear", "Sideways", "Bull"]
BEAR, SIDEWAYS, BULL = 0, 1, 2

RETURN_TYPES: tuple[str, ...] = ("simple", "log")


def label_regimes(
    close: pd.Series,
    window: int = 20,  # config/default.yaml labelling.window: 20 trading days of trailing return
    threshold: float = 0.05,  # config/default.yaml labelling.threshold: +/-5% trailing return
    return_type: str = "simple",  # config/default.yaml labelling.return_type: companion uses simple
) -> pd.Series:
    """Label each day from its trailing ``window``-day return.

    Bull (2)     : rolling return >  +threshold   (strict)
    Bear (0)     : rolling return <  -threshold   (strict)
    Sideways (1) : otherwise, including a return of exactly +/-threshold

    ``return_type`` selects the rolling return: ``"simple"`` is
    ``close.pct_change(window)`` (the companion script), ``"log"`` is
    ``log(close / close.shift(window))`` (the Pine indicator).

    Returns an int ``Series`` indexed like ``close`` minus the first ``window`` rows,
    which are dropped because their rolling return is undefined.

    Raises ``ValueError`` for an unknown ``return_type``, ``window < 1``,
    ``threshold <= 0``, a non-``DatetimeIndex``, or an empty series.
    """
    if return_type not in RETURN_TYPES:
        raise ValueError(f"return_type must be one of {RETURN_TYPES}, got {return_type!r}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(f"close must have a DatetimeIndex, got {type(close.index).__name__}")
    if close.empty:
        raise ValueError("close is empty; need at least window + 1 rows to label any day")

    if return_type == "simple":
        rolling_return = close.pct_change(window)
    else:
        rolling_return = np.log(close / close.shift(window))

    labels = pd.Series(SIDEWAYS, index=close.index, dtype=int)
    labels[rolling_return > threshold] = BULL
    labels[rolling_return < -threshold] = BEAR
    return labels.loc[rolling_return.notna()]
