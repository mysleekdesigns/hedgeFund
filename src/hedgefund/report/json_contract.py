"""The companion skill's JSON contract: ``analyze()`` returns exactly its dict.

Field names, types, nesting and the ``[Bear, Sideways, Bull]`` index order are frozen
(``.claude/rules/json-contract.md``). The one Phase 1 addition is the additive
``params.return_type``. Every value is computed the same way as the companion script
(``float()`` casts, ``str(date)``) so the reproduction test matches to 6 decimals.
"""

import json

import pandas as pd

from hedgefund.backtest.walk_forward import walk_forward_backtest
from hedgefund.regime.hmm import hmm_summary
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, STATES, label_regimes
from hedgefund.regime.markov import build_transition_matrix, stationary_distribution
from hedgefund.regime.signal import next_state_probabilities, signal_from_matrix

FRAMEWORK = "Roan (@RohOnChain)"
DISCLAIMER = "Backtests are historical, not forward-looking."

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

# Defaults mirror config/default.yaml (plan.md section 5): 20-day window, +/-5% threshold,
# one trading year of training before the walk-forward starts, simple returns.
DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.05
DEFAULT_MIN_TRAIN = 252
DEFAULT_RETURN_TYPE = "simple"

# The companion requires at least one transition, i.e. two labelled days.
MIN_LABELS = 2


def analyze(
    close: pd.Series,
    *,
    source: str,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_train: int = DEFAULT_MIN_TRAIN,
    hmm: bool = True,
    return_type: str = DEFAULT_RETURN_TYPE,
) -> dict:
    """Run the whole framework and return the companion's structured dict.

    Raises ``ValueError`` when the series is too short to label at least two days.
    """
    close = close.dropna()
    labels = label_regimes(close, window=window, threshold=threshold, return_type=return_type)
    if len(labels) < MIN_LABELS:
        raise ValueError(
            "Not enough data to label regimes — need more rows than the "
            f"rolling window ({window}). Got {len(close)} price rows."
        )

    matrix = build_transition_matrix(labels)
    pi = stationary_distribution(matrix)

    current_state = int(labels.iloc[-1])
    next_probs = next_state_probabilities(matrix, current_state)
    bear_p = float(next_probs[BEAR])
    side_p = float(next_probs[SIDEWAYS])
    bull_p = float(next_probs[BULL])

    bt = walk_forward_backtest(close, labels, min_train=min_train)

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


def to_json(result: dict) -> str:
    """Serialise the contract dict exactly as the companion does.

    ``json.dumps`` emits ``NaN`` for ``walk_forward.sharpe`` / ``max_drawdown`` on short
    history, which the companion's consumers already accept. No other field may be NaN.
    """
    return json.dumps(result)
