"""The ``hmm`` section of the analyze dict: walk-forward HMM versus the threshold labeller.

Phase 4 (plan.md Fix 3 and Fix 5). The block keeps the companion's keys first
(``available``, ``regimes``, ``caveat``) and appends: the feature set, the winning fit,
the last bar's forward-filtered probabilities, the hysteresis decision, the walk-forward
summary, an agreement table against the threshold labeller, a "which detector flips
first" report and the net and gross P&L of the sign rule on each detector's held regime.
Degrade paths return ``{"available": False, "reason": ...}`` exactly as the companion
does, so the contract never changes shape for a consumer that only reads the companion
keys; a bad parameter raises ``ValueError`` instead of degrading.

``agreement``, ``first_to_flip`` and ``pnl_comparison`` are pure functions on integer
label Series (STATES order) with no I/O; ``hmm_block`` is the orchestration that runs
the walk-forward and assembles the dict. Everything emitted is JSON-ready: builtins
only, ``None`` for NaN.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from hedgefund.backtest.benchmarks import buy_and_hold
from hedgefund.backtest.costs import CostModel, apply_costs
from hedgefund.backtest.hmm_walk_forward import (
    DEFAULT_HMM_MIN_TRAIN,
    DEFAULT_REFIT_EVERY,
    DEFAULT_TRAIN_BARS,
    hmm_walk_forward_from_close,
)
from hedgefund.backtest.metrics import TRADING_DAYS_PER_YEAR, metric_set
from hedgefund.regime.hmm import (
    DEFAULT_N_ITER,
    DEFAULT_SEEDS,
    HMM_CAVEAT,
    REASON_DISABLED,
    fit_summary,
)
from hedgefund.regime.hmm_features import DEFAULT_VOL_LONG, DEFAULT_VOL_SHORT
from hedgefund.regime.hmm_filter import DEFAULT_CONFIRM_BARS, DEFAULT_ENTER, DEFAULT_EXIT
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, STATES

# config hmm.match_window: a flip of one detector is matched to the other's flip within
# +/-20 bars, the labeller window (plan.md Fix 3: compare which detector flips first).
DEFAULT_MATCH_WINDOW = 20

STATE_KEYS = ("bear", "sideways", "bull")  # JSON keys in STATES order
FLIP_TARGETS = (("bear", BEAR), ("bull", BULL))  # entries INTO Bear and INTO Bull

AGREEMENT_NOTE = (
    "rows = threshold labeller, cols = walk-forward HMM (filtered + hysteresis); common dates only"
)
FIRST_TO_FLIP_NOTE = (
    "positive lead = the HMM flipped first; unmatched entries are flips the other "
    "detector never made within the window"
)
STANDARDISATION_NOTE = "training-window statistics only (plan Fix 5)"
DECISION_RULE = (
    "hysteresis on the forward-filtered probability; smoothed and Viterbi paths are "
    "never used (rules/no-lookahead.md)"
)
WALK_FORWARD_NOTE = (
    "walk-forward: every refit uses bars strictly before the decision date; "
    "probabilities are filtered, not smoothed"
)
PNL_RULE = (
    "sign rule on each detector's held regime at t, earning t+1, same cost model for "
    "both; this is a direct regime-position comparison, not the Phase 3 matrix-signal "
    "walk-forward"
)
PNL_NOTE = (
    "net figures sit beside gross with the cost assumption stated (rules/no-lookahead.md); "
    "historical, not forward-looking"
)
# Sign rule on a held regime label: Bear -> short, Sideways -> flat, Bull -> long.
_POSITION_BY_STATE = np.zeros(len(STATES))
_POSITION_BY_STATE[BEAR] = -1.0
_POSITION_BY_STATE[SIDEWAYS] = 0.0
_POSITION_BY_STATE[BULL] = 1.0
MIN_PNL_DAYS = 2  # one return is a number, not a statistic: below two days every metric is null
# Keys of ``HMMWalkForwardResult.summary()`` that the block surfaces elsewhere.
_SUMMARY_KEYS_ALREADY_SURFACED = frozenset(
    {"available", "reason", "last_filtered", "last_regime", "last_raw_regime"}
)


def _json_ready(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays to builtins and non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_json_ready(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _validate_labels(name: str, labels: pd.Series) -> None:
    if not isinstance(labels, pd.Series):
        raise ValueError(f"{name} must be a pandas Series of int labels, got {type(labels)}")
    if len(labels) == 0:
        return
    values = labels.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(
            f"{name} must hold integer labels in STATES order, got dtype {values.dtype}"
        )
    if values.min() < 0 or values.max() >= len(STATES):
        raise ValueError(
            f"{name} must hold labels in [0, {len(STATES) - 1}], got "
            f"[{values.min()}, {values.max()}]"
        )


def _align(threshold: pd.Series, hmm: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Int arrays of both label series on their common (inner-joined, sorted) index."""
    _validate_labels("threshold", threshold)
    _validate_labels("hmm", hmm)
    common = threshold.index.intersection(hmm.index).sort_values()
    return (
        threshold.loc[common].to_numpy(dtype=int),
        hmm.loc[common].to_numpy(dtype=int),
    )


def agreement(threshold: pd.Series, hmm: pd.Series) -> dict:
    """Day-by-day agreement between the threshold labeller and the walk-forward HMM.

    Both Series hold integer labels in STATES order and are aligned on their common
    index (inner join). ``confusion`` rows are the threshold regime, columns the HMM
    regime. ``share_agree_by_regime`` is, per threshold regime, the share of those days
    on which the HMM says the same regime (``None`` when the threshold never says it).
    With no common dates ``n`` is 0, ``share_agree`` is ``None`` and the confusion
    matrix is all zeros.
    """
    t, h = _align(threshold, hmm)
    n = int(t.size)
    k = len(STATES)
    confusion = np.bincount(t * k + h, minlength=k * k).reshape(k, k) if n else np.zeros((k, k))
    confusion = confusion.astype(int)
    row_totals = confusion.sum(axis=1)
    by_regime = {
        key: (float(confusion[i, i] / row_totals[i]) if row_totals[i] else None)
        for i, key in enumerate(STATE_KEYS)
    }
    return {
        "n": n,
        "share_agree": float(np.trace(confusion) / n) if n else None,
        "confusion": confusion.tolist(),
        "share_agree_by_regime": by_regime,
        "note": AGREEMENT_NOTE,
    }


def _entries(labels: np.ndarray, target: int) -> np.ndarray:
    """Positions where ``labels`` enters ``target``: the bar is ``target`` and the previous
    bar is not. The first bar never counts (there is no previous bar to flip from)."""
    if labels.size < 2:
        return np.empty(0, dtype=int)
    is_target = labels == target
    return np.flatnonzero(is_target[1:] & ~is_target[:-1]) + 1


def _match_entries(
    threshold_pos: np.ndarray, hmm_pos: np.ndarray, match_window: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching in time order.

    Each threshold entry (ascending) pairs with the still-unmatched HMM entry nearest to
    it within ``match_window`` bars on either side (ties go to the earlier HMM entry).
    Taking the earliest candidate instead would hand a one-bar HMM flicker long before
    the threshold flip the match and overstate the HMM's lead. Returns the matched pairs
    ``(threshold_pos, hmm_pos)`` and the unmatched positions of each detector.
    """
    unmatched_hmm = list(int(p) for p in hmm_pos)
    pairs: list[tuple[int, int]] = []
    unmatched_threshold: list[int] = []
    for t_pos in (int(p) for p in threshold_pos):
        candidate = min(
            (h for h in unmatched_hmm if abs(h - t_pos) <= match_window),
            key=lambda h: (abs(h - t_pos), h),
            default=None,
        )
        if candidate is None:
            unmatched_threshold.append(t_pos)
            continue
        unmatched_hmm.remove(candidate)
        pairs.append((t_pos, candidate))
    return pairs, unmatched_threshold, unmatched_hmm


def first_to_flip(
    threshold: pd.Series, hmm: pd.Series, *, match_window: int = DEFAULT_MATCH_WINDOW
) -> dict:
    """Which detector flips first into Bear and into Bull, on the common index.

    An entry is a bar whose label is the target regime while the previous bar's is not.
    Entries are matched greedily in time order: each threshold entry pairs with the
    nearest unmatched HMM entry within ``match_window`` bars either side. The lead of a
    matched pair is ``threshold_pos - hmm_pos``: positive means the HMM flipped first.
    Entries with no partner inside the window are reported as unmatched, which is the
    honest way to count flips the other detector never made.
    """
    if match_window < 0:
        raise ValueError(f"match_window must be >= 0 bars, got {match_window}")
    t, h = _align(threshold, hmm)
    out: dict[str, Any] = {}
    for key, target in FLIP_TARGETS:
        t_pos = _entries(t, target)
        h_pos = _entries(h, target)
        pairs, t_unmatched, h_unmatched = _match_entries(t_pos, h_pos, match_window)
        leads = [t_at - h_at for t_at, h_at in pairs]
        out[key] = {
            "n_threshold_entries": int(t_pos.size),
            "n_hmm_entries": int(h_pos.size),
            "n_matched": len(pairs),
            "hmm_first": sum(1 for lead in leads if lead > 0),
            "threshold_first": sum(1 for lead in leads if lead < 0),
            "same_bar": sum(1 for lead in leads if lead == 0),
            "mean_lead_bars": float(np.mean(leads)) if leads else None,
            "median_lead_bars": float(np.median(leads)) if leads else None,
            "leads": leads,
            "hmm_unmatched": len(h_unmatched),
            "threshold_unmatched": len(t_unmatched),
        }
    out["match_window"] = int(match_window)
    out["note"] = FIRST_TO_FLIP_NOTE
    return out


def _null_metric_set() -> dict:
    """Every ``metric_set`` key as ``None`` with ``n_days`` 0: the too-short answer."""
    keys = metric_set(np.zeros(MIN_PNL_DAYS), np.zeros(MIN_PNL_DAYS))
    return {**{key: None for key in keys}, "n_days": 0}


def _costs_echo(model: CostModel) -> dict:
    return {
        "fee_bps": float(model.fee_bps),
        "slippage_bps": float(model.slippage_bps),
        "borrow_bps": float(model.borrow_bps),
        "funding_bps": float(model.funding_bps),
        "per_side_bps": float(model.per_side_bps),
    }


def pnl_comparison(
    close: pd.Series,
    threshold_labels: pd.Series,
    hmm_regimes: pd.Series,
    *,
    cost_model: CostModel | None = None,  # costs.profiles[<universe cost_profile>]; None = us_etfs
    periods_per_year: int = TRADING_DAYS_PER_YEAR,  # daily bars on the US equity calendar
) -> dict:
    """Net and gross P&L of the sign rule on each detector's held regime (plan Fix 5).

    On the dates both label Series share and for which ``close`` has the bar and the
    next bar, each detector holds Bull -> +1, Bear -> -1, Sideways -> 0 at ``t`` and
    earns the close-to-close simple return of ``t + 1`` (so the last common date drops
    out). Gross is ``position x asset return``; net applies ``cost_model`` through
    ``apply_costs`` to both detectors alike. Buy-and-hold is charged the same way as in
    the Phase 3 block (a constant +1 position pays the entry cost and funding only).
    This is a direct regime-position comparison, not the Phase 3 matrix-signal
    walk-forward. With fewer than ``MIN_PNL_DAYS`` common dates every metric is ``None``
    and ``n_days`` is 0; nothing here is ever NaN.
    """
    model = CostModel() if cost_model is None else cost_model
    _validate_labels("threshold_labels", threshold_labels)
    _validate_labels("hmm_regimes", hmm_regimes)
    if not isinstance(close, pd.Series):
        raise ValueError(f"close must be a pandas Series of prices, got {type(close)}")
    prices = close.dropna().astype(float)
    # Return earned by a position held at t: close[t+1] / close[t] - 1 on the next bar.
    next_return = prices.pct_change().shift(-1).dropna()
    common = (
        threshold_labels.index.intersection(hmm_regimes.index)
        .intersection(next_return.index)
        .sort_values()
    )
    n_days = len(common)
    costs = _costs_echo(model)
    if n_days < MIN_PNL_DAYS:
        null = _null_metric_set()
        return {
            "threshold": {"gross": dict(null), "net": dict(null)},
            "hmm": {"gross": dict(null), "net": dict(null)},
            "buy_and_hold": {"net": dict(null)},
            "costs": costs,
            "n_days": 0,
            "first_date": None,
            "last_date": None,
            "rule": PNL_RULE,
            "note": PNL_NOTE,
        }
    asset = next_return.loc[common].to_numpy(dtype=float)
    detectors: dict[str, dict] = {}
    for name, labels in (("threshold", threshold_labels), ("hmm", hmm_regimes)):
        positions = _POSITION_BY_STATE[labels.loc[common].to_numpy(dtype=int)]
        gross = positions * asset
        net = apply_costs(gross, positions, model, periods_per_year)
        detectors[name] = {
            "gross": metric_set(gross, positions, periods_per_year),
            "net": metric_set(net, positions, periods_per_year),
        }
    ones = np.ones_like(asset)
    bh_net = apply_costs(buy_and_hold(asset), ones, model, periods_per_year)
    return _json_ready(
        {
            **detectors,
            "buy_and_hold": {"net": metric_set(bh_net, ones, periods_per_year)},
            "costs": costs,
            "n_days": n_days,
            "first_date": common[0],
            "last_date": common[-1],
            "rule": PNL_RULE,
            "note": PNL_NOTE,
        }
    )


def _state_vector(values: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {key: _json_ready(arr[i]) for i, key in enumerate(STATE_KEYS)}


def hmm_block(
    close: pd.Series,
    threshold_labels: pd.Series,
    *,
    enabled: bool = True,
    seeds: Sequence[int] = DEFAULT_SEEDS,  # config hmm.seeds
    n_iter: int = DEFAULT_N_ITER,  # hmm.n_iter
    emission: str = "gaussian",  # hmm.emission
    enter: float = DEFAULT_ENTER,  # hmm.enter
    exit: float = DEFAULT_EXIT,  # hmm.exit
    confirm_bars: int = DEFAULT_CONFIRM_BARS,  # hmm.confirm_bars
    min_train: int = DEFAULT_HMM_MIN_TRAIN,  # backtest.min_train, shared with the backtest
    refit_every: int = DEFAULT_REFIT_EVERY,  # hmm.refit_every
    train_bars: int = DEFAULT_TRAIN_BARS,  # hmm.train_bars
    vol_short: int = DEFAULT_VOL_SHORT,  # hmm.vol_short
    vol_long: int = DEFAULT_VOL_LONG,  # hmm.vol_long
    match_window: int = DEFAULT_MATCH_WINDOW,  # hmm.match_window
    cost_model: CostModel | None = None,  # costs.profiles[<universe cost_profile>]; None = us_etfs
) -> dict:
    """The ``hmm`` value of the analyze dict.

    Companion keys first and unchanged in meaning: ``available``, then either ``reason``
    (disabled, hmmlearn missing, history too short, or ``"hmm runtime error: ..."`` for
    a non-``ValueError`` failure of the walk-forward) or ``regimes`` and ``caveat``. A
    ``ValueError`` (a bad parameter such as ``refit_every`` 0 or ``enter <= exit``)
    propagates so the CLI fails the command like it does for every other bad flag. The
    Phase 4 keys follow in a fixed order: ``features``, ``fit``,
    ``filtered_probabilities``, ``decision``, ``walk_forward``,
    ``agreement_with_threshold``, ``first_to_flip``, ``pnl_comparison``, ``note``.
    ``regimes`` and ``fit`` describe the last refit, the one that scores the final bar;
    ``filtered_probabilities`` and ``decision`` are the final bar's.
    """
    if not enabled:
        return {"available": False, "reason": REASON_DISABLED}
    try:
        result = hmm_walk_forward_from_close(
            close,
            vol_short=vol_short,
            vol_long=vol_long,
            min_train=min_train,
            refit_every=refit_every,
            train_bars=train_bars,
            seeds=tuple(int(s) for s in seeds),
            n_iter=n_iter,
            emission=emission,
            enter=enter,
            exit=exit,
            confirm_bars=confirm_bars,
        )
    except ValueError:
        raise  # a bad parameter is the caller's error, not a degrade path
    except Exception as exc:
        return {"available": False, "reason": f"hmm runtime error: {exc}"}
    if not result.available:
        return {"available": False, "reason": str(result.reason)}

    last_fit, last_scaler = result.fits[-1], result.scalers[-1]
    fit = fit_summary(last_fit, last_scaler.mean, last_scaler.std)
    regimes = fit.pop("regimes")
    summary = result.summary()
    walk_forward = {
        key: value for key, value in summary.items() if key not in _SUMMARY_KEYS_ALREADY_SURFACED
    }
    last_probs = result.filtered.iloc[-1].to_numpy(dtype=float)
    block = {
        "available": True,
        "regimes": regimes,
        "caveat": HMM_CAVEAT,
        "features": {
            "names": list(last_scaler.columns),
            "vol_short": int(vol_short),
            "vol_long": int(vol_long),
            "standardisation": STANDARDISATION_NOTE,
        },
        "fit": fit,
        "filtered_probabilities": _state_vector(last_probs),
        "decision": {
            "regime": STATES[int(result.regimes.iloc[-1])],
            "raw_regime": STATES[int(result.raw_regimes.iloc[-1])],
            "enter": float(enter),
            "exit": float(exit),
            "confirm_bars": int(confirm_bars),
            "rule": DECISION_RULE,
        },
        "walk_forward": walk_forward,
        "agreement_with_threshold": agreement(threshold_labels, result.regimes),
        "first_to_flip": first_to_flip(threshold_labels, result.regimes, match_window=match_window),
        "pnl_comparison": pnl_comparison(
            close, threshold_labels, result.regimes, cost_model=cost_model
        ),
        "note": WALK_FORWARD_NOTE,
    }
    return _json_ready(block)
