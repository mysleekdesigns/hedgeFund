"""The Phase 4 ``hmm`` block: agreement, first-to-flip and the graceful-degrade paths.

``agreement`` and ``first_to_flip`` are checked on hand-built label Series with known
answers (no fitting). ``hmm_block`` degrade paths run without hmmlearn or a fit; the one
available-path test needs hmmlearn and keeps the fit tiny (one seed, 50 iterations,
~500 rows) so it takes about a second.
"""

import json
import math
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import hedgefund.report.hmm_report as hmm_report
from hedgefund.backtest.costs import ZERO_COSTS, CostModel
from hedgefund.backtest.hmm_walk_forward import REASON_TOO_SHORT
from hedgefund.regime.hmm import HMM_CAVEAT, REASON_DISABLED, REASON_UNAVAILABLE
from hedgefund.regime.labels import BEAR, BULL, SIDEWAYS, STATES, label_regimes
from hedgefund.report.hmm_report import (
    DEFAULT_MATCH_WINDOW,
    agreement,
    first_to_flip,
    hmm_block,
    pnl_comparison,
)

SEED = 7
ROWS = 500  # ~500 rows: enough for a 100-bar first training window and a second refit
DAILY_VOL = 0.01
START_PRICE = 100.0
SMALL_MIN_TRAIN = 100
SMALL_REFIT = 100
TINY_SEEDS = (0,)
TINY_N_ITER = 50
WINDOW = 20
THRESHOLD = 0.05
PROB_TOL = 1e-9  # filtered probabilities are normalised in float64; sum to 1 to round-off

STATE_KEYS = ["bear", "sideways", "bull"]
COMPANION_HMM_KEYS = ["available", "regimes", "caveat"]
PHASE4_HMM_KEYS = [
    "features",
    "fit",
    "filtered_probabilities",
    "decision",
    "walk_forward",
    "agreement_with_threshold",
    "first_to_flip",
    "pnl_comparison",
    "note",
]
METRIC_KEYS = [  # the Phase 3 metric set (hedgefund.backtest.metrics.metric_set)
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "total_return",
    "annual_return",
    "hit_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "turnover",
    "time_in_market",
    "n_days",
]
FLIP_KEYS = [
    "n_threshold_entries",
    "n_hmm_entries",
    "n_matched",
    "hmm_first",
    "threshold_first",
    "same_bar",
    "mean_lead_bars",
    "median_lead_bars",
    "leads",
    "hmm_unmatched",
    "threshold_unmatched",
]


def _labels(values: list[int], start: int = 0) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=start + len(values))[start:]
    return pd.Series(values, index=index, dtype=int)


def _synthetic_close(rows: int = ROWS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


def _walk(node: object, path: str, visit: Callable[[object, str], None]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}", visit)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", visit)
    else:
        visit(node, path)


def _assert_no_nan(node: object, path: str) -> None:
    def visit(value: object, where: str) -> None:
        if isinstance(value, float):
            assert math.isfinite(value), where

    _walk(node, path, visit)


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #
def test_agreement_of_identical_series_is_one_with_a_diagonal_confusion() -> None:
    labels = _labels([BEAR, BEAR, SIDEWAYS, BULL, BULL, BULL])
    out = agreement(labels, labels.copy())
    assert out["n"] == 6
    assert out["share_agree"] == 1.0
    assert out["confusion"] == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    assert out["share_agree_by_regime"] == {"bear": 1.0, "sideways": 1.0, "bull": 1.0}


def test_agreement_confusion_rows_are_threshold_and_columns_are_hmm() -> None:
    threshold = _labels([BEAR, BEAR, SIDEWAYS, BULL])
    hmm = _labels([BULL, BEAR, SIDEWAYS, SIDEWAYS])
    out = agreement(threshold, hmm)
    # Bear->Bull once, Bear->Bear once, Sideways->Sideways once, Bull->Sideways once.
    assert out["confusion"] == [[1, 0, 1], [0, 1, 0], [0, 1, 0]]
    assert out["share_agree"] == pytest.approx(2 / 4)
    assert "rows = threshold labeller" in out["note"]


def test_agreement_by_regime_is_null_for_a_regime_the_threshold_never_says() -> None:
    threshold = _labels([SIDEWAYS, SIDEWAYS, BULL, BULL])
    hmm = _labels([SIDEWAYS, BEAR, BULL, SIDEWAYS])
    out = agreement(threshold, hmm)
    assert out["share_agree_by_regime"] == {"bear": None, "sideways": 0.5, "bull": 0.5}


def test_agreement_uses_the_common_index_only() -> None:
    threshold = _labels([BEAR, BEAR, BEAR, BEAR])  # days 0..3
    hmm = _labels([BEAR, BULL, BULL, BULL], start=2)  # days 2..5
    out = agreement(threshold, hmm)
    assert out["n"] == 2
    assert out["confusion"] == [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert out["share_agree"] == 0.5


def test_agreement_on_disjoint_indexes_is_empty_with_zero_confusion() -> None:
    threshold = _labels([BEAR, BULL])
    hmm = _labels([BEAR, BULL], start=10)
    out = agreement(threshold, hmm)
    assert out["n"] == 0
    assert out["share_agree"] is None
    assert out["confusion"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert out["share_agree_by_regime"] == {"bear": None, "sideways": None, "bull": None}


def test_agreement_rejects_labels_outside_states() -> None:
    with pytest.raises(ValueError, match="threshold"):
        agreement(_labels([0, 3]), _labels([0, 1]))


# --------------------------------------------------------------------------- #
# first_to_flip
# --------------------------------------------------------------------------- #
def test_first_to_flip_counts_the_hmm_leading_by_three_bars() -> None:
    s, b = SIDEWAYS, BEAR
    hmm = _labels([s, s, b, b, b, b, b, b, s, s])  # enters Bear at bar 2
    threshold = _labels([s, s, s, s, s, b, b, b, s, s])  # enters Bear at bar 5
    bear = first_to_flip(threshold, hmm)["bear"]
    assert bear["n_threshold_entries"] == 1 and bear["n_hmm_entries"] == 1
    assert bear["n_matched"] == 1
    assert (bear["hmm_first"], bear["threshold_first"], bear["same_bar"]) == (1, 0, 0)
    assert bear["mean_lead_bars"] == 3.0 and bear["median_lead_bars"] == 3.0
    assert bear["leads"] == [3]
    assert bear["hmm_unmatched"] == 0 and bear["threshold_unmatched"] == 0


def test_first_to_flip_counts_the_threshold_leading_with_a_negative_lead() -> None:
    s, u = SIDEWAYS, BULL
    threshold = _labels([s, u, u, u, u, u, s, s])  # enters Bull at bar 1
    hmm = _labels([s, s, s, u, u, u, s, s])  # enters Bull at bar 3
    bull = first_to_flip(threshold, hmm)["bull"]
    assert (bull["hmm_first"], bull["threshold_first"], bull["same_bar"]) == (0, 1, 0)
    assert bull["leads"] == [-2]
    assert bull["mean_lead_bars"] == -2.0


def test_first_to_flip_same_bar_has_zero_lead() -> None:
    s, b = SIDEWAYS, BEAR
    labels = _labels([s, s, s, b, b, s])
    bear = first_to_flip(labels, labels.copy())["bear"]
    assert (bear["hmm_first"], bear["threshold_first"], bear["same_bar"]) == (0, 0, 1)
    assert bear["leads"] == [0] and bear["mean_lead_bars"] == 0.0


def test_first_to_flip_leaves_entries_beyond_the_window_unmatched() -> None:
    s, b = SIDEWAYS, BEAR
    n = 40
    hmm_values = [s] * n
    hmm_values[2] = b  # enters Bear at bar 2
    threshold_values = [s] * n
    threshold_values[30] = b  # enters Bear at bar 30, 28 bars later
    out = first_to_flip(_labels(threshold_values), _labels(hmm_values), match_window=5)
    bear = out["bear"]
    assert bear["n_matched"] == 0
    assert bear["mean_lead_bars"] is None and bear["median_lead_bars"] is None
    assert bear["leads"] == []
    assert bear["hmm_unmatched"] == 1 and bear["threshold_unmatched"] == 1
    assert out["match_window"] == 5


def test_first_to_flip_matching_is_greedy_and_one_to_one() -> None:
    s, b = SIDEWAYS, BEAR
    n = 30
    # HMM enters Bear at bars 4 and 8; threshold enters at bars 6 and 10.
    # Greedy in time order: threshold 6 sees HMM 4 and 8 at the same distance and takes
    # the earlier (4, lead 2), so threshold 10 is left with HMM 8 (lead 2) rather than
    # both pairing with 8.
    hmm_values = [s] * n
    hmm_values[4] = b
    hmm_values[8] = b
    threshold_values = [s] * n
    threshold_values[6] = b
    threshold_values[10] = b
    bear = first_to_flip(_labels(threshold_values), _labels(hmm_values), match_window=4)["bear"]
    assert bear["n_matched"] == 2
    assert bear["leads"] == [2, 2]
    assert bear["hmm_first"] == 2
    assert bear["hmm_unmatched"] == 0 and bear["threshold_unmatched"] == 0


def test_first_to_flip_matches_the_nearest_entry_not_the_earliest() -> None:
    """A one-bar HMM flicker at bar 5 must not steal the match from the real HMM flip at
    bar 19 when the threshold flips at bar 20: the honest lead is 1, not 15."""
    s, b = SIDEWAYS, BEAR
    n = 40
    hmm_values = [s] * n
    hmm_values[5] = b  # one-bar false alarm
    hmm_values[19] = b  # the real flip, one bar before the threshold
    threshold_values = [s] * n
    threshold_values[20] = b
    bear = first_to_flip(_labels(threshold_values), _labels(hmm_values), match_window=20)["bear"]
    assert bear["leads"] == [1]
    assert bear["hmm_first"] == 1 and bear["mean_lead_bars"] == 1.0
    assert bear["hmm_unmatched"] == 1 and bear["threshold_unmatched"] == 0


def test_first_to_flip_nearest_tie_goes_to_the_earlier_hmm_entry() -> None:
    s, b = SIDEWAYS, BEAR
    n = 20
    hmm_values = [s] * n
    hmm_values[8] = b
    hmm_values[12] = b
    threshold_values = [s] * n
    threshold_values[10] = b
    bear = first_to_flip(_labels(threshold_values), _labels(hmm_values), match_window=5)["bear"]
    assert bear["leads"] == [2]
    assert bear["hmm_unmatched"] == 1


def test_first_to_flip_never_counts_the_first_bar_as_an_entry() -> None:
    b, s = BEAR, SIDEWAYS
    labels = _labels([b, b, s, s])  # starts in Bear; no entry
    out = first_to_flip(labels, labels.copy())
    assert out["bear"]["n_threshold_entries"] == 0
    assert out["bear"]["n_hmm_entries"] == 0


def test_first_to_flip_reports_both_targets_window_and_note() -> None:
    labels = _labels([SIDEWAYS, BEAR, SIDEWAYS, BULL])
    out = first_to_flip(labels, labels.copy())
    assert list(out) == ["bear", "bull", "match_window", "note"]
    assert out["match_window"] == DEFAULT_MATCH_WINDOW
    assert list(out["bear"]) == FLIP_KEYS and list(out["bull"]) == FLIP_KEYS
    assert "positive lead = the HMM flipped first" in out["note"]


def test_first_to_flip_rejects_a_negative_window() -> None:
    labels = _labels([SIDEWAYS, BEAR])
    with pytest.raises(ValueError, match="match_window"):
        first_to_flip(labels, labels, match_window=-1)


def test_agreement_and_first_to_flip_are_json_ready() -> None:
    threshold = _labels([SIDEWAYS, BEAR, BEAR, SIDEWAYS, BULL, BULL])
    hmm = _labels([BEAR, BEAR, SIDEWAYS, SIDEWAYS, SIDEWAYS, BULL])
    text = json.dumps({"a": agreement(threshold, hmm), "f": first_to_flip(threshold, hmm)})
    assert "NaN" not in text
    assert isinstance(json.loads(text)["a"]["confusion"][0][0], int)


# --------------------------------------------------------------------------- #
# hmm_block degrade paths (no fitting)
# --------------------------------------------------------------------------- #
def test_hmm_block_disabled_is_the_companion_dict_exactly() -> None:
    close = _synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    assert hmm_block(close, labels, enabled=False) == {
        "available": False,
        "reason": REASON_DISABLED,
    }


def test_hmm_block_without_hmmlearn_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hmmlearn", None)
    close = _synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    out = hmm_block(
        close, labels, min_train=SMALL_MIN_TRAIN, refit_every=SMALL_REFIT, seeds=TINY_SEEDS
    )
    assert out == {"available": False, "reason": REASON_UNAVAILABLE}


def test_hmm_block_on_a_short_history_reports_too_short() -> None:
    close = _synthetic_close(rows=120)  # 100 feature rows < min_train + 30
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    out = hmm_block(close, labels, min_train=SMALL_MIN_TRAIN, seeds=TINY_SEEDS)
    assert out["available"] is False
    assert out["reason"].startswith(REASON_TOO_SHORT)
    assert list(out) == ["available", "reason"]


@pytest.mark.parametrize(
    "bad",
    [
        {"emission": "bogus"},
        {"refit_every": 0},
        {"enter": 0.3, "exit": 0.4},
        {"confirm_bars": 0},
        {"vol_short": 20, "vol_long": 10},
    ],
    ids=["emission", "refit_every", "enter_exit", "confirm_bars", "vol_windows"],
)
def test_hmm_block_lets_a_bad_parameter_value_error_propagate(bad: dict) -> None:
    close = _synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    with pytest.raises(ValueError):
        hmm_block(close, labels, seeds=TINY_SEEDS, **bad)


def test_hmm_block_wraps_a_non_value_error_failure_as_a_runtime_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("every Baum-Welch restart failed")

    monkeypatch.setattr(hmm_report, "hmm_walk_forward_from_close", explode)
    close = _synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    out = hmm_block(close, labels, seeds=TINY_SEEDS)
    assert out == {
        "available": False,
        "reason": "hmm runtime error: every Baum-Welch restart failed",
    }


# --------------------------------------------------------------------------- #
# pnl_comparison (hand-built close path, no fitting)
# --------------------------------------------------------------------------- #
PNL_CLOSE = [100.0, 110.0, 99.0, 99.0, 108.9]  # next-bar returns: +10%, -10%, 0%, +10%
PNL_THRESHOLD = [BULL, BULL, BEAR, SIDEWAYS, BULL]  # positions +1, +1, -1, 0 (last drops)
PNL_HMM = [BEAR, BULL, BULL, BULL, BULL]  # positions -1, +1, +1, +1
PNL_TOL = 1e-12  # products of a handful of exact decimals; float round-off only


def _pnl_inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.bdate_range("2020-01-01", periods=len(PNL_CLOSE), name="date")
    close = pd.Series(PNL_CLOSE, index=index, name="close")
    return close, pd.Series(PNL_THRESHOLD, index=index), pd.Series(PNL_HMM, index=index)


def test_pnl_comparison_gross_returns_follow_the_sign_rule_on_each_detector() -> None:
    close, threshold, hmm = _pnl_inputs()
    out = pnl_comparison(close, threshold, hmm, cost_model=ZERO_COSTS)
    assert out["n_days"] == 4
    assert out["first_date"] == "2020-01-01" and out["last_date"] == "2020-01-06"
    # threshold: +10%, -10%, 0 (short a flat bar), 0 (flat) -> 1.1 * 0.9 - 1
    assert out["threshold"]["gross"]["total_return"] == pytest.approx(1.1 * 0.9 - 1, abs=PNL_TOL)
    # hmm: -10% (short the up bar), -10%, 0, +10% -> 0.9 * 0.9 * 1.1 - 1
    assert out["hmm"]["gross"]["total_return"] == pytest.approx(0.9 * 0.9 * 1.1 - 1, abs=PNL_TOL)
    assert out["buy_and_hold"]["net"]["total_return"] == pytest.approx(
        1.1 * 0.9 * 1.1 - 1, abs=PNL_TOL
    )
    assert out["threshold"]["gross"]["time_in_market"] == pytest.approx(3 / 4)
    assert out["hmm"]["gross"]["time_in_market"] == 1.0


def test_pnl_comparison_zero_costs_make_net_equal_gross() -> None:
    close, threshold, hmm = _pnl_inputs()
    out = pnl_comparison(close, threshold, hmm, cost_model=ZERO_COSTS)
    for detector in ("threshold", "hmm"):
        assert out[detector]["net"] == out[detector]["gross"]
    assert out["costs"] == {
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "borrow_bps": 0.0,
        "funding_bps": 0.0,
        "per_side_bps": 0.0,
    }


def test_pnl_comparison_costs_reduce_net_total_return_monotonically() -> None:
    close, threshold, hmm = _pnl_inputs()
    previous = {
        name: pnl_comparison(close, threshold, hmm, cost_model=ZERO_COSTS)[name]["net"][
            "total_return"
        ]
        for name in ("threshold", "hmm", "buy_and_hold")
    }
    for fee in (5.0, 20.0, 50.0):
        out = pnl_comparison(close, threshold, hmm, cost_model=CostModel(fee, 0.0, 0.0, 0.0))
        for name in ("threshold", "hmm", "buy_and_hold"):
            assert out[name]["net"]["total_return"] < previous[name], (name, fee)
            previous[name] = out[name]["net"]["total_return"]
        assert out["threshold"]["gross"]["total_return"] == pytest.approx(1.1 * 0.9 - 1)
        assert out["costs"]["per_side_bps"] == fee


def test_pnl_comparison_defaults_to_the_us_etfs_cost_model() -> None:
    close, threshold, hmm = _pnl_inputs()
    default = CostModel()
    out = pnl_comparison(close, threshold, hmm)
    assert out["costs"]["per_side_bps"] == default.per_side_bps
    assert out["costs"]["borrow_bps"] == default.borrow_bps


def test_pnl_comparison_too_short_is_all_null_with_zero_days() -> None:
    close, threshold, hmm = _pnl_inputs()
    out = pnl_comparison(close, threshold.iloc[:1], hmm, cost_model=ZERO_COSTS)  # 1 return
    assert out["n_days"] == 0
    assert out["first_date"] is None and out["last_date"] is None
    for block in (out["threshold"]["gross"], out["hmm"]["net"], out["buy_and_hold"]["net"]):
        assert list(block) == METRIC_KEYS
        assert block["n_days"] == 0
        assert all(block[key] is None for key in METRIC_KEYS if key != "n_days")
    assert "not the Phase 3" in out["rule"]


def test_pnl_comparison_uses_only_common_dates_with_a_next_close() -> None:
    close, threshold, hmm = _pnl_inputs()
    out = pnl_comparison(close, threshold, hmm.iloc[1:], cost_model=ZERO_COSTS)
    assert out["n_days"] == 3
    assert out["first_date"] == "2020-01-02"
    assert out["hmm"]["gross"]["total_return"] == pytest.approx(0.9 * 1.1 - 1, abs=PNL_TOL)


def test_pnl_comparison_is_json_ready_with_the_phase3_metric_keys() -> None:
    close, threshold, hmm = _pnl_inputs()
    out = pnl_comparison(close, threshold, hmm)
    assert list(out) == [
        "threshold",
        "hmm",
        "buy_and_hold",
        "costs",
        "n_days",
        "first_date",
        "last_date",
        "rule",
        "note",
    ]
    assert list(out["threshold"]["gross"]) == METRIC_KEYS
    _assert_no_nan(out, "pnl")
    assert json.loads(json.dumps(out)) == out


def test_pnl_comparison_rejects_labels_outside_states() -> None:
    close, threshold, hmm = _pnl_inputs()
    with pytest.raises(ValueError, match="hmm_regimes"):
        pnl_comparison(close, threshold, hmm.replace({BULL: 7}))


# --------------------------------------------------------------------------- #
# hmm_block available path (one tiny fit)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def available_block() -> dict:
    pytest.importorskip("hmmlearn")
    close = _synthetic_close()
    labels = label_regimes(close, window=WINDOW, threshold=THRESHOLD, return_type="simple")
    return hmm_block(
        close,
        labels,
        min_train=SMALL_MIN_TRAIN,
        refit_every=SMALL_REFIT,
        seeds=TINY_SEEDS,
        n_iter=TINY_N_ITER,
    )


def test_available_block_keys_are_companion_first_then_phase4_in_order(
    available_block: dict,
) -> None:
    assert list(available_block) == [*COMPANION_HMM_KEYS, *PHASE4_HMM_KEYS]
    assert available_block["available"] is True
    assert available_block["caveat"] == HMM_CAVEAT
    assert [r["label"] for r in available_block["regimes"]] == STATES
    assert "regimes" not in available_block["fit"]


def test_available_block_has_no_nan_and_round_trips_through_json(available_block: dict) -> None:
    _assert_no_nan(available_block, "hmm")
    assert json.loads(json.dumps(available_block)) == available_block


def test_available_block_decision_and_probabilities(available_block: dict) -> None:
    decision = available_block["decision"]
    assert decision["regime"] in STATES and decision["raw_regime"] in STATES
    assert (decision["enter"], decision["exit"], decision["confirm_bars"]) == (0.6, 0.4, 2)
    assert "filtered" in decision["rule"]
    probs = available_block["filtered_probabilities"]
    assert list(probs) == STATE_KEYS
    assert sum(probs.values()) == pytest.approx(1.0, abs=PROB_TOL)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_available_block_features_fit_and_walk_forward(available_block: dict) -> None:
    features = available_block["features"]
    assert features["names"] == ["log_return", "vol_10", "vol_20"]
    assert (features["vol_short"], features["vol_long"]) == (10, 20)
    fit = available_block["fit"]
    assert fit["seeds"] == list(TINY_SEEDS) and fit["covariance_type"] == "full"
    assert len(fit["transition_matrix"]) == len(STATES)
    for row in fit["transition_matrix"]:
        assert sum(row) == pytest.approx(1.0, abs=PROB_TOL)
    wf = available_block["walk_forward"]
    assert list(wf) == [
        "n_days",
        "n_refits",
        "first_date",
        "last_date",
        "regime_counts",
        "mean_log_likelihood",
        "params",
    ]
    assert wf["n_days"] > 0 and wf["n_refits"] >= 1
    assert sum(wf["regime_counts"].values()) == wf["n_days"]


def test_available_block_agreement_and_first_to_flip_use_the_decision_dates(
    available_block: dict,
) -> None:
    agree = available_block["agreement_with_threshold"]
    assert agree["n"] == available_block["walk_forward"]["n_days"]
    assert sum(sum(row) for row in agree["confusion"]) == agree["n"]
    flip = available_block["first_to_flip"]
    assert flip["match_window"] == DEFAULT_MATCH_WINDOW
    for key in ("bear", "bull"):
        entry = flip[key]
        assert (
            entry["n_matched"] == entry["hmm_first"] + entry["threshold_first"] + entry["same_bar"]
        )
        assert entry["n_matched"] + entry["threshold_unmatched"] == entry["n_threshold_entries"]
        assert entry["n_matched"] + entry["hmm_unmatched"] == entry["n_hmm_entries"]
    assert "filtered, not smoothed" in available_block["note"]


def test_available_block_pnl_comparison_covers_the_decision_dates(available_block: dict) -> None:
    pnl = available_block["pnl_comparison"]
    # The last decision date has no next close, so one day fewer than the walk-forward.
    assert pnl["n_days"] == available_block["walk_forward"]["n_days"] - 1
    assert pnl["hmm"]["net"]["n_days"] == pnl["n_days"]
    assert pnl["costs"]["per_side_bps"] == CostModel().per_side_bps
    assert pnl["threshold"]["net"]["total_return"] <= pnl["threshold"]["gross"]["total_return"]
    assert pnl["hmm"]["net"]["total_return"] <= pnl["hmm"]["gross"]["total_return"]
