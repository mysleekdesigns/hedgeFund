"""``render`` reproduces the companion's ``_print_pretty`` text line for line."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.report.json_contract import PHASE2_FIELDS, PHASE3_FIELDS, analyze, backtest_only
from hedgefund.report.terminal import (
    BACKTEST_HEADER,
    DIAGONAL_COLUMNS,
    DSR_HYBRID_CAVEAT,
    FOOTER,
    GAP_EXPLANATION,
    HONEST_HEADER,
    NULL_OVERLAP_CAVEAT,
    print_backtest,
    print_pretty,
    render,
    render_backtest,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SEED = 3
ROWS = 400
DAILY_VOL = 0.01
START_PRICE = 100.0
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20
LONG_ROWS = 700  # about 2.8 years: leaves in-sample history beyond the two-year holdout
SMALL_MIN_TRAIN = 60  # so the walk-forward and the sensitivity grid have days to score


def _synthetic_close(rows: int = ROWS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name="date")
    return pd.Series(START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")


@pytest.fixture(scope="module")
def companion() -> dict:
    with (FIXTURES / "spy_10y.companion.json").open() as fh:
        return json.load(fh)["result"]


@pytest.fixture(scope="module")
def text(companion: dict) -> str:
    return render(companion)


@pytest.fixture(scope="module")
def full_result() -> dict:
    return analyze(
        _synthetic_close(),
        source="synthetic",
        hmm=False,
        null_paths=FAST_NULL_PATHS,
        bootstrap_resamples=FAST_RESAMPLES,
    )


@pytest.fixture(scope="module")
def full_text(full_result: dict) -> str:
    return render(full_result)


def test_header_lines(text: str) -> None:
    assert text.startswith("\nmarkov-regime — source=spy_10y.csv window=20 threshold=0.05\n")
    assert "  2511 rows | 2016-10-20 -> 2026-10-16\n" in text


def test_transition_matrix_block(text: str) -> None:
    assert "Transition matrix (rows = from, cols = to):\n" in text
    assert "                 Bear  Sideways      Bull\n" in text
    assert "       Bear    77.54%    22.46%     0.00%\n" in text
    assert "   Sideways     2.14%    93.03%     4.83%\n" in text
    assert "       Bull     0.00%    28.11%    71.89%\n" in text


def test_persistence_lines_use_two_decimal_percentages(text: str) -> None:
    assert "Persistence diagonal (how sticky each regime is):\n" in text
    assert "  Bear -> Bear:         77.54%\n" in text
    assert "  Sideways -> Sideways: 93.03%\n" in text
    assert "  Bull -> Bull:         71.89%\n" in text


def test_stationary_distribution_lines(text: str) -> None:
    assert "Stationary distribution (long-run regime mix):\n" in text
    assert "       Bear: 7.51%\n" in text
    assert "   Sideways: 78.92%\n" in text
    assert "       Bull: 13.57%\n" in text


def test_current_regime_and_signal_lines(text: str) -> None:
    assert "Current regime: Sideways\n" in text
    assert "   Bull: 4.83%   Bear: 2.14%   Sideways: 93.03%\n" in text
    assert "Signal (bull_prob - bear_prob): +0.0270\n" in text


def test_walk_forward_lines(text: str) -> None:
    assert "Walk-forward backtest (matrix re-estimated every step, no lookahead):\n" in text
    assert "  Sharpe (annualised): 0.246\n" in text
    assert "  Max drawdown:        -33.91%\n" in text
    assert "  Trades evaluated:    2238\n" in text


def test_hmm_skipped_line(text: str) -> None:
    assert "\nHMM skipped: disabled via --no-hmm (observable model above is unaffected).\n" in text


def test_hmm_available_block(companion: dict) -> None:
    with_hmm = copy.deepcopy(companion)
    with_hmm["hmm"] = {
        "available": True,
        "regimes": [
            {"label": "Bear", "latent_state": 2, "mean_daily_return": -0.0012345},
            {"label": "Sideways", "latent_state": 0, "mean_daily_return": 0.0003},
            {"label": "Bull", "latent_state": 1, "mean_daily_return": 0.0021},
        ],
        "caveat": "local maxima",
    }
    text = render(with_hmm)
    assert "Hidden Markov Model (Baum-Welch + Viterbi):\n" in text
    assert "  Bear      (latent state 2): -0.123% mean daily return\n" in text
    assert "  Sideways  (latent state 0): +0.030% mean daily return\n" in text
    assert "  Bull      (latent state 1): +0.210% mean daily return\n" in text
    assert "  Note: local maxima\n" in text


def test_framework_footer(text: str) -> None:
    footer = (
        "\n----------------------------------------------------------------\n"
        " Framework: Roan (@RohOnChain). Refactored into a Claude Code\n"
        " plugin by Lewis Jackson. Backtests are historical, not forward-\n"
        " looking. Point the matrix at whatever you trade.\n"
        "----------------------------------------------------------------\n\n"
    )
    assert text.endswith(footer)


def test_nan_sharpe_renders_insufficient_data_lines(companion: dict) -> None:
    short = copy.deepcopy(companion)
    short["walk_forward"] = {
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "n_trades": 0,
    }
    text = render(short)
    assert "  Sharpe: NaN (insufficient data — try a longer history)\n" in text
    assert "  Max drawdown: NaN\n" in text
    assert "  Trades evaluated:    0\n" in text


def test_print_pretty_writes_render_verbatim(
    companion: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    print_pretty(companion)
    assert capsys.readouterr().out == render(companion)


# --------------------------------------------------------------------------- #
# Phase 2: "Honest statistics" section
# --------------------------------------------------------------------------- #
def test_companion_dict_renders_without_the_honest_statistics_section(text: str) -> None:
    assert HONEST_HEADER not in text
    assert GAP_EXPLANATION not in text


def test_full_result_contains_honest_statistics_header_and_columns(full_text: str) -> None:
    assert f"\n{HONEST_HEADER}\n" in full_text
    for column in DIAGONAL_COLUMNS:
        assert column in full_text, column
    assert f"\n{GAP_EXPLANATION}\n" in full_text


def test_honest_section_sits_between_hmm_block_and_footer(full_text: str) -> None:
    hmm_at = full_text.index("HMM skipped:")
    header_at = full_text.index(HONEST_HEADER)
    footer_at = full_text.index(FOOTER[1])
    assert hmm_at < header_at < footer_at
    assert full_text.endswith("\n".join(FOOTER) + "\n")


def test_three_diagonals_render_in_percent(full_result: dict, full_text: str) -> None:
    overlap = full_result["persistence_diagonal"]["sideways"] * 100
    implied = full_result["implied_window_step_diagonal"]["sideways"] * 100
    stride = full_result["stride_diagonal"]["sideways"] * 100
    section = full_text[full_text.index(HONEST_HEADER) :]
    line = next(ln for ln in section.splitlines() if ln.startswith("   Sideways") and "%" in ln)
    assert f"{overlap:.2f}%" in line
    assert f"{implied:.2f}%" in line
    assert f"{stride:.2f}%" in line


def test_honest_section_names_every_block(full_text: str) -> None:
    assert (
        "Run-length persistence, 1 - 1/mean episode length (on the stride sequence):" in full_text
    )
    assert "Null percentiles of the overlapping diagonal" in full_text
    assert "label-shuffle" in full_text and "iid" in full_text
    assert "Labeller-induced null" in full_text and "excess 95% CI" in full_text
    assert "95% block-bootstrap CI (20 resamples, 20-day blocks):" in full_text
    assert "effective n" in full_text
    assert "Drift: rolling (756 bars) vs full-history diagonal, TVD = " in full_text
    assert "EW hl=250" in full_text and "shrunk a=20" in full_text
    assert "Mixing time (TVD <= 0.01): " in full_text
    assert "Anderson-Goodman first-order dependence test (stride sequence): chi2 = " in full_text
    assert "Homogeneity test (stride sequence, 2 sub-periods): chi2 = " in full_text
    assert "matrix stable at 5%" in full_text or "matrix NOT stable at 5%" in full_text


def test_null_values_render_as_na(full_result: dict) -> None:
    with_nulls = copy.deepcopy(full_result)
    with_nulls["stride_diagonal"]["bear"] = None
    with_nulls["run_length_persistence"]["bull"] = None
    with_nulls["null_percentile"]["labeller_induced"]["excess_ci"]["bear"] = [None, None]
    with_nulls["mixing_time"] = None
    with_nulls["tests"]["anderson_goodman"] = None
    with_nulls["tests"]["homogeneity"] = None
    text = render(with_nulls)
    assert "n/a" in text
    assert "       Bull: n/a\n" in text
    assert "n/a to n/a" in text
    assert "Mixing time (TVD <= 0.01): not reached within max steps" in text
    assert "Anderson-Goodman first-order dependence test (stride sequence): n/a\n" in text
    assert "Homogeneity test (stride sequence): n/a (series too short)\n" in text


def test_null_percentile_table_carries_the_overlap_caveat(full_text: str) -> None:
    # Label-shuffle and iid nulls destroy the window overlap, so a smoothed labeller
    # always scores 100 against them; the line must sit between that table and the
    # labeller-induced null that is the honest comparison.
    assert NULL_OVERLAP_CAVEAT.startswith("  These two nulls ignore the window overlap")
    assert (NULL_OVERLAP_CAVEAT + "\n  Labeller-induced null (") in full_text
    header = full_text.index("Null percentiles of the overlapping diagonal")
    assert header < full_text.index(NULL_OVERLAP_CAVEAT)


def test_homogeneity_verdict_wording(full_result: dict) -> None:
    stable = copy.deepcopy(full_result)
    stable["tests"]["homogeneity"]["stable"] = True
    assert "-> matrix stable at 5%\n" in render(stable)
    unstable = copy.deepcopy(full_result)
    unstable["tests"]["homogeneity"]["stable"] = False
    assert "-> matrix NOT stable at 5% (regime probabilities drift)\n" in render(unstable)


def test_print_pretty_writes_full_result_verbatim(
    full_result: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    print_pretty(full_result)
    assert capsys.readouterr().out == render(full_result)


def test_phase3_field_is_consumed_by_render(full_result: dict) -> None:
    """Dropping ``backtest`` removes the Phase 3 section; the rest still renders."""
    assert PHASE3_FIELDS == ["backtest"]
    broken = copy.deepcopy(full_result)
    del broken["backtest"]
    text = render(broken)
    assert BACKTEST_HEADER not in text
    assert HONEST_HEADER in text


def test_phase2_fields_are_all_consumed_by_render(full_result: dict) -> None:
    """Every Phase 2 key is read by ``render`` (dropping one must break rendering)."""
    for field in PHASE2_FIELDS:
        if field in ("statistics_params", "mixing_epsilon", "ew_half_life", "shrinkage_alpha"):
            continue  # echoed parameters / labels only; rendering survives their absence
        broken = copy.deepcopy(full_result)
        del broken[field]
        if field == "stride_matrix":
            assert HONEST_HEADER not in render(broken)
            continue
        with pytest.raises(KeyError):
            render(broken)


# --------------------------------------------------------------------------- #
# Phase 3: "Backtest" section
# --------------------------------------------------------------------------- #
def test_companion_dict_renders_without_the_backtest_section(text: str) -> None:
    assert BACKTEST_HEADER not in text


def test_backtest_section_sits_after_honest_statistics_and_before_footer(full_text: str) -> None:
    honest_at = full_text.index(HONEST_HEADER)
    backtest_at = full_text.index(BACKTEST_HEADER)
    footer_at = full_text.index(FOOTER[1])
    assert honest_at < backtest_at < footer_at


def test_backtest_section_states_costs_turnover_benchmarks_and_null_percentile(
    full_result: dict, full_text: str
) -> None:
    section = full_text[full_text.index(BACKTEST_HEADER) :]
    costs = full_result["backtest"]["params"]["costs"]
    assert f"Costs: {costs['per_side_bps']:.1f} bps per side" in section
    assert f"borrow {costs['borrow_bps']:.0f} bps/yr on shorts" in section
    assert "annual cost drag" in section
    assert "Turnover (one-way notional per year):" in section
    assert "time in market:" in section
    assert "buy-and-hold" in section and "constant 50%" in section and "long-only" in section
    assert "strategy net" in section
    assert "gross pct" in section and "net pct" in section and "null turnover" in section
    for label in ("random-sign", "label-shuffle", "permutation"):
        assert label in section, label
    assert "Probabilistic Sharpe (net):" in section and "Deflated Sharpe:" in section
    assert "15 trials" in section
    assert "Holdout (2y, from " in section and "in-sample net Sharpe" in section
    assert "best in-sample cell" in section
    assert "Cost sensitivity (per-side bps -> net):" in section
    assert "Sensitivity: in-sample net Sharpe, rows = window, cols = threshold" in section
    assert "Detection lag, median bars from turning point to flip" in section
    assert "hindsight diagnostic" in section


def test_backtest_section_never_shows_gross_without_net(full_result: dict, full_text: str) -> None:
    section = full_text[full_text.index(BACKTEST_HEADER) :]
    gross = full_result["backtest"]["gross"]["sharpe"]
    net = full_result["backtest"]["net"]["sharpe"]
    line = next(ln for ln in section.splitlines() if ln.strip().startswith("Sharpe (annualised)"))
    assert f"{gross:.3f}" in line and f"{net:.3f}" in line
    header = next(ln for ln in section.splitlines() if "gross" in ln and "net" in ln)
    assert header.index("gross") < header.index("net")


def test_backtest_sensitivity_heatmap_has_window_rows_and_threshold_columns(
    full_result: dict, full_text: str
) -> None:
    section = full_text[full_text.index("Sensitivity: in-sample") :]
    sens = full_result["backtest"]["sensitivity"]
    header = section.splitlines()[1]
    for t in sens["thresholds"]:
        assert f"{t:.2f}" in header
    for w in sens["windows"]:
        assert any(ln.strip().startswith(str(w)) for ln in section.splitlines()[2:]), w


def test_backtest_none_values_render_as_na(full_result: dict) -> None:
    with_nulls = copy.deepcopy(full_result)
    bt = with_nulls["backtest"]
    bt["gross"]["sharpe"] = None
    bt["net"]["sharpe"] = None
    bt["net"]["min_track_record"] = None
    bt["annual_cost"] = None
    bt["null"] = None
    bt["holdout"] = None
    bt["psr"] = None
    bt["dsr"]["dsr"] = None
    bt["benchmarks"]["buy_and_hold"]["sharpe"] = None
    bt["cost_sensitivity"][0]["sharpe"] = None
    bt["sensitivity"]["net_sharpe"][0][0] = None
    bt["sensitivity"]["best"] = {
        "window": None,
        "threshold": None,
        "net_sharpe": None,
        "holdout_gross_sharpe": None,
        "holdout_net_sharpe": None,
    }
    bt["sensitivity"]["share_positive"] = None
    bt["lag"]["zscore"] = None
    text = render(with_nulls)
    section = text[text.index(BACKTEST_HEADER) :]
    sharpe_line = next(ln for ln in section.splitlines() if "Sharpe (annualised)" in ln)
    assert sharpe_line.count("n/a") == 2
    assert "annual cost drag n/a" in section
    assert "no paths (history too short)" in section
    sign_line = next(ln for ln in section.splitlines() if ln.strip().startswith("random-sign"))
    assert sign_line.count("n/a") == 5  # gross pct, gross p, net pct, net p, null turnover
    assert "Holdout (2y, from n/a): in-sample net Sharpe n/a (0 days)" in section
    # Without a holdout block the default cell falls back to params so the line stays explicit.
    assert "scores the default cell (window 20, threshold 0.05)" in section
    assert "best in-sample cell: n/a (no cell could be scored)" in section
    assert "Probabilistic Sharpe (net): n/a | Deflated Sharpe: n/a" in section
    assert "Best cell: n/a (no cell could be scored)" in section
    assert "n/a of cells positive" in section
    zscore_line = next(ln for ln in section.splitlines() if ln.strip().startswith("z-score"))
    assert zscore_line.count("n/a") == 5  # crash, recovery, undetected, flips, false flips


def test_null_table_prints_gross_and_net_side_by_side_with_null_turnover(
    full_result: dict, full_text: str
) -> None:
    """W1: never a net-only null; the gross pair and the null's turnover sit on the same row."""
    section = full_text[full_text.index(BACKTEST_HEADER) :]
    null = full_result["backtest"]["null"]
    header = next(ln for ln in section.splitlines() if "gross pct" in ln)
    assert header.index("gross pct") < header.index("net pct") < header.index("null turnover")
    for label, key in (
        ("random-sign", "random_sign"),
        ("label-shuffle", "label_shuffle"),
        ("permutation", "permutation"),
    ):
        entry = null[key]
        line = next(ln for ln in section.splitlines() if ln.strip().startswith(label))
        assert f"{entry['percentile_gross']:.1f}" in line, key
        assert f"{entry['p_value_gross']:.4f}" in line, key
        assert f"{entry['percentile_net']:.1f}" in line, key
        assert f"{entry['p_value_net']:.4f}" in line, key
        assert f"{entry['null_mean_turnover']:.1f}" in line, key
    strategy_turnover = full_result["backtest"]["net"]["turnover"]
    assert f"strategy turnover {strategy_turnover:.1f}/yr" in section
    assert "gross charges nothing" in section


def test_dsr_line_is_followed_by_the_conservative_hybrid_caveat(full_text: str) -> None:
    section = full_text[full_text.index(BACKTEST_HEADER) :]
    dsr_at = section.index("Deflated Sharpe:")
    assert "Conservative hybrid, not textbook DSR" in DSR_HYBRID_CAVEAT
    assert section.index(DSR_HYBRID_CAVEAT) > dsr_at
    assert section.index(DSR_HYBRID_CAVEAT) < section.index("Holdout (")


def test_holdout_line_names_the_default_cell_and_the_best_cell_holdout() -> None:
    """W3: the in-sample winner's out-of-sample Sharpe sits next to the default cell's.

    Needs a history long enough for the in-sample grid to score a cell (the 400-row
    fixture leaves under two years in sample), so it builds its own slim result.
    """
    slim = backtest_only(
        _synthetic_close(LONG_ROWS),
        source="synthetic",
        min_train=SMALL_MIN_TRAIN,
        null_paths=FAST_NULL_PATHS,
    )
    text = render_backtest(slim)
    section = text[text.index(BACKTEST_HEADER) :]
    bt = slim["backtest"]
    cell = bt["holdout"]["default_cell"]
    best = bt["sensitivity"]["best"]
    line = next(ln for ln in section.splitlines() if "best in-sample cell" in ln)
    assert f"default cell (window {cell['window']}, threshold {cell['threshold']})" in line
    assert f"best in-sample cell (window {best['window']}, threshold {best['threshold']})" in line
    assert f"in-sample net {best['net_sharpe']:.3f}" in line
    holdout_net = best["holdout_net_sharpe"]
    expected = f"{holdout_net:.3f}" if holdout_net is not None else "n/a"
    assert f"holdout net {expected}" in line


def test_lag_table_prints_flips_and_false_flip_rate_when_present(
    full_result: dict, full_text: str
) -> None:
    section = full_text[full_text.index("Detection lag") :]
    header = section.splitlines()[1]
    assert "flips" in header and "false flips" in header
    crash = full_result["backtest"]["lag"]["threshold"]["crash"]
    recovery = full_result["backtest"]["lag"]["threshold"]["recovery"]
    if "n_flips" not in crash:
        pytest.skip("lag module does not emit flip counts yet")
    line = next(ln for ln in section.splitlines() if ln.strip().startswith("threshold (plain)"))
    assert f"{crash['n_flips']}/{recovery['n_flips']}" in line

    def rate(value: object) -> str:
        return f"{value * 100:.2f}%" if value is not None else "n/a"

    assert f"{rate(crash['false_flip_rate'])}/{rate(recovery['false_flip_rate'])}" in line


def test_lag_table_renders_na_for_flip_keys_the_lag_module_has_not_emitted(
    full_result: dict,
) -> None:
    older = copy.deepcopy(full_result)
    for name in ("threshold", "zscore", "ew", "ensemble", "hysteresis"):
        entry = older["backtest"]["lag"][name]
        if entry is None:
            continue
        for direction in ("crash", "recovery"):
            for key in ("n_flips", "false_flips", "false_flip_rate"):
                entry[direction].pop(key, None)
    text = render(older)
    section = text[text.index("Detection lag") :]
    line = next(ln for ln in section.splitlines() if ln.strip().startswith("threshold (plain)"))
    assert line.rstrip().endswith("n/a/n/a")  # false-flip column
    assert line.count("n/a/n/a") >= 2  # flips and false-flip columns both degrade gracefully


def test_render_backtest_prints_header_section_and_footer() -> None:
    slim = backtest_only(_synthetic_close(), source="synthetic", null_paths=FAST_NULL_PATHS)
    text = render_backtest(slim)
    assert text.startswith(
        "\nmarkov-regime backtest — source=synthetic window=20 threshold=0.05 min_train=252\n"
    )
    assert f"  {ROWS} rows | 2020-01-01 -> " in text
    assert BACKTEST_HEADER in text
    assert HONEST_HEADER not in text  # no Phase 2 statistics in the slim dict
    assert text.endswith("\n".join(FOOTER) + "\n")


def test_render_backtest_matches_the_section_inside_render(full_result: dict) -> None:
    slim = {
        key: full_result[key]
        for key in ("source", "rows", "date_start", "date_end", "params", "backtest")
    }
    section_in_full = render(full_result)
    section_in_full = section_in_full[section_in_full.index(BACKTEST_HEADER) :]
    slim_text = render_backtest(slim)
    assert slim_text[slim_text.index(BACKTEST_HEADER) :] == section_in_full


def test_print_backtest_writes_render_backtest_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    slim = backtest_only(_synthetic_close(), source="synthetic", null_paths=FAST_NULL_PATHS)
    print_backtest(slim)
    assert capsys.readouterr().out == render_backtest(slim)
