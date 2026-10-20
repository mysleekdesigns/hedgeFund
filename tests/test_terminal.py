"""``render`` reproduces the companion's ``_print_pretty`` text line for line."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.report.json_contract import PHASE2_FIELDS, analyze
from hedgefund.report.terminal import (
    DIAGONAL_COLUMNS,
    FOOTER,
    GAP_EXPLANATION,
    HONEST_HEADER,
    NULL_OVERLAP_CAVEAT,
    print_pretty,
    render,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SEED = 3
ROWS = 400
DAILY_VOL = 0.01
START_PRICE = 100.0
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20


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
