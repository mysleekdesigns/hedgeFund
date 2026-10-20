"""``render`` reproduces the companion's ``_print_pretty`` text line for line."""

import copy
import json
from pathlib import Path

import pytest

from hedgefund.report.terminal import print_pretty, render

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def companion() -> dict:
    with (FIXTURES / "spy_10y.companion.json").open() as fh:
        return json.load(fh)["result"]


@pytest.fixture(scope="module")
def text(companion: dict) -> str:
    return render(companion)


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
