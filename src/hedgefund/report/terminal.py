"""Pretty terminal output, line for line the companion script's ``_print_pretty``.

``render`` returns the plain text (no rich markup) so it can be diffed against the
companion's stdout; ``print_pretty`` writes it unchanged.
"""

import math

from hedgefund.regime.labels import STATES

PERCENT = 100  # probabilities are printed as percentages

FOOTER = [
    "",
    "----------------------------------------------------------------",
    " Framework: Roan (@RohOnChain). Refactored into a Claude Code",
    " plugin by Lewis Jackson. Backtests are historical, not forward-",
    " looking. Point the matrix at whatever you trade.",
    "----------------------------------------------------------------",
    "",
]


def _is_finite(value: object) -> bool:
    """``np.isfinite`` semantics for floats, ``False`` for ``None``/non-numbers."""
    return isinstance(value, int | float) and math.isfinite(value)


def render(result: dict) -> str:
    """Return the on-camera pretty report for an ``analyze()`` dict as one string."""
    a = result
    matrix = a["transition_matrix"]
    lines: list[str] = []

    lines.append("")
    lines.append(
        f"markov-regime — source={a['source']} "
        f"window={a['params']['window']} threshold={a['params']['threshold']}"
    )
    lines.append(f"  {a['rows']} rows | {a['date_start']} -> {a['date_end']}")

    lines.append("")
    lines.append("Transition matrix (rows = from, cols = to):")
    lines.append(f"            {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
    for i, from_state in enumerate(STATES):
        row = "  ".join(f"{matrix[i][j] * PERCENT:7.2f}%" for j in range(len(STATES)))
        lines.append(f"  {from_state:>9s}  {row}")

    pd_diag = a["persistence_diagonal"]
    lines.append("")
    lines.append("Persistence diagonal (how sticky each regime is):")
    lines.append(f"  Bear -> Bear:         {pd_diag['bear'] * PERCENT:.2f}%")
    lines.append(f"  Sideways -> Sideways: {pd_diag['sideways'] * PERCENT:.2f}%")
    lines.append(f"  Bull -> Bull:         {pd_diag['bull'] * PERCENT:.2f}%")

    sd = a["stationary_distribution"]
    lines.append("")
    lines.append("Stationary distribution (long-run regime mix):")
    lines.append(f"       Bear: {sd['bear'] * PERCENT:.2f}%")
    lines.append(f"   Sideways: {sd['sideways'] * PERCENT:.2f}%")
    lines.append(f"       Bull: {sd['bull'] * PERCENT:.2f}%")

    np_ = a["next_state_probabilities"]
    lines.append("")
    lines.append(f"Current regime: {a['current_regime']}")
    lines.append("Next-day probabilities from here:")
    lines.append(
        f"   Bull: {np_['bull'] * PERCENT:.2f}%   "
        f"Bear: {np_['bear'] * PERCENT:.2f}%   "
        f"Sideways: {np_['sideways'] * PERCENT:.2f}%"
    )
    lines.append(f"Signal (bull_prob - bear_prob): {a['signal']:+.4f}")

    wf = a["walk_forward"]
    lines.append("")
    lines.append("Walk-forward backtest (matrix re-estimated every step, no lookahead):")
    if _is_finite(wf["sharpe"]):
        lines.append(f"  Sharpe (annualised): {wf['sharpe']:.3f}")
    else:
        lines.append("  Sharpe: NaN (insufficient data — try a longer history)")
    if _is_finite(wf["max_drawdown"]):
        lines.append(f"  Max drawdown:        {wf['max_drawdown'] * PERCENT:.2f}%")
    else:
        lines.append("  Max drawdown: NaN")
    lines.append(f"  Trades evaluated:    {wf['n_trades']}")

    hmm = a["hmm"]
    if hmm.get("available"):
        lines.append("")
        lines.append("Hidden Markov Model (Baum-Welch + Viterbi):")
        for r in hmm["regimes"]:
            lines.append(
                f"  {r['label']:<9s} (latent state {r['latent_state']}): "
                f"{r['mean_daily_return'] * PERCENT:+.3f}% mean daily return"
            )
        lines.append(f"  Note: {hmm['caveat']}")
    else:
        lines.append("")
        lines.append(
            f"HMM skipped: {hmm.get('reason', 'unavailable')} "
            "(observable model above is unaffected)."
        )

    lines.extend(FOOTER)
    return "\n".join(lines) + "\n"


def print_pretty(result: dict) -> None:
    """Print the pretty report exactly as the companion script does."""
    print(render(result), end="")
