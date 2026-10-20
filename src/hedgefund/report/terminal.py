"""Pretty terminal output, line for line the companion script's ``_print_pretty``.

``render`` returns the plain text (no rich markup) so it can be diffed against the
companion's stdout; ``print_pretty`` writes it unchanged. When the dict carries the
Phase 2 fields (``stride_matrix`` present) an "Honest statistics" section is inserted
between the HMM block and the footer; a dict without them renders the companion text
unchanged.
"""

import math

from hedgefund.regime.labels import STATES

PERCENT = 100  # probabilities are printed as percentages
NA = "n/a"  # a statistic that could not be computed (JSON ``null``)
STATE_KEYS = ["bear", "sideways", "bull"]  # dict keys in STATES order

HONEST_HEADER = "Honest statistics (Phase 2):"
DIAGONAL_COLUMNS = ["overlapping 1-step", "implied P^window", "stride (window-step)"]
GAP_EXPLANATION = (
    "  Gap = window autocorrelation: consecutive labels share window-1 bars. "
    "Stride counts labels window bars apart."
)
NULL_OVERLAP_CAVEAT = (
    "  These two nulls ignore the window overlap, so 100 is expected for any smoothed "
    "labeller; the labeller-induced null below is the honest comparison."
)

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

    if "stride_matrix" in a:
        lines.extend(_honest_statistics_lines(a))

    lines.extend(FOOTER)
    return "\n".join(lines) + "\n"


def _pct(value: object) -> str:
    """``value`` as a two-decimal percentage, or ``n/a`` for ``None`` / non-finite."""
    return f"{value * PERCENT:.2f}%" if _is_finite(value) else NA


def _num(value: object, digits: int = 2) -> str:
    """``value`` with ``digits`` decimals (no percent scaling), or ``n/a``."""
    return f"{value:.{digits}f}" if _is_finite(value) else NA


def _pct_range(lower: object, upper: object) -> str:
    return f"{_pct(lower)} to {_pct(upper)}"


def _diag(matrix: list[list[float]], i: int) -> float:
    return matrix[i][i]


def _honest_statistics_lines(a: dict) -> list[str]:
    """The Phase 2 section: both diagonals side by side, nulls, CIs, drift, tests."""
    lines: list[str] = []
    matrix = a["transition_matrix"]
    stride_diag = a["stride_diagonal"]
    implied = a["implied_window_step_diagonal"]
    window = a["params"]["window"]

    lines.append("")
    lines.append(HONEST_HEADER)
    lines.append(
        f"  Persistence diagonal, three ways (window = {window} bars, "
        f"{a['stride_n_transitions']} stride transitions):"
    )
    header = "".join(f"{col:>22s}" for col in DIAGONAL_COLUMNS)
    lines.append(f"  {'':>9s}{header}")
    for i, (state, key) in enumerate(zip(STATES, STATE_KEYS, strict=True)):
        cells = [_pct(_diag(matrix, i)), _pct(implied[key]), _pct(stride_diag[key])]
        lines.append(f"  {state:>9s}" + "".join(f"{c:>22s}" for c in cells))
    lines.append(GAP_EXPLANATION)
    lines.append(
        "  Implied P^window is the overlapping matrix raised to the window power: what the "
        "stride diagonal would be if the daily chain were first-order Markov."
    )

    rl = a["run_length_persistence"]
    lines.append("")
    lines.append("Run-length persistence, 1 - 1/mean episode length (on the stride sequence):")
    for state, key in zip(STATES, STATE_KEYS, strict=True):
        lines.append(f"  {state:>9s}: {_pct(rl[key])}")

    null = a["null_percentile"]
    shuffle, iid, induced = null["label_shuffle"], null["iid"], null["labeller_induced"]
    lines.append("")
    lines.append(
        f"Null percentiles of the overlapping diagonal ({shuffle['n_paths']} paths; "
        "100 = above every null draw):"
    )
    lines.append(f"  {'':>9s}{'label-shuffle':>16s}{'iid':>16s}")
    for state, key in zip(STATES, STATE_KEYS, strict=True):
        cells = [_num(shuffle["diagonal_percentile"][key]), _num(iid["diagonal_percentile"][key])]
        lines.append(f"  {state:>9s}" + "".join(f"{c:>16s}" for c in cells))
    lines.append(NULL_OVERLAP_CAVEAT)
    lines.append(
        f"  Labeller-induced null ({induced['n_paths']} iid price paths, same labeller): "
        "excess persistence = real - null mean"
    )
    lines.append(f"  {'':>9s}{'null mean':>12s}{'excess':>12s}{'excess 95% CI':>26s}")
    for state, key in zip(STATES, STATE_KEYS, strict=True):
        lo, hi = induced["excess_ci"][key]
        cells = [
            _pct(induced["null_diagonal_mean"][key]),
            _pct(induced["excess_persistence"][key]),
            _pct_range(lo, hi),
        ]
        lines.append(f"  {state:>9s}{cells[0]:>12s}{cells[1]:>12s}{cells[2]:>26s}")

    ci = a["confidence_intervals"]
    lower, upper = ci["transition_matrix"]["lower"], ci["transition_matrix"]["upper"]
    sd_lo, sd_hi = ci["stationary_distribution"]["lower"], ci["stationary_distribution"]["upper"]
    eff = a["effective_n"]
    lines.append("")
    lines.append(
        f"{ci['ci'] * PERCENT:.0f}% block-bootstrap CI ({ci['n_resamples']} resamples, "
        f"{ci['block_days']}-day blocks):"
    )
    lines.append(f"  {'':>9s}{'diagonal':>24s}{'stationary':>24s}{'effective n':>14s}")
    for i, (state, key) in enumerate(zip(STATES, STATE_KEYS, strict=True)):
        cells = [
            _pct_range(_diag(lower, i), _diag(upper, i)),
            _pct_range(sd_lo[key], sd_hi[key]),
            _num(eff[key], 1),
        ]
        lines.append(f"  {state:>9s}{cells[0]:>24s}{cells[1]:>24s}{cells[2]:>14s}")
    lines.append("  Effective n = transitions / window (overlapping counts are not independent).")

    rolling, ew, shrunk = a["rolling_matrix"], a["ew_matrix"], a["shrunk_matrix"]
    lines.append("")
    lines.append(
        f"Drift: rolling ({a['rolling_bars']} bars) vs full-history diagonal, "
        f"TVD = {_num(a['drift_tvd'], 4)} (worst row):"
    )
    lines.append(
        f"  {'':>9s}{'full':>12s}{'rolling':>12s}"
        f"{'EW hl=' + _num(a['ew_half_life'], 0):>14s}"
        f"{'shrunk a=' + _num(a['shrinkage_alpha'], 0):>14s}"
    )
    for i, state in enumerate(STATES):
        cells = [
            _pct(_diag(matrix, i)),
            _pct(_diag(rolling, i)),
            _pct(_diag(ew, i)),
            _pct(_diag(shrunk, i)),
        ]
        lines.append(f"  {state:>9s}{cells[0]:>12s}{cells[1]:>12s}{cells[2]:>14s}{cells[3]:>14s}")

    lines.append("")
    mixing = a["mixing_time"]
    if mixing is None:
        lines.append(
            f"Mixing time (TVD <= {a['mixing_epsilon']}): not reached within max steps "
            "(periodic or reducible chain)"
        )
    else:
        lines.append(
            f"Mixing time (TVD <= {a['mixing_epsilon']}): {mixing} steps until P^n "
            "is at the stationary mix from every start"
        )
    tests = a["tests"]
    ag = tests["anderson_goodman"]
    if ag is None:
        lines.append("Anderson-Goodman first-order dependence test (stride sequence): n/a")
    else:
        lines.append(
            f"Anderson-Goodman first-order dependence test (stride sequence): "
            f"chi2 = {_num(ag['statistic'])}, df = {ag['df']}, p = {_num(ag['p_value'], 4)}"
        )
    hg = tests["homogeneity"]
    if hg is None:
        lines.append("Homogeneity test (stride sequence): n/a (series too short)")
    else:
        verdict = (
            "matrix stable at 5%"
            if hg["stable"]
            else "matrix NOT stable at 5% (regime probabilities drift)"
        )
        lines.append(
            f"Homogeneity test (stride sequence, {hg['n_periods']} sub-periods): "
            f"chi2 = {_num(hg['statistic'])}, "
            f"df = {hg['df']}, p = {_num(hg['p_value'], 4)} -> {verdict}"
        )
    return lines


def print_pretty(result: dict) -> None:
    """Print the pretty report exactly as the companion script does."""
    print(render(result), end="")
