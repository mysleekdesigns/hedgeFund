"""Pretty terminal output, line for line the companion script's ``_print_pretty``.

``render`` returns the plain text (no rich markup) so it can be diffed against the
companion's stdout; ``print_pretty`` writes it unchanged. When the dict carries the
Phase 2 fields (``stride_matrix`` present) an "Honest statistics" section is inserted
between the HMM block and the footer; a dict without them renders the companion text
unchanged. When it carries the Phase 3 ``backtest`` block a "Backtest (Phase 3)" section
follows, always showing gross and net side by side with the cost assumption stated
(``.claude/rules/no-lookahead.md``). ``render_backtest`` prints the slim dict behind
``hedgefund backtest``: header, the same section, footer. Any ``None`` value prints ``n/a``.
"""

import math
from collections.abc import Callable

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

BACKTEST_HEADER = "Backtest (Phase 3):"
BACKTEST_METRIC_ROWS = [  # (label, metric key, formatter) for the gross vs net table
    ("Sharpe (annualised)", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Calmar", "calmar", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Total return", "total_return", "pct"),
    ("Annual return", "annual_return", "pct"),
    ("Hit rate", "hit_rate", "pct"),
    ("Profit factor", "profit_factor", "num"),
    ("PSR (P[true Sharpe > 0])", "psr", "num"),
    ("Min track record (days)", "min_track_record", "days"),
]
BENCHMARK_ROWS = [  # (label, key under backtest.benchmarks); the strategy row comes first
    ("buy-and-hold", "buy_and_hold"),
    ("constant 50%", "constant_50"),
    ("long-only", "long_only"),
]
NULL_ROWS = [  # (label, key under backtest.null); each entry carries gross and net keys
    ("random-sign", "random_sign"),
    ("label-shuffle", "label_shuffle"),
    ("permutation", "permutation"),
]
DSR_HYBRID_CAVEAT = (
    "  Conservative hybrid, not textbook DSR: PSR of the default cell's full-sample net "
    "returns against SR0 from the in-sample grid; the default cell is not the argmax and "
    "its sample is longer than the grid cells'."
)
HMM_PHASE4_KEYS = (  # any of these under ``hmm`` turns on the Phase 4 lines
    "features",
    "fit",
    "filtered_probabilities",
    "decision",
    "walk_forward",
    "agreement_with_threshold",
    "first_to_flip",
    "pnl_comparison",
)
HMM_PNL_ROWS = (  # (label, key under hmm.pnl_comparison)
    ("threshold", "threshold"),
    ("HMM", "hmm"),
    ("buy-and-hold", "buy_and_hold"),
)
HMM_FLIP_LABELS = {"bear": "Bear", "bull": "Bull"}  # first_to_flip keys -> printed regime
LAG_LABELLER_LABELS = {
    "threshold": "threshold (plain)",
    "zscore": "z-score",
    "ew": "EW return",
    "ensemble": "ensemble",
    "hysteresis": "hysteresis",
}

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
        lines.append("Hidden Markov Model (Baum-Welch, walk-forward filtered):")
        for r in hmm["regimes"]:
            lines.append(
                f"  {r['label']:<9s} (latent state {r['latent_state']}): "
                f"{r['mean_daily_return'] * PERCENT:+.3f}% mean daily return"
            )
        lines.append(f"  Note: {hmm['caveat']}")
        lines.extend(_hmm_phase4_lines(hmm))
    else:
        lines.append("")
        lines.append(
            f"HMM skipped: {hmm.get('reason', 'unavailable')} "
            "(observable model above is unaffected)."
        )

    if "stride_matrix" in a:
        lines.extend(_honest_statistics_lines(a))
    if "backtest" in a:
        lines.extend(_backtest_lines(a["backtest"]))

    lines.extend(FOOTER)
    return "\n".join(lines) + "\n"


def render_backtest(result: dict) -> str:
    """Pretty text for a ``backtest_only()`` dict: header, the backtest section, footer."""
    a = result
    lines: list[str] = [
        "",
        (
            f"markov-regime backtest — source={a['source']} "
            f"window={a['params']['window']} threshold={a['params']['threshold']} "
            f"min_train={a['params']['min_train']}"
        ),
        f"  {a['rows']} rows | {a['date_start']} -> {a['date_end']}",
    ]
    lines.extend(_backtest_lines(a["backtest"]))
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


def _get(node: dict | None, *keys: str) -> object:
    """Walk nested dict keys, returning ``None`` as soon as a level is missing or ``None``."""
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _text(value: object) -> str:
    """A string field, ``n/a`` for ``None``."""
    return str(value) if value is not None else NA


def _hmm_phase4_lines(hmm: dict) -> list[str]:
    """The Phase 4 lines of the HMM block: features, fit, decision, walk-forward,
    agreement, first-to-flip, P&L comparison (net beside gross). A dict carrying only
    the companion keys (``available``, ``regimes``, ``caveat``) renders nothing here;
    every key is read with ``.get``."""
    if not any(key in hmm for key in HMM_PHASE4_KEYS):
        return []
    lines: list[str] = []
    features = hmm.get("features")
    names = _get(features, "names")
    lines.append(
        f"  Features: {', '.join(names) if isinstance(names, list) else NA} "
        f"(vol windows {_count(_get(features, 'vol_short'))}/"
        f"{_count(_get(features, 'vol_long'))}; "
        f"standardised with {_text(_get(features, 'standardisation'))})"
    )
    fit = hmm.get("fit")
    seeds = _get(fit, "seeds")
    lines.append(
        f"  Fit: best seed {_count(_get(fit, 'seed'))} of restarts "
        f"[{', '.join(str(s) for s in seeds) if isinstance(seeds, list) else NA}] | "
        f"log-likelihood {_num(_get(fit, 'log_likelihood'), 2)} | "
        f"emission {_text(_get(fit, 'emission'))} | "
        f"converged {_text(_get(fit, 'converged'))}"
    )
    decision = hmm.get("decision")
    probs = hmm.get("filtered_probabilities")
    lines.append(
        f"  Decision: {_text(_get(decision, 'regime'))} "
        f"(raw argmax {_text(_get(decision, 'raw_regime'))}) | filtered "
        f"Bear {_pct(_get(probs, 'bear'))} Sideways {_pct(_get(probs, 'sideways'))} "
        f"Bull {_pct(_get(probs, 'bull'))} | enter {_num(_get(decision, 'enter'))} "
        f"exit {_num(_get(decision, 'exit'))} confirm "
        f"{_count(_get(decision, 'confirm_bars'))} bars"
    )
    wf = hmm.get("walk_forward")
    lines.append(
        f"  Walk-forward: {_count(_get(wf, 'n_days'))} decision days, "
        f"{_count(_get(wf, 'n_refits'))} refits on past data only"
    )
    agree = hmm.get("agreement_with_threshold")
    lines.append(
        f"  Agreement with the threshold labeller: {_pct(_get(agree, 'share_agree'))} "
        f"of {_count(_get(agree, 'n'))} common days"
    )
    flip = hmm.get("first_to_flip")
    lines.append(
        f"  First to flip (matched within +/-{_count(_get(flip, 'match_window'))} bars; "
        "lead > 0 = HMM earlier):"
    )
    for key, label in HMM_FLIP_LABELS.items():
        entry = _get(flip, key)
        lines.append(
            f"    into {label:<5s} HMM first {_count(_get(entry, 'hmm_first'))} | "
            f"threshold first {_count(_get(entry, 'threshold_first'))} | "
            f"same bar {_count(_get(entry, 'same_bar'))} | "
            f"mean lead {_num(_get(entry, 'mean_lead_bars'), 1)} bars | "
            f"unmatched HMM {_count(_get(entry, 'hmm_unmatched'))}, "
            f"threshold {_count(_get(entry, 'threshold_unmatched'))}"
        )
    pnl = hmm.get("pnl_comparison")
    lines.append(
        f"  P&L comparison, sign rule on each held regime, same cost model "
        f"({_count(_get(pnl, 'n_days'))} days, "
        f"{_num(_get(pnl, 'costs', 'per_side_bps'), 1)} bps per side):"
    )
    for label, key in HMM_PNL_ROWS:
        entry = _get(pnl, key)
        lines.append(
            f"    {label:<13s}net Sharpe {_num(_get(entry, 'net', 'sharpe'), 3)} | "
            f"total {_pct(_get(entry, 'net', 'total_return'))} | "
            f"max DD {_pct(_get(entry, 'net', 'max_drawdown'))} | "
            f"turnover {_num(_get(entry, 'net', 'turnover'), 1)}/yr | "
            f"gross Sharpe {_num(_get(entry, 'gross', 'sharpe'), 3)}"
        )
    lines.append(f"  Note: {_text(hmm.get('note'))}")
    return lines


def _days(value: object) -> str:
    """A track-record length in whole days, ``n/a`` for ``None`` (undefined or infinite)."""
    return f"{value:.0f}" if _is_finite(value) else NA


def _count(value: object) -> str:
    """An integer count, ``n/a`` for ``None``."""
    return str(int(value)) if _is_finite(value) else NA


def _lag_pair(entry: dict | None, key: str, fmt: Callable[[object], str]) -> str:
    """``crash/recovery`` values of one lag statistic; ``n/a`` for a missing labeller.

    Uses ``.get`` semantics through ``_get`` so a key the lag module does not emit yet
    (``n_flips``, ``false_flips``, ``false_flip_rate``) prints ``n/a/n/a`` instead of raising.
    """
    if entry is None:
        return NA
    return f"{fmt(_get(entry, 'crash', key))}/{fmt(_get(entry, 'recovery', key))}"


def _fmt(value: object, kind: str) -> str:
    if kind == "pct":
        return _pct(value)
    if kind == "days":
        return _days(value)
    return _num(value, 3)


def _backtest_lines(bt: dict) -> list[str]:
    """The Phase 3 section: costs, gross vs net, benchmarks, nulls, PSR/DSR, holdout,
    cost sensitivity, the window x threshold heatmap and the lag diagnostic."""
    lines: list[str] = []
    params = bt["params"]
    costs = params["costs"]
    gross, net = bt["gross"], bt["net"]

    lines.append("")
    lines.append(BACKTEST_HEADER)
    lines.append(
        f"  Costs: {_num(costs['per_side_bps'], 1)} bps per side "
        f"(fee {_num(costs['fee_bps'], 1)} + slippage {_num(costs['slippage_bps'], 1)}), "
        f"borrow {_num(costs['borrow_bps'], 0)} bps/yr on shorts, "
        f"funding {_num(costs['funding_bps'], 0)} bps/yr on gross notional; "
        f"annual cost drag {_pct(bt['annual_cost'])}"
    )
    rolling = f" ({params['rolling_bars']} bars)" if params["refit"] == "rolling" else ""
    lines.append(
        f"  Rule: {params['rule']}, refit {params['refit']}{rolling}, "
        f"dead band {_num(params['dead_band'], 2)}, {net['n_days']} days evaluated"
    )
    lines.append(f"  {'':<26s}{'gross':>12s}{'net':>12s}")
    for label, key, kind in BACKTEST_METRIC_ROWS:
        lines.append(f"  {label:<26s}{_fmt(gross[key], kind):>12s}{_fmt(net[key], kind):>12s}")
    lines.append(
        f"  Turnover (one-way notional per year): {_num(net['turnover'], 1)} | "
        f"time in market: {_pct(net['time_in_market'])}"
    )

    lines.append("")
    lines.append("Benchmarks (net of the same costs):")
    lines.append(f"  {'':<14s}{'sharpe':>10s}{'max DD':>10s}{'total':>10s}{'turnover':>10s}")
    rows = [("strategy net", net)] + [
        (label, bt["benchmarks"][key]) for label, key in BENCHMARK_ROWS
    ]
    for label, m in rows:
        lines.append(
            f"  {label:<14s}{_num(m['sharpe'], 3):>10s}{_pct(m['max_drawdown']):>10s}"
            f"{_pct(m['total_return']):>10s}{_num(m['turnover'], 1):>10s}"
        )
    lines.append(f"  {bt['benchmarks']['note']}")

    null = bt["null"]
    lines.append("")
    n_paths = _get(null, "random_sign", "n_paths")
    paths = f"{n_paths} paths" if n_paths is not None else "no paths (history too short)"
    lines.append(
        f"Null models of the Sharpe, gross and net percentile side by side ({paths}; "
        "100 = above every null draw):"
    )
    lines.append(
        f"  {'':<16s}{'gross pct':>10s}{'gross p':>9s}{'net pct':>10s}{'net p':>9s}"
        f"{'null turnover':>15s}"
    )
    for label, key in NULL_ROWS:
        entry = _get(null, key)
        lines.append(
            f"  {label:<16s}{_num(_get(entry, 'percentile_gross'), 1):>10s}"
            f"{_num(_get(entry, 'p_value_gross'), 4):>9s}"
            f"{_num(_get(entry, 'percentile_net'), 1):>10s}"
            f"{_num(_get(entry, 'p_value_net'), 4):>9s}"
            f"{_num(_get(entry, 'null_mean_turnover'), 1):>15s}"
        )
    lines.append(
        f"  strategy turnover {_num(net['turnover'], 1)}/yr. "
        "p = (count(null >= real) + 1) / (N + 1); net charges every null path the cost "
        "model, gross charges nothing. A null that trades far more than the strategy pays "
        "far more in costs, so read the gross pair first."
    )

    dsr = bt["dsr"]
    lines.append("")
    lines.append(
        f"Probabilistic Sharpe (net): {_num(bt['psr'], 3)} | Deflated Sharpe: "
        f"{_num(dsr['dsr'], 3)} against SR0 = {_num(dsr['sr0'], 4)} per day from "
        f"{dsr['n_trials']} trials of the parameter search"
    )
    lines.append(DSR_HYBRID_CAVEAT)

    holdout = bt["holdout"]
    cutoff = _get(holdout, "cutoff")
    years = params["holdout_years"]
    lines.append(
        f"Holdout ({years}y, from {cutoff if cutoff is not None else NA}): "
        f"in-sample net Sharpe {_num(_get(holdout, 'in_sample', 'net', 'sharpe'), 3)} "
        f"({_get(holdout, 'in_sample', 'net', 'n_days') or 0} days) | "
        f"holdout net Sharpe {_num(_get(holdout, 'holdout', 'net', 'sharpe'), 3)} "
        f"({_get(holdout, 'holdout', 'net', 'n_days') or 0} days)"
    )
    default_w = _get(holdout, "default_cell", "window")
    default_t = _get(holdout, "default_cell", "threshold")
    if default_w is None:
        default_w, default_t = params["window"], params["threshold"]
    best = bt["sensitivity"]["best"]
    if best["window"] is not None:
        best_text = (
            f"best in-sample cell (window {best['window']}, threshold {best['threshold']}): "
            f"in-sample net {_num(best['net_sharpe'], 3)}, "
            f"holdout net {_num(best.get('holdout_net_sharpe'), 3)} "
            f"(gross {_num(best.get('holdout_gross_sharpe'), 3)})"
        )
    else:
        best_text = "best in-sample cell: n/a (no cell could be scored)"
    lines.append(
        f"  The line above scores the default cell (window {default_w}, threshold {default_t}); "
        f"{best_text}"
    )

    lines.append("")
    lines.append("Cost sensitivity (per-side bps -> net):")
    lines.append(f"  {'bps':>6s}{'sharpe':>10s}{'total':>10s}{'max DD':>10s}{'cost/yr':>10s}")
    for row in bt["cost_sensitivity"]:
        lines.append(
            f"  {row['bps_per_side']:>6d}{_num(row['sharpe'], 3):>10s}"
            f"{_pct(row['total_return']):>10s}{_pct(row['max_drawdown']):>10s}"
            f"{_pct(row['annual_cost']):>10s}"
        )

    sens = bt["sensitivity"]
    lines.append("")
    lines.append(
        f"Sensitivity: in-sample net Sharpe, rows = window, cols = threshold "
        f"({sens['n_trials']} trials, {_pct(sens['share_positive'])} of cells positive):"
    )
    lines.append(f"  {'window':>8s}" + "".join(f"{t:>8.2f}" for t in sens["thresholds"]))
    for w, row in zip(sens["windows"], sens["net_sharpe"], strict=True):
        lines.append(f"  {w:>8d}" + "".join(f"{_num(cell, 2):>8s}" for cell in row))
    best = sens["best"]
    if best["window"] is not None:
        lines.append(
            f"  Best cell: window {best['window']}, threshold {best['threshold']}, "
            f"net Sharpe {_num(best['net_sharpe'], 3)} (in-sample; the holdout never saw it)"
        )
    else:
        lines.append("  Best cell: n/a (no cell could be scored)")

    lag = bt["lag"]
    lines.append("")
    lines.append(
        f"Detection lag, median bars from turning point to flip (+/-{lag['order']} bar extrema); "
        "crash/recovery pairs:"
    )
    lines.append(
        f"  {'labeller':<18s}{'crash':>8s}{'recovery':>10s}{'undetected':>12s}{'flips':>12s}"
        f"{'false flips':>18s}"
    )
    for key, label in LAG_LABELLER_LABELS.items():
        entry = lag.get(key)
        lines.append(
            f"  {label:<18s}{_num(_get(entry, 'crash', 'median'), 1):>8s}"
            f"{_num(_get(entry, 'recovery', 'median'), 1):>10s}"
            f"{_lag_pair(entry, 'n_undetected', _count):>12s}"
            f"{_lag_pair(entry, 'n_flips', _count):>12s}"
            f"{_lag_pair(entry, 'false_flip_rate', _pct):>18s}"
        )
    lines.append(
        "  flips = label flips into Bear (crash) / Bull (recovery); false flips = share of "
        "those with no turning point in the preceding order bars."
    )
    lines.append(f"  Note: {lag['note']}.")
    return lines


def print_pretty(result: dict) -> None:
    """Print the pretty report exactly as the companion script does."""
    print(render(result), end="")


def print_backtest(result: dict) -> None:
    """Print the ``hedgefund backtest`` report."""
    print(render_backtest(result), end="")
