---
name: regime
description: >-
  Detect the market regime (Bull / Bear / Sideways) for any ticker or CSV price
  series and turn it into a signal, a confirmation gate, or a position-size
  filter, with the honest statistics the companion skill lacks. Use this whenever
  the user asks for regime detection, a Markov transition matrix, a stationary
  regime distribution, n-step regime probabilities, a no-lookahead walk-forward
  regime backtest, or an HMM regime read on a ticker (yfinance) or on their own
  CSV. Adds a non-overlapping stride matrix, labeller-induced null percentiles,
  block-bootstrap confidence intervals, drift and shrinkage matrices, a
  cost-aware walk-forward with gross and net metrics, benchmarks, permutation
  p-values, PSR/DSR, an untouched holdout, and a walk-forward filtered HMM.
  Framework by Roan (@RohOnChain), reproduced bit-for-bit then extended.
---

# Regime: honest Markov regime detection for any asset

One command answers: **what regime are we in, how sticky is it really, and would
trading it have survived costs and a null test?** The output is a strict superset of
the companion `markov-hedge-fund-method` skill, so anything built on that contract
keeps working and can start reading the honest fields.

Backtests are historical, not forward-looking.

## Install as a local plugin

```
/plugin marketplace add /Users/simonlacey/Documents/GitHub/hedgeFund
/plugin install hedgefund@hedgefund
```

The plugin root is the project checkout. The CLI is the `hedgefund` entry point of the
uv project, so every command runs through `uv run`.

## How to invoke

From the project root:

```bash
uv run hedgefund analyze --ticker SPY --json
```

When installed as a plugin, point uv at the plugin root instead of changing directory:

```bash
uv run --project ${CLAUDE_PLUGIN_ROOT} hedgefund analyze --ticker BTC-USD --json
uv run --project ${CLAUDE_PLUGIN_ROOT} hedgefund analyze --csv ./my_prices.csv --json
```

- `--json` prints exactly one JSON object to stdout and nothing else. On failure it
  prints `{"error": "..."}` and exits non-zero. Drop `--json` for the terminal report.
- `--csv` needs a date column and a close column; common names are auto-detected.
- Defaults come from `config/default.yaml`: `--window 20`, `--threshold 0.05`,
  `--years 10`, `--min-train 252`. Override with flags; `--config PATH` swaps the file.
- Fast first call: add `--no-hmm --null-paths 50 --bootstrap-resamples 50`. The
  walk-forward HMM and the Monte Carlo nulls are the slow parts. Use the full defaults
  for a read you intend to act on.
- Other flags: `--seed`, `--refit expanding|rolling|rolling:N`, `--rule
  sign|conviction|persistence|long_only`, `--dead-band`, `--cost-profile NAME`,
  `--hmm-seeds 0,1,2`, `--hmm-refit-every N`, `--hmm-emission gaussian|student_t|auto`.
- The first `uv run` resolves dependencies once; later calls are instant.

Secondary commands: `hedgefund backtest` (the cost-aware walk-forward alone),
`hedgefund scan --universe NAME --json` (every symbol of a universe file, ranked by
net Sharpe, written under `jobs/<run-id>/`), `hedgefund report <run-id>` (regenerate
the HTML of a cached run). See `reference/json-contract.md` for their objects.

Data sources are yfinance and CSV. Alpaca is data-only, paper endpoint, opt-in, and
not wired in yet. Nothing in this project places, replaces or cancels orders.

## The JSON contract

Index order is `["Bear","Sideways","Bull"]` everywhere: every 3x3 matrix, every
`{"bear","sideways","bull"}` object, every confusion table. The full field list with
types is in `reference/json-contract.md`.

### Companion fields (unchanged)

| Field | Type | Meaning |
|---|---|---|
| `source` | str | ticker or CSV path analysed |
| `rows` | int | price rows used |
| `date_start` | str | ISO date of the first row |
| `date_end` | str | ISO date of the last row |
| `params` | obj | `window`, `threshold`, `min_train`, `return_type` actually used |
| `states` | list | `["Bear","Sideways","Bull"]`, fixed index order 0, 1, 2 |
| `current_regime` | str | regime as of the last bar |
| `next_state_probabilities` | obj | `bear`, `sideways`, `bull`: P(next \| current) |
| `signal` | float | `bull - bear` in [-1, 1]; sign = direction, magnitude = conviction |
| `transition_matrix` | 3x3 | row = from, col = to, rows sum to 1 |
| `persistence_diagonal` | obj | P(stay) from **overlapping** daily labels, inflated by the window autocorrelation |
| `stationary_distribution` | obj | long-run share of each regime, sums to 1 |
| `walk_forward` | obj | `sharpe`, `max_drawdown`, `n_trades`; gross, no costs; NaN on short history |
| `hmm` | obj | `available` plus `regimes` and `caveat`, extended in Phase 4; or `reason` |
| `framework` | str | `"Roan (@RohOnChain)"` |
| `disclaimer` | str | `"Backtests are historical, not forward-looking."` |

### Extended fields a strategy should read

| Field | Type | Meaning |
|---|---|---|
| `stride_matrix` | 3x3 | matrix of `labels[::window]`: transitions counted `window` bars apart, no overlap |
| `stride_diagonal` | obj | P(stay) over one window step. **Use this for persistence**, not `persistence_diagonal` |
| `null_percentile.labeller_induced.excess_persistence` | obj | real diagonal minus the diagonal the labeller manufactures on iid noise; near zero means the stickiness is an artefact |
| `tests.anderson_goodman.p_value` | float or null | independence test on the stride chain; large means there is no Markov structure to trade |
| `rolling_matrix` | 3x3 | matrix from the trailing `rolling_bars` labels |
| `drift_tvd` | float or null | worst-row distance between the full and rolling matrices; large means the chain has moved |
| `shrunk_matrix` | 3x3 | rolling counts shrunk toward the full matrix; the matrix to size from when drift is high |
| `mixing_time` | int or null | bars until the chain forgets its start; `null` for a reducible chain |
| `confidence_intervals` | obj | block-bootstrap bounds on the matrix and the stationary vector |
| `backtest.gross` | obj | walk-forward metric set before costs (`sharpe`, `max_drawdown`, `turnover`, ...) |
| `backtest.net` | obj | the same after fees, slippage, borrow and funding. **Quote net, never gross alone** |
| `backtest.benchmarks` | obj | `buy_and_hold`, `constant_50`, `long_only` under the same cost model |
| `backtest.null.permutation` | obj | whole pipeline rerun on permuted returns; `p_value_net` is the significance number |
| `backtest.holdout` | obj or null | last `holdout_years` scored separately, never used for selection |
| `backtest.psr` | float or null | probabilistic Sharpe of the net returns |
| `backtest.dsr` | obj | Sharpe deflated for the parameter grid searched |
| `backtest.lag` | obj | crash and recovery detection lag. **Hindsight diagnostic**: never a signal |
| `hmm.decision` | obj | `regime` held by the walk-forward HMM with hysteresis on the filtered probability |
| `hmm.filtered_probabilities` | obj | last bar's P(state \| data up to now); never smoothed |
| `hmm.first_to_flip` | obj | which detector enters Bear or Bull first; positive lead = HMM first |
| `hmm.pnl_comparison` | obj | threshold vs HMM vs buy-and-hold, gross and net |

`backtest.null`, `backtest.holdout` and most `hmm` additions are `null` or absent on a
history shorter than `min_train + 30` bars. Check for `null` before reading.

## Composition: slot it into what the user already has

Run it, read a few fields, gate the user's existing logic. Read the JSON once:

```python
import json, subprocess


def regime(symbol: str) -> dict:
    out = subprocess.check_output(
        ["uv", "run", "hedgefund", "analyze", "--ticker", symbol, "--json"]
    )
    return json.loads(out)


r = regime("SPY")
```

### (a) Regime confirmation gate on an existing strategy

Companion form: only take longs when the chain agrees.

```python
if my_strategy_says_long and r["signal"] > 0:
    enter_long()
elif my_strategy_says_long:
    skip()
```

Honest form: require the edge to be real net of costs, the regime to be sticky without
overlap, and the two detectors to agree.

```python
null = r["backtest"]["null"]
hmm = r["hmm"]
edge_is_real = null is not None and null["permutation"]["p_value_net"] < 0.05
sticky = (r["stride_diagonal"]["bull"] or 0.0) > 0.5  # window-step, not daily
detectors_agree = (not hmm["available"]) or hmm["decision"]["regime"] == r["current_regime"]

if my_strategy_says_long and r["signal"] > 0 and edge_is_real and sticky and detectors_agree:
    enter_long()
else:
    skip()
```

### (b) Stationary bear share as a position-size / tail-risk filter

Companion form: scale size by the long-run bear share.

```python
bear_baseline = r["stationary_distribution"]["bear"]
size = base_size * (1.0 - bear_baseline)
```

Honest form: size from the matrix that reflects the recent chain, and cut size when
the chain has drifted or the walk-forward did not survive costs.

```python
drift = r["drift_tvd"] or 0.0
matrix = r["shrunk_matrix"] if drift > 0.2 else r["transition_matrix"]
row = r["states"].index(r["current_regime"])
p_bear_next = matrix[row][0]  # column 0 = Bear
size = base_size * (1.0 - bear_baseline) * (1.0 - p_bear_next) * (1.0 - drift)
if r["backtest"]["net"]["sharpe"] is None or r["backtest"]["net"]["sharpe"] <= 0:
    size = 0.0  # did not survive the cost model
```

### (c) Standalone signal

Companion form: position = signal.

```python
position = r["signal"]  # +0.6 is 60% long, -0.4 is 40% short, near 0 is flat
```

Honest form: trade the signal only when net metrics, the holdout and the null agree,
and never read `backtest.lag`, which looks at future bars by construction.

```python
bt = r["backtest"]
ok = (
    bt["null"] is not None
    and bt["null"]["permutation"]["p_value_net"] < 0.05
    and bt["holdout"] is not None
    and (bt["holdout"]["holdout"]["net"]["sharpe"] or 0.0) > 0.0
    and bt["net"]["sharpe"] > bt["benchmarks"]["buy_and_hold"]["sharpe"]
)
position = r["signal"] if ok else 0.0
```

## Notes

- **No lookahead.** Every matrix, null, bootstrap and HMM refit uses only bars before
  the decision date. The HMM reads filtered probabilities, never smoothed or Viterbi.
- **Persistence is overstated by the companion.** `persistence_diagonal` counts
  overlapping windows. Read `stride_diagonal` and the labeller-induced excess instead.
- **Gross never travels alone.** `backtest.gross` and `backtest.net` always ship together
  with `backtest.params.costs` echoed.
- **HMM degrades gracefully.** `hmm.available` is `false` with a `reason` when hmmlearn
  is missing, `--no-hmm` is set, or the history is too short; every other field is valid.
- Backtests are historical, not forward-looking.
