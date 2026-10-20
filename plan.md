# hedgeFund — Project Plan

**Status:** planning complete, reference plugin installed, no project code written yet
**Date:** 2026-10-19
**Source video:** "I Re-Created A Quant Trading Strategy With Claude Code (Insanely Cool)" — Lewis Jackson Investing, 27 min, premiered 2026-05-19
https://www.youtube.com/watch?v=ZVMTeDBmSrI
**Companion repo (MIT):** https://github.com/jackson-video-resources/markov-hedge-fund-method
**Underlying framework:** Roan (@RohOnChain)

---

## 1. What the video teaches

The video walks through a "hedge fund method" for regime detection: a three-state **Markov chain** fitted to an asset's price history, then used as a trade signal, a risk filter, and a backtested strategy. The chapter structure of the video is the method itself:

| Time  | Chapter                      | Core idea |
|-------|------------------------------|-----------|
| 0:00  | How Quants Actually Trade    | Quants quantify the market into states and calculate probabilities; they do not read trend lines or indicators. |
| 2:08  | Defining Market States       | Label every day by its **trailing 20-day return**: ≥ +5% → Bull, ≤ −5% → Bear, otherwise Sideways. |
| 4:57  | The Markov Property          | Tomorrow's regime depends only on today's regime, not the path that led here. |
| 6:59  | The Transition Matrix        | Count every day-to-day state transition, row-normalise → 3×3 matrix of P(next \| current) (maximum-likelihood estimate). |
| 8:04  | Stickiness and Persistence   | The diagonal of the matrix (P(stay)) is the "load-bearing" number. High persistence = regime is sticky = size up; low = size down. |
| 9:53  | Squaring the Matrix          | P² gives 2-day-ahead probabilities, Pⁿ gives n-day-ahead (Chapman-Kolmogorov). |
| 12:12 | Stationary Distribution      | As n grows, Pⁿ converges to the long-run regime mix. This is why ~28-day forecasts "collapse" to the baseline and carry no information. |
| 13:50 | Generating Trade Signals     | signal = P(Bull next) − P(Bear next) ∈ [−1, 1]. Sign = direction, magnitude = conviction. |
| 16:31 | Walk-Forward Backtesting     | Re-estimate the matrix at every step using only data before that day. No lookahead. Report Sharpe and max drawdown. |
| 18:54 | Hidden Markov Model          | Replace hand-picked ±5% labels with latent states learned by a Gaussian HMM (Baum-Welch + Viterbi) on daily returns. |
| 21:05 | Installing the Claude Skill  | A Claude Code plugin/skill that runs the whole pipeline on any ticker or CSV and returns JSON. |
| 24:53 | Pine Script on TradingView   | A v5 overlay indicator: regime ribbon, live transition matrix table, stationary-distribution table, regime banner. |

### The reference implementation (companion repo)

Verified live on 2026-10-19 (ignore the pinned video comment saying the GitHub is unavailable). Files:

| Path | Purpose |
|------|---------|
| `skills/regime/SKILL.md` | the Claude Code skill: invocation, JSON contract, composition patterns |
| `scripts/markov_regime.py` | the engine the skill calls (PEP 723 inline deps, run with `uv`) |
| `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` | plugin marketplace manifests |
| `markov-hedge-fund-method.md` | original on-camera one-shot install prompt (older variant, ±2% default) |
| `pine-script/markov-hedge-fund-method.pine` | TradingView v5 overlay |

Install in Claude Code:

```
/plugin marketplace add jackson-video-resources/markov-hedge-fund-method
/plugin install markov-hedge-fund-method@markov-hedge-fund-method
/markov-hedge-fund-method:regime
```

The repo ships a single-file Python module (`scripts/markov_regime.py`, PEP 723 inline deps, run with `uv`) exposing:

- `label_regimes(close, window=20, threshold=0.05)`
- `build_transition_matrix(labels)` — MLE counting
- `nstep_forecast(P, n)` — matrix power
- `stationary_distribution(P)` — left eigenvector for eigenvalue 1
- `signal_from_matrix(P, state)` — bull_prob − bear_prob
- `walk_forward_backtest(close, labels, min_train=252)` — incremental O(n) refit, ±1/0 position, next-day return, Sharpe + max DD
- `fit_hmm(returns, n_components=3)` — hmmlearn GaussianHMM, graceful degrade if unavailable
- `analyze(...)` — returns one JSON dict (the contract agents consume: `current_regime`, `signal`, `transition_matrix`, `persistence_diagonal`, `stationary_distribution`, `walk_forward`, `hmm`)

Data in: `--ticker` via yfinance or `--csv` with a date + close column. Defaults: window 20, threshold ±5%, 10 years, min_train 252.

The skill doc describes three composition patterns we should preserve: (a) regime **confirmation gate** on an existing strategy, (b) stationary bear share as a **position-size / tail-risk filter**, (c) **standalone signal** where position = signal.

### Known weaknesses called out by viewers (and worth fixing, not ignoring)

These came from the comment thread and are the main reason our build should go beyond a copy of the repo:

1. **Overlapping-window autocorrelation.** Labelling each bar from a 20-bar rolling return and then counting bar-to-bar transitions means consecutive transitions share 19/20 bars of data. The diagonal inflates to 80–95% on every asset. That is window autocorrelation, not regime persistence. Fix: stride/non-overlapping sampling, or compare against a null model.
2. **Lookahead in the Pine labels.** The Pine Script requires a regime to hold `min_regime_hold` bars before labelling, then draws the label back at the bar where the flip started. Looks great in hindsight, not tradeable live.
3. **Built-in lag.** A 20-day trailing return can take many days of decline before flipping to Bear; you are long through most of the drop.
4. **Non-stationarity.** A matrix fitted to 10 years is static. Probabilities drift; a single long-run matrix can be wrong for the current year.
5. **HMM on raw price is wrong.** Feed stationary features (returns, realised vol, normalised volume), not price levels. HMM regime changes also tend to be confirmed only after the move.
6. **No costs, no benchmarks.** The walk-forward reports Sharpe and max DD only. Need fees/slippage, win rate, per-horizon P&L, and comparison to buy-and-hold and to a shuffled-label null.

### Fixes for the known weaknesses

Each fix below names the root cause, what we will implement, how we prove it worked, and which phase owns it. Sources are listed at the end of this section.

#### Fix 1 — Overlapping-window autocorrelation (inflated diagonal)

**Root cause.** The label at day t and day t+1 are computed from windows that share 19 of 20 returns, so consecutive "transitions" are not independent draws. Any smoothed signal stays the same from one bar to the next most of the time. The 93% Sideways persistence we measured on SPY is mostly that.

**What we implement.**
- *Stride counting.* Count transitions only between labels w bars apart (t → t+w), so the two windows do not overlap. Report this as the primary matrix.
- *Run-length persistence.* Treat each regime as an episode. Estimate persistence as 1 − 1 / (mean episode length in bars). For a true first-order chain the episode length is geometric, so this gives the same answer as counting. Phase 2 finding: on the daily sequence it is algebraically the overlapping diagonal (same exits-per-visit ratio), so it is only overlap-free when computed on the stride-subsampled labels, where it equals the stride diagonal up to the censored last run. Reported on the stride sequence; do not treat it as an independent estimator.
- *Labeller-induced null.* Simulate iid returns with the asset's mean and volatility, run the identical 20-day labeller on them, and estimate the matrix. The diagonal this produces is the amount of "persistence" the labeller manufactures on its own. Report excess persistence = real − labeller-null, with a bootstrap interval.
- *Anderson-Goodman test.* The classical chi-square test of first-order Markov dependence against independence, applied to the stride-sampled sequence. Report the p-value so the Markov claim is tested rather than assumed.
- *Effective sample size.* For the overlapping matrix, use n / w as the effective count when building confidence intervals.

**How we verify.** On an iid synthetic series the overlapping diagonal must come out inflated and the stride and run-length estimates must come out near the marginal frequencies. On a synthetic chain with a known matrix, all three estimators must converge to the truth.

**Owner.** Phase 2.

#### Fix 2 — Back-dated labels in the Pine Script

**Root cause.** The overlay waits `min_regime_hold` confirmed bars, then draws the label at `bar_index − (min_regime_hold − 1)`, the bar where the flip first appeared. On a chart that looks like the flip was called early. In live trading it was only known `min_regime_hold` bars later.

**What we implement.**
- Draw the label on the bar where the hold condition is met (`bar_index`, offset 0), inside a `barstate.isconfirmed` guard. The label text can say "flip was at bar −3" so the information is kept without the visual lie.
- Keep an explicit "show hindsight labels" toggle, off by default, that back-dates with a hollow style and a "hindsight" tag so nobody mistakes it for a live signal.
- Count transitions and update tables only on confirmed bars (the original already does this; keep it).
- Add a `strategy()` variant of the script that enters on the next bar's open after a confirmed flip, with `process_orders_on_close=false`, so TradingView's own Strategy Tester shows what the signal would actually have earned.
- Validate with TradingView Bar Replay: step forward bar by bar and confirm no label ever appears on a bar that has already passed.

**How we verify.** Manual Bar Replay check on BTCUSDT daily and SPY daily. The strategy variant's trade list must show every entry one bar after a confirmed flip.

**Owner.** Phase 6.

#### Fix 3 — Built-in lag of the trailing-return labeller

**Root cause.** A 20-day trailing return needs enough down days to drag the window below −5%. During the first part of a decline the label still says Bull or Sideways. Detection lag is a property of any evidence-accumulating detector, not a bug, but it can be measured and shortened.

**What we implement.**
- *Measure first.* Lag analysis in the backtester: bars from each local peak or trough to the label flip, reported as a distribution, split by direction. Crash detection is usually slower than recovery detection.
- *Volatility-normalised thresholds.* Label on a z-score, trailing return divided by realised volatility scaled to the window, instead of a fixed ±5%. Thresholds then adapt: fewer bars needed in calm markets, fewer false flips in volatile ones.
- *Multi-window ensemble.* Compute labels at 5, 10, 20 and 60 days and vote or weight them. The short windows flip early; the long windows filter noise.
- *Exponentially weighted return.* Replace the simple trailing return with an exponentially weighted one (half-life about 10 days) so recent days count more.
- *Hysteresis.* Separate entry and exit thresholds, for example enter Bear below −5% and leave Bear above −2%, plus a minimum hold of 2 to 3 bars. This cuts whipsaw so the faster detectors above become usable.
- *Online HMM filter as a fast detector.* A forward-filter on daily returns reacts in a few days. Choose the probability threshold from a lag versus false-positive table (roughly: 0.3 gives about 1.5 days lag with about 20% false positives, 0.5 about 3 days and 6%, 0.7 about 5 days and 1%).
- *Budget for the lag that remains.* Size positions assuming the median lag in bars of adverse return will be absorbed before any flip.

**How we verify.** Every variant is scored on net P&L after costs and on its lag versus false-positive curve, not on label accuracy. The chosen default must beat the plain 20-day ±5% labeller on net Sharpe and shorten median crash-detection lag.

**Owner.** Phase 3 for measurement and position rules, Phase 2 for the alternative labellers, Phase 4 for the online filter.

#### Fix 4 — Non-stationarity of a single 10-year matrix

**Root cause.** Transition probabilities drift with volatility regimes, market structure and macro conditions. One matrix fitted to a decade averages over all of them and can be wrong for the current year.

**What we implement.**
- *Exponentially weighted counts.* Decay each transition count with a half-life (default about 250 bars) so the matrix tracks recent behaviour while using all history.
- *Rolling-window matrices* (default 3 years) next to the full-history matrix, with a drift metric (total variation distance between the two) on every report.
- *Dirichlet shrinkage.* Treat the full-history matrix as a prior with pseudo-count strength α. Posterior row = (recent counts + α × prior row) / (recent row total + α). This keeps rolling estimates stable when a regime has few recent observations.
- *Homogeneity test.* Chi-square test that the transition matrix is the same across sub-periods (for example by year). A rejection is reported as "matrix is not stable", which is useful information rather than a failure.
- *Walk-forward always.* The live signal uses the matrix as of yesterday, re-estimated daily. Nothing uses a matrix fitted on data after the decision date.
- *Reinterpret the stationary distribution.* Report it as the long-run mix implied by the current matrix, not as a forecast. The mixing time already tells us how quickly forecasts collapse to it.
- *Stretch goal: time-varying transition probabilities.* Transition probabilities as a logistic function of covariates such as realised volatility (Filardo 1994; Bazzi et al. 2017). Only if the simpler methods above leave a clear gap.

**How we verify.** On a synthetic chain whose matrix changes halfway through, the exponentially weighted and rolling estimates must track the change and the full-history estimate must not. The homogeneity test must reject on that series and not reject on a stationary one.

**Owner.** Phase 2.

#### Fix 5 — HMM fitted on the wrong inputs and read with hindsight

**Root cause.** Price levels are non-stationary, so an HMM on prices clusters on scale rather than regime. Even on returns, the state path produced by Viterbi or the smoothed probabilities uses future observations. Backtests built on those paths show a detector that is faster than anything available live.

**What we implement.**
- *Features.* Daily log return, 10-day realised volatility, optionally Parkinson range and a volume z-score. Standardise with statistics from the training window only. Do not feed rolling volatility alone, because the HMM then just clusters on it; use returns and volatility together with a full covariance matrix.
- *Fitting.* Several random restarts of Baum-Welch, keep the best log-likelihood. Order states by mean return so labels are stable. Consider Student-t emissions for heavy tails if Gaussian fits produce an implausible "crash" state that fires on single outliers.
- *Walk-forward retraining.* Refit on a sliding window (about 10 years of daily bars) before each decision date, or at least monthly, so the model never sees the day it is scoring.
- *Filtered, not smoothed.* The signal uses the forward-filtered probability P(state at t given data up to t). Smoothed probabilities and Viterbi paths are allowed only in diagnostic plots, clearly labelled.
- *Decision rule.* Hysteresis on the filtered probability (enter Bear above 0.6, leave below 0.4) and a 2 to 3 day confirmation to control whipsaw.
- *Comparison.* Report which detector flips first, the threshold labeller or the filtered HMM, and the net P&L of each.

**How we verify.** The no-lookahead test applies to the HMM path: perturb data after day t and assert the filtered probability up to t is unchanged. Smoothed and filtered paths must differ on the same data, which proves the two are not being confused.

**Owner.** Phase 4.

#### Fix 6 — No costs, no benchmarks, no significance

**Root cause.** The walk-forward reports a gross Sharpe and max drawdown for a strategy that can flip position daily. That hides turnover cost and gives no way to tell skill from luck.

**What we implement.**
- *Cost model.* Per-side cost = commission + half spread + slippage, charged on |Δposition|. Defaults: 2 bps per side for liquid US ETFs, 15 bps per side for crypto majors including funding, plus a borrow charge for short days. Report a sensitivity table at 0, 5, 10, 20 and 50 bps.
- *Turnover control.* Threshold rebalancing: only change position when the signal moves by more than a band. This is the standard remedy when daily rebalancing erodes a low-Sharpe strategy.
- *Benchmarks.* Buy-and-hold, constant 50% exposure, and the signal's own long-only version.
- *Permutation test.* Shuffle the return series (or the labels) N = 1,000 times, rerun the entire pipeline including the labeller and the costs on each shuffle, and report p = (count of nulls ≥ real + 1) / (N + 1). Running the whole pipeline on each shuffle keeps the selection effect inside the null. Report the null with and without costs, because random signals that trade more can make the cost-inclusive null look too easy to beat.
- *Probabilistic and Deflated Sharpe.* PSR gives the probability the true Sharpe exceeds zero given skew, kurtosis and sample length. DSR additionally deflates for the number of parameter combinations tried (window × threshold × rule). Report both next to the raw Sharpe.
- *Out-of-sample holdout.* The last two years are never used for parameter selection. Report in-sample and holdout results separately.
- *Parameter sensitivity map.* A window × threshold heatmap of net Sharpe. A strategy that only works at one cell is overfit.
- *Full metric set.* Sortino, Calmar, hit rate, average win and loss, profit factor, P&L by holding horizon (1, 5, 10, 20 days), time in market, turnover, minimum track record length.

**How we verify.** With zero costs the sign-only rule reproduces the Phase 1 numbers exactly. Costs reduce returns monotonically. On a synthetic series with no signal the permutation p-value is uniform and PSR is near 0.5.

**Owner.** Phase 3.

#### Sources consulted for the fixes

- Anderson and Goodman (1957), chi-square tests of Markov order and homogeneity; implemented in the R `markovchain` package as `assessIndependence`.
- Bailey and López de Prado (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management.
- Masters-style Monte Carlo permutation tests for strategy significance (susanpotter.net, buildalpha.com, 2026 write-ups).
- "Filtering vs. Smoothing: Why Backtests Lie About Regime Detection", Math and Markets, 2026: filtered versus smoothed probabilities, lag versus false-positive table, hysteresis.
- Filardo (1994) and Bazzi et al. (2017) on time-varying transition probabilities.
- TradingView Pine Script documentation on repainting and bar states; ValorAlgo and TraderOSHQ repainting guides.
- QuantPedia, "The Price of Transaction Costs"; QuanterLab and QuantInsti cost-modelling notes for basis-point ranges.
- HMM feature practice from QuantStart, BSIC, MDPI (sliding-window retraining) and recent arXiv work on heavy-tailed emissions.

---

## 2. What we will build

A Python research and signal platform, `hedgeFund`, that implements the Markov regime method faithfully, then hardens it against the critiques above, and exposes it three ways: a CLI/JSON contract (same shape as the video's skill so it composes with other agents), a Claude Code skill, and a local dashboard. An Alpaca paper-trading adapter is included as a later, opt-in phase for live regime tracking, never for automatic order placement without an explicit human step.

### Goals

- Reproduce the video's numbers on SPY and BTC-USD so we trust the base implementation.
- Make every evaluation honest: no lookahead anywhere, costs included, null-model comparison, confidence intervals.
- Support many assets and timeframes from one config.
- Keep the JSON contract from the companion skill so the engine drops into existing agents.

### Non-goals (for this plan)

- Automated live trading. Phase 7 is paper-only and signal-publishing; order placement stays a manual, human-confirmed action.
- Intraday data. Daily bars first; the design should allow other bar sizes later.
- A hosted web service. Local-first.

---

## 3. Architecture

```
hedgeFund/
├── plan.md
├── pyproject.toml                 # uv-managed, Python 3.12
├── README.md
├── config/
│   ├── default.yaml               # window, thresholds, costs, universe
│   └── universes/                 # e.g. us_etfs.yaml, crypto.yaml
├── src/hedgefund/
│   ├── data/
│   │   ├── base.py                # PriceSource protocol: get_daily(symbol, start, end) -> DataFrame
│   │   ├── yfinance_source.py
│   │   ├── alpaca_source.py       # uses the Alpaca market-data API (stocks + crypto bars)
│   │   ├── csv_source.py
│   │   └── cache.py               # parquet cache under ./cache (folder already exists)
│   ├── regime/
│   │   ├── labels.py              # rolling-return labeller (+ log-return variant, asymmetric thresholds)
│   │   ├── markov.py              # transition matrix, n-step, stationary dist, mixing time
│   │   ├── signal.py              # bull−bear signal, conviction scaling, persistence-weighted sizing
│   │   ├── hmm.py                 # GaussianHMM on feature matrix, multi-restart, state ordering
│   │   └── stats.py               # bootstrap CIs, stride sampling, null-model (shuffled labels), chi-square vs iid
│   ├── backtest/
│   │   ├── walk_forward.py        # expanding + rolling-window refits, no lookahead
│   │   ├── costs.py               # fees, slippage, borrow (shorts), funding (crypto)
│   │   ├── metrics.py             # Sharpe, Sortino, max DD, win rate, per-horizon P&L, turnover
│   │   └── benchmarks.py          # buy-and-hold, random-sign null, label-shuffle null
│   ├── report/
│   │   ├── json_contract.py       # same field names as the video's skill, plus extensions
│   │   ├── terminal.py            # pretty on-camera style output
│   │   └── html.py                # static report per run
│   ├── cli.py                     # `hedgefund analyze|backtest|scan|report`
│   └── jobs/                      # scheduled scans (folder ./jobs already exists)
├── skills/regime/SKILL.md         # Claude Code skill wrapping the CLI
├── pine/markov_regime.pine        # our corrected Pine overlay (no retroactive labels)
├── dashboard/                     # optional Streamlit app
├── tests/
│   ├── test_markov.py             # matrix rows sum to 1, stationary = left eigvec, Pⁿ converges
│   ├── test_no_lookahead.py       # perturb future prices, assert past signals unchanged
│   ├── test_backtest.py
│   └── fixtures/                  # small synthetic series with known transition matrix
└── webhooks/                      # (folder already exists) reserved for job notifications
```

Key design choices:

- **Pure functions in `regime/`** operating on numpy/pandas; no I/O. Everything above them is orchestration. This keeps the maths testable against synthetic chains with known matrices.
- **One `PriceSource` protocol**, so yfinance, Alpaca, and CSV are interchangeable. Alpaca is preferred when keys are present because the user already has the Alpaca MCP server configured and it gives clean stock + crypto bars.
- **JSON contract is a superset** of the companion skill's fields, so anything built against that skill keeps working.
- **Config over flags.** The CLI reads `config/default.yaml`, flags override.

---

## 4. Phases and checklists

Progress key: `[x]` done, `[ ]` not started. Each phase ends with an acceptance gate; do not start the next phase until the gate passes.

### Phase 0 — Scaffold and baseline (½ day)

**Already done (2026-10-19)**
- [x] Companion repo located and verified live.
- [x] All repo files audited for malicious code: none found (see section 1).
- [x] Plugin installed at user scope: `markov-hedge-fund-method@markov-hedge-fund-method`.
- [x] Installed files hash-matched to the audited copies.
- [x] Smoke run on SPY 10y succeeded (Sharpe 0.244, max DD −33.9%, Sideways persistence 93.0%).

**To do**
- [x] `uv init` in the project root, pin Python 3.12 in `pyproject.toml`.
- [x] Add dependencies: numpy, pandas, scipy, yfinance, pyarrow, pyyaml, typer, rich.
- [x] Add optional extras: `hmm` (hmmlearn), `alpaca` (alpaca-py), `dashboard` (streamlit).
- [x] Add dev dependencies: pytest, hypothesis, ruff, pre-commit.
- [x] Create the package layout from section 3 (`src/hedgefund/...`) with empty `__init__.py` files.
- [x] Create `config/default.yaml` with the defaults from section 5.
- [x] Create `tests/fixtures/` and freeze a SPY 10y close series as parquet (the reproduction baseline).
- [x] Write `README.md` with the educational-use disclaimer from the video.
- [x] Configure ruff, pytest, and pre-commit; confirm `uv run pytest` runs with zero tests.

**Gate:** `uv run pytest` and `uv run ruff check .` both pass on the empty scaffold.

### Phase 1 — Faithful reproduction (1 day)

- [x] Port `label_regimes` into `regime/labels.py` (simple-return variant, 20-day, ±5%).
- [x] Port `build_transition_matrix` into `regime/markov.py`.
- [x] Port `nstep_forecast` and `stationary_distribution` into `regime/markov.py`.
- [x] Port `signal_from_matrix` into `regime/signal.py`.
- [x] Port the incremental walk-forward into `backtest/walk_forward.py` (sign-only position, next-day return).
- [x] Port Sharpe and max drawdown into `backtest/metrics.py`.
- [x] Port `fit_hmm` into `regime/hmm.py` with the same lazy import and graceful degrade.
- [x] Port `load_csv` into `data/csv_source.py` and `fetch_ticker` into `data/yfinance_source.py`.
- [x] Define the `PriceSource` protocol in `data/base.py`.
- [x] Build `report/json_contract.py` emitting exactly the companion skill's field names.
- [x] Build `report/terminal.py` matching the on-camera pretty output.
- [x] Build `cli.py` with `hedgefund analyze --ticker|--csv [--json]`.
- [x] Test: synthetic chain from a known 3×3 matrix, estimate converges within tolerance.
- [x] Test: `stationary(P) @ P == stationary(P)` and rows of `P**50` equal the stationary vector.
- [x] Test: rows of every estimated matrix sum to 1.
- [x] Test: `analyze()` on the frozen SPY fixture reproduces the companion script's matrix, stationary mix, Sharpe and max DD to 6 decimals.

**Gate:** reproduction test passes against the frozen fixture and against a fresh run of the installed plugin on the same dates.

### Phase 2 — Honest statistics (1–2 days)

- [x] Add stride (non-overlapping) transition counting as an option in `regime/markov.py`.
- [x] Add effective-sample-size estimate for overlapping counts.
- [x] Add label-shuffle null model: refit on shuffled labels N times, report the diagonal's percentile.
- [x] Add iid null model with the same marginal regime frequencies.
- [x] Add block-bootstrap confidence intervals for every matrix cell and the stationary vector.
- [x] Add rolling-window matrix fits (configurable, default 3 years) alongside the full-history fit.
- [x] Add a drift metric between rolling and full-history matrices.
- [x] Add mixing time: steps until `P**n` is within ε of the stationary vector.
- [x] Add run-length persistence estimator (1 − 1 / mean episode length) next to the stride estimate. (Fix 1)
- [x] Add the labeller-induced null: iid returns with matched mean and vol, same labeller, report excess persistence with CI. (Fix 1)
- [x] Add the Anderson-Goodman chi-square test of first-order dependence on the stride sequence. (Fix 1)
- [x] Add exponentially weighted transition counts with a configurable half-life. (Fix 4)
- [x] Add Dirichlet shrinkage of rolling counts toward the full-history matrix. (Fix 4)
- [x] Add the chi-square homogeneity test across sub-periods, reported as matrix stability. (Fix 4)
- [x] Add alternative labellers: z-score threshold, exponentially weighted return, multi-window ensemble, hysteresis with min-hold. (Fix 3)
- [x] Test: on a synthetic chain whose matrix changes halfway, EW and rolling estimates track the change and the full-history estimate does not. (Fix 4)
- [x] Extend the JSON contract with `stride_matrix`, `null_percentile`, `confidence_intervals`, `rolling_matrix`, `mixing_time`.
- [x] Test: on an iid synthetic series the overlapping diagonal is inflated and the stride diagonal is not.
- [x] Test: bootstrap CIs contain the true matrix on a synthetic chain at the nominal rate.

**Gate:** the SPY report shows both diagonals side by side with the gap explained, and the null percentile is computed.

### Phase 3 — Backtester upgrade (1–2 days)

- [ ] Add rolling-window refit mode to the walk-forward (expanding stays the default).
- [ ] Add position rules: sign-only, conviction-scaled, persistence-weighted, long-only.
- [ ] Add `backtest/costs.py`: fee bps, slippage bps, short borrow, crypto funding, configurable per universe.
- [ ] Add turnover and exposure reporting.
- [ ] Extend `metrics.py`: Sortino, Calmar, hit rate, average win/loss, P&L by holding horizon (1, 5, 10, 20 days).
- [ ] Add `backtest/benchmarks.py`: buy-and-hold, random-sign null, label-shuffle null (1,000 paths).
- [ ] Report the strategy's percentile rank against the null paths on every run.
- [ ] Add lag analysis: average days from a local peak or trough to the label flip.
- [ ] Report lag as a distribution split by direction (crash versus recovery), per labeller variant. (Fix 3)
- [ ] Add threshold rebalancing (signal dead-band) to control turnover. (Fix 6)
- [ ] Add cost sensitivity table at 0, 5, 10, 20 and 50 bps per side. (Fix 6)
- [ ] Add constant-50%-exposure and long-only benchmarks. (Fix 6)
- [ ] Add the permutation test: 1,000 shuffles through the full pipeline, p = (count + 1) / (N + 1), with and without costs. (Fix 6)
- [ ] Add Probabilistic Sharpe and Deflated Sharpe with the number of trials recorded by the parameter search. (Fix 6)
- [ ] Add the two-year out-of-sample holdout and report it separately. (Fix 6)
- [ ] Add the window × threshold sensitivity heatmap. (Fix 6)
- [ ] Add profit factor, minimum track record length and time in market to the metric set. (Fix 6)
- [ ] Test: on a no-signal synthetic series the permutation p-value is uniform and PSR is near 0.5. (Fix 6)
- [ ] Add `hedgefund backtest` CLI command with cost and rule flags.
- [ ] Test: no-lookahead, mutate prices after day t and assert every signal up to t is bit-identical.
- [ ] Test: with zero costs the sign-only rule reproduces Phase 1 numbers exactly.
- [ ] Test: costs reduce returns monotonically as bps increase.

**Gate:** every backtest report includes costs, turnover, buy-and-hold, and the null percentile; the no-lookahead test passes.

### Phase 4 — HMM done properly (1 day)

- [ ] Build a feature matrix: daily log return, 10-day and 20-day realised volatility, optional normalised volume and range.
- [ ] Standardise features using training-window statistics only. (Fix 5)
- [ ] Use a full covariance matrix; never fit on rolling volatility alone. (Fix 5)
- [ ] Add an online forward filter that updates P(state) one bar at a time for the live signal. (Fix 3, Fix 5)
- [ ] Add hysteresis (enter 0.6, exit 0.4) and a 2 to 3 bar confirmation on the filtered probability. (Fix 5)
- [ ] Evaluate Student-t emissions if Gaussian fits produce a single-outlier crash state. (Fix 5)
- [ ] Test: smoothed and filtered paths differ on the same data, proving they are not confused. (Fix 5)
- [ ] Multi-restart Baum-Welch (configurable seeds), keep the best log-likelihood.
- [ ] Order latent states by mean return so labels are stable across fits.
- [ ] Walk-forward HMM: refit on past data only, decode the current state with filtered probabilities.
- [ ] Agreement report between HMM regimes and threshold regimes, including which flips first.
- [ ] Keep graceful degrade when hmmlearn is missing; JSON reports `available: false` with a reason.
- [ ] Test: no-lookahead check on the walk-forward HMM path.
- [ ] Test: with a fixed seed, repeated fits produce identical state sequences.

**Gate:** HMM section of the report runs walk-forward with no lookahead and degrades cleanly without hmmlearn.

### Phase 5 — Multi-asset scan and reports (1 day)

- [ ] Add `config/universes/us_etfs.yaml` and `config/universes/crypto.yaml`.
- [ ] Add `data/cache.py`: parquet cache under `./cache`, keyed by source, symbol, and date range.
- [ ] Add `hedgefund scan --universe <file>` running the full pipeline per symbol.
- [ ] Ranked scan table: current regime, signal, persistence, stationary bear share, net Sharpe, null percentile.
- [ ] Add `report/html.py`: matrix heatmap, regime ribbon over price, equity curve vs benchmarks, CI bars.
- [ ] Write run manifests (parameters, data hashes, timestamps) to `./jobs`.
- [ ] Add `hedgefund report <run-id>` to regenerate HTML from a cached run.
- [ ] Test: scan on a two-symbol universe produces two cached runs and one ranked table.

**Gate:** a scan over the ETF universe completes from cold cache and again from warm cache, with the warm run making no network calls.

### Phase 6 — Claude Code skill and Pine Script (½ day)

- [ ] Write `skills/regime/SKILL.md` wrapping `hedgefund analyze --json`.
- [ ] Document the extended JSON contract and the three composition patterns (gate, size filter, standalone).
- [ ] Add `.claude-plugin/plugin.json` so the project installs as a local plugin.
- [ ] Write `pine/markov_regime.pine`: labels placed on the bar where the hold condition is met, never back-dated.
- [ ] Add a hindsight-labels toggle, off by default, that draws back-dated labels in a distinct hollow style tagged "hindsight". (Fix 2)
- [ ] Add a `strategy()` variant entering on the next bar open after a confirmed flip, with `process_orders_on_close=false`. (Fix 2)
- [ ] Manual check with Bar Replay on BTCUSDT and SPY daily: no label appears on a past bar. (Fix 2)
- [ ] Add optional stride counting to the Pine overlay.
- [ ] Keep the matrix table, stationary table, banner and ribbon from the original.
- [ ] Manual check: overlay compiles on TradingView v5 and labels never appear on past bars.

**Gate:** invoking the skill from Claude Code returns valid JSON in one call; the Pine overlay compiles.

### Phase 7 — Alpaca paper integration, opt-in (1 day)

- [ ] Confirm with the owner that this phase is wanted before starting.
- [ ] Add `data/alpaca_source.py` for daily stock and crypto bars via the paper endpoint.
- [ ] Read Alpaca keys from the environment only; never write them to config or cache.
- [ ] Add a daily job that recomputes regimes for a watchlist and writes a signal file.
- [ ] Post a run summary to consumers registered in `./webhooks`.
- [ ] Add a paper-only sizing helper that turns `signal` into a target allocation and prints proposed orders.
- [ ] Document that order placement is a separate, explicit human action; nothing submits orders automatically.
- [ ] Test: Alpaca source and yfinance source return aligned closes for the same symbol and dates.

**Gate:** the daily job runs end to end on paper data and produces a signal file without placing any order.

### Phase 8 — Dashboard, optional (1 day)

- [ ] Streamlit app under `dashboard/` reading cached runs.
- [ ] Controls: symbol, window, threshold, refit mode, cost profile.
- [ ] Views: regime ribbon, matrix heatmap with CIs, stationary mix, `P**n` convergence animation.
- [ ] Views: walk-forward equity vs buy-and-hold and null band.
- [ ] Launch command `hedgefund dashboard`.

**Gate:** dashboard opens from a cached scan with no network calls.

---

## 5. Key parameters and defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| window | 20 trading days | video default; expose 10/20/60 |
| threshold | ±5% simple return | Pine uses log return; support both. The original onboarding prompt used ±2%. |
| years | 10 | |
| min_train | 252 | one year before walk-forward starts |
| refit | expanding | also `rolling:756` (3y) |
| fee_bps / slippage_bps | 1 / 2 (ETFs), 10 / 10 (crypto) | configurable per universe |
| null_paths | 1000 | for percentile ranking |
| bootstrap_blocks | 20-day blocks, 500 resamples | |

---

## 6. Definition of done

The project is done when every gate in section 4 has passed. The gates, in order:

1. Scaffold lints and tests run clean.
2. Reproduction matches the companion script on the frozen fixture.
3. Both diagonals and the null percentile appear in the report.
4. Backtests include costs, benchmarks, and pass the no-lookahead test.
5. Walk-forward HMM has no lookahead and degrades cleanly.
6. Scans run cold and warm, with the warm run offline.
7. Skill returns JSON in one call; Pine overlay compiles with no back-dated labels.
8. (Opt-in) Paper job produces a signal file without placing orders.
9. (Optional) Dashboard opens from cache.

---

## 7. Risks and open questions

- **Yahoo rate limits.** yfinance is flaky; Alpaca bars or cached parquet mitigate. Decide: is Alpaca the primary source once keys exist?
- **The edge may not exist.** After costs and null-model comparison, the standalone signal may rank near the median. That is a valid outcome; the project's value is the honest measurement and the regime filter, which is how the video itself frames it ("the matrix is the honest measurement").
- **hmmlearn build issues** on some machines; keep it an optional extra.
- **Scope of Phase 7.** Confirm with the owner whether paper-trading integration is wanted at all, or whether the project stays a research tool.
- **Existing folders** (`cache/`, `jobs/`, `webhooks/`) are empty; the plan assumes they were created for this purpose and reuses them.

---

## 8. Suggested order of work

Phases 0 → 1 → 2 → 3 are the core and should be done in sequence. Phase 4 (HMM) and Phase 5 (scan/report) are independent of each other and can follow in either order. Phases 6, 7, 8 are additive and can be picked by priority.
