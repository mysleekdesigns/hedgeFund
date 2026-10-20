# hedgeFund

**Status: Phases 0 to 6 complete (regime engine, walk-forward backtests, HMM, multi-asset scan, HTML reports, Claude Code skill, Pine overlay). Phase 7 adds the opt-in Alpaca paper integration: a daily job that writes a signal file and a sizing helper that prints proposals. See `plan.md`.**

Python research platform for Markov regime detection (Bull / Bear / Sideways) with honest, no-lookahead backtests. It will reproduce the "hedge fund method" from Lewis Jackson's video faithfully, then fix the method's known weaknesses: overlapping-window autocorrelation, back-dated Pine labels, detection lag, non-stationarity, HMMs fitted on the wrong inputs, and backtests with no costs, benchmarks, or significance tests. The engine will keep the companion skill's JSON contract so anything built against that skill keeps working.

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use this software at your own risk.

Backtests are historical, not forward-looking.

Nothing in this repository places, modifies, or cancels orders. Any broker integration is paper-trading only and opt-in.

## Credit

- Framework: Roan (@RohOnChain on X).
- Plugin and on-camera install: Lewis Jackson (YouTube @lewisjackson).
- Video: "I Re-Created A Quant Trading Strategy With Claude Code (Insanely Cool)", Lewis Jackson Investing, premiered 2026-05-19 (the companion repo calls it video 1 of the Quant Series). https://www.youtube.com/watch?v=ZVMTeDBmSrI
- Companion repo: `jackson-video-resources/markov-hedge-fund-method` on GitHub, MIT licensed. https://github.com/jackson-video-resources/markov-hedge-fund-method

This project is an independent reimplementation. It is not affiliated with or endorsed by the authors above.

## What it will do

The method, as planned in `plan.md`. Phase 1 implements the six steps below; later phases add the fixes.

- Labels every trading day by its trailing 20-day simple return: at or above +5% is Bull, at or below -5% is Bear, otherwise Sideways.
- Estimates a 3x3 transition matrix by maximum likelihood: count every day-to-day state transition and row-normalise.
- Forecasts n days ahead with matrix powers (P to the n), per Chapman-Kolmogorov.
- Computes the stationary distribution, the long-run regime mix that long-horizon forecasts collapse to.
- Produces a signal = P(Bull) - P(Bear) in [-1, 1]. Sign is direction, magnitude is conviction.
- Runs a walk-forward backtest that re-estimates the matrix at every step using only data before that day, and reports Sharpe and max drawdown.

### What we will fix

1. Overlapping-window autocorrelation inflates the matrix diagonal to 80 to 95% on every asset. Fix: stride (non-overlapping) counting, run-length persistence, a labeller-induced null model, and the Anderson-Goodman test of Markov dependence.
2. The original Pine Script draws regime labels back at the bar where a flip started, which is not tradeable live. Fix: labels placed on the bar where the hold condition is met, a hindsight toggle that is off by default, and a strategy variant that enters on the next bar's open.
3. A 20-day trailing return flips to Bear only after much of a decline. Fix: measure the lag first, then volatility-normalised thresholds, a multi-window ensemble, exponentially weighted returns, hysteresis, and an online HMM filter as a fast detector.
4. One matrix fitted to 10 years ignores drift. Fix: exponentially weighted counts, rolling 3-year matrices with a drift metric, Dirichlet shrinkage toward the full-history prior, and a chi-square homogeneity test.
5. An HMM fitted on price levels clusters on scale, and smoothed or Viterbi paths use future data. Fix: stationary features standardised on the training window, full covariance, walk-forward refits, and filtered (not smoothed) probabilities for the live signal.
6. The original backtest has no costs, benchmarks, or significance. Fix: a per-side cost model with a sensitivity table, buy-and-hold and constant-exposure benchmarks, a 1,000-path permutation test, Probabilistic and Deflated Sharpe, and a two-year out-of-sample holdout.

## Quick start

Requires `uv`. The project pins Python 3.12 and `uv` installs it for you.

```
uv sync
uv run pytest
uv run ruff check .
uv run hedgefund version
```

The analysis command:

```
uv run hedgefund analyze --ticker SPY --json
```

Optional extras:

```
uv sync --extra hmm         # hmmlearn for the Gaussian HMM
uv sync --extra alpaca      # alpaca-py for Alpaca market data (paper endpoint only)
uv sync --extra dashboard   # streamlit for the local dashboard
```

## Scan and report

`hedgefund scan` runs the full `analyze` pipeline over every symbol of a universe and ranks the results. `hedgefund report` regenerates the HTML of a finished run without fetching anything.

```
uv run hedgefund scan --universe us_etfs                 # cold run: fetches, caches, analyses, ranks, renders HTML
uv run hedgefund scan --universe us_etfs --offline       # warm run: served from the cache; a miss is an error
uv run hedgefund scan --universe crypto --json           # one JSON object: run_id, table, html path, ...
uv run hedgefund scan --universe my.yaml --source csv --csv-dir data/   # your own <SYMBOL>.csv files
uv run hedgefund report 20260101T120000Z_us_etfs         # rewrite jobs/<run-id>/html/ from the cached run
```

## Dashboard

`hedgefund dashboard` opens the optional Streamlit app over the runs under `jobs/`. It reads only the cached scan output and makes no network calls: the server binds `localhost` only, sends no telemetry, and skips Streamlit's startup IP lookups and email prompt. The same settings live in `.streamlit/config.toml` so a direct `uv run streamlit run dashboard/app.py` behaves the same way. Install the extra first with `uv sync --extra dashboard`.

```
uv run hedgefund dashboard                              # http://localhost:8501 over ./jobs
uv run hedgefund dashboard --jobs-dir /path/to/jobs     # another runs directory (passed via HEDGEFUND_JOBS_DIR)
uv run hedgefund dashboard --port 8600 --headless       # no browser; for servers and CI
uv run hedgefund dashboard --json                       # dry run: print the command, launch nothing
```

`--universe` is a name under `config/universes/` or a path to a YAML file with `name`, `cost_profile` and `symbols`; the cost profile comes from that file. The analysis flags of `analyze` (`--window`, `--threshold`, `--min-train`, `--no-hmm`, `--null-paths`, `--seed`, `--refit`, `--rule`, ...) apply to every symbol. `--no-html` skips the report.

Layout written by a scan:

```
cache/<source>/<SYMBOL>_<start>_<end>.parquet   # one validated daily frame per symbol and date window
jobs/<run-id>/manifest.json                     # parameters, per-symbol data hashes, cache paths, cache hits, errors
jobs/<run-id>/results/<SYMBOL>.json             # the full analyze JSON per symbol
jobs/<run-id>/table.json                        # the ranked table
jobs/<run-id>/html/index.html, <SYMBOL>.html    # heatmap, regime ribbon, equity curves, CI bars
```

To check the cache is warm, run the same scan again with `--offline`: it must succeed, and every `cache_hit` in the new manifest must be `true`. A missing cache file makes `--offline` fail naming the symbol, and `report` fails the same way when a run's cached frame has been deleted; re-run the scan to refresh it.

Backtests are historical, not forward-looking. A ranked table is a research artefact, not a trade list; nothing here places orders.

## Configuration

`config/default.yaml` holds the defaults. CLI flags override them. `config/universes/*.yaml` list symbols and their cost profiles.

| Parameter | Default | Notes |
|-----------|---------|-------|
| window | 20 trading days | video default; 10 and 60 planned |
| threshold | +/-5% simple return | log-return variant planned |
| years | 10 | |
| min_train | 252 | one year of history before the walk-forward starts |
| refit | expanding | also `rolling:756` (3 years) |
| fee_bps / slippage_bps | 1 / 2 (ETFs), 10 / 10 (crypto) | configurable per universe |
| null_paths | 1000 | for percentile ranking against the null |
| bootstrap_blocks | 20-day blocks, 500 resamples | |

## Layout

```
hedgeFund/
├── plan.md                  # phases, checklists, acceptance gates (source of truth)
├── pyproject.toml           # uv-managed, Python 3.12
├── config/                  # default.yaml and universes/
├── src/hedgefund/
│   ├── data/                # PriceSource protocol, yfinance, CSV, Alpaca, parquet cache
│   ├── regime/              # labels, markov, signal, hmm, stats (pure functions, no I/O)
│   ├── backtest/            # walk_forward, costs, metrics, benchmarks
│   ├── report/              # json_contract, terminal, html
│   ├── cli.py               # hedgefund analyze | backtest | scan | report | daily | propose | dashboard
│   ├── jobs/                # daily job, webhook run summaries
│   └── paper/               # sizing helper: signal -> printed proposal, never an order
├── skills/regime/           # Claude Code skill wrapping the CLI
├── pine/                    # corrected Pine overlay, no back-dated labels
├── dashboard/               # optional Streamlit app
├── tests/                   # offline tests and fixtures/
└── webhooks/                # JSON registrations for daily run summaries (real files git-ignored)
```

## Data sources

Three sources implement the same `PriceSource` protocol and are selected with `--source` (or `data.source` in the config).

- `yfinance` is the default and needs no API key.
- `csv` reads your own `<SYMBOL>.csv` files from `--csv-dir`, with a date column and a close column.
- `alpaca` reads daily stock and crypto bars through `alpaca-py` (`uv sync --extra alpaca`). Market data comes read-only from `data.alpaca.markets` on the free IEX feed with split and dividend adjustment. Dates line up with yfinance exactly (both index by the New York trading date); IEX daily closes are IEX-only prints that approximate the consolidated close yfinance reports, typically within cents on liquid names, and `feed: sip` (paid subscription) is needed for equality. Crypto bars are New York calendar days, not the UTC days yfinance uses. Account reads use the paper endpoint only (`paper-api.alpaca.markets`). Keys are read from the `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment variables at call time and are never written to config, cache, fixtures, logs, or JSON output.

Downloaded bars are cached as parquet under `./cache`. Tests run offline against the frozen series in `tests/fixtures/`.

## Daily job and paper proposals (Phase 7, opt-in)

Phase 7 is opt-in and paper-only. It adds two commands. Neither one talks to an order endpoint.

`hedgefund daily` recomputes regimes for a watchlist and writes a signal file:

```
uv run hedgefund daily --universe us_etfs --source alpaca   # or --source yfinance; no keys needed
```

Each run writes `jobs/<run_id>/signal.json` and refreshes `jobs/signal_latest.json`, a stable copy that `hedgefund propose` reads. The signal file carries:

| Field | Meaning |
|-------|---------|
| `as_of` | date of the last bar the regimes were computed on |
| `symbols.<SYM>.current_regime` | Bull, Bear or Sideways on `as_of` |
| `symbols.<SYM>.signal` | P(Bull) - P(Bear) in [-1, 1] from the walk-forward matrix as of `as_of` |
| `symbols.<SYM>.stay_probability` | probability the current regime persists one more day |
| `symbols.<SYM>.position` | the backtest rule's target in {-1, 0, +1} |
| `symbols.<SYM>.last_close` | last close used |
| `symbols.<SYM>.error` | per-symbol failure message, otherwise null; one bad symbol never aborts the run |
| `orders_placed` | always `false` |

Webhooks: drop a JSON registration file in `./webhooks/` and the job POSTs a run summary to it after every run. A registration has a `name`, an https `url`, `events` set to `["daily"]`, and optionally `token_env`, the name of an environment variable whose value is sent as the `X-Hedgefund-Token` header. That name must start with `HEDGEFUND_WEBHOOK_`; `APCA_*` and any other variable are refused, so a registration can never forward the broker keys. The token itself never appears in a file, redirects are refused, and the job output names only the webhook host, never its URL. Real registrations are git-ignored; a failed POST is logged and does not fail the run.

`hedgefund propose` turns the latest signal file into proposed orders and prints them:

```
uv run hedgefund propose --equity 100000          # size against a stated equity
uv run hedgefund propose --paper-equity           # read the paper account's equity instead (APCA_* keys)
uv run hedgefund propose --equity 100000 --json   # the same proposal as one JSON object
```

Sizing follows the `paper` section of `config/default.yaml`: total |weight| is capped at `max_gross` (1.0, unlevered), a Bear signal proposes a short only when `allow_short` is true, and proposals under `min_trade_notional` are dropped as dust.

### Order placement is a human action

Nothing in hedgeFund submits, replaces, cancels or exercises orders. The daily job writes a signal file. The sizing helper prints proposals. Placing an order is a separate, explicit action the owner takes outside this tool, in the broker's own interface or client. There is no command, flag or config key that changes this.

- Alpaca is paper-only (`paper-api.alpaca.markets`). The live endpoint is blocked by a hook in `.claude/settings.json`, as are the order-side Alpaca MCP tools.
- Keys come from the environment at call time and are never persisted to config, cache, fixtures, logs, signal files or JSON output.
- Every signal file, proposal and report carries the disclaimer: backtests are historical, not forward-looking.

## Development

- Tests are offline. `uv run pytest` makes no network calls.
- Every estimator ships with a no-lookahead perturbation test: mutate prices after day t and assert that every output up to t is bit-identical.
- `plan.md` is the source of truth for phases, checklists, and acceptance gates. A phase does not start until the previous gate passes.
- Commits follow Conventional Commits with the phase as scope, for example `feat(phase-2): add stride transition counting`.
- Pre-commit runs ruff for linting and formatting. Run `uv run ruff check .` and `uv run ruff format .` before committing.
- Tooling is `uv` only. Do not `pip install` into the system Python.

## License

MIT.
