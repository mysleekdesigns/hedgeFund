# hedgeFund

**Status: Phase 0 complete (scaffold, config, frozen SPY fixture). The regime engine, backtester, and data sources arrive in Phases 1 to 5; see `plan.md`.**

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

Phase 1 adds the analysis command:

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
│   ├── cli.py               # hedgefund analyze | backtest | scan | report
│   └── jobs/                # scheduled scans
├── skills/regime/           # Claude Code skill wrapping the CLI
├── pine/                    # corrected Pine overlay, no back-dated labels
├── dashboard/               # optional Streamlit app
├── tests/                   # offline tests and fixtures/
└── webhooks/                # reserved for job notifications
```

Today only the scaffold exists. Modules fill in phase by phase as listed in `plan.md`.

## Data sources (planned, Phases 1 and 5)

- yfinance is the default and needs no API key.
- CSV files with a date column and a close column.
- Alpaca market data is optional and uses the paper endpoint only. Keys are read from the `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment variables at call time and are never written to config, cache, fixtures, logs, or JSON output.

Downloaded bars are cached as parquet under `./cache`. Tests run offline against the frozen series in `tests/fixtures/`.

## Development

- Tests are offline. `uv run pytest` makes no network calls.
- Every estimator ships with a no-lookahead perturbation test: mutate prices after day t and assert that every output up to t is bit-identical.
- `plan.md` is the source of truth for phases, checklists, and acceptance gates. A phase does not start until the previous gate passes.
- Commits follow Conventional Commits with the phase as scope, for example `feat(phase-2): add stride transition counting`.
- Pre-commit runs ruff for linting and formatting. Run `uv run ruff check .` and `uv run ruff format .` before committing.
- Tooling is `uv` only. Do not `pip install` into the system Python.

## License

MIT.
