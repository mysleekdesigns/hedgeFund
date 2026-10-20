---
name: phase5-review-context
description: Non-obvious facts from the Phase 5 (scan, cache, manifest, HTML) review on 2026-10-20 — the four warnings (partial bar, offline abort, slash symbols, CSV staleness) and how they were fixed, the CSV-fingerprint key semantics, the HEAD-vs-worktree recipe needs config/ too, probe timings
metadata:
  type: project
---

Phase 5 reviewed 2026-10-20 (uncommitted diff on develop, base a123e87): 1138 tests pass; `analyze --json` and `backtest --json` byte-identical to HEAD on the SPY fixture (null_paths 20, resamples 20, no-hmm); `report <run-id>` regenerates HTML byte-identical to the scan's own output; `equity_curves` ends exactly at 1 + backtest.net.total_return and the benchmarks at 1 + their total_return. No criticals; warnings were the four below.

**Gaps found on first pass; all four verified FIXED the same day (re-run, not read). Kept so the failure modes stay known:**
- `CachedSource` passes concrete bounds to `YFinanceSource.get_daily`, which adds ONE_DAY to `end`, so the cache requests `end = tomorrow` and stores TODAY's intraday partial bar; `fetch_ticker` (used by `analyze --ticker`) requests `end = today` exclusive and stops at yesterday. Verified with a fake `yfinance` module: cache window `('2016-10-20','2026-10-21')` last row 2026-10-20 vs fetch_ticker last row 2026-10-19. A scan run during market hours freezes a non-close in the cache for the rest of the UTC day.
- `scan --offline` aborts the whole run (exit 1, orphan `jobs/<id>/results/` without manifest or table) when any symbol has no cache file, because `CacheMissError` is a `LookupError` and `scan_universe` only records `ValueError`. A cold run with one yfinance failure therefore makes the warm offline run of the same universe impossible until that symbol is fetched.
- `report <run-id>` keys results by file stem (`read_results`) but the manifest `data` by raw symbol, so any symbol containing `/`, `\` or `:` (Alpaca `BTC/USD`) fails with "records no cache_path for BTC-USD". The scan's own HTML write is unaffected (it uses in-memory keys).
- CSV source behind the cache: key is `(csv, symbol, start, end)` with day resolution, so editing a CSV and re-scanning the same day silently serves the stale parquet.

**Post-fix semantics (2026-10-20, second pass)**
- `scan_universe` catches `(ValueError, LookupError)` per symbol and writes manifest + table in a `finally`, so an offline miss is an error row (exit 0) and a RuntimeError mid-scan still leaves manifest/table (the propagated error is not in `errors`).
- Cache window ends at `default_end - ONE_DAY`; cache and `fetch_ticker` now request the same Yahoo window and end on the same bar. Key: `SPY_<start>_<yesterday>.parquet`.
- Manifest `data[symbol].result_file` + `read_results` maps stems back to raw symbols; `BTC/USD` scan + `report` work, index links `BTC-USD.html`.
- CSV source: `CsvDirSource.fingerprint` (12-hex sha256 prefix of file bytes) is appended to the key, so an edited CSV is a miss with a new sha256 in the manifest. Consequence: a CSV scan with the file ABSENT keys without the fingerprint and misses even when a parquet exists, and old content versions are never evicted. yfinance keys are unchanged.

**Semantics confirmed**
- Table `persistence` = stride diagonal of the current regime, `persistence_source` says `stride`/`overlapping` (fallback only when the stride row is null). `null_percentile` in the table = `backtest.null.permutation.percentile_net` (Sharpe null), NOT the top-level analyze `null_percentile` (labeller-induced persistence null); on SPY both the net and gross permutation percentiles were 20.0 with comparable turnover (null 19.3 vs strategy 17.0), so the net figure is fair for this null.
- Ranking key is `(errored, sharpe is None, -sharpe, symbol)`; unscored rows still get a rank after scored ones.
- Manifest `params` echoes every analyze kwarg plus `cost_model`, but not `years`, `source`, `offline`, `cache_dir` or the config path; `data[symbol].start/end` carry the cache window. `created_at` is `+00:00`, `git_commit` is `rev-parse HEAD`.
- `write_result` uses plain `json.dumps` (NaN allowed, like `to_json`); table and manifest use `allow_nan=False`.
- HTML: no `<script>`, everything through `html.escape`, regime ribbon uses `label_regimes` at the bar (trailing only), equity curves re-run `run_walk_forward` from `backtest.params` (min_train, refit, rolling_bars, rule, dead_band) and the same cost conventions as `backtest_report` (constant positions pay entry via `apply_costs`). Log axis clips at 1e-6.
- plan.md Phase 5 boxes were still `[ ]` at review time and the `config/universes/*.yaml` files already existed at HEAD.

**Review mechanics**
- HEAD-vs-worktree CLI comparison needs `git archive HEAD src config | tar -x -C <scratch>/head` (config/default.yaml is resolved relative to the package, so without `config` the archived CLI fails with "config file not found"). Then `sys.path.insert(0, "<scratch>/head/src")`, set `sys.argv`, call `hedgefund.cli.main()`.
- Fast end-to-end scan: write `<SYMBOL>.csv` files from the fixture, a YAML universe, then `hedgefund scan --universe <yaml> --source csv --csv-dir … --cache-dir … --jobs-dir … --no-hmm --null-paths 20 --bootstrap-resamples 20 --json`; ~25 s for SPY + two truncated copies. Prove the warm run is offline by moving the CSV dir away before `--offline`.
- A 100-row symbol yields `net_sharpe null`, stride persistence 1.0, stationary_bear 0.0 and still ranks (last among scored); the `SHORT` row is a handy edge case.

**Why:** these decide whether the Phase 5 gate ("warm run makes no network calls") is actually satisfiable and whether scan numbers match `analyze` for the same ticker on the same day.
**How to apply:** when Phase 6/7 reads a scan run or adds an Alpaca source, check (1) the partial-bar window is fixed or the cache end is yesterday, (2) `--offline` tolerates cache misses per symbol, (3) `report` maps stems back to symbols (store the file name in the manifest). See [[phase4-review-context]] for the HMM timing (47 s per symbol, so a 10-ETF scan with HMM on is ~15 min) and [[phase3-review-context]] for the null conventions.
