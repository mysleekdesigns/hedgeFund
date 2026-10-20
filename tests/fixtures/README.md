# Test fixtures

Fixtures are frozen once created. Add new ones; never overwrite an existing file
(the `.claude/rules/testing.md` convention). Tests run offline against these files only.

| File | What | Provenance |
|------|------|------------|
| `spy_10y.parquet` | SPY daily OHLCV, adjusted, 2016-10-20 to 2026-10-16 (2511 rows) | `spy_10y.provenance.json` |

## `spy_10y.parquet`

The Phase 1 reproduction baseline. Pulled with `scripts/freeze_spy_fixture.py`, which
mirrors the companion script's fetch (`yfinance.download(auto_adjust=True)`, `end` = today
normalised to UTC, `start` = `end` minus ten years). Adjusted closes drift as dividends
accrue, so a fresh download will not match this file bit for bit; that is the point of
freezing it.

Schema: tz-naive, sorted, unique `DatetimeIndex` named `date`; float64 columns
`open`, `high`, `low`, `close`; int64 `volume`. The `close` column is what the regime
code consumes; the rest are kept for the Phase 4 HMM features (realised vol, volume).

The provenance JSON records the exact download arguments, the pull timestamp, row count,
date span, the SHA-256 of the parquet file, and the library versions used to write it.

## Reproducing the companion script on this fixture (Phase 1 gate)

A fresh `--ticker SPY` run of the installed plugin can never match this file: its `end`
is "today" and Yahoo's dividend-adjusted closes move every ex-dividend date. The last bar
here is Friday 2026-10-16 (pulled 2026-10-20 00:18 UTC, so Monday 2026-10-19 was not yet
available). Compare on the same input instead:

1. Export the fixture's close column to CSV (`date,close`), for example from Python:
   `pd.read_parquet("tests/fixtures/spy_10y.parquet")[["close"]].to_csv("spy_10y.csv")`.
2. Run the plugin on that CSV, not on a ticker:
   `uv run ~/.claude/plugins/marketplaces/markov-hedge-fund-method/scripts/markov_regime.py --csv spy_10y.csv --json`.
3. Compare our `analyze()` output against that JSON to 6 decimals.

Compare `close` values, not whole DataFrames: the parquet keeps a `datetime64[ms]` index
and a `columns.name` of `Price` from yfinance, neither of which changes any number.

## `spy_10y.companion.json`

The companion script's `--json` output on the fixture above, produced exactly as the three
steps describe (`--csv spy_10y.csv --no-hmm --json`, `source` normalised to `spy_10y.csv`).
The `result` key holds the contract dict; `_provenance` records how it was made. The Phase 1
reproduction test compares our `analyze()` against `result` at 6 decimals. Frozen; never
regenerate it, because the fixture it was made from never changes either.
