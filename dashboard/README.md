# hedgeFund dashboard

A Streamlit view over cached `hedgefund scan` runs. Everything on the page is read from
`./jobs/<run_id>/` (the scan's manifest, result JSON and table) and from the parquet price
cache the scan wrote. The app imports no data source module and opens no network connection.
Backtests are historical, not forward-looking.

## Launch

```sh
uv run hedgefund scan --universe us_etfs   # once, to produce a run
uv run hedgefund dashboard                 # or: uv run streamlit run dashboard/app.py
```

`hedgefund dashboard --jobs-dir PATH` points the app at another jobs directory by setting
`HEDGEFUND_JOBS_DIR`, which `streamlit run` also honours. With no runs under the jobs
directory the page shows a warning and stops.

## Controls (sidebar)

Run, symbol, window, threshold, refit mode (expanding or rolling, with the rolling bar count),
cost profile, and the number of random-sign null paths used for the equity band (the slider
only appears when the scan simulated more than the 20-path minimum). Defaults are
the scan's own parameters. With the defaults, the matrix, intervals, stationary mix, mixing
time and signal come straight from the result JSON (green badge). Any other choice recomputes
them from the cached prices with the run's bootstrap settings (blue badge). Equity curves and
the null band are always rebuilt from the cached closes with the selected cost profile.

## Views

1. **Regime ribbon.** Close price over one coloured band per regime run. Each day's colour
   uses only data up to that day.
2. **Transition matrix with confidence intervals.** 3x3 heatmap, rows are the from-state,
   columns the to-state. Cells read `p [lower, upper]` from the block bootstrap; the caption
   gives the interval level, block length and resample count.
3. **Stationary mix.** The long-run state mix implied by the current matrix, with bootstrap
   error bars. It is not a forecast.
4. **P**n convergence.** A slider (or the Play button, which animates n from 0 upward) shows
   the n-step matrix next to the stationary row, plus the max-row total-variation distance to
   the stationary distribution against n.
5. **Walk-forward equity.** Strategy net and gross of costs, buy-and-hold, and the shaded
   2.5 to 97.5 percentile band of random-sign null strategies net of the same costs. The band
   is drawn with the scan's own seed: at the run's full path count it is the same draw the
   JSON null percentile ranks against, and with fewer paths (the default is 200, for speed) it
   is the leading subset of that draw. The caption names the path counts and the seed, and a
   metric beside the chart shows the JSON random-sign `percentile_net` so the ranking stays visible (the scan table's `null_percentile` column is the permutation null, a different test). The
   caption also states the cost model in basis points. Histories too short for the
   walk-forward show a warning instead.

The footer repeats the disclaimer and the run metadata: run id, creation time, universe, cost
profile and the data source per symbol.

## Tests

```sh
uv run pytest -q tests/test_dashboard_app.py
```

The tests drive the page through `streamlit.testing.v1.AppTest` against an offline scan over
synthetic CSVs, so no server starts and no port is bound.
