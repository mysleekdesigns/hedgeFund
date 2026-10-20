# JSON contract

## Contents
- Companion fields (must never change)
- Planned additions by phase
- Serialisation rules

## Companion fields (must never change)

Emitted by `markov_regime.py --json` and by `hedgefund analyze --json`. `states` index order 0, 1, 2 = Bear, Sideways, Bull everywhere.

| Field | Type | Meaning |
|---|---|---|
| `source` | str | ticker or CSV path analysed |
| `rows` | int | price rows used |
| `date_start`, `date_end` | str | ISO dates of the window |
| `params` | obj | `window`, `threshold`, `min_train` actually used |
| `states` | list[str] | `["Bear","Sideways","Bull"]` |
| `current_regime` | str | regime as of the last bar |
| `next_state_probabilities` | obj | `bear`, `sideways`, `bull`: P(next \| current) |
| `signal` | float | `bull - bear` in [-1, 1] |
| `transition_matrix` | 3×3 list[float] | row = from, col = to, rows sum to 1; a state never seen as a source gives an all-zero row (companion convention, kept for bit-identity) |
| `persistence_diagonal` | obj | `bear`, `sideways`, `bull`: P(stay) |
| `stationary_distribution` | obj | `bear`, `sideways`, `bull`, sums to 1 |
| `walk_forward` | obj | `sharpe`, `max_drawdown`, `n_trades`; first two may be NaN on short history |
| `hmm` | obj | `available: true` → `regimes` (list of `label`, `latent_state`, `mean_daily_return`) and `caveat`; `available: false` → `reason` |
| `framework` | str | `"Roan (@RohOnChain)"` |
| `disclaimer` | str | `"Backtests are historical, not forward-looking."` |

On failure: `{"error": "..."}` on stdout, non-zero exit.

## Planned additions by phase

Additive only. Keep names stable once shipped.

Phase 1 additions (shipped): `params.return_type` str, `"simple"` or `"log"`, the rolling-return definition used by the labeller (companion default `"simple"`).

Phase 2 (honest statistics):
- `stride_matrix` 3×3, transitions counted `window` bars apart
- `run_length_persistence` obj `bear`, `sideways`, `bull` = 1 − 1 / mean episode length
- `null_percentile` obj: diagonal percentile vs label-shuffle null and labeller-induced null; include `excess_persistence` with CI
- `confidence_intervals` obj: block-bootstrap intervals for each matrix cell and stationary entry; `effective_n`
- `rolling_matrix` 3×3 (default 3y) and `drift_tvd` total variation distance to the full-history matrix
- `ew_matrix` 3×3 with `half_life`; `shrunk_matrix` with `alpha`
- `mixing_time` int, steps until `P^n` is within ε of `π`
- `tests` obj: `anderson_goodman_p`, `homogeneity_p`

Phase 3 (backtester): nest under `backtest`:
- `costs` (`fee_bps`, `slippage_bps`, `borrow_bps`, `funding_bps`), `net` and `gross` metric sets (sharpe, sortino, calmar, max_drawdown, hit_rate, avg_win, avg_loss, profit_factor, turnover, time_in_market, min_track_record)
- `benchmarks` (`buy_and_hold`, `constant_50`, `long_only`), `null` (`percentile`, `p_value_gross`, `p_value_net`, `n_paths`)
- `psr`, `dsr` with `n_trials`; `holdout` (same metric set on the last 2 years); `lag` (distribution by direction); `sensitivity` (window × threshold net Sharpe grid)

Phase 4 (HMM): extend `hmm` with `features`, `filtered_probabilities` (last bar), `decision` (hysteresis state), `agreement_with_threshold`, `first_to_flip`.

## Serialisation rules

- One JSON object on stdout, nothing else. Pretty printing is fine; no log lines.
- Keep the companion's NaN behaviour for `walk_forward.sharpe` and `walk_forward.max_drawdown`. For new fields emit `null`, never NaN.
- Floats as JSON numbers, never strings. Dates as ISO `YYYY-MM-DD`.
