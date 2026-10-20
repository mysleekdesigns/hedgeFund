# JSON contract

## Contents
- Companion fields (must never change)
- Phase 1 addition (shipped)
- Phase 2 additions (shipped)
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

## Phase 1 addition (shipped)

`params.return_type` str, `"simple"` or `"log"`, the rolling-return definition used by the labeller (companion default `"simple"`).

## Phase 2 additions (shipped)

Appended after `disclaimer`, in this order. Every vector is a `{"bear","sideways","bull"}` object and every matrix a 3×3 list in the companion's index order. Floats are JSON numbers or `null` (a state never visited, or a statistic the series is too short for); never NaN.

| Field | Type | Meaning |
|---|---|---|
| `stride_matrix` | 3×3 | Transition matrix of the subsampled chain `labels[::window]`: transitions counted `window` bars apart so consecutive windows do not overlap (Fix 1). It is the `window`-step matrix, not a 1-step matrix. |
| `stride_diagonal` | obj | Diagonal of `stride_matrix`; `null` when the stride sequence has fewer than two labels or a state's stride row is all zeros (never seen as a source) |
| `implied_window_step_diagonal` | obj | Diagonal of `transition_matrix ** window` (Chapman-Kolmogorov). What the stride diagonal would be if the daily chain were first-order Markov. Compare three numbers: overlapping 1-step, implied P^window, stride window-step. The gap between the overlapping 1-step and the stride diagonal is the window autocorrelation. |
| `stride_n_transitions` | int | Transitions in the stride sequence (`len(labels[::window]) - 1`) |
| `run_length_persistence` | obj | `1 − 1 / mean episode length`, computed on the stride sequence `labels[::window]`. On daily labels this equals the overlapping diagonal algebraically (same exits-per-visit ratio), so the daily version carries the same overlap inflation and is not emitted. `null` for a state with no episode. |
| `effective_n` | obj | Per-row effective transition count of the overlapping matrix, `row total / window` (rules/no-lookahead.md) |
| `null_percentile` | obj | `label_shuffle` and `iid` (each: `diagonal_percentile` obj = 100 · P(null diagonal ≤ real), `null_diagonal_mean` obj, `n_paths`); `labeller_induced` (`null_diagonal_mean`, `null_diagonal_lower`, `null_diagonal_upper`, `diagonal_percentile`, `excess_persistence` = real − null mean, `excess_ci` obj of `[lo, hi]`, `n_paths`). The labeller-induced null runs the same labeller on iid Gaussian returns with matched mean and vol (Fix 1). `excess_ci` is a bootstrap interval on the excess: block-bootstrap draws of the real diagonal minus null draws, so it carries both the real diagonal's sampling error and the null spread. The `label_shuffle` and `iid` nulls destroy the window overlap, so a smoothed labeller sits at percentile 100 under them by construction; the labeller-induced null is the honest comparison. |
| `confidence_intervals` | obj | `ci` (0.95), `block_days`, `n_resamples`, `transition_matrix` `{"lower": 3×3, "upper": 3×3}`, `stationary_distribution` `{"lower": obj, "upper": obj}` from the circular block bootstrap |
| `rolling_matrix` | 3×3 | Matrix from the trailing `rolling_bars` labels (Fix 4) |
| `rolling_bars` | int | Trailing window used, default 756 (3 trading years) |
| `drift_tvd` | float or null | Worst-row total variation distance between `transition_matrix` and `rolling_matrix`, in [0, 1]. Rows that are all zeros in either matrix (a state absent from the window) are skipped; `null` if no row remains |
| `ew_matrix` | 3×3 | Exponentially weighted transition counts, row-normalised |
| `ew_half_life` | float | Half-life in bars of the EW counts, default 250 |
| `shrunk_matrix` | 3×3 | Dirichlet shrinkage of the rolling counts toward `transition_matrix` as prior with `shrinkage_alpha` pseudo-counts per row |
| `shrinkage_alpha` | float | Prior strength, default 20 |
| `mixing_time` | int or null | Smallest `n` with every row of `P^n` within `mixing_epsilon` TVD of the stationary vector; `null` if not reached within the step bound (periodic or reducible chain) |
| `mixing_epsilon` | float | TVD tolerance, default 0.01 |
| `tests` | obj | `anderson_goodman` (`statistic`, `p_value`, `df`, `n_transitions`, `note: "on the stride sequence"`) tests first-order dependence vs independence on `labels[::window]`; `homogeneity` (`statistic`, `p_value`, `df`, `n_periods`, `stable` = p ≥ 0.05, `note: "on the stride sequence"`) tests one matrix across contiguous sub-periods of `labels[::window]` (Fix 4); default 2 periods because the stride sequence has only about 125 labels on 10 years. Both tests run on the stride sequence because overlapping daily labels violate their independence assumption and inflate the statistic. Either is `null` when the series is too short. |
| `statistics_params` | obj | Echo of `null_paths`, `bootstrap_resamples`, `block_days`, `rolling_bars`, `ew_half_life`, `shrinkage_alpha`, `mixing_epsilon`, `homogeneity_periods`, `seed` as used |

CLI: `--null-paths`, `--bootstrap-resamples` and `--seed` override the `statistics` block of `config/default.yaml` for fast runs; the remaining statistics values come from the config only.

## Planned additions by phase

Additive only. Keep names stable once shipped.

Phase 3 (backtester): nest under `backtest`:
- `costs` (`fee_bps`, `slippage_bps`, `borrow_bps`, `funding_bps`), `net` and `gross` metric sets (sharpe, sortino, calmar, max_drawdown, hit_rate, avg_win, avg_loss, profit_factor, turnover, time_in_market, min_track_record)
- `benchmarks` (`buy_and_hold`, `constant_50`, `long_only`), `null` (`percentile`, `p_value_gross`, `p_value_net`, `n_paths`)
- `psr`, `dsr` with `n_trials`; `holdout` (same metric set on the last 2 years); `lag` (distribution by direction); `sensitivity` (window × threshold net Sharpe grid)

Phase 4 (HMM): extend `hmm` with `features`, `filtered_probabilities` (last bar), `decision` (hysteresis state), `agreement_with_threshold`, `first_to_flip`.

## Serialisation rules

- One JSON object on stdout, nothing else. Pretty printing is fine; no log lines.
- Keep the companion's NaN behaviour for `walk_forward.sharpe` and `walk_forward.max_drawdown`. For new fields emit `null`, never NaN.
- Floats as JSON numbers, never strings. Dates as ISO `YYYY-MM-DD`.
