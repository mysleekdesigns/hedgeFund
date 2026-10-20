# JSON contract of `hedgefund analyze --json`

Every path below is read from the single object that `hedgefund analyze --json` prints.
Dotted names are nested keys (`backtest.net.sharpe`). `states` index order 0, 1, 2 is
Bear, Sideways, Bull in every matrix and every `{"bear","sideways","bull"}` object.
Floats are JSON numbers or `null`; NaN is only kept where the companion emits it
(`walk_forward.sharpe`, `walk_forward.max_drawdown` on short history).

On failure the command prints `{"error": "..."}` on stdout and exits non-zero.

## Contents

- Companion fields (never change)
- Phase 1 addition
- Phase 2 additions: honest matrix statistics
- Phase 3 additions: `backtest`
- Phase 4 additions: walk-forward `hmm`
- Secondary commands (`backtest`, `scan`, `report`)
- Serialisation rules

## Companion fields (never change)

Identical names, types and order to the companion `markov_regime.py --json`.

| Field | Type | Meaning |
|---|---|---|
| `source` | str | ticker or CSV path analysed |
| `rows` | int | price rows used |
| `date_start` | str | ISO date of the first row |
| `date_end` | str | ISO date of the last row |
| `params` | obj | `window`, `threshold`, `min_train` actually used, plus `return_type` (Phase 1) |
| `states` | list[str] | `["Bear","Sideways","Bull"]` |
| `current_regime` | str | regime as of the last bar |
| `next_state_probabilities` | obj | `bear`, `sideways`, `bull`: P(next \| current) |
| `signal` | float | `bull - bear` in [-1, 1] |
| `transition_matrix` | 3x3 | row = from, col = to, rows sum to 1; a state never seen as a source gives an all-zero row (companion convention) |
| `persistence_diagonal` | obj | `bear`, `sideways`, `bull`: P(stay), counted on **overlapping** daily labels, so it is inflated by the window autocorrelation (see `stride_diagonal`) |
| `stationary_distribution` | obj | `bear`, `sideways`, `bull`, sums to 1 |
| `walk_forward` | obj | `sharpe`, `max_drawdown`, `n_trades`; gross, sign rule, no costs; the first two may be NaN on short history |
| `hmm` | obj | `available: true` with `regimes` and `caveat`, extended in Phase 4; `available: false` with `reason` otherwise |
| `framework` | str | `"Roan (@RohOnChain)"` |
| `disclaimer` | str | `"Backtests are historical, not forward-looking."` |

## Phase 1 addition

| Field | Type | Meaning |
|---|---|---|
| `params.return_type` | str | `"simple"` or `"log"`: the rolling-return definition of the labeller (default `"simple"`) |

## Phase 2 additions: honest matrix statistics

Appended after `disclaimer`. These fix the overlap inflation of the companion matrix
(plan.md Fix 1) and expose drift (Fix 4).

| Field | Type | Meaning |
|---|---|---|
| `stride_matrix` | 3x3 | Transition matrix of the subsampled chain `labels[::window]`: transitions counted `window` bars apart, so consecutive windows do not overlap. It is the `window`-step matrix, not a 1-step matrix |
| `stride_diagonal` | obj | Diagonal of `stride_matrix`. **Use this for persistence.** `null` for a state never seen as a source in the stride sequence |
| `implied_window_step_diagonal` | obj | Diagonal of `transition_matrix ** window` (Chapman-Kolmogorov): what the stride diagonal would be if the daily chain were first-order Markov |
| `stride_n_transitions` | int | Transitions in the stride sequence |
| `run_length_persistence` | obj | `1 - 1 / mean episode length` on the stride sequence; `null` for a state with no episode |
| `effective_n` | obj | Per-row effective transition count of the overlapping matrix, `row total / window` |
| `null_percentile` | obj | Three nulls for the overlapping diagonal. `label_shuffle` and `iid` (each `diagonal_percentile`, `null_diagonal_mean`, `n_paths`) destroy the window overlap, so a smoothed labeller sits at percentile 100 by construction. `labeller_induced` runs the same labeller on iid Gaussian returns with matched mean and vol: `null_diagonal_mean`, `null_diagonal_lower`, `null_diagonal_upper`, `diagonal_percentile`, `excess_persistence` (real minus null mean), `excess_ci` (`[lo, hi]` per state), `n_paths`. **The labeller-induced null is the honest comparison** |
| `null_percentile.labeller_induced.excess_persistence` | obj | `bear`, `sideways`, `bull`: how much stickier the real chain is than what the labeller alone manufactures. Near zero means the "persistence" is an artefact |
| `confidence_intervals` | obj | `ci` (0.95), `block_days`, `n_resamples`, `transition_matrix` `{"lower","upper"}` (3x3 each), `stationary_distribution` `{"lower","upper"}` from the circular block bootstrap |
| `rolling_matrix` | 3x3 | Matrix from the trailing `rolling_bars` labels |
| `rolling_bars` | int | Trailing window used, default 756 |
| `drift_tvd` | float or null | Worst-row total variation distance between `transition_matrix` and `rolling_matrix`, in [0, 1]; all-zero rows skipped |
| `ew_matrix` | 3x3 | Exponentially weighted transition counts, row-normalised |
| `ew_half_life` | float | Half-life in bars of the EW counts, default 250 |
| `shrunk_matrix` | 3x3 | Dirichlet shrinkage of the rolling counts toward `transition_matrix` with `shrinkage_alpha` pseudo-counts per row. The matrix to size from when `drift_tvd` is large |
| `shrinkage_alpha` | float | Prior strength, default 20 |
| `mixing_time` | int or null | Smallest `n` with every row of `P^n` within `mixing_epsilon` of the stationary vector; `null` if not reached (periodic or reducible chain) |
| `mixing_epsilon` | float | TVD tolerance, default 0.01 |
| `tests` | obj | `anderson_goodman` (`statistic`, `p_value`, `df`, `n_transitions`, `note`): first-order dependence vs independence on the stride sequence. `homogeneity` (`statistic`, `p_value`, `df`, `n_periods`, `stable`, `note`): one matrix across contiguous sub-periods of the stride sequence. Either is `null` when the series is too short |
| `tests.anderson_goodman.p_value` | float or null | Small means the stride chain is not independent draws, so a transition matrix is worth reading at all |
| `statistics_params` | obj | Echo of `null_paths`, `bootstrap_resamples`, `block_days`, `rolling_bars`, `ew_half_life`, `shrinkage_alpha`, `mixing_epsilon`, `homogeneity_periods`, `seed` |

## Phase 3 additions: `backtest`

One top-level key. On a history shorter than `min_train + 30` labelled days every metric
is `null`, `n_days` is 0, `backtest.null` and `backtest.holdout` are `null`, and the
sensitivity grid is all `null` cells. Gross and net always ship together with the cost
assumption echoed.

A *metric set* is the object `sharpe`, `sortino`, `calmar`, `max_drawdown`,
`total_return`, `annual_return`, `hit_rate`, `avg_win`, `avg_loss`, `profit_factor`,
`turnover` (one-way notional per year), `time_in_market`, `n_days`.

| Field | Type | Meaning |
|---|---|---|
| `backtest.params` | obj | Echo of `window`, `threshold`, `return_type`, `min_train`, `refit` (`expanding` or `rolling`), `rolling_bars`, `rule` (`sign`, `conviction`, `persistence`, `long_only`), `dead_band`, `periods_per_year`, `horizons`, `null_paths`, `seed`, `holdout_years`, `sensitivity`, `cost_sensitivity_bps`, `lag_order`, `costs` (`fee_bps`, `slippage_bps`, `borrow_bps`, `funding_bps`, `per_side_bps`) |
| `backtest.gross` | obj | Metric set of `position x asset return` plus `psr` and `min_track_record`. With the `sign` rule and zero costs `gross.sharpe` equals the companion `walk_forward.sharpe` |
| `backtest.net` | obj | Same keys after costs. **Quote this one, never gross alone** |
| `backtest.annual_cost` | float | Yearly return drag of the cost model |
| `backtest.horizon_pnl` | obj | Keyed `"1"`, `"5"`, `"10"`, `"20"`: `mean`, `median`, `hit_rate`, `n` of holding each decision `h` days |
| `backtest.cost_sensitivity` | list | One row per level in `cost_sensitivity_bps`: `bps_per_side`, `sharpe`, `total_return`, `max_drawdown`, `annual_cost` |
| `backtest.benchmarks` | obj | `buy_and_hold`, `constant_50`, `long_only`, each a net metric set under the same cost model, plus `note` |
| `backtest.null` | obj or null | `random_sign`, `label_shuffle`, `permutation`, `note`. Each null carries `p_value_net`, `percentile_net`, `null_mean_net`, `null_std_net` (null paths charged the same costs), `p_value_gross`, `percentile_gross`, `null_mean_gross`, `null_std_gross`, `real_gross`, `real_net`, `n_paths`, `null_mean_turnover`. `p = (count(null >= real) + 1) / (N + 1)` |
| `backtest.null.permutation` | obj | Reruns the whole pipeline (labeller, walk-forward, costs) on permuted daily returns. **`p_value_net` is the single significance number to gate on** |
| `backtest.holdout` | obj or null | `cutoff` (ISO), `holdout_years`, `in_sample` and `holdout` each `{"gross","net"}` metric sets, `default_cell` (`window`, `threshold`). The holdout never informs parameter selection |
| `backtest.sensitivity` | obj | `windows`, `thresholds`, `net_sharpe` grid (rows = windows), `n_trials`, `best` (`window`, `threshold`, `net_sharpe`, `holdout_gross_sharpe`, `holdout_net_sharpe`), `share_positive`, `sharpe_variance`, `note`. Scored on the in-sample close only |
| `backtest.psr` | float or null | Probabilistic Sharpe ratio of the net returns against zero |
| `backtest.dsr` | obj | `dsr`, `sr0`, `n_trials`, `sharpe_variance`, `note`. A conservative hybrid, not textbook DSR: the note says exactly how |
| `backtest.lag` | obj | Per labeller variant (`threshold`, `zscore`, `ew`, `ensemble`, `hysteresis`): `crash` and `recovery` lag statistics plus `n_flips`, `false_flips`, `false_flip_rate`. **Hindsight diagnostic**: turning points use future bars. Never feed it into a signal |
| `backtest.disclaimer` | str | `"Backtests are historical, not forward-looking."` |

## Phase 4 additions: walk-forward `hmm`

The companion `hmm` object is extended in place. When `available` is `false` the object
is exactly `{"available": false, "reason": "..."}` (`"disabled via --no-hmm"`,
`"hmmlearn not installed or failed to compile"`, `"history too short for the
walk-forward HMM..."`, or `"hmm runtime error: ..."`) and nothing below exists. When
`available` is `true` the HMM is fitted walk-forward on `log_return` and two realised
volatilities, refitted every `refit_every` bars on past bars only, and the decision reads
the **forward-filtered** probability with hysteresis. Smoothed probabilities and Viterbi
paths are never used.

| Field | Type | Meaning |
|---|---|---|
| `hmm.regimes` | list | Companion shape (`label`, `latent_state`, `mean_daily_return`) for the last refit, Bear / Sideways / Bull order |
| `hmm.caveat` | str | Companion caveat |
| `hmm.features` | obj | `names`, `vol_short`, `vol_long`, `standardisation` |
| `hmm.fit` | obj | Winning restart of the last refit: `seed`, `seeds`, `log_likelihood`, `restart_log_likelihoods`, `converged`, `n_iter`, `emission`, `df`, `covariance_type`, `floored_states`, `feature_names`, `transition_matrix`, `crash_state`, `student_t` |
| `hmm.filtered_probabilities` | obj | `bear`, `sideways`, `bull`: the last bar's P(state_t \| data <= t), sums to 1 |
| `hmm.decision` | obj | `regime` (hysteresis decision on the last bar), `raw_regime` (plain argmax), `enter`, `exit`, `confirm_bars`, `rule` |
| `hmm.decision.regime` | str | **The HMM's held regime.** Compare with `current_regime` for a two-detector agreement gate |
| `hmm.walk_forward` | obj | `n_days`, `n_refits`, `first_date`, `last_date`, `regime_counts`, `mean_log_likelihood`, `params` |
| `hmm.agreement_with_threshold` | obj | `n`, `share_agree`, `confusion` (3x3 int, rows = threshold, cols = HMM), `share_agree_by_regime`, `note` |
| `hmm.first_to_flip` | obj | `bear` and `bull` entries: `n_threshold_entries`, `n_hmm_entries`, `n_matched`, `hmm_first`, `threshold_first`, `same_bar`, `mean_lead_bars`, `median_lead_bars` (positive = the HMM flipped first), `leads`, `hmm_unmatched`, `threshold_unmatched`; top-level `match_window`, `note` |
| `hmm.pnl_comparison` | obj | `threshold` and `hmm` each `{"gross","net"}` metric sets from holding the sign rule on each detector's held regime; `buy_and_hold` (`net`), `costs`, `n_days`, `first_date`, `last_date`, `rule`, `note` |
| `hmm.note` | str | Every refit uses bars strictly before the decision date; probabilities are filtered, not smoothed |

## Secondary commands

These print their own objects, not the `analyze` object.

- `hedgefund backtest --ticker SYM | --csv PATH --json` prints the slim object `source`,
  `rows`, `date_start`, `date_end`, `params`, `backtest`, `framework`, `disclaimer`, with
  no Phase 2 statistics and no HMM. Extra flags: `--rolling-bars`, `--fee-bps`,
  `--slippage-bps`, `--borrow-bps`, `--funding-bps`.
- `hedgefund scan --universe <name|path> --json` runs `analyze` per symbol of a universe
  file through the parquet cache, writes `jobs/<run_id>/` (`manifest.json`, `table.json`,
  `results/<symbol>.json` holding the unchanged `analyze` dict, `html/index.html`) and
  prints `run_id`, `run_dir`, `universe`, `cost_profile`, `symbols`, `table`, `errors`,
  `framework`, `disclaimer`. Table rows are ranked by **net** Sharpe and carry `rank`,
  `symbol`, `current_regime`, `signal`, `persistence` (from `stride_diagonal`, falling
  back to the overlapping diagonal, flagged by `persistence_source`), `stationary_bear`,
  `net_sharpe`, `null_percentile` (`backtest.null.permutation.percentile_net`), `rows`,
  `date_end`, `error`.
- `hedgefund report <run-id> --json` regenerates the HTML from a cached run and prints
  `{"run_id", "html"}`.

## Serialisation rules

- One JSON object on stdout, nothing else. No log lines.
- Companion NaN behaviour is kept for `walk_forward.sharpe` and `walk_forward.max_drawdown`.
  Every new field emits `null`, never NaN.
- Floats as JSON numbers, never strings. Dates as ISO `YYYY-MM-DD`.
- Additive only: names never change once shipped.
