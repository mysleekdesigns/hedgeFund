# JSON contract

## Contents
- Companion fields (must never change)
- Phase 1 addition (shipped)
- Phase 2 additions (shipped)
- Phase 3 additions (shipped)
- Phase 4 additions (shipped)
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

## Phase 3 additions (shipped)

One top-level key, `backtest`, appended after `statistics_params`. Everything inside is a JSON number, string, list, object or `null`; NaN and infinity are never emitted (an infinite minimum track record is `null`). Gross and net always ship together with the cost assumption echoed (rules/no-lookahead.md). On a history too short for the walk-forward (`min_train + 30` labelled days) every metric is `null`, `n_days` is 0, `null` and `holdout` are `null`, the sensitivity grid is all `null` cells and `lag` is still computed.

A *metric set* is the object `sharpe`, `sortino`, `calmar`, `max_drawdown`, `total_return`, `annual_return`, `hit_rate`, `avg_win`, `avg_loss`, `profit_factor`, `turnover` (one-way notional per year, the first position entered from flat), `time_in_market` (share of days with a non-zero position), `n_days`.

| Field | Type | Meaning |
|---|---|---|
| `params` | obj | Echo of `window`, `threshold`, `return_type`, `min_train`, `refit` (`expanding` or `rolling`), `rolling_bars`, `rule` (`sign`, `conviction`, `persistence`, `long_only`), `dead_band`, `periods_per_year`, `horizons`, `null_paths`, `seed`, `holdout_years`, `sensitivity` (`windows`, `thresholds`), `cost_sensitivity_bps`, `lag_order`, and `costs` (`fee_bps`, `slippage_bps`, `borrow_bps`, `funding_bps`, `per_side_bps` = fee + slippage). Fee and slippage are per-side bps on \|Δposition\|; borrow (short notional) and funding (gross notional) are annual bps. `dead_band` semantics: `0` is the Phase 1 behaviour (rebalance on every change); for the `sign` and `long_only` rules the band is hysteresis on the **signal** around zero (the position only flips once the signal has crossed zero by more than the band); for the `conviction` and `persistence` rules it is a band in **position space** (a new target is ignored while it sits within the band of the held position) |
| `gross` | obj | Metric set of `position × asset return` plus `psr` (probabilistic Sharpe, P[true Sharpe > 0]) and `min_track_record` (days until PSR reaches 95%, `null` when infinite). With the `sign` rule and zero costs `gross.sharpe` equals the companion `walk_forward.sharpe` exactly |
| `net` | obj | Same keys after `apply_costs` with `params.costs` |
| `annual_cost` | float | Mean cost per day × `periods_per_year`: the yearly return drag of the cost model |
| `horizon_pnl` | obj | Keyed by the horizon as a string (`"1"`, `"5"`, `"10"`, `"20"`): `mean`, `median`, `hit_rate`, `n` of holding each decision `h` days (overlapping samples). `"1"` reproduces the gross per-day returns |
| `cost_sensitivity` | list | One row per level in `cost_sensitivity_bps`: `bps_per_side`, `sharpe`, `total_return`, `max_drawdown`, `annual_cost` with that level as a pure per-side charge (level 0 = gross) |
| `benchmarks` | obj | `buy_and_hold`, `constant_50` (50% exposure), `long_only` (the strategy's positions clipped at 0), each a metric set net of the same cost model, plus `note`. Buy-and-hold and constant 50% hold a fixed position so they pay the entry cost on day one (and funding, if any) and nothing else |
| `null` | obj or null | Every null is reported **with and without costs** from the same seeded paths (Fix 6). `random_sign` (coin-flip ±1 positions) and `label_shuffle` (permuted label values pushed through the walk-forward) each carry: the original keys `percentile`, `p_value`, `n_paths`, `null_mean`, `null_std`, which rank the real **net** Sharpe against null paths that pay the same cost model; `percentile_net`, `p_value_net`, `null_mean_net`, `null_std_net`, exact aliases of those; `percentile_gross`, `p_value_gross`, `null_mean_gross`, `null_std_gross`, which rank the real **gross** Sharpe against the same paths charged nothing; `real_gross`, `real_net`; and `null_mean_turnover`, the mean one-way annual turnover of the null paths (comparable to `net.turnover`; a coin-flipper trades about `periods_per_year` units a year, so charging it costs makes its net null far easier to beat than its gross null, which is why the two pairs differ). `permutation` reruns the whole pipeline (labeller, walk-forward, costs) on permuted daily returns: `p_value_gross`, `p_value_net`, `percentile_gross`, `percentile_net`, `real_gross`, `real_net`, `null_mean_gross`, `null_mean_net`, `null_std_gross`, `null_std_net`, `n_paths`, `null_mean_turnover`. Percentile = 100 × share of null draws strictly below the real statistic; `p = (count(null ≥ real) + 1) / (N + 1)`. `note` repeats this. `null` when the history is too short |
| `holdout` | obj or null | `cutoff` (ISO date, `holdout_years` before the last date), `holdout_years`, `in_sample` and `holdout` each `{"gross": metric set, "net": metric set}`, and `default_cell` `{"window", "threshold"}` naming the cell those metric sets score (the default parameters, never the in-sample best; see `sensitivity.best.holdout_net_sharpe` for that one). Days before the cutoff are in sample; the holdout never informs parameter selection (Fix 6) |
| `sensitivity` | obj | `windows`, `thresholds`, `net_sharpe` (rows = windows, cols = thresholds, cells `null` when a cell cannot be scored), `n_trials`, `best` (`window`, `threshold`, `net_sharpe` at the in-sample argmax, plus `holdout_gross_sharpe` and `holdout_net_sharpe`: the best cell's labeller and walk-forward re-run on the **full** close with the same refit / rule / dead band / min_train, costs applied, Sharpe of the returns on dates at or after `holdout.cutoff` only; all five `null` if no cell is finite, the holdout pair `null` when the cell cannot run or the holdout slice is too short), `share_positive`, `sharpe_variance` (per-period, for the DSR), `note`. The grid is scored on the in-sample close only; the best cell's holdout figure is the out-of-sample check that exposes selection (compare it with `holdout.holdout.net.sharpe` of the default cell) |
| `psr` | float or null | Probabilistic Sharpe ratio of the net returns against zero (Bailey and López de Prado 2012) |
| `dsr` | obj | `dsr` (PSR against the expected maximum Sharpe of `n_trials` zero-skill trials), `sr0` (that per-period benchmark), `n_trials` (= `sensitivity.n_trials`), `sharpe_variance`, `note`. **Conservative hybrid, not textbook DSR**: `dsr` is the PSR of the *default* `(window, threshold)` cell's *full-sample* net returns against `sr0` built from the *in-sample* grid's Sharpe variance; the default cell is not the in-sample argmax and its sample is longer than the grid cells', so the two sides are not the same experiment. `note` says exactly this. When the grid has fewer than two finite cells the variance is taken as 0.0, so `sr0` = 0 and `dsr` = `psr` |
| `lag` | obj | Per labeller variant (`threshold`, `zscore`, `ew`, `ensemble`, `hysteresis`): `crash` (peak → first Bear label) and `recovery` (trough → first Bull label), each `n`, `n_undetected`, `mean`, `median`, `p25`, `p75`, `max`, `lags` (list of int bars), `n_flips` (int, entries into Bear for `crash` / Bull for `recovery`), `false_flips` (int, entries with no matching peak / trough within the preceding `order` bars) and `false_flip_rate` (`false_flips / n_flips`, `null` when `n_flips` is 0), plus the variant's `order` and `note`; a variant is `null` when it has no labels on the history. Top-level `order` and `note`. **Hindsight diagnostic**: turning points use `order` future bars; never feed it into a signal (Fix 3) |
| `disclaimer` | str | `"Backtests are historical, not forward-looking."` |

`hedgefund backtest --ticker SYM | --csv PATH [--json]` emits the slim object `source`, `rows`, `date_start`, `date_end`, `params` (companion style: `window`, `threshold`, `min_train`, `return_type`), `backtest`, `framework`, `disclaimer` with no Phase 2 statistics and no HMM.

CLI flags (both commands unless noted): `--refit expanding|rolling|rolling:N` (N sets `rolling_bars`), `--rule`, `--dead-band`, `--cost-profile` (a name under `costs.profiles`; default is the `cost_profile` of the config's `universe`). `backtest` only: `--rolling-bars`, `--fee-bps`, `--slippage-bps`, `--borrow-bps`, `--funding-bps` (each overrides the profile), `--null-paths`, `--seed`. `analyze` only: `--no-hmm`, `--hmm-seeds` (comma-separated Baum-Welch seeds, e.g. `0,1,2`), `--hmm-refit-every` (bars between walk-forward HMM refits), `--hmm-emission gaussian|student_t|auto` (an unknown value is an error naming it); the remaining HMM values come from the `hmm` block of `config/default.yaml`, which is read only when the HMM runs (`--no-hmm` skips it entirely). Everything else comes from the `backtest`, `costs` and `hmm` blocks of `config/default.yaml`.

## Phase 4 additions (shipped)

The companion `hmm` object is extended in place; its top-level position and its companion keys do not move. When `available` is `false` the object is exactly `{"available": false, "reason": "..."}` as before: `reason` is `"disabled via --no-hmm"`, `"hmmlearn not installed or failed to compile"`, a string starting with `"history too short for the walk-forward HMM"` (fewer than `min_train + 30` feature rows), or `"hmm runtime error: ..."` for a non-`ValueError` failure inside the walk-forward. A bad HMM parameter (`refit_every` 0, `enter <= exit`, `confirm_bars` 0, `vol_long <= vol_short`, an unknown emission) is **not** a degrade path: it raises `ValueError`, so the CLI prints `{"error": "..."}` and exits non-zero like every other bad flag. When `available` is `true` the keys come in this order: `available`, `regimes`, `caveat` (companion, unchanged in type), then the additions below (`features`, `fit`, `filtered_probabilities`, `decision`, `walk_forward`, `agreement_with_threshold`, `first_to_flip`, `pnl_comparison`, `note`). `regimes` and `fit` describe the **last refit**, the model that scores the final bar; `filtered_probabilities` and `decision` are the **final bar's**. Everything is a JSON number, string, list, object or `null`; NaN is never emitted.

The HMM is fitted walk-forward (plan.md Fix 5): a multi-restart, full-covariance Gaussian HMM on `log_return`, `vol_<vol_short>`, `vol_<vol_long>` standardised with training-window statistics only, refitted every `refit_every` bars on at most `train_bars` past bars, with the signal read from the **forward-filtered** probability P(state_t | data ≤ t) and a hysteresis rule. Smoothed probabilities and Viterbi paths are never used (rules/no-lookahead.md). `min_train` is the companion `params.min_train` (`backtest.min_train`), shared with the walk-forward backtest.

| Field | Type | Meaning |
|---|---|---|
| `regimes` | list | Companion shape (`label`, `latent_state`, `mean_daily_return`) for the last refit, in Bear / Sideways / Bull order; `latent_state` is the index after ordering by mean return, so it is always 0, 1, 2; `mean_daily_return` is de-standardised back to a daily log return |
| `features` | obj | `names` (list[str], column order of the feature matrix), `vol_short`, `vol_long` (int, realised-volatility windows in bars), `standardisation` (str: training-window statistics only) |
| `fit` | obj | The winning restart of the last refit: `seed` (int), `seeds` (list[int], every restart tried), `log_likelihood` (float, Gaussian, on the training block), `restart_log_likelihoods` (obj keyed by seed as a string; `null` for a restart that failed), `converged` (bool), `n_iter` (int, Baum-Welch iterations run), `emission` (`"gaussian"` or `"student_t"`), `df` (float or `null`, Student-t degrees of freedom), `covariance_type` (`"full"`), `floored_states` (list[int], empty for a healthy fit: states whose fitted covariance was rank-deficient, typically a state holding a single outlier bar, and had hmmlearn's `min_covar` added to its diagonal so the filter can use it), `feature_names` (list[str]), `transition_matrix` (3×3, Bear / Sideways / Bull order, rows sum to 1), `crash_state` (`flagged` bool, `state` int or `null`, `occupancy` list[float], `mean_run_length` list[float or null], `max_occupancy`, `max_mean_run`: the single-outlier-state diagnostic), `student_t` (`null` unless evaluated: `gaussian_log_likelihood`, `by_df` obj, `best_df`, `student_t_log_likelihood`, `prefer_student_t`, `note`) |
| `filtered_probabilities` | obj | `bear`, `sideways`, `bull`: the last bar's forward-filtered P(state_t \| data ≤ t), sums to 1 |
| `decision` | obj | `regime` (str, the hysteresis decision on the last bar), `raw_regime` (str, plain argmax of the filtered row), `enter`, `exit` (float, hysteresis thresholds), `confirm_bars` (int), `rule` (str, says the decision is on the filtered probability and never on smoothed or Viterbi paths) |
| `walk_forward` | obj | `n_days` (int, decision dates scored), `n_refits` (int), `first_date`, `last_date` (ISO, `null` when nothing scored), `regime_counts` (obj keyed `Bear`, `Sideways`, `Bull`: held-regime day counts), `mean_log_likelihood` (float or `null`, mean per-bar filtered log-likelihood), `params` (echo of `min_train`, `refit_every`, `train_bars`, `seeds`, `n_iter`, `emission`, `enter`, `exit`, `confirm_bars`, `feature_names`) |
| `agreement_with_threshold` | obj | Day-by-day agreement on the dates both detectors label: `n` (int, common dates), `share_agree` (float in [0, 1] or `null` when `n` is 0), `confusion` (3×3 list[int], rows = threshold regime, cols = HMM decision, Bear / Sideways / Bull order), `share_agree_by_regime` (`bear`, `sideways`, `bull`: share of the days the threshold says that regime on which the HMM agrees; `null` when the threshold never says it), `note` |
| `first_to_flip` | obj | Which detector flips first (plan.md Fix 3 and Fix 5). `bear` and `bull` each describe entries **into** that regime (a bar labelled the target whose previous bar is not; the first bar never counts): `n_threshold_entries`, `n_hmm_entries`, `n_matched` (int; each threshold entry is paired greedily, in time order, with the **nearest** unmatched HMM entry within `match_window` bars either side, ties to the earlier one, so a one-bar HMM flicker long before the threshold flip stays unmatched instead of inflating the HMM's lead), `hmm_first`, `threshold_first`, `same_bar` (int, matched pairs by who was earlier), `mean_lead_bars`, `median_lead_bars` (float or `null` with no match; lead = threshold bar − HMM bar, **positive = the HMM flipped first**), `leads` (list[int], one per matched pair), `hmm_unmatched`, `threshold_unmatched` (int, flips the other detector never made within the window). Top level: `match_window` (int) and `note` |
| `pnl_comparison` | obj | Net and gross P&L of each detector (plan.md Fix 5 "Comparison"). On the dates both label series share and for which the next close exists, each detector holds the sign rule on its **held** regime at `t` (Bull +1, Bear −1, Sideways 0) and earns the close-to-close simple return of `t+1`; the last common date drops. `threshold` and `hmm` each carry `gross` and `net` Phase 3 metric sets (same keys as `backtest.gross`, without `psr` / `min_track_record`); net applies the same cost model to both through `apply_costs`. `buy_and_hold` carries `net` only, charged like the Phase 3 benchmark (constant +1 pays the entry cost and funding). `costs` echoes `fee_bps`, `slippage_bps`, `borrow_bps`, `funding_bps`, `per_side_bps`; `n_days` (int), `first_date`, `last_date` (ISO or `null`), `rule` (says this is a direct regime-position comparison, not the Phase 3 matrix-signal walk-forward) and `note`. With fewer than 2 common dates every metric is `null` and `n_days` is 0; never NaN |
| `note` | str | Says every refit uses bars strictly before the decision date and that the probabilities are filtered, not smoothed |

## Planned additions by phase

Additive only. Keep names stable once shipped. Phase 4 has shipped; later phases add top-level keys only.

## Serialisation rules

- One JSON object on stdout, nothing else. Pretty printing is fine; no log lines.
- Keep the companion's NaN behaviour for `walk_forward.sharpe` and `walk_forward.max_drawdown`. For new fields emit `null`, never NaN.
- Floats as JSON numbers, never strings. Dates as ISO `YYYY-MM-DD`.
