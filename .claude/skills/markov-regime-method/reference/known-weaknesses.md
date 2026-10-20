# Known weaknesses and their fixes

## Contents
- Fix 1: overlapping-window autocorrelation (Phase 2)
- Fix 2: back-dated Pine labels (Phase 6)
- Fix 3: labeller lag (Phases 2, 3, 4)
- Fix 4: non-stationary matrix (Phase 2)
- Fix 5: HMM inputs and hindsight (Phase 4)
- Fix 6: costs, benchmarks, significance (Phase 3)

Full detail lives in `plan.md` section 1. This file is the implementer's summary: what to build and how to prove it.

## Fix 1: overlapping-window autocorrelation (Phase 2)

Root cause: labels at t and t+1 share window−1 bars, so consecutive transitions are not independent. Diagonal inflates to 80 to 95% on every asset.

Build: stride counting (t → t+window) as the primary matrix; run-length persistence 1 − 1/mean episode length; labeller-induced null (iid returns with matched mean and vol through the same labeller) with bootstrap CI on excess persistence; Anderson-Goodman chi-square on the stride sequence; effective n = n / window for overlapping CIs.

Prove: on iid synthetic returns the overlapping diagonal is inflated and the stride and run-length estimates sit near the marginal frequencies. On a synthetic chain with a known matrix all three converge to the truth.

## Fix 2: back-dated Pine labels (Phase 6)

Root cause: the overlay waits `min_regime_hold` bars then draws the label at the bar where the flip started.

Build: label at `bar_index` inside `barstate.isconfirmed`; hindsight toggle off by default with hollow style; `strategy()` variant entering next bar open with `process_orders_on_close=false`.

Prove: Bar Replay on BTCUSDT and SPY daily shows no label on a passed bar; strategy trade list shows every entry one bar after a confirmed flip. Manual check, recorded in the commit message.

## Fix 3: labeller lag (Phases 2, 3, 4)

Root cause: a 20-day trailing return needs many down days before crossing −5%.

Build: lag measurement (bars from local peak or trough to flip, by direction); z-score thresholds; multi-window ensemble (5, 10, 20, 60); exponentially weighted return (half-life ≈ 10); hysteresis (enter −5%, exit −2%, min hold 2 to 3 bars); online HMM forward filter with a lag versus false-positive table; size for the median lag.

Prove: each variant scored on net P&L and its lag versus false-positive curve. The chosen default beats plain 20-day ±5% on net Sharpe and shortens median crash-detection lag.

## Fix 4: non-stationary matrix (Phase 2)

Root cause: one matrix over 10 years averages across volatility regimes.

Build: exponentially weighted counts (half-life ≈ 250 bars); rolling 3-year matrix with total-variation drift metric; Dirichlet shrinkage toward the full-history prior with pseudo-count α; chi-square homogeneity test across sub-periods reported as "matrix stable: yes/no"; walk-forward always; stationary distribution reported as long-run mix, not forecast.

Prove: on a synthetic chain whose matrix changes halfway, EW and rolling estimates track the change and full-history does not; homogeneity test rejects there and not on a stationary series.

## Fix 5: HMM inputs and hindsight (Phase 4)

Root cause: price levels are non-stationary; smoothed or Viterbi paths use future data.

Build: features = log return, 10- and 20-day realised vol, optional range and volume z-score, standardised with training-window stats, full covariance; multi-restart Baum-Welch keeping best log-likelihood; states ordered by mean return; sliding-window walk-forward refit; signal from filtered probabilities only; hysteresis 0.6 / 0.4 with 2 to 3 bar confirmation; Student-t emissions if a single-outlier crash state appears.

Prove: perturbation test on the filtered path; smoothed and filtered paths differ on the same data; fixed seed gives identical state sequences.

## Fix 6: costs, benchmarks, significance (Phase 3)

Root cause: gross Sharpe and max DD only, for a strategy that can flip daily.

Build: per-side cost = commission + half spread + slippage on |Δposition| (2 bps ETFs, 15 bps crypto, borrow for shorts) with a 0/5/10/20/50 bps sensitivity table; signal dead-band rebalancing; benchmarks buy-and-hold, constant 50%, long-only; permutation test through the full pipeline, p = (count ≥ real + 1)/(N + 1), with and without costs; Probabilistic and Deflated Sharpe with the trial count from the parameter search; 2-year out-of-sample holdout; window × threshold heatmap; Sortino, Calmar, hit rate, avg win/loss, profit factor, P&L by horizon, time in market, turnover, min track record length.

Prove: zero costs reproduce Phase 1 numbers exactly; costs reduce returns monotonically; on a no-signal series the permutation p-value is uniform and PSR ≈ 0.5.
