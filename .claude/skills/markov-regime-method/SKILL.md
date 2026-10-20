---
name: markov-regime-method
description: Domain knowledge for the Markov regime method this repo implements. Trailing-return labelling, transition-matrix MLE, n-step forecast, stationary distribution, bull-minus-bear signal, walk-forward backtest, Gaussian HMM, the companion JSON contract, and the six known weaknesses with their fixes. Use when implementing or reviewing anything in src/hedgefund/regime, backtest, report, or the Pine overlay, or when the user mentions regimes, transition matrices, persistence, stationary distribution, or HMM.
user-invocable: false
---

# Markov regime method

Three-state chain fitted to an asset's daily closes, used as a signal, a risk filter, and a backtested strategy. Reproduce it faithfully first (Phase 1), then harden it (Phases 2 to 4).

## Core definitions

- **Label** each day by trailing simple return over `window` bars (default 20): `r_t = close_t / close_{t-window} - 1`. `r_t >= +threshold` is Bull, `r_t <= -threshold` is Bear, else Sideways. Default threshold 0.05. State index order is fixed: `["Bear", "Sideways", "Bull"]` = 0, 1, 2.
- **Transition matrix** `P[i, j] = count(i → j) / count(i → *)`, the maximum-likelihood estimate from consecutive labels. Rows sum to 1.
- **n-step forecast** `P^n` (Chapman-Kolmogorov). Row `s` of `P^n` is the distribution n days ahead given state `s` today.
- **Stationary distribution** `π` with `π P = π`, the left eigenvector for eigenvalue 1, normalised to sum 1. It is the long-run regime mix implied by `P`, not a forecast; `P^n` collapses to it after the mixing time.
- **Persistence** is the diagonal `P[s, s]`. High means sticky regime. With overlapping windows most of it is window autocorrelation; see Fix 1.
- **Signal** `= P[current, Bull] - P[current, Bear]` in [-1, 1]. Sign is direction, magnitude is conviction.
- **Walk-forward backtest**: at each day t ≥ `min_train` (252), fit `P` on labels up to t, take position `sign(signal_t)`, earn `close_{t+1} / close_t - 1`. Incremental O(n) count update is bit-identical to refitting from scratch. Report Sharpe (annualised, √252) and max drawdown.
- **HMM**: Gaussian HMM, 3 components, on daily returns (Phase 1) and later on a feature matrix (Phase 4). Order states by mean return. Degrade gracefully when hmmlearn is missing.

## Module map (from plan.md section 3)

| Concern | Module |
|---|---|
| labels, alternative labellers | `src/hedgefund/regime/labels.py` |
| matrix, `P^n`, stationary, mixing time, stride, EW counts, shrinkage | `src/hedgefund/regime/markov.py` |
| signal, conviction and persistence sizing | `src/hedgefund/regime/signal.py` |
| HMM fit, filter, state ordering | `src/hedgefund/regime/hmm.py` |
| bootstrap, nulls, Anderson-Goodman, homogeneity tests | `src/hedgefund/regime/stats.py` |
| walk-forward, costs, metrics, benchmarks | `src/hedgefund/backtest/` |
| JSON contract, terminal, HTML | `src/hedgefund/report/` |

`regime/` is pure: arrays in, arrays out, no I/O.

## Reference files

- **JSON contract**, every field and type the companion skill emits plus the planned additions: [reference/json-contract.md](reference/json-contract.md)
- **Known weaknesses and fixes**, root cause, implementation, verification, owning phase for Fix 1 to Fix 6: [reference/known-weaknesses.md](reference/known-weaknesses.md)

## Reference implementation

The audited companion script is installed at `~/.claude/plugins/cache/markov-hedge-fund-method/markov-hedge-fund-method/1.0.0/scripts/markov_regime.py` (PEP 723, run with `uv run`). Read it as reference when porting; do not copy its lookahead-free walk-forward loosely, port it exactly and prove equality to 6 decimals on the frozen SPY fixture.

## Composition patterns to preserve

1. **Confirmation gate**: take an existing strategy's long only when `signal > 0`.
2. **Tail-risk size filter**: scale size by `1 - stationary_distribution.bear`.
3. **Standalone**: `position = signal`.
