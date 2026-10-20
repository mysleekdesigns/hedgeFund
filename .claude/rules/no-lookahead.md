---
paths:
  - "src/hedgefund/regime/**"
  - "src/hedgefund/backtest/**"
---

# No lookahead, honest estimation

- The signal that earns the return of day t+1 is computed from data through day t only. Matrices, labels, feature scalers, and HMM parameters used at t are fitted on data at or before t.
- Every new estimator or position rule ships with the perturbation test in `tests/test_no_lookahead.py`: mutate prices after day t and assert every output up to t is bit-identical.
- Standardise features with statistics from the training window only.
- HMM signals use forward-filtered probabilities P(state_t | data ≤ t). Smoothed probabilities and Viterbi paths appear only in diagnostics and are labelled as such.
- Report the stride (non-overlapping) matrix next to the overlapping one and say which is which. Confidence intervals for overlapping counts use effective sample size n / window.
- Assert rows of every transition matrix sum to 1 and every probability is in [0, 1].
- Costs are charged on |Δposition| per side. Never report a gross-only Sharpe; the net figure sits beside it with the cost assumption stated.
- Order latent HMM states by mean return so labels are stable across fits.
- The stationary distribution is the long-run mix implied by the current matrix, not a forecast. Say so in docstrings and reports.
