---
name: honest-backtesting
description: Checklist and report template for evaluating a trading signal honestly. Use when writing or reviewing backtest code, when reporting Sharpe, drawdown, or P&L for any strategy, or when the user asks whether a result is real, significant, or overfit. Covers costs, benchmarks, null models, PSR/DSR, holdout, and lag.
---

# Honest backtesting

A result is reportable only when every line below is present. Missing lines are reported as missing, not skipped.

## Checklist

```
Backtest honesty:
- [ ] No lookahead: perturbation test passes for every estimator in the path
- [ ] Costs: per-side bps stated; net metrics beside gross; sensitivity at 0/5/10/20/50 bps
- [ ] Turnover and time in market reported
- [ ] Benchmarks: buy-and-hold, constant 50% exposure, long-only version of the signal
- [ ] Null: percentile vs 1,000 full-pipeline shuffles, p = (count ≥ real + 1) / (N + 1), with and without costs
- [ ] Significance: Probabilistic Sharpe and Deflated Sharpe with the number of trials from the parameter search
- [ ] Holdout: last 2 years never used for parameter selection, reported separately
- [ ] Sensitivity: window × threshold grid of net Sharpe; a one-cell win is overfit
- [ ] Lag: bars from local peak or trough to flip, by direction
- [ ] Disclaimer present: historical, not forward-looking
```

## Rules of thumb

- Default to the conclusion "no edge after costs" until the null percentile, PSR, and holdout all agree otherwise. The project's value is the honest measurement and the regime filter, not a winning standalone signal (plan.md section 7).
- Run the permutation null through the whole pipeline including the labeller and costs, so selection effects stay inside the null.
- A random signal that trades more looks easier to beat once costs are included. Always show the gross null beside the net null.
- Never tune on the holdout. If you looked at it, it is no longer a holdout; say so.
- Report distributions (lag, null) not just means.

## Report template

```markdown
## <symbol> <labeller> <rule>  (<date_start> to <date_end>, costs <bps> bps/side)

| metric | gross | net | buy&hold | const 50% | long-only |
|---|---|---|---|---|---|
| Sharpe | | | | | |
| Sortino | | | | | |
| max DD | | | | | |
| hit rate | | | | | |
| turnover /yr | | | | | |

Null percentile: <p> (N=1000, net), <p> (gross). PSR: <x>. DSR (<n> trials): <y>.
Holdout (<start>–<end>): Sharpe <net>, max DD <dd>.
Lag to Bear flip: median <n> bars (crash), <m> bars (recovery).
Sensitivity: <n>/<total> cells with net Sharpe > 0.
Backtests are historical, not forward-looking.
```
