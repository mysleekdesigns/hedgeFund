---
name: phase4-review-context
description: Non-obvious facts from the Phase 4 (walk-forward HMM) review on 2026-10-20 — SPY timing and fit numbers, why the Student-t "prefer" flag is not a tail test, the first_to_flip earliest-match bias, which degrade paths swallow config errors, and the probe recipes
metadata:
  type: project
---

Phase 4 reviewed 2026-10-20 (uncommitted diff on develop, base c7dbc4c): suite passes (gate runner reported 999 tests), non-HMM contract on the SPY fixture byte-identical to HEAD with `hmm=False`. No criticals; four warnings (first_to_flip earliest-match bias, Student-t comparison at the Gaussian scale, no HMM net P&L despite Fix 5 asking for it, config errors swallowed as "hmm runtime error").

**SPY fixture numbers worth re-using (defaults: seeds 0,1,2; refit 21; train_bars 2520; min_train 252; features log_return, vol_10, vol_20)**
- Full walk-forward ~47 s: 2239 decision days, 107 refits, all converged, no crash state flagged, held regimes Bull 887 / Bear 734 / Sideways 618. One 3-seed full-cov fit on 2491 rows is ~1.35 s (seed 2 wins by ~1 nat over seeds 0/1); one pure-numpy `forward_filter` over 2491 rows is ~0.28 s.
- Full-sample fit: transmat diagonal 0.977 / 0.964 / 0.974, de-standardised state means +0.012% / +0.063% / +0.084% per day (all three positive: "Bear" is the low-return high-vol state, not a negative-mean state).
- Student-t evaluated at the Gaussian EM scale on SPY: by_df {3: −4787, 5: −4639, 10: −4551, 30: −4542} vs Gaussian −4596. Monotone toward the Gaussian limit and the largest df wins by +55 nats, so `prefer_student_t=True` only says the EM covariance is slightly tight (t covariance is df/(df−2)·Σ), not that tails are heavy. Default emission is gaussian and `auto` never fired on SPY, so the live signal was unaffected.

**Semantics confirmed**
- Walk-forward: refit(t) trains on rows [max(0, t − train_bars), t), scaler on that block only, filter re-seeded from the block's last filtered row, then one `filter_step` on row t. Hysteresis state is carried across refits and starts at Sideways. Perturbation test (MIN_TRAIN 60, refit_every 50, cuts 81/150/301/450/598) straddles refit boundaries at close idx 80, 130, 180, ….
- `first_to_flip` pairs each threshold entry with the EARLIEST unmatched HMM entry inside ±match_window, so a one-bar HMM flicker 15 bars early reports lead 15 and leaves the real entry unmatched. Any quoted "HMM leads by N bars" from this field is biased upward until it is nearest-match.
- `hmm_block` wraps every exception as `{"available": False, "reason": "hmm runtime error: ..."}` with exit 0, including bad options (`--hmm-refit-every 0`, enter <= exit, confirm_bars 0). The backtest flags, by contrast, fail the command.
- Rank-deficient state covariances are floored with hmmlearn's `model.min_covar` (1e-3), not the 1e-7 hmmlearn uses when scoring, so for a floored state the forward-filter likelihood differs from `model.score`; `HMMFit.floored_states` / `fit_summary["floored_states"]` records it. The contract reference did not list the field at review time.
- Companion `hmm.regimes[].mean_daily_return` is now a de-standardised daily LOG return from the last refit's block and `latent_state` is always 0/1/2; names and types unchanged.

**Review mechanics**
- Probe scripts: `UV_CACHE_DIR=~/.claude/tmp/uv-cache LOKY_MAX_CPU_COUNT=1 uv run --no-sync python <path>`; the SPY fixture is `tests/fixtures/spy_10y.parquet` (one column, read with `pd.read_parquet(...).iloc[:, 0]`). The HEAD-vs-worktree comparison from [[phase3-review-context]] still works with `hmm=False`.
- `hmm.py` was still being edited by the fit implementer while I reviewed; re-read it before the final verdict rather than trusting the first read.

**Why:** these decide whether the Phase 4 "which detector flips first" and Student-t claims are trustworthy, and they are invisible from the diff.
**How to apply:** when a later phase quotes an HMM lead or P&L, check (1) the matching is nearest not earliest, (2) any Student-t switch rescales the scale matrix by (df−2)/df or is diagnostic only, (3) an HMM-driven position has the Phase 3 net metric set, turnover and null beside it. See [[phase3-review-context]] for the null/cost conventions and [[phase2-review-context]] for the stride rule.
