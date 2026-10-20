---
name: phase2-review-context
description: Non-obvious facts established during the Phase 2 (honest statistics) review on 2026-10-19 — calibration numbers for the homogeneity / Anderson-Goodman tests and the block bootstrap, the run-length = diagonal identity, and which conventions the Phase 2 fields inherit
metadata:
  type: project
---

Phase 2 reviewed 2026-10-19 (uncommitted diff on develop, base 89b10aa): 430 tests pass, companion reproduction untouched. One critical (homogeneity test applied to the overlapping daily sequence), several warnings.

**Calibration facts worth re-using (iid Gaussian prices, 2520 bars, 20-day ±5% labeller, 100-200 seeds)**
- Anderson-Goodman homogeneity test on the *daily* overlapping label sequence with 4 periods rejects ~30% of the time at the 5% level. On `labels[::20]` it rejects ~1-2% (4 or 2 periods). Power on a series whose first half is regime-switching and second half iid: stride/2 periods 0.98, stride/4 periods 0.84. So any chi-square test on labels must run on the stride sequence, as the dependence test already does.
- Anderson-Goodman dependence test on the stride sequence is calibrated (~6% rejection on iid).
- Circular block bootstrap (within-block counting, 20-day blocks) covers the diagonal of iid-labelled prices at ~0.90-0.98 and a true Markov chain per cell at 0.89-0.97; the implementer's 0.85 floor is loose but the method is sound. n/window "effective n" is far more conservative than the block CI (SPY Bear: eff n 9.3, CI width ±7pp).
- Run-length persistence `1 - 1/mean episode` equals the transition-matrix diagonal algebraically (exits-per-visit) up to the censored last run, on any sequence. Computing it on the stride sequence just reproduces the stride diagonal; the plan's premise that it is "not inflated by overlap" is false and the implementer documented that.
- The SPY labeller-induced null (Gaussian, matched mean) gives Bull excess persistence of about −11pp: positive drift makes the iid paths sit in Bull longer than SPY does. Not a bug.

**Conventions the Phase 2 fields inherit**
- `stride_diagonal` emits 0.0 (not null) for a zero stride row, like the companion diagonal; the null blocks use NaN→null for the same case. `drift_tvd` is 0.5 whenever one matrix has a zero row the other does not.
- `labeller_induced.excess_ci` is `real − null band`, not a bootstrap interval of the excess; it ignores the real diagonal's own sampling error.

**Review mechanics**
- `git diff` of this phase is >30KB per file group; read the current files with Read instead and only diff the tests.
- Probe scripts live in the session scratchpad and run with `UV_CACHE_DIR=~/.claude/tmp/uv-cache uv run python <path>`; a 200-seed calibration loop over `label_regimes` + a chi-square test takes under a minute.

**Why:** these numbers took the longest to establish and decide whether the headline "matrix NOT stable" verdict on SPY is trustworthy.
**How to apply:** when Phase 3+ adds any test or CI on label sequences, check it runs on the stride sequence or on block resamples, never on raw daily overlapping counts. See [[phase1-review-context]] for the zero-row convention and [[phase0-review-context]] for the uv prefix.
