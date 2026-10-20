---
name: phase3-review-context
description: Non-obvious facts from the Phase 3 (backtester upgrade) review on 2026-10-19 — SPY null-model numbers and why they look the way they do, the cost-artefact in the random-sign null, dead-band semantics per rule, lag-metric blind spot, runtime per walk-forward pass, and the HEAD-vs-worktree contract comparison recipe
metadata:
  type: project
---

Phase 3 reviewed 2026-10-19 (uncommitted diff on develop, base f31dd06): 744 tests pass, companion + Phase 2 fields byte-identical on the SPY fixture, `backtest` appended as the 36th top-level key. No criticals; four warnings (net-only random-sign/label-shuffle nulls, dead band inert for the sign rule, no holdout figure for the grid's best cell, lag table has no false-positive count).

**SPY fixture numbers worth re-using (sign rule, 20/0.05, us_etfs 3 bps per side)**
- Gross Sharpe 0.246, net 0.216, turnover 18.8 units/yr, annual cost drag 0.56%, time in market 97.6%, short 8.9% of days. Buy-and-hold net Sharpe 0.825.
- Permutation null (shuffled returns, full pipeline) mean gross Sharpe ~0.64 and label-shuffle null mean ~0.83: both nulls are mostly long and inherit SPY's drift, so the strategy sits at the 0th percentile with p = 1. That is an honest result, not a bug.
- Random-sign null: gross percentile 75.5 (p 0.25) but net percentile 96.4 (p 0.037). The null trades ~252 units/yr vs the strategy's 18.8, so the cost charge alone manufactures the "significance". Any null with turnover unlike the strategy's must be shown gross as well as net.
- Sensitivity grid best cell (20, 0.10) in-sample net 0.70 but holdout 0.35, same as the default cell's 0.35; the in-sample best does not generalise.
- One labeller + walk-forward pass on 2511 rows is ~12 ms, so 1000 permutation + 1000 label-shuffle paths add ~30 s to `analyze`.

**Semantics confirmed**
- `dead_band` compares the *target position* with the held one. For the sign rule targets are ±1/0, so any band < 1 is a no-op and a band in [1, 2) only blocks exits to flat; a band >= 2 never trades. It is a real signal band only for conviction/persistence.
- Detection lag measures time to the first Bear/Bull label after a ±order extremum, i.e. regime *entries* only. Hysteresis only changes exits, so its lag row is identical to the plain threshold row by construction; no false-positive count exists.
- `CostModel()` defaults are (1, 2, 0, 0) while config `us_etfs` has borrow 30, so `analyze()` called from Python with `cost_model=None` differs slightly from the CLI for any strategy that shorts.
- DSR is PSR of the *default* cell's full-sample net returns against SR0 from the 15 in-sample grid cells; the default cell is not the grid argmax, so it is a conservative hybrid rather than textbook DSR.

**Review mechanics**
- Before/after contract check: `git archive HEAD src | tar -x -C <scratch>/head`, then run a script with `sys.path.insert(0, "<scratch>/head/src")` under `uv run --no-sync python`; the archived copy shadows the editable install. `analyze()` needs `source=` as a keyword.
- `hedgefund backtest --csv <export> --json --null-paths 5` runs in a few seconds and is the quickest end-to-end smoke test; `--refit rolling:N` and the per-bps flags work.

**Why:** these are the numbers and conventions that decide whether the Phase 3 significance claims are trustworthy, and they are invisible from the diff.
**How to apply:** when Phase 4+ adds a detector or position rule, check (1) its null is reported gross and net when its turnover differs from the strategy's, (2) the dead band actually applies to its targets, (3) any lag claim is paired with a false-positive rate. See [[phase2-review-context]] for the stride-sequence rule and [[phase1-review-context]] for the zero-row convention.
