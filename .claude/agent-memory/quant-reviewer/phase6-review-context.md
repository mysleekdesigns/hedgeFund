---
name: phase6-review-context
description: Non-obvious facts from the Phase 6 (skill + Pine) review on 2026-10-20 — the inherited stationary-table mislabel in both overlays, which Pine regressions the static tests miss (mutation probe), HMM reference paths verified against a real run, plugin install-string format, zsh `echo ====` trap
metadata:
  type: project
---

Phase 6 reviewed 2026-10-20 (untracked files on develop, base 52f695a): 40 Phase 6 tests pass; every field path in `skills/regime/SKILL.md` and `skills/regime/reference/json-contract.md` (102 paths) resolves in a real `analyze --json` run on the SPY CSV with the HMM ON (the `test_skill.py` walk is vacuous for `hmm.*` because it runs `--no-hmm` and `_walk` returns early on `available: false`). All documented flags exist in `cli.py`; `POSITION_RULES = (sign, conviction, persistence, long_only)`, `EMISSION_CHOICES = (gaussian, student_t, auto)`, `--refit rolling:N`.

**Pine facts that are invisible from the diff**
- Internal regime code is 0=Side, 1=Bull, 2=Bear. The matrix table remaps display→internal (`disp_to_internal`), but the stationary table reads `M[0]` as Bull, `M[1]` as Bear, `M[2]` as Side, i.e. row 0 of P^50 in INTERNAL order. So BULL shows π_Side, BEAR shows π_Bull, SIDE shows π_Bear. The original companion overlay (lines 318-320) has the identical bug; the Yaniv Haver fix only touched the matrix card. On SPY it would print BULL 79% / BEAR 14% / SIDE 8%.
- Stride counting `bar_index % w == 0` with `regime[w] -> regime` is the same phase-0 subsample as Python `labels[::w]` (first count at bar 2w because `regime` is na for bar < w); no off-by-one.
- `regime != last_lbl_regime` with `last_lbl_regime` initially na is inherited verbatim from the original; whether Pine evaluates `int != na` as true decides if the first label state ever initialises. Cannot verify offline.
- Unverifiable offline: `array<int> x = cond ? arrA : arrB` (array ternary) compiling in v5; "hollow" label is rendered as `color.new(color.white, 100)` fill + tinted text (no hollow label style exists).

**Static Pine test gaps (mutation probe via `tests/test_pine.py` helpers) — first pass; all fixed same day except two residuals below**
Caught: back-dating through a variable or `bar_index[k]`, count update moved outside the guard, `regime[lookback_window - 1]`. Missed: `label.set_x(lb, bar_index - k)` after creation, hindsight label drawn without the `show_hindsight_labels` branch, `if not barstate.isconfirmed`, `(barstate.isconfirmed or true)`, `last_lbl_regime :=` moved to top level. The guard walk only checks that SOME enclosing `if` header contains the token.

**Post-fix (2026-10-20, second pass, 53 tests)**: stationary reads go through `disp_to_internal` and a test pins it; `label.set_*` movers banned; hindsight label must sit under an `if ... show_hindsight_labels` header; `last_*_regime :=` must be confirmed-guarded; guard regex `(?<!not )barstate\.isconfirmed(?!\s+or\b)`; `test_skill.py` runs the HMM on (500 rows, refit 100) and walks every `hmm.*` row. Residual misses: `if held or barstate.isconfirmed` (token after `or`) and `regime[lookback_window - 1]` (the substring test matches the comment on line ~217).

**Plugin mechanics**
- Install string is `<plugin>@<marketplace>` (companion is registered as `markov-hedge-fund-method@markov-hedge-fund-method` in `~/.claude/plugins/installed_plugins.json`); cache path `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>`. Our manifests mirror the companion shape exactly (`source: "."`, `skills/<name>/SKILL.md`). `source: "."` copies the whole checkout into the cache, so `uv run --project ${CLAUDE_PLUGIN_ROOT}` resolves a fresh venv in the cached copy.

**Review mechanics**
- Full `analyze --json` with HMM on the SPY CSV (`--null-paths 20 --bootstrap-resamples 20`) is ~1 min and is the only way to exercise the `hmm.*` reference rows.
- zsh: `echo ====` fails with "=== not found" (`=cmd` expansion). Quote it or use `echo "----"`.

**Why:** these decide whether the Pine overlay's numbers can be trusted and whether the skill docs are honest; none of it shows in the diff.
**How to apply:** if Phase 6 is revisited, check (1) the stationary table mapping was fixed, (2) the static tests gained a `show_hindsight_labels` ancestry check and a `label.set_x` ban, (3) the Bar Replay note is in the commit. See [[phase5-review-context]] for the scan/cache conventions and [[phase2-review-context]] for the stride rule.
