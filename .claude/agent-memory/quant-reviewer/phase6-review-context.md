---
name: phase6-review-context
description: Non-obvious facts from the Phase 6 (skill + Pine) reviews on 2026-10-20 — Pine semantics settled from the official docs (int/int division is float, array<int> syntax), what the static Pine tests catch and miss, strategy cost model vs config, HMM reference paths verified against a real run, plugin install-string format, zsh `echo ====` trap
metadata:
  type: project
---

Phase 6 reviewed 2026-10-20 (two passes on untracked files, then a third "be the compiler" pass on the committed tree at d069a69): 41 Pine static tests pass; every field path in `skills/regime/SKILL.md` and `skills/regime/reference/json-contract.md` (102 paths) resolves in a real `analyze --json` run on the SPY CSV with the HMM ON. All documented flags exist in `cli.py`; `POSITION_RULES = (sign, conviction, persistence, long_only)`, `EMISSION_CHOICES = (gaussian, student_t, auto)`, `--refit rolling:N`.

**Pine semantics settled from the official docs (tradingview.com/pine-script-docs, pages come back gzip-compressed to curl; decompress before grepping)**
- `int / int` is NOT truncating: docs state "5/2 = 2.5"; wrap in `int()` to truncate. So `array.get(counts, i) / row_sum` on int arrays row-normalises correctly. Do not re-raise this.
- `array<int> x = ...` type-template syntax is the recommended v5 form; `int[]` is deprecated.
- Still unverifiable offline: `cond ? arrA : arrB` (array ternary, overlay line ~238) and `int != na_var` (inherited from the original overlay line 217, which demonstrably labels on TradingView, so it evaluates true in practice).
- `strategy()` header args are `const`; commission/slippage cannot be `input`s. The strategy ships `commission_value = 0.03` (%) = 3 bps/side, which matches the `us_etfs` profile (1 fee + 2 slip) but is ~7x below the `crypto` profile (10 + 10) on BTCUSDT.

**Pine facts that are invisible from the diff**
- Internal regime code is 0=Side, 1=Bull, 2=Bear. Matrix card AND stationary row now go through `disp_to_internal`; the original companion overlay (plugin cache `.../pine-script/markov-hedge-fund-method.pine` lines 318-320) still prints the Sideways share under BULL.
- Stride counting `bar_index % w == 0` with `regime[w] -> regime` is the same phase-0 subsample as Python `labels[::w]`; no off-by-one.
- "Hollow" hindsight label is rendered as `color.new(color.white, 100)` fill + tinted text (no hollow label style exists).
- Overlay ribbon (line ~203) falls through to Sideways grey when `regime` is na (first `lookback_window` bars); the strategy variant masks na. Cosmetic, inherited.

**Static Pine test coverage (mutation probe via `tests/test_pine.py` helpers)**
Caught: back-dating through a variable or `bar_index[k]`, count update moved outside the guard, `regime[lookback_window - 1]` (comments are now stripped before the substring test), `label.set_*` movers, hindsight label outside the `show_hindsight_labels` branch, `last_*_regime :=` outside the guard, `if not barstate.isconfirmed`, `... or barstate.isconfirmed`, `barstate.isconfirmed or ...`. Missed: `calc_on_order_fills` is not pinned by `test_pine_strategy.py` (only `calc_on_every_tick`); a `strategy.*` call in the `else` of the guard is not rejected by the strategy walker.

**Plugin mechanics**
- Install string is `<plugin>@<marketplace>` (companion is `markov-hedge-fund-method@markov-hedge-fund-method` in `~/.claude/plugins/installed_plugins.json`); cache path `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>`. `source: "."` copies the whole checkout into the cache.

**Review mechanics**
- Full `analyze --json` with HMM on the SPY CSV (`--null-paths 20 --bootstrap-resamples 20`) is ~1 min and is the only way to exercise the `hmm.*` reference rows.
- zsh: `echo ====` fails with "=== not found" (`=cmd` expansion). Quote it or use `echo "----"`.
- Process substitution `diff <(...) ...` is blocked in the sandbox ("/dev/fd/11: Operation not permitted"); write temp files to `$TMPDIR` instead.

**Why:** these decide whether the Pine overlay's numbers can be trusted and whether the skill docs are honest; none of it shows in the diff.
**How to apply:** the six Pine boxes in plan.md stay unticked until the owner's manual TradingView compile + Bar Replay; when that lands, check the commit names both tickers. See [[phase5-review-context]] for the scan/cache conventions and [[phase2-review-context]] for the stride rule.
