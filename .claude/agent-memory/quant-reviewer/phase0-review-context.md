---
name: phase0-review-context
description: Non-obvious facts confirmed during the Phase 0 scaffold review (2026-10-19) that later reviews of Phase 1+ depend on: sandbox uv cache, companion script location, fixture quirks, ruff friction points
metadata:
  type: project
---

Phase 0 scaffold reviewed 2026-10-19: gate passed (26 tests, ruff clean, lock in sync). No criticals.

**Facts future reviews need**
- Run uv as `UV_CACHE_DIR=/Users/simonlacey/.claude/tmp/uv-cache uv run ...`; the sandbox cannot write uv's default cache.
- Reference implementation lives at `~/.claude/plugins/marketplaces/markov-hedge-fund-method/scripts/markov_regime.py` (unpinned PEP 723 deps, so it resolves to the same pandas 3.x as the project).
- `tests/fixtures/spy_10y.parquet`: pulled 2026-10-20T00:18 UTC, last bar 2026-10-16 (Monday 10-19 bar absent), index `datetime64[ms]`, `columns.name == "Price"` leaked from yfinance. A fresh `--ticker` plugin run will never match it; the Phase 1 gate's "fresh run of the installed plugin on the same dates" must go through the plugin's `--csv` on an export of the fixture.
- plan.md is internally inconsistent on default costs: section 5 says 1+2 bps ETF / 10+10 crypto per side; Fix 6 text says 2 bps ETF / 15 bps crypto. Config follows section 5. Phase 3 reviewer should check which one the cost model and docs cite.
- ruff selects `N` and `B`: uppercase matrix names (`P`, `P_t`) trip N806 and `typer.Option(...)` defaults trip B008; python.md forbids `# noqa`, so Phase 1 ports must rename and use `Annotated`.
- `tests/conftest.py` has an autouse socket guard (raises `NetworkAccessError` on connect) added after this review; `tests/test_offline_guard.py` proves it.

**Why:** these are the things that bit or nearly bit the Phase 0 review and are not visible from the diff of a later phase.
**How to apply:** check the Phase 1 reproduction test's input path and tolerance against the fixture facts above before anything else.
