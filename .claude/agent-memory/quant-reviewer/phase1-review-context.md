---
name: phase1-review-context
description: Non-obvious facts confirmed during the Phase 1 (faithful port) review on 2026-10-19 — how to run a fresh companion comparison offline, the known ulp-level diffs, and companion conventions the port inherits (zero rows, first-day drawdown, full-sample HMM)
metadata:
  type: project
---

Phase 1 reviewed 2026-10-19: 243 tests pass, ruff clean, port reproduces the companion on the SPY fixture to 8.9e-16 and the pretty terminal text byte-for-byte. No criticals.

**How to do the fresh-companion comparison without network**
- Run the plugin with the project venv: `uv run python ~/.claude/plugins/marketplaces/markov-hedge-fund-method/scripts/markov_regime.py --csv <export> --no-hmm --json`. Its yfinance/hmmlearn imports are lazy, so the project's numpy/pandas/hmmlearn suffice.
- Export the fixture with `close.rename("close").rename_axis("date").to_csv(path)`; pandas writes shortest-round-trip floats, so `load_csv` (which uses `float_precision="round_trip"`) reads them back bit-identical to the parquet.
- The companion reads the same CSV with pandas' default parser, which lands 252 of 2511 closes one ulp off. That is the whole source of the ~1e-15 Sharpe / drawdown gap; the port is arithmetically identical (verified with a verbatim oracle in tests/test_backtest.py).

**Companion conventions the port deliberately inherits (do not flag as bugs in later phases, but Phase 2/3 must handle them)**
- A state never seen as a source gets an all-zero matrix row (not uniform). `validate_transition_matrix` accepts it; `signal_from_matrix` returns 0 (flat) for it. The JSON contract text says "rows sum to 1", which is false for that row.
- `max_drawdown` starts the equity curve at the first period's equity, so a first-day loss is not a drawdown.
- Sharpe is gross, ddof=1, sqrt(252); no costs, turnover, buy-and-hold or null in Phase 1 (that is Phase 3 scope).
- `hmm_summary` is a full-sample Baum-Welch fit (diagnostic only, not in the perturbation registry). Repeated fits differ in the last digits with multithreaded BLAS; tests compare at rtol 1e-9.

**Port-only deviations (stricter than companion)**
- `label_regimes` rejects threshold <= 0 and non-DatetimeIndex; the companion accepts `--threshold 0`.
- `params.return_type` is an additive contract field (documented in the skill's json-contract.md).
- `--config` replaces the default YAML wholesale; a partial file fails with "config is missing ...".

**Why:** these took the most time to establish and are invisible from a later diff.
**How to apply:** Phase 2 review should check that stride/null/bootstrap estimators handle the all-zero row and that any new JSON field emits null, not NaN. See [[phase0-review-context]] for the uv cache prefix and fixture quirks.
