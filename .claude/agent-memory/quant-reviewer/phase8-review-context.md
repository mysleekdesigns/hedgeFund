---
name: phase8-review-context
description: Non-obvious facts from the Phase 8 (Streamlit dashboard) review on 2026-10-20 — streamlit headless startup hits checkip.amazonaws.com, email prompt on first non-headless run, cost model read from live config not the manifest, app null band uses seed 0 not the run's seed, importorskip hides the gate test, probe recipe for recompute-vs-JSON
metadata:
  type: project
---

Phase 8 reviewed 2026-10-20 (uncommitted working tree on develop, base ddb6f68). 44 dashboard tests pass in 3 s with 0 skips (streamlit 1.66.0 and altair 6.3.0 are in .venv), ruff clean. Data-layer numbers are sound: forcing the recompute branch with the run's own params reproduces the JSON matrix, both CI matrices, stationary vector and its CIs, mixing time and signal bit-for-bit; curve ends equal 1 + JSON total_return exactly; truncating closes leaves labels and curves identical on the prefix (expanding and rolling).

**Gaps found (status at review time; re-check before trusting):**
- `hedgefund dashboard --headless` makes an HTTP GET to `checkip.amazonaws.com` at startup: streamlit `web/bootstrap._print_url` calls `net_util.get_external_ip()` when `server.headless` is true and neither `server.address` nor `browser.serverAddress` is manually set; every launch also does a UDP `connect(("8.8.8.8", 1))` for the internal IP. Fix is `--server.address localhost` in the argv (single-URL branch skips both). Verified from the installed source, not run.
- Non-headless first run with no `~/.streamlit/credentials.toml` shows the email prompt on stdin (`server.showEmailPrompt` default true); pass `--server.showEmailPrompt false`.
- `compute_symbol_view` builds the CostModel from the live `config/default.yaml`, not `manifest.params["cost_model"]`; editing a profile after the scan changes the curves while the "From the scan's result JSON" badge still shows (probe: fee 1->50 bps moved strategy_net end 0.889 -> 0.770).
- `dashboard/app.py` passes `seed=NULL_SEED=0` and a 200-path default, so the band is NOT the JSON's null when the scan used another seed or path count; the data.py docstring says it is. Config default seed is 0, so it coincides by accident for default scans.
- `tests/test_dashboard_app.py` starts with `pytest.importorskip("streamlit")`; streamlit is only in the optional `dashboard` extra (not the dev group), so a fresh `uv sync && uv run pytest` skips all 16 app tests including the no-socket gate test. CLI tests stub `find_spec`.
- plan.md Phase 8 boxes were all still `[ ]` in the diff (same as Phase 7; workflow.md says flip in the same change).

**Semantics confirmed:** `WalkForwardResult.asset_returns` and `strategy_returns` share the earned-date index, `positions` the decision-date index, so `_with_start(lower, positions.index[0], asset_returns.index)` aligns with `equity_curves`. Manifest key for the walk-forward window is `backtest_rolling_bars` (statistics `rolling_bars` is the rolling-matrix window). `analyze` labels `close.dropna()` and takes `current_state = labels.iloc[-1]`, same as the dashboard. The conftest `no_network` guard patches `socket.socket.connect`, so `REFUSED_ATTEMPTS == []` after an in-process `AppTest.run()` is a real proof for the Python side; browser telemetry is a separate matter (`--browser.gatherUsageStats false` is passed by the CLI but not by the README's direct `streamlit run`). The random-sign band is not prefix-stable as history grows (draw shape `(n_paths, T)` row-major) — a property of the null, not lookahead.

**Probe recipe (offline, ~5 s):** import `tests/conftest.py` as a module, call `_write_dashboard_csv` for AAA/BBB, run `scan --source csv ... --no-hmm --null-paths 20 --bootstrap-resamples 20 --seed 7 --json` through `CliRunner`, `data.load_run`, then monkeypatch `data.controls_match_run = lambda r, c: False` to force the recompute branch and compare to the matching view. Config drift: `dataclasses.replace(run, config=cfg2)`.

**Why:** the Phase 8 gate is literally "no network calls", and the launcher is the only place that violates it.
**How to apply:** when the fixes land, re-run the probe, check `--server.address` is in `dashboard_command_line`, and that the app seed defaults to `manifest.params["seed"]`. See [[phase5-review-context]] for the scan/cache conventions the dashboard reads and [[phase3-review-context]] for the null conventions.
