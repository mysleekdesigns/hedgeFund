# hedgeFund

Python research platform for Markov regime detection (Bull / Bear / Sideways) with honest, no-lookahead backtests. `plan.md` is the source of truth: phases, checklists, acceptance gates. Read it before starting any phase work.

## Commands

- `uv run pytest` runs the test suite (offline, no network)
- `uv run ruff check .` and `uv run ruff format .`
- `uv run hedgefund analyze --ticker SPY --json` once Phase 1 lands

## Hard constraints

- Nothing in this repo places orders. Alpaca is paper-only and opt-in; Phase 7 starts only after the owner confirms it.
- No lookahead anywhere. Every estimator ships with the perturbation test (mutate prices after day t, outputs up to t are bit-identical).
- Keep the companion skill's JSON contract. Only add fields, never rename or remove.
- Flip `plan.md` checkboxes as items complete. Do not start a phase before the previous gate passes.

## Where things live

- `.claude/rules/` holds path-scoped conventions (Python, testing, data sources, no-lookahead, Pine, JSON contract).
- Skills: `/next-phase` (work the next plan phase with parallel subagents, then commit, push, merge everywhere), `/phase-status`, `/run-gate`, `/commit-phase`, plus background knowledge in `markov-regime-method` and `honest-backtesting`.
- Subagents: `phase-implementer` (parallel workers), `quant-reviewer` (read-only adversarial review), `gate-runner`, `test-debugger`, `quant-researcher`.
- Hooks in `.claude/settings.json` block secrets, destructive git, live Alpaca endpoints, and order-placing MCP tools; they format Python after edits and run the test gate before a turn ends.
