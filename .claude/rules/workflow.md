# Workflow

- `plan.md` is the source of truth for scope, phases, checklists, and acceptance gates. Phases 0, 1, 2, 3 run in order. Phases 4 and 5 are independent. Phases 6, 7, 8 follow by priority.
- Do not start a phase until the previous phase's gate passes. State which gate you ran and show its output, not a claim that it passed.
- When a checklist item is finished, flip its `[ ]` to `[x]` in `plan.md` in the same change. Never flip a box for work that is not verified.
- Tooling is `uv` only: `uv run pytest`, `uv run ruff check .`, `uv add <pkg>`. Never `pip install` into the system Python (it is 3.9; the project pins 3.12).
- Work on branch `develop`. Conventional Commits with the phase as scope, for example `feat(phase-2): add stride transition counting`.
- Before claiming a phase complete run `/run-gate`. Commit with `/commit-phase`.
- Scope investigations with a subagent (`quant-researcher`, Explore) rather than reading many files into the main conversation.
