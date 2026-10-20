---
name: run-gate
description: Run the acceptance gate for a plan.md phase (ruff, pytest, and the phase-specific checks) in an isolated subagent and report pass or fail with evidence. Use before claiming a phase complete, before committing, or when asked whether the gate passes.
argument-hint: [phase-number]
context: fork
agent: gate-runner
background: false
---

Run the acceptance gate for hedgeFund phase: $ARGUMENTS

If no phase number was given, read `plan.md` and use the first phase that still has unchecked `- [ ]` items.

Steps:
1. Read the phase's checklist and its `**Gate:**` line in `plan.md`.
2. From the project root run `uv run ruff check .` and `uv run pytest -q -p no:cacheprovider`. If `pyproject.toml` does not exist, report "Phase 0 not started" and stop.
3. Run the phase-specific gate described in `plan.md`, using only the cache and `tests/fixtures/`; never call live network services.
4. Cross-check every `[x]` item in that phase against the code and tests. List any box ticked without evidence.

Return, in this order: `Gate: PASS` or `Gate: FAIL`; each command with its relevant output; unevidenced checkboxes; the single most useful next action. Do not edit code, tests, or `plan.md`.
