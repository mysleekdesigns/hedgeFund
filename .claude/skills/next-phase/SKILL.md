---
name: next-phase
description: Work the next unfinished phase of plan.md end to end. Splits the checklist into independent packages, runs phase-implementer subagents in parallel, reviews and gates the result, ticks plan.md, commits, pushes, and merges to every branch. Manual only.
disable-model-invocation: true
argument-hint: [phase-number]
allowed-tools: Agent Read Edit Write Grep Glob Bash(uv run pytest*) Bash(uv run ruff*) Bash(uv add *) Bash(uv sync*) Bash(uv init*) Bash(git add *) Bash(git commit *) Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git branch *) Bash(git push *) Bash(${CLAUDE_SKILL_DIR}/scripts/merge-all.sh *)
---

# Next phase

Work phase `$ARGUMENTS` of `plan.md`. If no number was given, use the first phase with unchecked items. Run subagents in parallel wherever packages are independent: send every Agent call for a stage in one message.

Phase progress right now:

!`awk '/^### Phase/ { if (name != "") printf "  %s: %d/%d\n", name, done, total; name = $0; sub(/^### /, "", name); done = 0; total = 0; next } /^- \[x\]/ { done++; total++ } /^- \[ \]/ { total++ } END { if (name != "") printf "  %s: %d/%d\n", name, done, total }' plan.md`

Working tree: !`git status --short | wc -l | tr -d ' '` dirty paths on branch !`git branch --show-current`.

Copy this checklist into your first message and keep it updated:

```
Next phase progress:
- [ ] 1. Preconditions: clean tree, previous gate green, phase scope read
- [ ] 2. Work packages defined (disjoint file ownership, interfaces fixed)
- [ ] 3. Implementers launched in parallel and all returned
- [ ] 4. Integration: full ruff + pytest green (test-debugger if not)
- [ ] 5. Review (quant-reviewer) and gate (gate-runner) run in parallel, findings fixed
- [ ] 6. plan.md checkboxes ticked, only for verified items
- [ ] 7. Commit, push, merge to all branches, push
```

## 1. Preconditions

- The working tree must be clean. If it is not, stop and tell the user what is uncommitted.
- Phase 7 needs the owner's explicit go-ahead (plan.md section 4). If the target is Phase 7 and the user did not say so in this conversation, ask with AskUserQuestion before doing anything else.
- Unless the target is Phase 0, confirm the previous phase's gate is green: launch `gate-runner` for the previous phase. If it fails, fix that first or stop and report; do not start the new phase on a red gate.
- Read in full: the phase's checklist and **Gate** line, `plan.md` section 3 (architecture) and section 5 (defaults), and every "Fix N" subsection the checklist cites.

## 2. Work packages

Partition the unchecked items into packages such that no two packages create or edit the same file. Group by module ownership from section 3. Typical shapes:

- Phase 0: (a) `pyproject.toml` + ruff/pytest config + package layout, (b) `config/default.yaml` + universes, (c) `README.md`, (d) `tests/fixtures/` baseline. Package (a) must finish before the others can run tests, so launch it first alone, then the rest in parallel.
- Phase 1: (a) `regime/labels.py` + `regime/markov.py` + their tests, (b) `regime/signal.py` + `backtest/walk_forward.py` + `backtest/metrics.py` + tests, (c) `data/` sources + `PriceSource`, (d) `report/` + `cli.py`, (e) `regime/hmm.py`. Fix the function signatures in the prompts so (b) and (d) can code against (a) before it lands.
- Phases 2 to 5: one package per module file plus its test file. Contract changes in `report/json_contract.py` belong to one package only.

Before launching, write the shared interfaces (exact signatures, return shapes, state order `["Bear","Sideways","Bull"]`) into every prompt that depends on them. If a formula or API detail is uncertain, launch `quant-researcher` for it in the same message as the implementers and pass its answer on when it returns.

Items that need a manual check (Pine Bar Replay, TradingView compile) are never assigned to an implementer and never ticked; list them for the user at the end.

## 3. Launch implementers

Use `subagent_type: phase-implementer`. One Agent call per package, all in one message. Each prompt contains: phase number, the checklist items quoted verbatim, the exact files the package owns, the interfaces it must expose or consume, and the test command for its own tests. Do not re-run their work yourself while they run; wait for every result.

When a package reports an unfinished item or a dependency on another package, either re-dispatch a follow-up to the same agent with the missing piece or fold it into step 4.

## 4. Integration

Run from the project root:

```
uv run ruff check . && uv run pytest -q -p no:cacheprovider
```

If anything is red, launch `test-debugger` (several in parallel if failures are in unrelated files, each told which files it may touch). Repeat until green. Never fix a failure by skipping, xfailing, or loosening a tolerance.

## 5. Review and gate, in parallel

In one message launch both:
- `quant-reviewer` with the phase number and `git diff --stat` output, asking for correctness and plan gaps only.
- `gate-runner` for this phase.

Fix every Critical and Warning finding and anything the gate reports, re-running step 4 after fixes. Re-run `gate-runner` until it reports `Gate: PASS`. If three rounds do not get the gate green, stop and report what is still failing instead of forcing it.

## 6. Tick plan.md

Flip `[ ]` to `[x]` only for items that have code and a passing test, or that the gate verified. Leave manual-check items unticked. Do not touch other phases.

## 7. Commit, push, merge everywhere

1. `git add -A` then review `git status --short`; nothing under `cache/`, no `.env*`, no stray scratch files.
2. Commit with a Conventional Commit, scope `phase-<n>`, body listing the completed items and the gate command with its result. End the message with the attribution trailer the session requires, if any.
3. Run the merge fan-out:

```
${CLAUDE_SKILL_DIR}/scripts/merge-all.sh
```

It pushes the current branch, then fast-forwards or merges every other local and remote branch to it and pushes each one. It refuses to run on a dirty tree and exits 2 on a conflict without pushing the conflicted branch; resolve that by hand and run it again.

## Final report

Phase and items completed; items left and why (manual checks listed separately); the gate command and its last lines; reviewer findings fixed; commit hash; branches updated with their new heads; the next phase and its first three items.
