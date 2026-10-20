---
name: commit-phase
description: Stage and commit the current work with a Conventional Commit scoped to the plan.md phase, after the checks pass and plan.md checkboxes are updated. Manual only.
disable-model-invocation: true
argument-hint: [phase-number] [short summary]
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) Bash(git diff *) Bash(uv run pytest*) Bash(uv run ruff*)
---

Commit the current hedgeFund work. Arguments: $ARGUMENTS (phase number, then an optional summary).

1. Run `git status --short` and `git diff --stat`. Stop if there is nothing to commit.
2. If `pyproject.toml` exists, run `uv run ruff check .` and `uv run pytest -q -p no:cacheprovider`. If either fails, stop and report the output; do not commit red.
3. Compare the diff with the phase checklist in `plan.md`. Every finished item must be `[x]`; every ticked item must be in the diff or already committed. Fix `plan.md` first if not.
4. Stage only files that belong to this change. Never stage `.env*`, `cache/`, or `.claude/settings.local.json`.
5. Commit with a Conventional Commit message. Type is `feat`, `fix`, `test`, `docs`, `chore`, or `refactor`; scope is `phase-<n>`. The body lists the checklist items completed and the gate command that passed, for example:

```
feat(phase-2): add stride transition counting and run-length persistence

- Count transitions window bars apart (Fix 1)
- Run-length persistence 1 - 1/mean episode length (Fix 1)
- Tests: iid series shows inflated overlapping diagonal, stride does not

Gate: uv run pytest -q (41 passed), uv run ruff check . (clean)
```

6. Report the commit hash and the one-line subject. Do not push; pushing is a separate decision.
