---
name: phase-status
description: Show progress against plan.md. Per-phase checkbox counts, the next unchecked items, and the acceptance gate for the current phase. Use when the user asks where we are, what is next, what to work on, or which phase is current.
allowed-tools: Read
---

# Phase status

Phase progress right now (done/total checkboxes):

!`awk '/^### Phase/ { if (name != "") printf "  %s: %d/%d\n", name, done, total; name = $0; sub(/^### /, "", name); done = 0; total = 0; next } /^- \[x\]/ { done++; total++ } /^- \[ \]/ { total++ } END { if (name != "") printf "  %s: %d/%d\n", name, done, total }' plan.md`

Next unchecked items, in plan order:

!`awk '/^### Phase/ { phase = $0; sub(/^### /, "", phase) } /^- \[ \]/ { n++; if (n <= 12) printf "  [%s] %s\n", phase, substr($0, 7) }' plan.md`

Working tree:

!`git status --short | head -20`

## What to report

1. **Current phase**: the first phase with unchecked items. Phases 0 to 3 are sequential; 4 and 5 are independent; 6 to 8 by priority.
2. **Gate for that phase**: quote the `**Gate:**` line from `plan.md` for it.
3. **Next three items** to pick up, with the module each lands in (plan.md section 3).
4. **Blockers**: anything in the working tree or the previous gate that must land first.

Do not edit `plan.md` from this skill. If an item looks done but is unchecked, say so and suggest `/run-gate` to verify before ticking it.
