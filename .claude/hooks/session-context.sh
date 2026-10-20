#!/usr/bin/env bash
# SessionStart hook: print branch, working-tree state, and plan.md phase progress.
# Plain-text stdout from a SessionStart hook is added to Claude's context. Keep it fast and offline.
set -u

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" || exit 0

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "n/a")
dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

echo "hedgeFund session context (from .claude/hooks/session-context.sh)"
echo "Branch: $branch. Uncommitted paths: $dirty."

if [ -f pyproject.toml ]; then
  echo "Scaffold present. Checks: 'uv run pytest' and 'uv run ruff check .'."
else
  echo "No pyproject.toml yet, so Phase 0 (scaffold) is the current phase."
fi

if [ -f plan.md ]; then
  echo "plan.md phase progress (done/total checkboxes):"
  awk '
    /^### Phase/ {
      if (name != "") printf "  %s: %d/%d\n", name, done, total
      name = $0; sub(/^### /, "", name); done = 0; total = 0; next
    }
    /^- \[x\]/ { done++; total++ }
    /^- \[ \]/ { total++ }
    END { if (name != "") printf "  %s: %d/%d\n", name, done, total }
  ' plan.md
  echo "Run /phase-status for the next unchecked items and the current gate."
fi
exit 0
