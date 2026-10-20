#!/usr/bin/env bash
# PostToolUse hook (Edit|Write|MultiEdit): ruff format + ruff check --fix on the edited .py file.
# No-op until pyproject.toml configures ruff (Phase 0). Remaining violations are returned to
# Claude as additionalContext so it fixes them instead of moving on.
set -u

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

root="${CLAUDE_PROJECT_DIR:-$PWD}"
[ -f "$root/pyproject.toml" ] || exit 0
grep -q 'ruff' "$root/pyproject.toml" || exit 0
command -v uv >/dev/null 2>&1 || exit 0
[ -f "$file" ] || exit 0

cd "$root" || exit 0
# Fix first (removes unused imports etc.), then format so the result is tidy.
remaining=$(uv run --quiet ruff check --fix --quiet "$file" 2>&1 || true)
uv run --quiet ruff format --quiet "$file" >/dev/null 2>&1 || true

if [ -n "$remaining" ]; then
  jq -n --arg f "$file" --arg o "$remaining" '{
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: ("ruff auto-fixed what it could in " + $f + ". These issues remain and must be fixed by hand:\n" + $o)
    }
  }'
fi
exit 0
