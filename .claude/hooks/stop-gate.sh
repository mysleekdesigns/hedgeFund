#!/usr/bin/env bash
# Stop hook: when Python files changed since the last commit, run ruff + pytest before the turn ends.
# Blocks the stop (decision: block) with the failure output so Claude fixes it. Honors
# stop_hook_active to avoid loops; Claude Code also caps consecutive blocks at 8.
set -u

input=$(cat)
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
[ "$active" = "true" ] && exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$root" || exit 0
[ -f pyproject.toml ] || exit 0
command -v uv >/dev/null 2>&1 || exit 0

changed=$(git status --porcelain -- '*.py' pyproject.toml 2>/dev/null)
[ -z "$changed" ] && exit 0

failures=""

if grep -q 'ruff' pyproject.toml; then
  lint=$(uv run --quiet ruff check . 2>&1)
  if [ $? -ne 0 ]; then
    failures="${failures}ruff check . failed:\n$(printf '%s' "$lint" | tail -n 30)\n\n"
  fi
fi

if [ -d tests ]; then
  tests_out=$(uv run --quiet pytest -q -x -p no:cacheprovider 2>&1)
  rc=$?
  # rc 5 = no tests collected; that is fine on an empty scaffold.
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then
    failures="${failures}uv run pytest -q -x failed (exit $rc):\n$(printf '%s' "$tests_out" | tail -n 40)\n"
  fi
fi

if [ -n "$failures" ]; then
  reason=$(printf "Python files changed this session and the gate is red. Fix before finishing (do not suppress or skip tests):\n\n${failures}")
  jq -n --arg r "$reason" '{decision: "block", reason: $r}'
fi
exit 0
