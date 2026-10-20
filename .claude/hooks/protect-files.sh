#!/usr/bin/env bash
# PreToolUse hook (Edit|Write|MultiEdit): block edits to files Claude must not touch.
# Exit 2 blocks the tool call and shows stderr to Claude as the reason.
set -u

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$file" ] && exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
case "$file" in
  "$root"/*) rel="${file#"$root"/}" ;;
  *)         rel="$file" ;;
esac

block() {
  echo "Blocked by .claude/hooks/protect-files.sh: $1" >&2
  exit 2
}

case "$rel" in
  .env|.env.*|*/.env|*/.env.*)
    block "'$rel' holds secrets. API keys come from environment variables only; never write them to files (plan.md Phase 7)." ;;
  .git/*)
    block "never edit .git internals directly; use git commands." ;;
  uv.lock)
    block "uv.lock is generated. Run 'uv add <pkg>' or 'uv lock' instead." ;;
  cache/.gitkeep) ;;
  cache/*)
    block "cache/ holds generated parquet. Regenerate it through the data layer, do not hand-edit." ;;
  tests/fixtures/*.parquet)
    if [ -f "$root/$rel" ]; then
      block "'$rel' is a frozen reproduction baseline (Phase 1 gate). Creating a new fixture is fine; overwriting an existing one is not. Ask the owner if the baseline must change."
    fi ;;
esac

exit 0
