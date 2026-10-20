#!/usr/bin/env bash
# PreToolUse hook (Bash): deny live-trading endpoints, secret leaks, and destructive commands;
# ask before history-rewriting git. Decisions are returned as JSON on stdout with exit 0.
set -u

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
[ -z "$cmd" ] && exit 0

deny() {
  jq -n --arg r "$1" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
  exit 0
}
ask() {
  jq -n --arg r "$1" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"ask",permissionDecisionReason:$r}}'
  exit 0
}

# --- deny: live Alpaca trading endpoint (paper-api.alpaca.markets is the only allowed host)
if printf '%s' "$cmd" | grep -Eq 'https?://api\.alpaca\.markets'; then
  deny "Live Alpaca endpoint. This repo is paper-only (paper-api.alpaca.markets) and never places orders automatically. See plan.md Non-goals."
fi
if printf '%s' "$cmd" | grep -Eq 'APCA_API_BASE_URL=["'"'"']?https?://api\.alpaca'; then
  deny "APCA_API_BASE_URL points at the live endpoint. Paper only."
fi

# --- deny: writing secrets into tracked files
if printf '%s' "$cmd" | grep -Eq '(>|>>|tee)[[:space:]]*["'"'"']?[^[:space:]]*\.env(\.[A-Za-z0-9_-]+)?["'"'"']?([[:space:]]|$)'; then
  deny "Writing to a .env file. Keys stay in the shell environment; nothing in this repo persists them."
fi
if printf '%s' "$cmd" | grep -Eq 'APCA_API_(KEY_ID|SECRET_KEY)=[A-Za-z0-9]{8,}'; then
  deny "Command embeds an Alpaca API key literal. Read keys from the environment instead."
fi

# --- deny: catastrophic rm
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])rm[[:space:]]+(-[A-Za-z]*[rR][A-Za-z]*[[:space:]]+)?(-[A-Za-z]*f[A-Za-z]*[[:space:]]+)?(-[A-Za-z]*[rR][A-Za-z]*[[:space:]]+)?["'"'"']?(/|~|\$HOME|\.git|\.|\*|\.\.)["'"'"']?([[:space:]]|$|/\*)'; then
  deny "Recursive delete of a root, home, repo root, or .git. Delete specific paths instead."
fi
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])rm[[:space:]]+-[A-Za-z]*r[A-Za-z]*[[:space:]]+.*tests/fixtures'; then
  deny "tests/fixtures holds the frozen reproduction baseline. Do not delete it."
fi

# --- ask: history rewriting or discarding work
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[[:space:]].*(--force|-f([[:space:]]|$)|--force-with-lease)'; then
  ask "Force push rewrites remote history. Confirm this is intended."
fi
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+(reset[[:space:]]+--hard|clean[[:space:]]+-[A-Za-z]*[fd]|checkout[[:space:]]+--[[:space:]]+\.|restore[[:space:]]+\.|stash[[:space:]]+drop|branch[[:space:]]+-D)'; then
  ask "This git command discards uncommitted work or branches. Confirm."
fi

exit 0
