#!/usr/bin/env bash
# PreToolUse hook for Alpaca MCP tools that place, change, or cancel orders or alter the account.
# Always denies. Order placement is a separate, explicit human action (plan.md Non-goals, Phase 7).
set -u

input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // "unknown"')

jq -n --arg t "$tool" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("\($t) is blocked by .claude/hooks/block-order-tools.sh. hedgeFund never submits, replaces, cancels, or exercises orders and never changes account configuration. Read-only Alpaca tools (bars, quotes, positions, account info, watchlists) remain available. If the owner wants to trade, they do it themselves outside this session.")
  }
}'
exit 0
