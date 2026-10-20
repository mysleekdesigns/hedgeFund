# .claude layout

Project-scoped Claude Code configuration for hedgeFund, following Anthropic's published guidance for rules, skills, subagents, and hooks.

| Piece | Where | Loads | Purpose |
|---|---|---|---|
| Rules | `rules/*.md` | at launch, or when a matching file is read or edited (`paths` frontmatter) | short, enforceable conventions per area |
| Skills | `skills/<name>/SKILL.md` | description always, body on invoke | workflows (`/phase-status`, `/run-gate`, `/commit-phase`) and background knowledge (`markov-regime-method`, `honest-backtesting`) |
| Subagents | `agents/*.md` | on delegation | `quant-reviewer`, `gate-runner`, `test-debugger`, `quant-researcher`, each with a restricted tool set |
| Hooks | `settings.json` + `hooks/*.sh` | deterministically on events | block secrets, live endpoints, destructive git, Alpaca order tools; ruff after edits; test gate before a turn ends; phase context at session start |

Rules of thumb used here:

- Advisory guidance goes in rules and skills. Anything that must hold every time is a hook.
- Rules are path-scoped so they cost no context until relevant. `trading-safety.md` and `workflow.md` load unconditionally.
- Skills with side effects (`commit-phase`) set `disable-model-invocation: true`; knowledge skills set `user-invocable: false`.
- Reviewer and gate subagents are read-only (`tools: Read, Grep, Glob, Bash`), so the agent grading the work cannot change it.
- Hooks use exec form (`command: bash`, `args: [...]`) with `${CLAUDE_PROJECT_DIR}` so they run from any working directory without needing the executable bit.

Validate after editing: `claude plugin validate .claude/agents` and `claude plugin validate .claude/skills`.
