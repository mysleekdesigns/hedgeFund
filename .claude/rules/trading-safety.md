# Trading safety

- This repo is a research and signal tool. Nothing in it places, replaces, cancels, or exercises orders. Order placement is a separate, explicit human action outside Claude Code.
- Alpaca: paper endpoint only (`paper-api.alpaca.markets`). API keys are read from environment variables at call time and never written to config, cache, fixtures, logs, or JSON output.
- Phase 7 (Alpaca paper integration) starts only after the owner confirms it in the conversation. Until then, do not add `alpaca_source.py` or any order-sizing helper.
- Order-side Alpaca MCP tools are denied by a hook. Do not route around it with `curl`, the SDK, or a script.
- Every report and JSON output carries the disclaimer: backtests are historical, not forward-looking.
