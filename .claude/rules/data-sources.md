---
paths:
  - "src/hedgefund/data/**"
  - "config/**"
---

# Data sources and config

- Every source implements the `PriceSource` protocol: `get_daily(symbol, start, end) -> DataFrame` with a tz-naive, sorted, unique `DatetimeIndex` and a float `close` column (plus `open`, `high`, `low`, `volume` when available).
- Parquet cache lives under `./cache`, keyed by source, symbol, start, end. A warm cache makes zero network calls; the Phase 5 gate tests this.
- yfinance is rate-limited and flaky: retry with backoff, prefer the cache, and never call it from tests.
- Alpaca keys come from `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in the environment, paper endpoint only. Never log, print, or persist them.
- Config over flags: `config/default.yaml` holds defaults (window 20, threshold ±5%, years 10, min_train 252, costs per universe). CLI flags override; nothing is hard-coded twice.
- Universe files in `config/universes/` list symbols and their cost profile; the scan reads them, code never embeds tickers.
