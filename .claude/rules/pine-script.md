---
paths:
  - "pine/**"
---

# Pine Script rules

- Pine v5 overlay. Labels are drawn on `bar_index` (offset 0) inside a `barstate.isconfirmed` guard. Never back-date a label by default; the label text may say "flip was at bar −N".
- A "show hindsight labels" toggle exists, off by default, draws hollow labels tagged "hindsight".
- Transition counts and tables update only on confirmed bars.
- The `strategy()` variant enters on the next bar's open after a confirmed flip with `process_orders_on_close=false`.
- Before marking any Pine checklist item done, note in the commit that Bar Replay on BTCUSDT daily and SPY daily showed no label on an already-passed bar. This is a manual check; say so.
