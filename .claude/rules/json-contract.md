---
paths:
  - "src/hedgefund/report/**"
  - "src/hedgefund/cli.py"
  - "skills/**"
---

# JSON contract

- Output is a superset of the companion skill's contract. Existing field names, types, and the `["Bear","Sideways","Bull"]` index order never change. New fields are additive.
- `--json` prints exactly one JSON object to stdout and nothing else. Errors print `{"error": "..."}` and exit non-zero.
- Phase 2 additions go under their own keys: `stride_matrix`, `null_percentile`, `confidence_intervals`, `rolling_matrix`, `mixing_time`. Phase 3 additions go under `backtest` (net metrics, costs, benchmarks, permutation p-value, PSR/DSR, holdout).
- Every output includes `framework` and `disclaimer` as in the companion.
- Field reference: `.claude/skills/markov-regime-method/reference/json-contract.md`. Update it in the same change when the contract grows.
