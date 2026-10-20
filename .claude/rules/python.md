---
paths:
  - "**/*.py"
---

# Python conventions

- Python 3.12, managed by `uv`. Type hints on every public function. `from __future__ import annotations` is not needed.
- `src/hedgefund/regime/` is pure: numpy or pandas in, numpy or pandas out, no I/O, no globals, no logging. Orchestration (data loading, caching, printing) lives above it.
- numpy for maths. pandas only at the boundaries where a `DatetimeIndex` matters. Vectorise; the only intentional Python loop over a price series is the incremental walk-forward refit.
- Optional dependencies (`hmmlearn`, `alpaca-py`, `streamlit`) are imported lazily inside the function that needs them and degrade to `{"available": False, "reason": "..."}`.
- Parameters have defaults that match `config/default.yaml` and a one-line comment justifying the value. No unexplained constants.
- Random seeds are explicit function parameters, never module-level state.
- Fix ruff findings; do not add `# noqa` or widen the ignore list to make them go away.
- Raise `ValueError` with a message naming the bad input and the expected shape; do not silently coerce.
