---
paths:
  - "tests/**"
---

# Testing conventions

- Tests run offline. No network, no yfinance, no Alpaca. Use `tests/fixtures/` parquet or synthetic series generated in the test.
- Synthetic chains come from a known 3×3 matrix with an explicit seed; assert the estimator converges within a stated tolerance and say why that tolerance.
- Use `hypothesis` for invariants: rows sum to 1, stationary vector is a left eigenvector, Pⁿ converges, costs reduce returns monotonically.
- The reproduction test compares against the companion script's output to 6 decimals. Do not loosen the tolerance to make it pass; find the discrepancy.
- Mark permutation and bootstrap tests that need N ≥ 1000 with `@pytest.mark.slow` and run a small N by default.
- One behaviour per test, named `test_<behaviour>`. Prefer plain asserts over mocks; nothing in `regime/` needs mocking because it is pure.
- Fixtures under `tests/fixtures/` are frozen once created. Add new ones; never overwrite.
