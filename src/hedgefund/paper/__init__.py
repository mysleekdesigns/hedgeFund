"""Paper-only sizing helpers (plan.md Phase 7, opt-in).

Everything here computes and prints *proposed* orders from a signal file. Nothing in
this package submits, replaces or cancels an order. Placing an order is a separate,
explicit human action outside hedgeFund (``.claude/rules/trading-safety.md``).
"""
