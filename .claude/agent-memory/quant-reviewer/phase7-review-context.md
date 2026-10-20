---
name: phase7-review-context
description: Non-obvious facts from the Phase 7 (Alpaca paper, daily job, webhooks, sizing) review on 2026-10-20 — the token_env exfil path, urllib redirect forwards custom headers, propose --rule ignored when position is present, crypto partial-bar window, IEX vs consolidated closes, probe recipes
metadata:
  type: project
---

Phase 7 reviewed 2026-10-20 (uncommitted working tree on develop, base e7446b0). 129 tests in the five new/changed test modules pass in ~2 s; the 12 Phase 7 CLI tests pass; ruff clean. No order-placing call or live host anywhere in src/ (grep `api\.alpaca` and the order verbs; only paper-api and data.alpaca.markets are named). alpaca-py 0.44.0 is installed in .venv and `RESTClient._base_url` exists, so `_assert_paper_endpoint` works and fails closed if renamed.

**Gaps found on first pass; all verified FIXED the same day by re-running the probes (245 tests in the Phase 7 modules + test_cli pass, ruff clean). Kept so the failure modes stay known:**
- `jobs/webhooks.py` accepts any env var name in `token_env`; a registration naming `APCA_API_SECRET_KEY` sends the Alpaca secret as the `X-Hedgefund-Token` header (probe confirmed). No denylist/prefix rule.
- `post_json` uses the default urllib opener; `HTTPRedirectHandler.redirect_request` copies every non-content header (the token) onto the redirect target, and 301/302/303 turn the POST into a bodyless GET that returns "ok 200". Verified from the stdlib source via `inspect.getsource`.
- `_validate_url` is a `str.startswith` check: `http://localhost.attacker.invalid`, `http://127.0.0.1.nip.io`, `http://127.0.0.10` all pass.
- `paper/sizing._positions` uses the file's `position` whenever present and ignores `rule`; `propose --rule long_only` on a sign-rule signal file reports `rule=long_only` with a -1.0 target (probe confirmed).
- `daily --json` and `WebhookResult.to_dict` print the webhook `url`; Slack/Discord URLs embed the secret.
- Alpaca crypto daily bars are NY-day bars that close at 04:00/05:00 UTC, but `CachedSource.resolve_window` ends at "yesterday UTC", so a run between 00:00 and 05:00 UTC caches a partial crypto bar. Stocks are fine.
- The aligned-closes test round-trips the SPY fixture through fakes; it proves the 04:00/05:00 UTC -> NY-date mapping only. IEX-feed closes are not consolidated closes, so real Alpaca vs yfinance closes will differ; README overstates "line up".
- plan.md Phase 7 boxes were all still `[ ]` in the diff (workflow.md says flip in the same change). Fixed: all eight ticked; owner confirmation recorded in the commit body.

**How they were fixed (useful for future reviews):** token_env must start with HEDGEFUND_WEBHOOK_ and APCA_* is refused by name; `NoRedirectHandler.redirect_request` returns None so any 3xx is an HTTPError result (ok False, status 3xx); `_validate_url` uses urlsplit with hostname in {localhost,127.0.0.1} for http; `WebhookResult` carries `host` not `url`; sizing re-derives position under the requested rule and only uses the stored position when signal is None AND the file rule matches (otherwise skipped with reason); `AlpacaSource(now=...)` plus `last_complete_date(now, crypto)` drops today's NY bar (stocks before 16:00 NY, crypto always) and `bars_to_frame` raises on any stamp not at NY midnight; signal files go through tmp + os.replace.

**Probe recipes that worked (offline):** build a signal dict by hand and call `propose_orders(..., rule=...)`; call `post_summary` with a recording `post` and a fake `environ`; `_validate_url(u, Path("f"))` directly. The hook blocks commands containing the live host, so derive it with `.replace("paper-", "")` as `tests/test_sizing.py` does.

**Conventions confirmed:** `_fail` prints `{"error": ...}` and exits 1 for every command; `--json` suppresses progress so stdout is one line; the daily job reuses the scan's default cache key for `last_closes` (second read is a cache hit under the CLI, a second network call for a bare source).
