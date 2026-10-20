"""The daily job: rescan a watchlist, write a signal file, tell the registered consumers.

``run_daily`` runs ``scan_universe`` on a universe, reads the last close of every symbol
that produced a result, and writes two identical JSON files: ``jobs/<run_id>/signal.json``
and ``jobs/signal_latest.json`` (the file ``hedgefund propose`` reads by default). When a
webhooks directory is given, the run *summary* is POSTed to every consumer registered
there (``hedgefund.jobs.webhooks``); a failing consumer is recorded, never raised.

The signal file is a signal, not an order. Nothing in hedgeFund submits orders
(``.claude/rules/trading-safety.md``); turning a position into an order is a separate,
explicit human action. The file says so in ``note`` and ``orders_placed``.

Backtests are historical, not forward-looking.
"""

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hedgefund.data.base import PriceSource, close_series
from hedgefund.jobs.manifest import DEFAULT_JOBS_DIR, JSON_INDENT, json_ready, utc_now
from hedgefund.jobs.scan import REGIME_KEYS, ScanResult, _finite_or_none, _get, scan_universe
from hedgefund.jobs.universe import Universe
from hedgefund.jobs.webhooks import WebhookResult, load_webhooks, post_json, post_summary
from hedgefund.regime.signal import POSITION_RULES, position_from_signal
from hedgefund.report.json_contract import DISCLAIMER, FRAMEWORK

SIGNAL_FILE = "signal.json"
LATEST_FILE = "signal_latest.json"
TEMP_SUFFIX = ".tmp"  # in-progress write next to the target, renamed over it when complete
DEFAULT_WEBHOOKS_DIR = DEFAULT_JOBS_DIR.parent / "webhooks"  # ./webhooks at the repo root
DEFAULT_EVENT = "daily"  # the event name consumers subscribe to for this job
ORDERS_NOTE = (
    "This file is a signal, not an order. Nothing in hedgeFund submits orders; "
    "placing an order is a separate, explicit human action."
)
# Per-symbol keys of the signal file, in the order they are written.
SYMBOL_KEYS = (
    "current_regime",
    "signal",
    "stay_probability",
    "position",
    "persistence",
    "persistence_source",
    "last_close",
    "date_end",
    "rows",
    "net_sharpe",
    "error",
)


def _validate_rule(rule: str) -> str:
    if rule not in POSITION_RULES:
        raise ValueError(f"rule must be one of {POSITION_RULES}, got {rule!r}")
    return rule


def _position(signal: float | None, stay: float | None, rule: str) -> float | None:
    """``position_from_signal`` when both inputs exist, else ``None`` (never a guess)."""
    if signal is None:
        return None
    if stay is None:
        return None
    return _finite_or_none(position_from_signal(signal, stay, rule))


def _symbol_entry(row: dict, result: dict | None, last_close: float | None, rule: str) -> dict:
    """One ``symbols`` entry from a scan table row, its analyze dict and the last close."""
    regime = row.get("current_regime")
    stay = None
    if result is not None and regime is not None:
        stay = _finite_or_none(_get(result, "persistence_diagonal", REGIME_KEYS[str(regime)]))
    signal = _finite_or_none(row.get("signal"))
    return {
        "current_regime": None if regime is None else str(regime),
        "signal": signal,
        "stay_probability": stay,
        "position": _position(signal, stay, rule),
        "persistence": _finite_or_none(row.get("persistence")),
        "persistence_source": row.get("persistence_source"),
        "last_close": _finite_or_none(last_close),
        "date_end": None if row.get("date_end") is None else str(row["date_end"]),
        "rows": None if row.get("rows") is None else int(row["rows"]),
        "net_sharpe": _finite_or_none(row.get("net_sharpe")),
        "error": None if row.get("error") is None else str(row["error"]),
    }


def build_signal(
    scan: ScanResult,
    *,
    last_close: dict[str, float],
    rule: str = "sign",  # config backtest.rule; the Phase 1 +1 / 0 / -1 rule
    generated_at: str,
    source: str | None = None,
) -> dict:
    """The signal-file dict for a finished scan.

    ``as_of`` is the latest ``date_end`` over the symbols that produced a result (``None``
    when none did). Every universe symbol gets a ``symbols`` entry, errored ones with
    ``error`` set and ``position`` ``None``. ``position`` is
    ``position_from_signal(signal, stay_probability, rule)`` where ``stay_probability`` is
    the overlapping 1-step diagonal of the current regime. ``source`` defaults to the
    manifest's data source name. Keys are sorted so the file is stable across runs.
    """
    _validate_rule(rule)
    symbols = {}
    for row in scan.table:
        symbol = str(row["symbol"])
        symbols[symbol] = _symbol_entry(row, scan.results.get(symbol), last_close.get(symbol), rule)
    date_ends = [e["date_end"] for e in symbols.values() if e["date_end"] is not None]
    if source is None:
        sources = [str(entry["source"]) for entry in scan.manifest.data.values()]
        source = sources[0] if sources else None
    return {
        "as_of": max(date_ends) if date_ends else None,
        "generated_at": generated_at,
        "run_id": scan.run_id,
        "run_dir": str(scan.run_dir),
        "universe": scan.manifest.universe,
        "cost_profile": scan.manifest.cost_profile,
        "source": source,
        "rule": rule,
        "symbols": dict(sorted(symbols.items())),
        "orders_placed": False,
        "note": ORDERS_NOTE,
        "framework": FRAMEWORK,
        "disclaimer": DISCLAIMER,
    }


def summary_payload(signal: dict) -> dict:
    """The compact run summary posted to webhooks: no paths, no environment values."""
    return {
        "event": DEFAULT_EVENT,
        "as_of": signal["as_of"],
        "generated_at": signal["generated_at"],
        "run_id": signal["run_id"],
        "universe": signal["universe"],
        "rule": signal["rule"],
        "symbols": {
            symbol: {
                "current_regime": entry["current_regime"],
                "signal": entry["signal"],
                "position": entry["position"],
                "error": entry["error"],
            }
            for symbol, entry in signal["symbols"].items()
        },
        "orders_placed": False,
        "note": ORDERS_NOTE,
        "disclaimer": DISCLAIMER,
    }


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a sibling temp file and ``os.replace`` it over ``path``.

    A consumer polling the file sees either the previous complete JSON or the new one,
    never a truncated write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=TEMP_SUFFIX, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_signal(signal: dict, jobs_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Write ``jobs_dir/<run_id>/signal.json`` and ``jobs_dir/signal_latest.json`` atomically."""
    text = json.dumps(json_ready(signal), indent=JSON_INDENT, allow_nan=False) + "\n"
    signal_path = Path(jobs_dir) / run_id / SIGNAL_FILE
    latest_path = Path(jobs_dir) / LATEST_FILE
    _write_atomic(signal_path, text)
    _write_atomic(latest_path, text)
    return signal_path, latest_path


def read_signal(path: Path) -> dict:
    """Read a signal file; ``ValueError`` names the path when it is missing or not an object."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"signal file not found at {path}; run `hedgefund daily` first")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "symbols" not in data:
        raise ValueError(f"signal file {path} must be a JSON object with a 'symbols' key")
    return data


def last_closes(source: PriceSource, symbols: list[str]) -> dict[str, float]:
    """The last close of each symbol, re-read from ``source`` (a warm cache serves hits)."""
    closes = {}
    for symbol in symbols:
        frame = source.get_daily(symbol)
        close = close_series(frame, symbol)
        closes[symbol] = float(close.iloc[-1])
    return closes


@dataclass
class DailyJobResult:
    """What ``run_daily`` produced: the scan, the signal dict, its paths, the webhook results."""

    scan: ScanResult
    signal: dict
    signal_path: Path
    latest_path: Path
    webhook_results: list[WebhookResult]


def run_daily(
    universe: Universe,
    source: PriceSource,
    cfg: dict,
    *,
    analyze_kwargs: dict,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    webhooks_dir: Path | None = None,
    now: datetime | None = None,
    progress: Callable[[str], None] | None = None,
    post: Callable[..., int] | None = None,
    rule: str = "sign",  # config backtest.rule; the Phase 1 +1 / 0 / -1 rule
    environ: dict[str, str] | None = None,
) -> DailyJobResult:
    """Scan ``universe``, write the signal files, post the summary to the webhooks.

    ``webhooks_dir`` ``None`` posts nothing. ``post`` replaces the HTTP call (tests inject a
    recorder); ``environ`` replaces ``os.environ`` for token lookups. A webhook failure is
    recorded in ``webhook_results`` and never raised. ``now`` fixes ``generated_at`` and
    the run id.
    """
    _validate_rule(rule)
    created = utc_now(now)
    scan = scan_universe(
        universe,
        source,
        cfg,
        analyze_kwargs=analyze_kwargs,
        jobs_dir=jobs_dir,
        now=created,
        progress=progress,
    )
    closes = last_closes(source, list(scan.results))
    signal = build_signal(
        scan,
        last_close=closes,
        rule=rule,
        generated_at=created.isoformat(timespec="seconds"),
        source=source.name,
    )
    signal_path, latest_path = write_signal(signal, jobs_dir, scan.run_id)
    webhook_results: list[WebhookResult] = []
    if webhooks_dir is not None:
        webhooks = load_webhooks(webhooks_dir)
        webhook_results = post_summary(
            webhooks,
            summary_payload(signal),
            event=DEFAULT_EVENT,
            environ=environ,
            post=post_json if post is None else post,
        )
        if progress is not None:
            for result in webhook_results:
                progress(render_webhook_result(result))
    return DailyJobResult(
        scan=scan,
        signal=signal,
        signal_path=signal_path,
        latest_path=latest_path,
        webhook_results=webhook_results,
    )


def render_webhook_result(result: WebhookResult) -> str:
    """``webhook <name> (<host>): ok 200`` or ``... FAILED <error>``; the URL is never shown."""
    where = f"{result.name} ({result.host})" if result.host else result.name
    if result.ok:
        return f"webhook {where}: ok {result.status}"
    return f"webhook {where}: FAILED {result.error}"
