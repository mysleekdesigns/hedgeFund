"""The per-symbol scan pipeline and the ranked scan table.

``scan_universe`` runs the full ``analyze`` pipeline on every symbol of a universe,
through whatever ``PriceSource`` it is handed (normally the parquet-backed
``CachedSource`` so a warm run makes no network calls), and writes one run directory
under ``jobs/``: ``results/<symbol>.json``, ``table.json`` and ``manifest.json``. One
symbol failing with ``ValueError`` (too little history, a missing CSV) is recorded in the
manifest's ``errors`` and in its table row; the scan continues. Any other exception
propagates.

The table ranks symbols by net Sharpe, the only honest ordering (gross never ships
alone; ``.claude/rules/no-lookahead.md``). ``persistence`` is the stride (overlap-free,
plan.md Fix 1) diagonal of the current regime, falling back to the overlapping 1-step
diagonal when the stride row is empty; ``persistence_source`` says which. Every table
value is a JSON number, string or ``null``, never NaN.

Backtests are historical, not forward-looking.
"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hedgefund.backtest.costs import cost_model_from_config
from hedgefund.data.base import PriceSource, close_series
from hedgefund.jobs.manifest import (
    DEFAULT_JOBS_DIR,
    RunManifest,
    git_commit,
    json_ready,
    new_run_id,
    run_dir,
    utc_now,
    write_manifest,
    write_result,
    write_table,
)
from hedgefund.jobs.universe import Universe
from hedgefund.regime.labels import STATES
from hedgefund.report.json_contract import DISCLAIMER, FRAMEWORK, analyze

TABLE_COLUMNS = [
    "rank",
    "symbol",
    "current_regime",
    "signal",
    "persistence",
    "stationary_bear",
    "net_sharpe",
    "null_percentile",
    "rows",
    "date_end",
    "error",
]
PERSISTENCE_STRIDE = "stride"  # window-step diagonal of labels[::window] (Fix 1)
PERSISTENCE_OVERLAPPING = "overlapping"  # the 1-step diagonal of the daily (overlapping) fit
# ``current_regime`` is a ``STATES`` entry ("Bear"); the diagonal objects use lower-case keys.
REGIME_KEYS = {state: state.lower() for state in STATES}

# Fixed-width rendering: (column, width, right-aligned) in display order.
RENDER_COLUMNS = [
    ("rank", 4, True),
    ("symbol", 10, False),
    ("current_regime", 10, False),
    ("signal", 8, True),
    ("persistence", 12, True),
    ("stationary_bear", 10, True),
    ("net_sharpe", 11, True),
    ("null_percentile", 9, True),
    ("rows", 6, True),
    ("date_end", 11, False),
    ("error", 0, False),
]
RENDER_DIGITS = 3  # probabilities and Sharpe to three decimals
NA = "n/a"
OVERLAP_MARK = "*"  # suffix on a persistence value that fell back to the overlapping diagonal
OVERLAP_NOTE = (
    f"persistence = stride (window-step, overlap-free) diagonal of the current regime; "
    f"{OVERLAP_MARK} marks the overlapping 1-step fallback"
)
RANK_NOTE = "ranked by net Sharpe (costs charged); null Sharpe and errors sort last"


def _finite_or_none(value: object) -> float | None:
    """``float(value)`` when finite, ``None`` for ``None`` / NaN / inf (never emit NaN)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _get(node: object, *keys: str) -> object:
    """``node[k1][k2]...`` returning ``None`` as soon as a level is missing or ``None``."""
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def table_row(symbol: str, result: dict) -> dict:
    """One scan table row from an ``analyze`` dict. ``rank`` is filled by ``rank_table``."""
    regime = str(result["current_regime"])
    regime_key = REGIME_KEYS[regime]
    stride = _finite_or_none(_get(result, "stride_diagonal", regime_key))
    if stride is not None:
        persistence, persistence_source = stride, PERSISTENCE_STRIDE
    else:
        persistence = _finite_or_none(_get(result, "persistence_diagonal", regime_key))
        persistence_source = PERSISTENCE_OVERLAPPING
    return {
        "rank": None,
        "symbol": symbol,
        "current_regime": regime,
        "signal": _finite_or_none(result["signal"]),
        "persistence": persistence,
        "persistence_source": persistence_source,
        "stationary_bear": _finite_or_none(_get(result, "stationary_distribution", "bear")),
        "net_sharpe": _finite_or_none(_get(result, "backtest", "net", "sharpe")),
        "null_percentile": _finite_or_none(
            _get(result, "backtest", "null", "permutation", "percentile_net")
        ),
        "rows": int(result["rows"]),
        "date_end": str(result["date_end"]),
        "error": None,
    }


def error_row(symbol: str, message: str) -> dict:
    """A table row for a symbol whose fetch or analysis raised ``ValueError``."""
    row = {column: None for column in TABLE_COLUMNS}
    row["symbol"] = symbol
    row["persistence_source"] = None
    row["error"] = message
    return row


def _sort_key(row: dict) -> tuple:
    errored = row.get("error") is not None
    sharpe = _finite_or_none(row.get("net_sharpe"))
    unscored = sharpe is None
    # Descending Sharpe, so negate; ties and unscored rows fall back to the symbol.
    return (errored, unscored, -(sharpe if sharpe is not None else 0.0), row["symbol"])


def rank_table(rows: list[dict]) -> list[dict]:
    """Sort by net Sharpe descending and number the scored rows 1..n.

    ``None`` / non-finite Sharpe rows follow the scored ones (by symbol), errored rows come
    last with ``rank`` ``None``. The input rows are not mutated.
    """
    ordered = sorted((dict(row) for row in rows), key=_sort_key)
    rank = 0
    for row in ordered:
        if row.get("error") is None:
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = None
    return ordered


def _cell(row: dict, column: str) -> str:
    value = row.get(column)
    if value is None:
        return "" if column == "error" else NA
    if column == "persistence":
        text = f"{float(value):.{RENDER_DIGITS}f}"
        return (
            text + OVERLAP_MARK
            if row.get("persistence_source") == PERSISTENCE_OVERLAPPING
            else text
        )
    if column == "null_percentile":
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return f"{value:.{RENDER_DIGITS}f}"
    return str(value)


def render_table(table: list[dict]) -> str:
    """Plain-text ranked table: one header row, one row per symbol, two footnotes."""
    lines = []
    header = []
    for column, width, right in RENDER_COLUMNS:
        header.append(column.rjust(width) if right else column.ljust(width))
    lines.append(" ".join(header).rstrip())
    for row in table:
        cells = []
        for column, width, right in RENDER_COLUMNS:
            text = _cell(row, column)
            cells.append(text.rjust(width) if right else text.ljust(width))
        lines.append(" ".join(cells).rstrip())
    lines.append("")
    lines.append(RANK_NOTE)
    lines.append(OVERLAP_NOTE)
    lines.append(DISCLAIMER)
    return "\n".join(lines)


@dataclass
class ScanResult:
    """What ``scan_universe`` produced and where it wrote it."""

    run_id: str
    manifest: RunManifest
    results: dict[str, dict]  # symbol -> analyze dict, successes only
    table: list[dict]  # ranked rows, every symbol (errored ones carry ``error``)
    run_dir: Path


def _fallback_digest(frame: pd.DataFrame) -> str:
    """sha256 over the index (int64 ns) and float64 close bytes, the cache module's recipe."""
    index_bytes = np.ascontiguousarray(frame.index.as_unit("ns").asi8, dtype=np.int64).tobytes()
    close_bytes = np.ascontiguousarray(frame["close"].to_numpy(dtype=np.float64)).tobytes()
    return hashlib.sha256(index_bytes + close_bytes).hexdigest()


def _digest(frame: pd.DataFrame) -> str:
    try:
        from hedgefund.data.cache import frame_digest
    except ImportError:
        return _fallback_digest(frame)
    return frame_digest(frame)


def _data_entry(source: PriceSource, symbol: str, frame: pd.DataFrame, events: list) -> dict:
    """The manifest ``data`` entry: from the cache event when there is one, else the frame."""
    event = next((e for e in reversed(events) if getattr(e, "symbol", None) == symbol), None)
    if event is not None:
        return {
            "source": source.name,
            "start": str(pd.Timestamp(event.start).date()),
            "end": str(pd.Timestamp(event.end).date()),
            "rows": int(event.rows),
            "sha256": str(event.digest),
            "cache_path": str(event.path),
            "cache_hit": bool(event.hit),
            "result_file": None,
        }
    return {
        "source": source.name,
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
        "rows": len(frame),
        "sha256": _digest(frame),
        "cache_path": None,
        "cache_hit": None,
        "result_file": None,
    }


def scan_universe(
    universe: Universe,
    source: PriceSource,
    cfg: dict,
    *,
    analyze_kwargs: dict,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    now: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> ScanResult:
    """Run ``analyze`` on every symbol of ``universe`` and write the run directory.

    ``analyze_kwargs`` are passed through unchanged (window, threshold, min_train, hmm,
    null_paths, seed, ...); the cost model is ``cost_model_from_config(cfg,
    universe.cost_profile)``. ``now`` fixes the run id and ``created_at`` (tests pass it).
    ``progress`` receives one line per symbol. A ``ValueError`` (too little history, a
    missing CSV) or ``LookupError`` (``CacheMissError`` under ``--offline``) from the
    source or from ``analyze`` is recorded in ``manifest.errors`` and the table row's
    ``error``; the scan continues. Anything else propagates, but ``table.json`` and
    ``manifest.json`` are still written for the symbols finished so far, so a run
    directory never holds results without a manifest. When ``source`` exposes ``events``
    (``CachedSource``) the manifest records the cache path and hit flag per symbol;
    otherwise it hashes the frame itself.
    """
    cost_model = cost_model_from_config(cfg, universe.cost_profile)
    created = utc_now(now)
    run_id = new_run_id(universe.name, created)
    directory = run_dir(jobs_dir, run_id)
    has_events = hasattr(source, "events")

    results: dict[str, dict] = {}
    rows: list[dict] = []
    data: dict[str, dict] = {}
    errors: dict[str, str] = {}
    total = len(universe.symbols)

    def _write_run_files() -> tuple[list[dict], RunManifest]:
        table = rank_table(rows)
        manifest = RunManifest(
            run_id=run_id,
            created_at=created.isoformat(timespec="seconds"),
            universe=universe.name,
            universe_file=str(universe.path),
            cost_profile=universe.cost_profile,
            symbols=list(universe.symbols),
            params={**json_ready(analyze_kwargs), "cost_model": json_ready(cost_model)},
            data=data,
            git_commit=git_commit(),
            errors=errors,
        )
        write_table(run_id, table, jobs_dir)
        write_manifest(manifest, jobs_dir)
        return table, manifest

    try:
        for i, symbol in enumerate(universe.symbols, start=1):
            if progress is not None:
                progress(f"[{i}/{total}] {symbol}")
            n_events = len(source.events) if has_events else 0
            try:
                frame = source.get_daily(symbol)
                events = list(source.events[n_events:]) if has_events else []
                data[symbol] = _data_entry(source, symbol, frame, events)
                close = close_series(frame, symbol)
                result = analyze(close, source=symbol, cost_model=cost_model, **analyze_kwargs)
            except (ValueError, LookupError) as exc:
                errors[symbol] = str(exc)
                rows.append(error_row(symbol, str(exc)))
                if progress is not None:
                    progress(f"[{i}/{total}] {symbol}: error: {exc}")
                continue
            results[symbol] = result
            rows.append(table_row(symbol, result))
            data[symbol]["result_file"] = write_result(run_id, symbol, result, jobs_dir).name
    finally:
        # Written even when an unexpected exception escapes, so the run dir is never
        # results-only; the exception still propagates.
        table, manifest = _write_run_files()
    return ScanResult(
        run_id=run_id, manifest=manifest, results=results, table=table, run_dir=directory
    )


def scan_json(scan: ScanResult) -> dict:
    """The ``hedgefund scan --json`` object: run id, where it lives, the ranked table."""
    return {
        "run_id": scan.run_id,
        "run_dir": str(scan.run_dir),
        "universe": scan.manifest.universe,
        "cost_profile": scan.manifest.cost_profile,
        "symbols": list(scan.manifest.symbols),
        "table": scan.table,
        "errors": dict(scan.manifest.errors),
        "framework": FRAMEWORK,
        "disclaimer": DISCLAIMER,
    }
