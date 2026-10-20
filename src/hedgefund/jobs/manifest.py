"""Run manifests: what a scan ran, on which data, when, under ``./jobs/<run_id>/``.

A run directory holds ``manifest.json`` (parameters, per-symbol data hashes, timestamps,
version and git commit), ``results/<symbol>.json`` (the full ``analyze`` dict per
symbol) and ``table.json`` (the ranked scan table). ``hedgefund report <run-id>`` rebuilds
the HTML from these files and the cache without touching the network. Everything written
is plain JSON: numbers, strings, lists, objects or ``null``, never NaN.

Backtests are historical, not forward-looking.
"""

import json
import math
import subprocess
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hedgefund import __version__
from hedgefund.report.json_contract import DISCLAIMER

# src/hedgefund/jobs/manifest.py -> jobs -> hedgefund -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOBS_DIR = REPO_ROOT / "jobs"
MANIFEST_FILE = "manifest.json"
TABLE_FILE = "table.json"
RESULTS_DIR = "results"
RESULT_SUFFIX = ".json"
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"  # compact ISO 8601 basic format, always UTC
GIT_TIMEOUT_SECONDS = 5  # a hung git must never stall a scan
JSON_INDENT = 2
# Characters a symbol may carry that are unsafe in a file name (``BTC/USD``).
UNSAFE_FILENAME_CHARS = "/\\:"


def utc_now(now: datetime | None = None) -> datetime:
    """``now`` as a tz-aware UTC datetime; ``None`` reads the clock, naive means UTC."""
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def new_run_id(universe_name: str, now: datetime | None = None) -> str:
    """``<YYYYMMDDTHHMMSSZ>_<universe>`` in UTC, so ids sort chronologically."""
    return f"{utc_now(now):{RUN_ID_TIME_FORMAT}}_{universe_name}"


def run_dir(jobs_dir: Path, run_id: str) -> Path:
    """Directory of one run: ``jobs_dir / run_id``."""
    return Path(jobs_dir) / run_id


def result_filename(symbol: str) -> str:
    """File name of a symbol's result, with path-unsafe characters replaced by ``-``."""
    safe = "".join("-" if ch in UNSAFE_FILENAME_CHARS else ch for ch in symbol)
    return f"{safe}{RESULT_SUFFIX}"


def git_commit(cwd: Path | None = None) -> str | None:
    """``git rev-parse HEAD`` of the repo, ``None`` on any failure (never raises)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT if cwd is None else cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        # No git binary, no repo, a sandbox denial or a timeout: the manifest just lacks it.
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def json_ready(value: object) -> object:
    """Recursively convert a parameter value to plain JSON types.

    Tuples and sets become lists, numpy scalars become Python numbers, NaN and infinity
    become ``None``, ``Path`` becomes a string and a dataclass (``CostModel``) becomes a
    dict. Everything else is returned unchanged.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return utc_now(value).isoformat(timespec="seconds")
    return value


@dataclass
class RunManifest:
    """Everything needed to reproduce or audit one scan run."""

    run_id: str
    created_at: str  # ISO 8601 UTC
    universe: str
    universe_file: str
    cost_profile: str
    symbols: list[str]
    params: dict  # every analyze kwarg used, JSON-ready; the CostModel as a dict
    # symbol -> {"source", "start", "end", "rows", "sha256", "cache_path", "cache_hit",
    #            "result_file"} (result_file is None for a symbol that produced no result)
    data: dict[str, dict]
    hedgefund_version: str = __version__
    git_commit: str | None = None
    errors: dict[str, str] = field(default_factory=dict)  # symbol -> message; successes absent
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        """Plain JSON-ready dict with every field, in declaration order."""
        return json_ready(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        """Inverse of ``to_dict``; unknown keys are ignored so older readers keep working.

        Raises ``ValueError`` naming the first required key that is missing.
        """
        kwargs: dict = {}
        for f in fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.default is MISSING and f.default_factory is MISSING:
                raise ValueError(f"manifest is missing required key {f.name!r}")
        return cls(**kwargs)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=JSON_INDENT, allow_nan=False) + "\n")
    return path


def _read_json(path: Path, what: str) -> object:
    if not path.is_file():
        raise ValueError(f"{what} not found at {path}")
    return json.loads(path.read_text())


def write_manifest(manifest: RunManifest, jobs_dir: Path = DEFAULT_JOBS_DIR) -> Path:
    """Write ``run_dir/manifest.json`` and return its path."""
    return _write_json(run_dir(jobs_dir, manifest.run_id) / MANIFEST_FILE, manifest.to_dict())


def read_manifest(run_id: str, jobs_dir: Path = DEFAULT_JOBS_DIR) -> RunManifest:
    """Read ``run_dir/manifest.json``; ``ValueError`` naming ``run_id`` when missing."""
    path = run_dir(jobs_dir, run_id) / MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"run {run_id!r} has no {MANIFEST_FILE} under {Path(jobs_dir)}")
    data = _read_json(path, f"manifest for run {run_id!r}")
    if not isinstance(data, dict):
        raise ValueError(f"manifest for run {run_id!r} must be a JSON object, got {type(data)}")
    return RunManifest.from_dict(data)


def write_result(run_id: str, symbol: str, result: dict, jobs_dir: Path = DEFAULT_JOBS_DIR) -> Path:
    """Write ``run_dir/results/<symbol>.json`` (the full ``analyze`` dict)."""
    path = run_dir(jobs_dir, run_id) / RESULTS_DIR / result_filename(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``to_json`` semantics: the companion's walk_forward NaN is allowed, nothing else is.
    path.write_text(json.dumps(result) + "\n")
    return path


def read_results(run_id: str, jobs_dir: Path = DEFAULT_JOBS_DIR) -> dict[str, dict]:
    """Every ``results/*.json`` of the run keyed by the **raw** symbol, in manifest order.

    File names are sanitised (``BTC/USD`` -> ``BTC-USD.json``), so the manifest's
    ``symbols`` list is used to map each file back to the symbol ``manifest.data`` and
    ``manifest.errors`` use. Without a manifest the file stems are the keys. A result
    file no manifest symbol maps to is keyed by its stem so nothing is silently dropped.
    """
    directory = run_dir(jobs_dir, run_id) / RESULTS_DIR
    if not directory.is_dir():
        return {}
    manifest_path = run_dir(jobs_dir, run_id) / MANIFEST_FILE
    symbols: list[str] = []
    if manifest_path.is_file():
        symbols = list(read_manifest(run_id, jobs_dir).symbols)
    by_file = {result_filename(symbol): symbol for symbol in symbols}
    out: dict[str, dict] = {}
    for symbol in symbols:
        path = directory / result_filename(symbol)
        if path.is_file():
            out[symbol] = json.loads(path.read_text())
    for path in sorted(directory.glob(f"*{RESULT_SUFFIX}")):
        if path.name not in by_file:
            out[path.stem] = json.loads(path.read_text())
    return out


def write_table(run_id: str, table: list[dict], jobs_dir: Path = DEFAULT_JOBS_DIR) -> Path:
    """Write ``run_dir/table.json`` (a list of ranked rows, never NaN)."""
    return _write_json(run_dir(jobs_dir, run_id) / TABLE_FILE, table)


def read_table(run_id: str, jobs_dir: Path = DEFAULT_JOBS_DIR) -> list[dict]:
    """Read ``run_dir/table.json``; ``ValueError`` naming ``run_id`` when missing."""
    path = run_dir(jobs_dir, run_id) / TABLE_FILE
    if not path.is_file():
        raise ValueError(f"run {run_id!r} has no {TABLE_FILE} under {Path(jobs_dir)}")
    table = _read_json(path, f"table for run {run_id!r}")
    if not isinstance(table, list):
        raise ValueError(f"table for run {run_id!r} must be a JSON list, got {type(table)}")
    return table


def list_runs(jobs_dir: Path = DEFAULT_JOBS_DIR) -> list[str]:
    """Run ids under ``jobs_dir`` that hold a ``manifest.json``, sorted ascending."""
    directory = Path(jobs_dir)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if (p / MANIFEST_FILE).is_file())
