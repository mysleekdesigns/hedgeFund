"""Run manifests round-trip through JSON and the run directory layout is stable."""

import json
import math
import re
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from hedgefund import __version__
from hedgefund.backtest.costs import CostModel
from hedgefund.jobs import manifest as manifest_module
from hedgefund.jobs.manifest import (
    DEFAULT_JOBS_DIR,
    MANIFEST_FILE,
    RESULTS_DIR,
    TABLE_FILE,
    RunManifest,
    git_commit,
    json_ready,
    list_runs,
    new_run_id,
    read_manifest,
    read_results,
    read_table,
    result_filename,
    run_dir,
    utc_now,
    write_manifest,
    write_result,
    write_table,
)
from hedgefund.report.json_contract import DISCLAIMER

FIXED_NOW = datetime(2026, 10, 20, 14, 30, 5, tzinfo=UTC)
FIXED_RUN_ID = "20261020T143005Z_us_etfs"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _manifest(run_id: str = FIXED_RUN_ID) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=FIXED_NOW.isoformat(timespec="seconds"),
        universe="us_etfs",
        universe_file="config/universes/us_etfs.yaml",
        cost_profile="us_etfs",
        symbols=["SPY", "QQQ"],
        params={"window": 20, "threshold": 0.05, "cost_model": json_ready(CostModel())},
        data={
            "SPY": {
                "source": "yfinance",
                "start": "2016-10-20",
                "end": "2026-10-20",
                "rows": 2500,
                "sha256": "a" * 64,
                "cache_path": "cache/yfinance/SPY_2016-10-20_2026-10-20.parquet",
                "cache_hit": True,
            }
        },
        git_commit="b" * 40,
        errors={"QQQ": "Not enough data"},
    )


def test_default_jobs_dir_is_repo_jobs() -> None:
    assert DEFAULT_JOBS_DIR.name == "jobs"
    assert (DEFAULT_JOBS_DIR.parent / "pyproject.toml").is_file()
    assert (DEFAULT_JOBS_DIR / ".gitignore").read_text().split() == ["*", "!.gitignore"]


def test_new_run_id_uses_utc() -> None:
    assert new_run_id("us_etfs", FIXED_NOW) == FIXED_RUN_ID
    # The same instant expressed in UTC+2 gives the same id.
    plus_two = FIXED_NOW.astimezone(timezone(timedelta(hours=2)))
    assert new_run_id("us_etfs", plus_two) == FIXED_RUN_ID
    # A naive datetime is taken as UTC.
    assert new_run_id("us_etfs", FIXED_NOW.replace(tzinfo=None)) == FIXED_RUN_ID
    # Reading the clock still produces the documented shape.
    assert re.fullmatch(r"\d{8}T\d{6}Z_crypto", new_run_id("crypto"))


def test_utc_now_is_tz_aware_utc() -> None:
    assert utc_now().tzinfo == UTC
    assert utc_now(FIXED_NOW.replace(tzinfo=None)) == FIXED_NOW


def test_run_dir_layout(tmp_path: Path) -> None:
    assert run_dir(tmp_path, FIXED_RUN_ID) == tmp_path / FIXED_RUN_ID
    assert result_filename("SPY") == "SPY.json"
    assert result_filename("BTC/USD") == "BTC-USD.json"


def test_manifest_round_trip() -> None:
    manifest = _manifest()
    d = manifest.to_dict()
    assert d["hedgefund_version"] == __version__
    assert d["disclaimer"] == DISCLAIMER
    assert d["params"]["cost_model"] == {
        "fee_bps": 1.0,
        "slippage_bps": 2.0,
        "borrow_bps": 30.0,
        "funding_bps": 0.0,
    }
    # JSON-serialisable without NaN and back to an equal dataclass.
    text = json.dumps(d, allow_nan=False)
    assert RunManifest.from_dict(json.loads(text)) == manifest
    # Unknown keys are ignored; a missing required key is named.
    assert RunManifest.from_dict({**d, "future_key": 1}) == manifest
    with pytest.raises(ValueError, match="'symbols'"):
        RunManifest.from_dict({k: v for k, v in d.items() if k != "symbols"})


def test_json_ready_converts_parameter_types() -> None:
    out = json_ready(
        {
            "tuple": (1, 2),
            "np_int": np.int64(3),
            "np_float": np.float64(0.5),
            "nan": float("nan"),
            "inf": np.inf,
            "path": Path("cache/x.parquet"),
            "cost": CostModel(10.0, 10.0, 0.0, 0.0),
            "when": FIXED_NOW,
            "nested": {"seeds": (0, 1, 2)},
        }
    )
    assert out["tuple"] == [1, 2]
    assert out["np_int"] == 3 and isinstance(out["np_int"], int)
    assert out["np_float"] == 0.5 and isinstance(out["np_float"], float)
    assert out["nan"] is None and out["inf"] is None
    assert out["path"] == "cache/x.parquet"
    assert out["cost"] == {
        "fee_bps": 10.0,
        "slippage_bps": 10.0,
        "borrow_bps": 0.0,
        "funding_bps": 0.0,
    }
    assert out["when"] == "2026-10-20T14:30:05+00:00"
    assert out["nested"] == {"seeds": [0, 1, 2]}
    json.dumps(out, allow_nan=False)


def test_write_and_read_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_manifest(manifest, tmp_path)
    assert path == tmp_path / FIXED_RUN_ID / MANIFEST_FILE
    assert read_manifest(FIXED_RUN_ID, tmp_path) == manifest
    with pytest.raises(ValueError, match="nope"):
        read_manifest("nope", tmp_path)


def test_write_and_read_results(tmp_path: Path) -> None:
    assert read_results(FIXED_RUN_ID, tmp_path) == {}
    spy_path = write_result(FIXED_RUN_ID, "SPY", {"rows": 3, "x": 1.5}, tmp_path)
    write_result(FIXED_RUN_ID, "QQQ", {"rows": 4}, tmp_path)
    assert spy_path == tmp_path / FIXED_RUN_ID / RESULTS_DIR / "SPY.json"
    results = read_results(FIXED_RUN_ID, tmp_path)
    assert list(results) == ["QQQ", "SPY"]
    assert results["SPY"] == {"rows": 3, "x": 1.5}
    # The companion's NaN in walk_forward is allowed through, like ``to_json``.
    write_result(FIXED_RUN_ID, "NAN", {"walk_forward": {"sharpe": float("nan")}}, tmp_path)
    assert math.isnan(read_results(FIXED_RUN_ID, tmp_path)["NAN"]["walk_forward"]["sharpe"])


def test_read_results_maps_sanitised_files_back_to_raw_symbols(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.symbols = ["BTC/USD", "SPY", "NEVER_RAN"]
    write_manifest(manifest, tmp_path)
    path = write_result(FIXED_RUN_ID, "BTC/USD", {"rows": 7}, tmp_path)
    write_result(FIXED_RUN_ID, "SPY", {"rows": 3}, tmp_path)
    write_result(FIXED_RUN_ID, "ORPHAN", {"rows": 1}, tmp_path)
    assert path.name == "BTC-USD.json"
    results = read_results(FIXED_RUN_ID, tmp_path)
    # Manifest order for known symbols, raw names as keys, orphans by stem, no NEVER_RAN.
    assert list(results) == ["BTC/USD", "SPY", "ORPHAN"]
    assert results["BTC/USD"] == {"rows": 7}


def test_write_and_read_table(tmp_path: Path) -> None:
    table = [{"rank": 1, "symbol": "SPY", "net_sharpe": 0.5}, {"rank": None, "symbol": "QQQ"}]
    path = write_table(FIXED_RUN_ID, table, tmp_path)
    assert path == tmp_path / FIXED_RUN_ID / TABLE_FILE
    assert read_table(FIXED_RUN_ID, tmp_path) == table
    with pytest.raises(ValueError, match="nope"):
        read_table("nope", tmp_path)
    with pytest.raises(ValueError):
        write_table("bad", [{"net_sharpe": float("nan")}], tmp_path)


def test_list_runs_sorted_and_only_with_manifest(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "absent") == []
    later = new_run_id("crypto", FIXED_NOW + timedelta(days=1))
    earlier = new_run_id("crypto", FIXED_NOW - timedelta(days=1))
    for run_id in (later, FIXED_RUN_ID, earlier):
        write_manifest(_manifest(run_id), tmp_path)
    (tmp_path / "no_manifest_here").mkdir()
    write_table("table_only", [], tmp_path)
    assert list_runs(tmp_path) == [earlier, FIXED_RUN_ID, later]


def test_git_commit_is_sha_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = git_commit()
    assert commit is None or GIT_SHA.fullmatch(commit)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("no git here")

    monkeypatch.setattr(manifest_module.subprocess, "run", boom)
    assert git_commit() is None

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(manifest_module.subprocess, "run", timeout)
    assert git_commit() is None
