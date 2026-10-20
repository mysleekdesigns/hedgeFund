"""A scan over a two-symbol universe writes two results, one ranked table, one manifest.

The price source is a fake that serves seeded geometric random walks, so the test runs
offline and finishes in a few seconds. Small ``null_paths`` / ``bootstrap_resamples``
keep ``analyze`` fast; every assertion is structural.
"""

import json
import math
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.costs import cost_model_from_config
from hedgefund.config import load_config
from hedgefund.data.base import INDEX_NAME
from hedgefund.jobs.manifest import (
    MANIFEST_FILE,
    RESULTS_DIR,
    TABLE_FILE,
    read_manifest,
    read_results,
    read_table,
)
from hedgefund.jobs.scan import (
    OVERLAP_MARK,
    PERSISTENCE_OVERLAPPING,
    PERSISTENCE_STRIDE,
    TABLE_COLUMNS,
    ScanResult,
    error_row,
    rank_table,
    render_table,
    scan_json,
    scan_universe,
    table_row,
)
from hedgefund.jobs.universe import Universe
from hedgefund.report.json_contract import DISCLAIMER, FRAMEWORK, analyze

ROWS = 600  # enough for min_train (252) + the walk-forward's 30-day minimum
SHORT_ROWS = 10  # fewer than the labeller window -> analyze raises ValueError
DAILY_VOL = 0.01
START_PRICE = 100.0
MIN_TRAIN = 252
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20
ANALYZE_KWARGS = {
    "hmm": False,
    "null_paths": FAST_NULL_PATHS,
    "bootstrap_resamples": FAST_RESAMPLES,
    "min_train": MIN_TRAIN,
}
FIXED_NOW = datetime(2026, 10, 20, 9, 0, 0, tzinfo=UTC)
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
SEEDS = {"AAA": 1, "BBB": 2, "SHORT": 3}


def _frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, rows)
    index = pd.bdate_range("2020-01-01", periods=rows, name=INDEX_NAME)
    close = START_PRICE * np.cumprod(1.0 + returns)
    return pd.DataFrame({"close": close.astype(np.float64)}, index=index)


class FakeSource:
    """A ``PriceSource`` serving seeded synthetic series; ``SHORT`` is too short to label."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_daily(
        self, symbol: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        self.calls.append(symbol)
        if symbol == "MISSING":
            raise ValueError(f"no data for {symbol}")
        rows = SHORT_ROWS if symbol == "SHORT" else ROWS
        return _frame(rows, SEEDS.get(symbol, 0))


class FakeCachedSource(FakeSource):
    """Same series, plus the ``events`` list a ``CachedSource`` exposes (duck typed)."""

    def __init__(self, cache_dir: Path) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.events: list[SimpleNamespace] = []

    def get_daily(
        self, symbol: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        frame = super().get_daily(symbol, start, end)
        self.events.append(
            SimpleNamespace(
                symbol=symbol,
                path=self.cache_dir / "fake" / f"{symbol}_2020-01-01_2022-04-20.parquet",
                hit=symbol == "BBB",
                start=pd.Timestamp("2020-01-01"),
                end=pd.Timestamp("2022-04-20"),
                rows=len(frame),
                digest="c" * 64,
            )
        )
        return frame


def _universe(tmp_path: Path, symbols: tuple[str, ...], profile: str = "us_etfs") -> Universe:
    path = tmp_path / "two.yaml"
    path.write_text(f"name: two\ncost_profile: {profile}\nsymbols: [{', '.join(symbols)}]\n")
    return Universe(name="two", cost_profile=profile, symbols=symbols, path=path)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def aaa_result() -> dict:
    close = _frame(ROWS, SEEDS["AAA"])["close"]
    return analyze(close, source="AAA", **ANALYZE_KWARGS)


def _assert_json_number_or_null(value: object) -> None:
    assert value is None or isinstance(value, int | float | str)
    if isinstance(value, float):
        assert math.isfinite(value)


# --- table_row / rank_table / render_table -------------------------------------------


def test_table_row_picks_stride_persistence_of_current_regime(aaa_result: dict) -> None:
    row = table_row("AAA", aaa_result)
    assert set(TABLE_COLUMNS) <= set(row)
    assert row["rank"] is None and row["error"] is None
    assert row["symbol"] == "AAA"
    regime_key = aaa_result["current_regime"].lower()
    stride = aaa_result["stride_diagonal"][regime_key]
    if stride is not None:
        assert row["persistence"] == stride
        assert row["persistence_source"] == PERSISTENCE_STRIDE
    else:
        assert row["persistence"] == aaa_result["persistence_diagonal"][regime_key]
        assert row["persistence_source"] == PERSISTENCE_OVERLAPPING
    assert row["signal"] == aaa_result["signal"]
    assert row["stationary_bear"] == aaa_result["stationary_distribution"]["bear"]
    assert row["net_sharpe"] == aaa_result["backtest"]["net"]["sharpe"]
    assert row["null_percentile"] == aaa_result["backtest"]["null"]["permutation"]["percentile_net"]
    assert row["rows"] == ROWS
    assert row["date_end"] == aaa_result["date_end"]
    for value in row.values():
        _assert_json_number_or_null(value)


def test_table_row_falls_back_to_overlapping_when_stride_is_null(aaa_result: dict) -> None:
    regime_key = aaa_result["current_regime"].lower()
    patched = json.loads(json.dumps(aaa_result))
    patched["stride_diagonal"][regime_key] = None
    row = table_row("AAA", patched)
    assert row["persistence"] == aaa_result["persistence_diagonal"][regime_key]
    assert row["persistence_source"] == PERSISTENCE_OVERLAPPING


def test_table_row_nan_and_missing_null_block_become_none(aaa_result: dict) -> None:
    patched = json.loads(json.dumps(aaa_result))
    patched["backtest"]["net"]["sharpe"] = float("nan")
    patched["backtest"]["null"] = None
    patched["signal"] = float("inf")
    row = table_row("AAA", patched)
    assert row["net_sharpe"] is None
    assert row["null_percentile"] is None
    assert row["signal"] is None
    json.dumps(row, allow_nan=False)


def test_rank_table_orders_by_net_sharpe_with_nulls_and_errors_last() -> None:
    rows = [
        {"symbol": "LOW", "net_sharpe": -0.5, "error": None},
        {"symbol": "NONE", "net_sharpe": None, "error": None},
        {"symbol": "HIGH", "net_sharpe": 1.5, "error": None},
        error_row("BAD", "Not enough data"),
        {"symbol": "MID", "net_sharpe": 0.2, "error": None},
        {"symbol": "NAN", "net_sharpe": float("nan"), "error": None},
    ]
    ranked = rank_table(rows)
    assert [r["symbol"] for r in ranked] == ["HIGH", "MID", "LOW", "NAN", "NONE", "BAD"]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4, 5, None]
    # The inputs are left untouched.
    assert rows[0]["symbol"] == "LOW" and "rank" not in rows[0]


def test_error_row_has_every_column() -> None:
    row = error_row("BAD", "boom")
    assert set(TABLE_COLUMNS) <= set(row)
    assert row["error"] == "boom" and row["symbol"] == "BAD" and row["rank"] is None
    assert all(row[c] is None for c in TABLE_COLUMNS if c not in ("symbol", "error"))


def test_render_table_has_header_symbols_and_disclaimer(aaa_result: dict) -> None:
    overlapping = json.loads(json.dumps(aaa_result))
    overlapping["stride_diagonal"] = {"bear": None, "sideways": None, "bull": None}
    table = rank_table(
        [table_row("AAA", aaa_result), table_row("OVL", overlapping), error_row("BAD", "boom")]
    )
    text = render_table(table)
    header = text.splitlines()[0]
    for column in TABLE_COLUMNS:
        assert column in header
    for symbol in ("AAA", "OVL", "BAD"):
        assert symbol in text
    assert "boom" in text
    assert DISCLAIMER in text
    assert OVERLAP_MARK in text


# --- scan_universe ----------------------------------------------------------------------


def test_scan_two_symbols_writes_two_results_one_table_one_manifest(
    tmp_path: Path, cfg: dict
) -> None:
    jobs_dir = tmp_path / "jobs"
    universe = _universe(tmp_path, ("AAA", "BBB"))
    source = FakeSource()
    lines: list[str] = []
    scan = scan_universe(
        universe,
        source,
        cfg,
        analyze_kwargs=ANALYZE_KWARGS,
        jobs_dir=jobs_dir,
        now=FIXED_NOW,
        progress=lines.append,
    )
    assert isinstance(scan, ScanResult)
    assert scan.run_id == "20261020T090000Z_two"
    assert scan.run_dir == jobs_dir / scan.run_id
    assert source.calls == ["AAA", "BBB"]
    assert len(lines) == 2

    # Two results, one table, one manifest on disk.
    results_dir = scan.run_dir / RESULTS_DIR
    assert sorted(p.name for p in results_dir.iterdir()) == ["AAA.json", "BBB.json"]
    assert (scan.run_dir / TABLE_FILE).is_file()
    assert (scan.run_dir / MANIFEST_FILE).is_file()
    assert read_results(scan.run_id, jobs_dir).keys() == {"AAA", "BBB"}
    assert read_results(scan.run_id, jobs_dir)["AAA"] == scan.results["AAA"]

    # One ranked table with two rows, ranks 1 and 2, Sharpe descending, no NaN.
    table = read_table(scan.run_id, jobs_dir)
    assert table == scan.table
    assert len(table) == 2
    assert [r["rank"] for r in table] == [1, 2]
    assert {r["symbol"] for r in table} == {"AAA", "BBB"}
    sharpes = [r["net_sharpe"] for r in table]
    assert all(s is not None for s in sharpes) and sharpes[0] >= sharpes[1]
    assert all(r["error"] is None for r in table)
    json.dumps(table, allow_nan=False)

    # The manifest hashes both series and records the parameters actually used.
    manifest = read_manifest(scan.run_id, jobs_dir)
    assert manifest == scan.manifest
    assert manifest.symbols == ["AAA", "BBB"]
    assert manifest.errors == {}
    assert set(manifest.data) == {"AAA", "BBB"}
    for symbol, entry in manifest.data.items():
        assert SHA256_HEX.fullmatch(entry["sha256"])
        assert entry["source"] == "fake"
        assert entry["rows"] == ROWS
        assert entry["start"] == "2020-01-01"
        assert entry["end"] == scan.results[symbol]["date_end"]
        assert entry["cache_path"] is None and entry["cache_hit"] is None
        assert entry["result_file"] == f"{symbol}.json"
    assert manifest.data["AAA"]["sha256"] != manifest.data["BBB"]["sha256"]
    assert manifest.created_at == "2026-10-20T09:00:00+00:00"
    assert manifest.params["hmm"] is False
    assert manifest.params["min_train"] == MIN_TRAIN
    assert manifest.params["cost_model"] == asdict(cost_model_from_config(cfg, "us_etfs"))
    assert manifest.universe_file == str(universe.path)
    assert manifest.cost_profile == "us_etfs"
    manifest_dict = json.loads((scan.run_dir / MANIFEST_FILE).read_text())
    assert manifest_dict["disclaimer"] == DISCLAIMER


def test_scan_digest_matches_a_rerun_and_changes_with_the_data(tmp_path: Path, cfg: dict) -> None:
    universe = _universe(tmp_path, ("AAA",))
    first = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "j1"
    )
    second = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "j2"
    )
    assert first.manifest.data["AAA"]["sha256"] == second.manifest.data["AAA"]["sha256"]
    assert first.results["AAA"] == second.results["AAA"]


def test_scan_uses_the_universe_cost_profile(tmp_path: Path, cfg: dict) -> None:
    universe = _universe(tmp_path, ("AAA",), profile="crypto")
    scan = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "jobs"
    )
    crypto = cost_model_from_config(cfg, "crypto")
    assert scan.manifest.params["cost_model"] == asdict(crypto)
    costs = scan.results["AAA"]["backtest"]["params"]["costs"]
    assert costs["fee_bps"] == crypto.fee_bps and costs["slippage_bps"] == crypto.slippage_bps


def test_scan_records_value_errors_and_continues(tmp_path: Path, cfg: dict) -> None:
    jobs_dir = tmp_path / "jobs"
    universe = _universe(tmp_path, ("SHORT", "AAA", "MISSING"))
    scan = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=jobs_dir
    )
    assert set(scan.manifest.errors) == {"SHORT", "MISSING"}
    assert "Not enough data" in scan.manifest.errors["SHORT"]
    assert "MISSING" in scan.manifest.errors["MISSING"]
    assert list(scan.results) == ["AAA"]
    # SHORT fetched fine so its data is hashed; MISSING never produced a frame.
    assert set(scan.manifest.data) == {"SHORT", "AAA"}
    assert scan.manifest.data["SHORT"]["result_file"] is None
    assert scan.manifest.data["AAA"]["result_file"] == "AAA.json"
    assert [r["symbol"] for r in scan.table] == ["AAA", "MISSING", "SHORT"]
    assert [r["rank"] for r in scan.table] == [1, None, None]
    assert scan.table[1]["error"] and scan.table[2]["error"]
    assert sorted(p.name for p in (scan.run_dir / RESULTS_DIR).iterdir()) == ["AAA.json"]
    assert read_manifest(scan.run_id, jobs_dir).errors == scan.manifest.errors


def test_scan_records_lookup_errors_like_an_offline_cache_miss(tmp_path: Path, cfg: dict) -> None:
    class OfflineMiss(FakeSource):
        def get_daily(self, symbol: str, start: object = None, end: object = None) -> pd.DataFrame:
            if symbol == "COLD":
                raise LookupError(f"offline: no cached data for {symbol}")
            return super().get_daily(symbol)

    jobs_dir = tmp_path / "jobs"
    universe = _universe(tmp_path, ("COLD", "AAA", "BBB"))
    scan = scan_universe(
        universe, OfflineMiss(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=jobs_dir
    )
    assert list(scan.manifest.errors) == ["COLD"]
    assert "offline" in scan.manifest.errors["COLD"]
    assert {r["symbol"] for r in scan.table[:2]} == {"AAA", "BBB"}
    assert scan.table[2]["symbol"] == "COLD"
    assert [r["rank"] for r in scan.table] == [1, 2, None]
    assert scan.table[2]["error"] is not None
    assert set(scan.results) == {"AAA", "BBB"}
    assert read_manifest(scan.run_id, jobs_dir).errors == {"COLD": scan.manifest.errors["COLD"]}
    assert read_table(scan.run_id, jobs_dir) == scan.table


def test_scan_writes_manifest_and_table_before_an_unexpected_error_escapes(
    tmp_path: Path, cfg: dict
) -> None:
    class Broken(FakeSource):
        def get_daily(self, symbol: str, start: object = None, end: object = None) -> pd.DataFrame:
            if symbol == "BOOM":
                raise RuntimeError("disk on fire")
            return super().get_daily(symbol)

    jobs_dir = tmp_path / "jobs"
    universe = _universe(tmp_path, ("AAA", "BOOM", "BBB"))
    with pytest.raises(RuntimeError, match="disk on fire"):
        scan_universe(universe, Broken(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=jobs_dir)
    # The run dir is never results-only: AAA finished, so its manifest and table exist.
    [run_id] = (p.name for p in jobs_dir.iterdir())
    manifest = read_manifest(run_id, jobs_dir)
    assert manifest.symbols == ["AAA", "BOOM", "BBB"]
    assert list(manifest.data) == ["AAA"]
    assert manifest.errors == {}
    table = read_table(run_id, jobs_dir)
    assert [r["symbol"] for r in table] == ["AAA"] and table[0]["rank"] == 1
    assert read_results(run_id, jobs_dir).keys() == {"AAA"}


def test_scan_unknown_cost_profile_raises_before_fetching(tmp_path: Path, cfg: dict) -> None:
    universe = _universe(tmp_path, ("AAA",), profile="nope")
    source = FakeSource()
    with pytest.raises(ValueError, match="nope"):
        scan_universe(
            universe, source, cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "jobs"
        )
    assert source.calls == []


def test_scan_reads_cache_events_when_the_source_has_them(tmp_path: Path, cfg: dict) -> None:
    universe = _universe(tmp_path, ("AAA", "BBB"))
    source = FakeCachedSource(tmp_path / "cache")
    scan = scan_universe(
        universe, source, cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "jobs"
    )
    aaa, bbb = scan.manifest.data["AAA"], scan.manifest.data["BBB"]
    assert aaa["cache_hit"] is False and bbb["cache_hit"] is True
    assert aaa["cache_path"].endswith("AAA_2020-01-01_2022-04-20.parquet")
    assert aaa["sha256"] == "c" * 64
    assert aaa["start"] == "2020-01-01" and aaa["end"] == "2022-04-20"
    assert aaa["rows"] == ROWS
    # The manifest file carries plain strings, so ``report`` can reopen the cache files.
    on_disk = json.loads((scan.run_dir / MANIFEST_FILE).read_text())
    assert isinstance(on_disk["data"]["AAA"]["cache_path"], str)


def test_scan_json_shape(tmp_path: Path, cfg: dict) -> None:
    universe = _universe(tmp_path, ("AAA", "SHORT"))
    scan = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=tmp_path / "jobs"
    )
    payload = scan_json(scan)
    assert list(payload) == [
        "run_id",
        "run_dir",
        "universe",
        "cost_profile",
        "symbols",
        "table",
        "errors",
        "framework",
        "disclaimer",
    ]
    assert payload["run_id"] == scan.run_id
    assert payload["run_dir"] == str(scan.run_dir)
    assert payload["universe"] == "two" and payload["cost_profile"] == "us_etfs"
    assert payload["symbols"] == ["AAA", "SHORT"]
    assert payload["table"] == scan.table
    assert set(payload["errors"]) == {"SHORT"}
    assert payload["framework"] == FRAMEWORK and payload["disclaimer"] == DISCLAIMER
    json.dumps(payload, allow_nan=False)


def test_scan_symbol_with_slash_round_trips_under_its_raw_name(tmp_path: Path, cfg: dict) -> None:
    jobs_dir = tmp_path / "jobs"
    universe = Universe(
        name="perps", cost_profile="crypto", symbols=("BTC/USD", "AAA"), path=tmp_path / "p.yaml"
    )
    scan = scan_universe(
        universe, FakeSource(), cfg, analyze_kwargs=ANALYZE_KWARGS, jobs_dir=jobs_dir
    )
    assert (scan.run_dir / RESULTS_DIR / "BTC-USD.json").is_file()
    assert scan.manifest.data["BTC/USD"]["result_file"] == "BTC-USD.json"
    results = read_results(scan.run_id, jobs_dir)
    assert set(results) == {"BTC/USD", "AAA"}
    assert results["BTC/USD"] == scan.results["BTC/USD"]
    assert [r["symbol"] for r in scan.table if r["symbol"] == "BTC/USD"] == ["BTC/USD"]
