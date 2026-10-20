"""``run_daily`` end to end, offline: a fake price source, a tiny universe, a recorded webhook.

The price source serves seeded geometric random walks (the ``test_scan.py`` setup), so the
job finishes in seconds; small ``null_paths`` / ``bootstrap_resamples`` keep ``analyze``
fast. Every assertion is structural: files exist and agree, positions follow
``position_from_signal``, nothing was ordered, and the webhook token never leaks.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedgefund.config import load_config
from hedgefund.data.base import INDEX_NAME
from hedgefund.jobs.daily import (
    LATEST_FILE,
    ORDERS_NOTE,
    SIGNAL_FILE,
    SYMBOL_KEYS,
    TEMP_SUFFIX,
    DailyJobResult,
    build_signal,
    read_signal,
    render_webhook_result,
    run_daily,
    summary_payload,
    write_signal,
)
from hedgefund.jobs.universe import Universe
from hedgefund.jobs.webhooks import TOKEN_HEADER
from hedgefund.regime.signal import position_from_signal
from hedgefund.report.json_contract import DISCLAIMER, FRAMEWORK

ROWS = 600  # enough for min_train (252) + the walk-forward's 30-day minimum
SHORT_ROWS = 10  # fewer than the labeller window -> analyze raises ValueError
DAILY_VOL = 0.01
START_PRICE = 100.0
FAST_NULL_PATHS = 20
FAST_RESAMPLES = 20
ANALYZE_KWARGS = {
    "hmm": False,
    "null_paths": FAST_NULL_PATHS,
    "bootstrap_resamples": FAST_RESAMPLES,
    "min_train": 252,
}
FIXED_NOW = datetime(2026, 10, 20, 9, 0, 0, tzinfo=UTC)
SEEDS = {"AAA": 1, "BBB": 2}
SYMBOLS = ("AAA", "BBB", "SHORT")
TOKEN_VAR = "HEDGEFUND_WEBHOOK_TEST"
SENTINEL_TOKEN = "sentinel-token-7f3a9c"  # a value no real file or payload would contain
SECRET_PATH = "T0SECRET/B0PATH/xyzSENTINEL"  # the secret part of a Slack / Discord hook URL
HOOK_URL = f"https://hooks.example.invalid/services/{SECRET_PATH}"
# Words that would betray a path or an environment value in the posted summary.
FORBIDDEN_SUMMARY_KEYS = ("path", "dir", "env", "token", "cache", "key", "secret")


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
        rows = SHORT_ROWS if symbol == "SHORT" else ROWS
        return _frame(rows, SEEDS.get(symbol, 0))


def _universe(tmp_path: Path, symbols: tuple[str, ...] = SYMBOLS) -> Universe:
    path = tmp_path / "three.yaml"
    path.write_text(f"name: three\ncost_profile: us_etfs\nsymbols: [{', '.join(symbols)}]\n")
    return Universe(name="three", cost_profile="us_etfs", symbols=symbols, path=path)


def _webhooks_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "webhooks"
    directory.mkdir()
    (directory / "consumer.json").write_text(
        json.dumps({"name": "consumer", "url": HOOK_URL, "token_env": TOKEN_VAR})
    )
    return directory


class Recorder:
    """A ``post`` stand-in recording ``(url, payload, headers)`` and returning a status."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status = status
        self.raises = raises
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url: str, payload: dict, *, headers: dict[str, str]) -> int:
        self.calls.append((url, payload, headers))
        if self.raises is not None:
            raise self.raises
        return self.status


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def job(tmp_path_factory: pytest.TempPathFactory, cfg: dict) -> dict:
    """One daily run over AAA, BBB and the too-short SHORT, with one token-bearing webhook."""
    root = tmp_path_factory.mktemp("daily")
    source = FakeSource()
    recorder = Recorder()
    result = run_daily(
        _universe(root),
        source,
        cfg,
        analyze_kwargs=ANALYZE_KWARGS,
        jobs_dir=root / "jobs",
        webhooks_dir=_webhooks_dir(root),
        now=FIXED_NOW,
        post=recorder,
        environ={TOKEN_VAR: SENTINEL_TOKEN},
    )
    return {"result": result, "source": source, "recorder": recorder, "jobs_dir": root / "jobs"}


def test_run_daily_writes_signal_and_latest_with_identical_json(job: dict) -> None:
    result: DailyJobResult = job["result"]
    assert result.signal_path == job["jobs_dir"] / result.scan.run_id / SIGNAL_FILE
    assert result.latest_path == job["jobs_dir"] / LATEST_FILE
    assert result.signal_path.is_file() and result.latest_path.is_file()
    assert result.signal_path.read_bytes() == result.latest_path.read_bytes()
    assert json.loads(result.signal_path.read_text()) == result.signal
    assert read_signal(result.latest_path) == result.signal


def test_run_daily_signal_has_every_universe_symbol_with_the_symbol_keys(job: dict) -> None:
    signal = job["result"].signal
    assert tuple(signal["symbols"]) == tuple(sorted(SYMBOLS))
    for entry in signal["symbols"].values():
        assert tuple(entry) == SYMBOL_KEYS
    assert signal["run_id"] == "20261020T090000Z_three"
    assert signal["generated_at"] == "2026-10-20T09:00:00+00:00"
    assert signal["universe"] == "three" and signal["cost_profile"] == "us_etfs"
    assert signal["source"] == "fake" and signal["rule"] == "sign"
    assert signal["as_of"] == max(
        e["date_end"] for e in signal["symbols"].values() if e["date_end"] is not None
    )


def test_run_daily_position_matches_position_from_signal(job: dict) -> None:
    result: DailyJobResult = job["result"]
    for symbol in ("AAA", "BBB"):
        entry = result.signal["symbols"][symbol]
        analyzed = result.scan.results[symbol]
        regime_key = analyzed["current_regime"].lower()
        assert entry["current_regime"] == analyzed["current_regime"]
        assert entry["signal"] == analyzed["signal"]
        assert entry["stay_probability"] == analyzed["persistence_diagonal"][regime_key]
        assert entry["position"] == position_from_signal(
            entry["signal"], entry["stay_probability"], "sign"
        )
        assert entry["position"] in (-1.0, 0.0, 1.0)
        assert entry["rows"] == ROWS and entry["error"] is None


def test_run_daily_last_close_is_the_final_close_of_the_source_frame(job: dict) -> None:
    result: DailyJobResult = job["result"]
    for symbol in ("AAA", "BBB"):
        expected = float(_frame(ROWS, SEEDS[symbol])["close"].iloc[-1])
        assert result.signal["symbols"][symbol]["last_close"] == expected
    # The scan read each symbol once and the last-close pass read the successes again.
    assert job["source"].calls == ["AAA", "BBB", "SHORT", "AAA", "BBB"]


def test_run_daily_errored_symbol_carries_error_and_no_position(job: dict) -> None:
    entry = job["result"].signal["symbols"]["SHORT"]
    assert entry["error"] is not None
    assert entry["position"] is None and entry["signal"] is None
    assert entry["last_close"] is None and entry["current_regime"] is None
    assert job["result"].scan.manifest.errors == {"SHORT": entry["error"]}


def test_run_daily_places_no_orders_and_says_so(job: dict) -> None:
    signal = job["result"].signal
    assert signal["orders_placed"] is False
    assert signal["note"] == ORDERS_NOTE
    assert signal["framework"] == FRAMEWORK and signal["disclaimer"] == DISCLAIMER
    assert "order" in ORDERS_NOTE.lower() and "human" in ORDERS_NOTE.lower()


def test_run_daily_posts_the_summary_with_the_token_header_only(job: dict) -> None:
    recorder: Recorder = job["recorder"]
    result: DailyJobResult = job["result"]
    assert len(recorder.calls) == 1
    url, payload, headers = recorder.calls[0]
    assert url == HOOK_URL
    assert headers == {TOKEN_HEADER: SENTINEL_TOKEN}
    assert payload == summary_payload(result.signal)
    assert [r.ok for r in result.webhook_results] == [True]
    assert result.webhook_results[0].status == 200


def test_run_daily_token_never_reaches_the_signal_file_payload_or_results(job: dict) -> None:
    result: DailyJobResult = job["result"]
    recorder: Recorder = job["recorder"]
    assert SENTINEL_TOKEN.encode() not in result.signal_path.read_bytes()
    assert SENTINEL_TOKEN.encode() not in result.latest_path.read_bytes()
    assert SENTINEL_TOKEN not in json.dumps(recorder.calls[0][1])
    assert SENTINEL_TOKEN not in json.dumps([r.to_dict() for r in result.webhook_results])
    manifest = (result.scan.run_dir / "manifest.json").read_bytes()
    assert SENTINEL_TOKEN.encode() not in manifest


def test_summary_payload_has_no_path_or_environment_keys(job: dict) -> None:
    payload = summary_payload(job["result"].signal)
    assert payload["orders_placed"] is False
    assert payload["note"] == ORDERS_NOTE and payload["disclaimer"] == DISCLAIMER
    assert set(payload["symbols"]) == set(SYMBOLS)
    assert payload["symbols"]["SHORT"]["error"] is not None
    assert payload["symbols"]["AAA"]["position"] in (-1.0, 0.0, 1.0)

    def keys(node: object) -> list[str]:
        if not isinstance(node, dict):
            return []
        out = []
        for key, value in node.items():
            out.append(str(key))
            out.extend(keys(value))
        return out

    for key in keys(payload):
        assert not any(word in key.lower() for word in FORBIDDEN_SUMMARY_KEYS), key
    assert str(job["result"].scan.run_dir) not in json.dumps(payload)


def test_run_daily_failing_post_is_recorded_not_raised(tmp_path: Path, cfg: dict) -> None:
    recorder = Recorder(raises=OSError(f"refused; headers were {{{SENTINEL_TOKEN}}}"))
    result = run_daily(
        _universe(tmp_path, ("AAA",)),
        FakeSource(),
        cfg,
        analyze_kwargs=ANALYZE_KWARGS,
        jobs_dir=tmp_path / "jobs",
        webhooks_dir=_webhooks_dir(tmp_path),
        now=FIXED_NOW,
        post=recorder,
        environ={TOKEN_VAR: SENTINEL_TOKEN},
    )
    assert result.signal_path.is_file()
    (hook,) = result.webhook_results
    assert hook.ok is False and hook.status is None
    assert "OSError" in hook.error
    assert SENTINEL_TOKEN not in hook.error


def test_run_daily_without_webhooks_dir_posts_nothing(tmp_path: Path, cfg: dict) -> None:
    recorder = Recorder()
    result = run_daily(
        _universe(tmp_path, ("AAA",)),
        FakeSource(),
        cfg,
        analyze_kwargs=ANALYZE_KWARGS,
        jobs_dir=tmp_path / "jobs",
        webhooks_dir=None,
        now=FIXED_NOW,
        post=recorder,
    )
    assert recorder.calls == []
    assert result.webhook_results == []
    assert result.signal["orders_placed"] is False


def test_run_daily_persistence_rule_scales_by_stay_probability(tmp_path: Path, cfg: dict) -> None:
    result = run_daily(
        _universe(tmp_path, ("AAA",)),
        FakeSource(),
        cfg,
        analyze_kwargs=ANALYZE_KWARGS,
        jobs_dir=tmp_path / "jobs",
        now=FIXED_NOW,
        rule="persistence",
    )
    entry = result.signal["symbols"]["AAA"]
    assert result.signal["rule"] == "persistence"
    assert entry["position"] == pytest.approx(entry["signal"] * entry["stay_probability"])


def test_build_signal_rejects_unknown_rule(job: dict) -> None:
    with pytest.raises(ValueError, match="rule"):
        build_signal(job["result"].scan, last_close={}, rule="yolo", generated_at="x")


def test_read_signal_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=str(tmp_path / "missing.json")):
        read_signal(tmp_path / "missing.json")


def test_run_daily_never_prints_or_serialises_the_webhook_url(job: dict) -> None:
    result: DailyJobResult = job["result"]
    (hook,) = result.webhook_results
    assert hook.url == HOOK_URL  # kept on the object
    assert "url" not in hook.to_dict()
    assert hook.to_dict()["host"] == "hooks.example.invalid"
    assert SECRET_PATH not in json.dumps(hook.to_dict())
    line = render_webhook_result(hook)
    assert SECRET_PATH not in line and "hooks.example.invalid" in line and "ok 200" in line
    assert SECRET_PATH.encode() not in result.signal_path.read_bytes()


def test_write_signal_replaces_atomically_and_leaves_no_temp_files(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    first = {"symbols": {}, "run_id": "r1", "orders_placed": False}
    second = {"symbols": {"AAA": {"position": 1.0}}, "run_id": "r2", "orders_placed": False}
    write_signal(first, jobs_dir, "r1")
    signal_path, latest_path = write_signal(second, jobs_dir, "r2")
    assert json.loads(latest_path.read_text()) == second
    assert json.loads(signal_path.read_text()) == second
    assert json.loads((jobs_dir / "r1" / SIGNAL_FILE).read_text()) == first
    assert list(jobs_dir.rglob(f"*{TEMP_SUFFIX}")) == []


def test_write_signal_failure_leaves_the_previous_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rename (disk full, permissions) keeps the old file and cleans the temp file."""
    import hedgefund.jobs.daily as daily

    jobs_dir = tmp_path / "jobs"
    good = {"symbols": {}, "run_id": "r1", "orders_placed": False}
    write_signal(good, jobs_dir, "r1")

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(daily.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        write_signal({"symbols": {}, "run_id": "r2", "orders_placed": False}, jobs_dir, "r2")
    assert json.loads((jobs_dir / LATEST_FILE).read_text()) == good
    assert json.loads((jobs_dir / "r1" / SIGNAL_FILE).read_text()) == good
    assert list(jobs_dir.rglob(f"*{TEMP_SUFFIX}")) == []
