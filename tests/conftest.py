"""Shared pytest configuration for the hedgefund test suite.

Tests run offline. Fixtures that need frozen data read from ``tests/fixtures/``. The
autouse guard below turns any attempt to open a network connection into a test failure,
so a test that forgets the fixture cannot silently hit Yahoo or Alpaca and pass.
"""

import socket
from collections.abc import Iterator

import pytest


class NetworkAccessError(RuntimeError):
    """Raised when a test tries to open a socket connection."""


# Every refused attempt is recorded here so a test can prove the guard, not the library's
# own error handling, is what stopped a call (yfinance swallows transport errors and logs
# a message whose wording depends on its internal caches).
REFUSED_ATTEMPTS: list[str] = []


def _refuse_connection(*args: object, **kwargs: object) -> None:
    REFUSED_ATTEMPTS.append(repr(args[:1]))
    raise NetworkAccessError(
        "Tests run offline. Use tests/fixtures/ or synthetic data instead of the network."
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound connections for every test.

    yfinance >= 1.0 talks to Yahoo through ``curl_cffi`` (libcurl), which never touches
    Python's ``socket`` module, so the socket patches alone would let a test that forgot to
    stub ``yfinance.download`` reach the network. Patch curl_cffi's request entry points too.
    """
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_connection)
    monkeypatch.setattr(socket, "create_connection", _refuse_connection)
    try:
        import curl_cffi
        import curl_cffi.requests
    except ImportError:  # optional transitive dependency of yfinance
        pass
    else:
        monkeypatch.setattr(curl_cffi.Curl, "perform", _refuse_connection)
        monkeypatch.setattr(curl_cffi.requests.Session, "request", _refuse_connection)
        monkeypatch.setattr(curl_cffi.requests.AsyncSession, "request", _refuse_connection)
    yield


# --------------------------------------------------------------------------- #
# Phase 8: a cached scan the dashboard data layer and app tests read offline
# --------------------------------------------------------------------------- #
DASHBOARD_SYMBOLS = ("AAA", "BBB")
DASHBOARD_COST_PROFILE = "us_etfs"
DASHBOARD_UNIVERSE = "pair"
DASHBOARD_ROWS = 600  # > min_train 252 + a two-year holdout so each symbol gets a net Sharpe
DASHBOARD_SEED = 11
DASHBOARD_DAILY_VOL = 0.01
DASHBOARD_START_PRICE = 50.0
DASHBOARD_NULL_PATHS = 20  # small Monte Carlo sizes keep the session fixture fast
DASHBOARD_RESAMPLES = 20


def _write_dashboard_csv(csv_dir, symbol: str, seed: int) -> None:
    """``<csv_dir>/<symbol>.csv`` with ``date, close`` columns ending yesterday (UTC).

    Ending yesterday keeps every row inside the cache's default window no matter when the
    test runs: the cache never stores today's partial bar.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DASHBOARD_DAILY_VOL, DASHBOARD_ROWS)
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    index = pd.bdate_range(end=end, periods=DASHBOARD_ROWS, name="date")
    close = pd.Series(DASHBOARD_START_PRICE * np.cumprod(1.0 + returns), index=index, name="close")
    close.to_csv(csv_dir / f"{symbol}.csv")


@pytest.fixture(scope="session")
def dashboard_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One offline ``hedgefund scan`` over two synthetic CSVs, shared by the dashboard tests.

    Session-scoped because the scan is the slow part. Returns the jobs dir, run id, run
    dir, cache dir, csv dir, the symbols and the cost profile the universe declared.
    """
    import json

    from typer.testing import CliRunner

    from hedgefund.cli import app

    root = tmp_path_factory.mktemp("dashboard")
    csv_dir = root / "csv"
    csv_dir.mkdir()
    for offset, symbol in enumerate(DASHBOARD_SYMBOLS):
        _write_dashboard_csv(csv_dir, symbol, DASHBOARD_SEED + offset)
    universe = root / f"{DASHBOARD_UNIVERSE}.yaml"
    universe.write_text(
        f"name: {DASHBOARD_UNIVERSE}\ncost_profile: {DASHBOARD_COST_PROFILE}\n"
        f"symbols: {list(DASHBOARD_SYMBOLS)}\n"
    )
    cache_dir = root / "cache"
    jobs_dir = root / "jobs"
    result = CliRunner().invoke(
        app,
        [
            "scan",
            "--universe",
            str(universe),
            "--source",
            "csv",
            "--csv-dir",
            str(csv_dir),
            "--cache-dir",
            str(cache_dir),
            "--jobs-dir",
            str(jobs_dir),
            "--no-hmm",
            "--null-paths",
            str(DASHBOARD_NULL_PATHS),
            "--bootstrap-resamples",
            str(DASHBOARD_RESAMPLES),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    run_id = json.loads(result.stdout)["run_id"]
    return {
        "jobs_dir": jobs_dir,
        "run_id": run_id,
        "run_dir": jobs_dir / run_id,
        "cache_dir": cache_dir,
        "csv_dir": csv_dir,
        "symbols": DASHBOARD_SYMBOLS,
        "cost_profile": DASHBOARD_COST_PROFILE,
    }
