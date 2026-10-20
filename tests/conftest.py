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
