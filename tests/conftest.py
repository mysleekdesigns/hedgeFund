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


def _refuse_connection(*args: object, **kwargs: object) -> None:
    raise NetworkAccessError(
        "Tests run offline. Use tests/fixtures/ or synthetic data instead of the network."
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound connections for every test."""
    monkeypatch.setattr(socket.socket, "connect", _refuse_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_connection)
    monkeypatch.setattr(socket, "create_connection", _refuse_connection)
    yield
