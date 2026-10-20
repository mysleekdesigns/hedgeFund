"""The autouse network guard in conftest.py must turn connection attempts into failures."""

import socket

import pytest

from conftest import REFUSED_ATTEMPTS, NetworkAccessError


def test_socket_connect_is_refused() -> None:
    with pytest.raises(NetworkAccessError), socket.socket() as sock:
        sock.connect(("127.0.0.1", 9))


def test_create_connection_is_refused() -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


def test_curl_cffi_requests_are_refused() -> None:
    curl_cffi = pytest.importorskip("curl_cffi")
    with pytest.raises(NetworkAccessError):
        curl_cffi.requests.Session().request("GET", "http://127.0.0.1:9/")
    with pytest.raises(NetworkAccessError):
        curl_cffi.Curl().perform()


def test_yfinance_download_is_refused_offline() -> None:
    # yfinance swallows the transport error and returns an empty frame, so prove the guard
    # fired by counting refused attempts rather than parsing yfinance's log wording.
    yfinance = pytest.importorskip("yfinance")
    REFUSED_ATTEMPTS.clear()
    frame = yfinance.download("SPY", start="2024-01-01", end="2024-01-10", progress=False)
    assert frame.empty
    assert REFUSED_ATTEMPTS, "yfinance.download completed without hitting the network guard"
