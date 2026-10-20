"""The autouse network guard in conftest.py must turn connection attempts into failures."""

import socket

import pytest

from conftest import NetworkAccessError


def test_socket_connect_is_refused() -> None:
    with pytest.raises(NetworkAccessError), socket.socket() as sock:
        sock.connect(("127.0.0.1", 9))


def test_create_connection_is_refused() -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
