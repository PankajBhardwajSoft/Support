"""Blocks outbound network connections, to prove runs need no network."""

from __future__ import annotations

import socket

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_create_connection = socket.create_connection


# Every blocked attempt is recorded, so a library that swallows the error
# still shows up as having tried to reach the network.
blocked_attempts: list[str] = []


class NetworkBlockedError(RuntimeError):
    pass


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # unix socket path
        return True
    host = address[0] if isinstance(address, tuple) and address else ""
    return host in {"127.0.0.1", "::1", "localhost"}


def _guard(address):
    if not _is_local(address):
        blocked_attempts.append(repr(address))
        raise NetworkBlockedError(f"Network access blocked: tried to connect to {address!r}")


def _blocked_connect(self, address):
    _guard(address)
    return _original_connect(self, address)


def _blocked_connect_ex(self, address):
    _guard(address)
    return _original_connect_ex(self, address)


def _blocked_create_connection(address, *args, **kwargs):
    _guard(address)
    return _original_create_connection(address, *args, **kwargs)


def block_network() -> None:
    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect_ex
    socket.create_connection = _blocked_create_connection


def unblock_network() -> None:
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.create_connection = _original_create_connection
