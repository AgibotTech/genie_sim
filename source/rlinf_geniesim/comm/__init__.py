# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pluggable comm layer: transport (server) + client (trainer) registries.

Mirror of ``physics/backends/__init__.py`` and ``render/bridges/__init__.py``
for the communication layer. Every concrete backend registers itself with
``@register_transport("name")`` / ``@register_client("name")``; the factories
``make_transport`` / ``make_client`` look the name up here and instantiate.

This package must stay importable on a plain Python host — no import-time
ROS / Isaac / rclpy dependency lives here. The SHM backend additionally needs
the ``posix_ipc`` wheel, but only when you actually construct/use the shm
transport or client (the import is lazy, done inside ``comm.shm.sem``); merely
importing ``rlinf_geniesim.comm`` does not require ``posix_ipc``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import ResultBuffer, SimClient, SimDims, SimHandler, SimTransport

# ---------------------------------------------------------------------------
# Registries + factories
# ---------------------------------------------------------------------------

_TRANSPORTS: dict[str, type[SimTransport]] = {}
_CLIENTS: dict[str, type[SimClient]] = {}


def register_transport(name: str):
    """Class decorator: register a concrete transport under ``name``."""

    def deco(cls: type[SimTransport]) -> type[SimTransport]:
        if name in _TRANSPORTS:
            raise ValueError(f"transport '{name}' already registered")
        if not issubclass(cls, SimTransport):
            raise TypeError(f"{cls.__name__} must subclass SimTransport")
        _TRANSPORTS[name] = cls
        return cls

    return deco


def make_transport(name: str, **cfg: Any) -> SimTransport:
    """Instantiate the transport registered under ``name``."""
    _load_builtin_transports()
    if name not in _TRANSPORTS:
        raise ValueError(f"unknown transport '{name}'; valid: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name](**cfg)


def list_transports() -> list[str]:
    """Return all registered transport names (sorted)."""
    _load_builtin_transports()
    return sorted(_TRANSPORTS)


def _load_builtin_transports() -> None:
    """Discover server transports when running from the server checkout."""
    for name in ("shm", "zmq"):
        module = f"{__name__}.{name}.transport"
        try:
            import_module(module)
        except ModuleNotFoundError as exc:
            # Client distributions omit transports; SHM additionally requires posix_ipc.
            if exc.name != module and not (name == "shm" and exc.name == "posix_ipc"):
                raise


def register_client(name: str):
    """Class decorator: register a concrete client under ``name``."""

    def deco(cls: type[SimClient]) -> type[SimClient]:
        if name in _CLIENTS:
            raise ValueError(f"client '{name}' already registered")
        if not issubclass(cls, SimClient):
            raise TypeError(f"{cls.__name__} must subclass SimClient")
        _CLIENTS[name] = cls
        return cls

    return deco


def make_client(name: str, **cfg: Any) -> SimClient:
    """Instantiate the client registered under ``name``."""
    if name not in _CLIENTS:
        raise ValueError(f"unknown client '{name}'; valid: {sorted(_CLIENTS)}")
    return _CLIENTS[name](**cfg)


def list_clients() -> list[str]:
    """Return all registered client names (sorted)."""
    return sorted(_CLIENTS)


# Client decorators need the registry functions to be defined first.
from . import shm, zmq  # noqa: E402,F401


__all__ = [
    "SimDims",
    "ResultBuffer",
    "SimHandler",
    "SimTransport",
    "SimClient",
    "register_transport",
    "make_transport",
    "list_transports",
    "register_client",
    "make_client",
    "list_clients",
]
