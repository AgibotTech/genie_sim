# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isolate generated stage and reload-control files by server identity.

Use the bound port for a stable tag across server restarts. Fall back to the
process ID when no port is available. An explicit RLINF_SERVER_TAG is retained
and inherited by child processes. Never share a default directory between
independent servers."""

from __future__ import annotations

import os
import re
from typing import Optional

#: Environment variable holding the tag, written once the bind string is
#: parsed (for example ``p5601``); a startup script may set it explicitly.
#: Subprocesses inherit it, which makes it visible in /proc when debugging.
_TAG_ENV = "RLINF_SERVER_TAG"

# The tag goes into a path, so allow [A-Za-z0-9_-] only.
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def parse_bind_port(bind: str) -> Optional[int]:
    """Extract the port from a transport bind string, or ``None``.

    Only the ``host:port`` form is recognised. Transports without a port and
    anything unparseable return ``None`` rather than a guess: a wrong guess
    would give two servers the same tag, which is what this module prevents.
    """
    tail = str(bind or "").rsplit(":", 1)
    if len(tail) != 2:
        return None
    try:
        port = int(tail[1])
    except ValueError:
        return None
    return port if 0 < port < 65536 else None


def set_server_tag_from_bind(bind: str) -> str:
    """Derive this process's tag from the bind string, store it, return it.

    An existing ``RLINF_SERVER_TAG`` is not overwritten: a value a startup
    script set explicitly wins. Nothing is written when no port can be parsed,
    leaving :func:`server_tag` to fall back to the pid.
    """
    existing = os.environ.get(_TAG_ENV, "").strip()
    if existing:
        return _SAFE.sub("_", existing)
    port = parse_bind_port(bind)
    if port is None:
        return server_tag()
    tag = f"p{port}"
    os.environ[_TAG_ENV] = tag
    return tag


def server_tag() -> str:
    """This process's server tag: ``RLINF_SERVER_TAG``, else ``pid<PID>``."""
    raw = os.environ.get(_TAG_ENV, "").strip()
    if raw:
        return _SAFE.sub("_", raw)
    return f"pid{os.getpid()}"


def default_multi_env_dir(num_envs: int) -> str:
    """The single definition of the multi-env artefact directory default.

    Used only when the caller supplies no ``multi_env_output_dir``. It is a
    function rather than a literal so the value cannot be duplicated and drift.
    """
    return f"/tmp/rlinf_multi_{int(num_envs)}env_{server_tag()}"


def default_stage_dir() -> str:
    """Return the per-server single-env stage directory without creating it."""
    return f"/tmp/rlinf_stage_{server_tag()}"


__all__ = [
    "parse_bind_port",
    "set_server_tag_from_bind",
    "server_tag",
    "default_multi_env_dir",
    "default_stage_dir",
]
