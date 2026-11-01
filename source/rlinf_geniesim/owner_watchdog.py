# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in lifetime binding for a server owned by another Linux process."""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

ENV_OWNER_PID = "GENIESIM_OWNER_PID"
ENV_OWNER_START = "GENIESIM_OWNER_START"


def process_start_time(pid: int) -> str | None:
    """Return a live process's start time, or None for a missing/zombie process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm may contain spaces and parentheses; field 22 follows state by 19 fields.
    fields = stat.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 and fields[0] not in ("Z", "X") else None


def start_owner_watchdog(interval_s: float = 1.0, stop_timeout_s: float = 15.0) -> threading.Event | None:
    """Watch the PID and start time supplied by spawn_server; unset means disabled.

    Return an event to cancel the watchdog after server cleanup. Owner loss sends
    SIGTERM, then SIGKILL if cleanup does not finish within stop_timeout_s.
    """
    raw_pid = os.environ.get(ENV_OWNER_PID)
    if raw_pid is None:
        return None
    start = os.environ.get(ENV_OWNER_START, "")
    if not raw_pid.isdigit() or int(raw_pid) <= 0 or not start.isdigit():
        raise ValueError("GENIESIM_OWNER_PID and GENIESIM_OWNER_START must identify a live owner")
    pid = int(raw_pid)
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.is_set():
            if process_start_time(pid) != start:
                print(f"[owner-watchdog] owner pid={pid} exited; stopping server", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                if not stopped.wait(stop_timeout_s):
                    os.kill(os.getpid(), signal.SIGKILL)
                return
            stopped.wait(interval_s)

    # PDEATHSIG follows the spawning thread, which may exit before its owner process.
    threading.Thread(target=watch, name="owner-watchdog", daemon=True).start()
    return stopped
