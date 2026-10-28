# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The launcher exiting must not leave a surviving engine child un-signalled."""

import os
import signal
import subprocess
import sys
from types import SimpleNamespace

from genie_sim_client import _process


def test_escalates_for_whole_group_after_launcher_exit(monkeypatch):
    signals = []
    # Model the actual failure: launcher has exited after TERM, engine stays
    # alive in the same group until KILL. poll() alone is not completion.
    proc = SimpleNamespace(pid=98765, poll=lambda: 0)
    alive = True

    def killpg(pgid, sig):
        nonlocal alive
        assert pgid == proc.pid
        if not alive:
            raise ProcessLookupError
        if sig:
            signals.append(sig)
        if sig == signal.SIGKILL:
            alive = False

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(_process, "_TEARDOWN_LADDER", ((signal.SIGINT, 0), (signal.SIGTERM, 0), (signal.SIGKILL, 0)))
    _process._terminate_process_group(proc)
    assert signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_real_owned_session_group_cleanup_is_idempotent():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _process._terminate_process_group(proc)
        proc.wait(timeout=2)
        assert _process._process_group_gone(proc, proc.pid)
        _process._terminate_process_group(proc)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
