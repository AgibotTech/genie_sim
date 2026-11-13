# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Graceful detached shutdown is distinct from signal escalation."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from geniesim_benchmark.newton import layout_batch


CHILD = """
import pathlib, signal, sys
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
def done(*args):
    (root / 'child-exited').write_text('clean')
    sys.exit(0)
signal.signal(signal.SIGINT, done if mode == 'normal' else signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == 'kill' else done)
(root / 'child-ready').write_text('ready')
while True:
    signal.pause()
"""

PARENT = """
import json, pathlib, signal, subprocess, sys
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGINT, (lambda *args: sys.exit(0)) if sys.argv[2] == 'normal' else signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', sys.argv[4], str(root), sys.argv[3]], start_new_session=True)
(root / 'child-pid').write_text(str(child.pid))
while True:
    signal.pause()
"""


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process ownership required")
@pytest.mark.parametrize(
    "parent_mode,child_mode,forced",
    [
        ("normal", "normal", False),
        ("normal", "term", True),
        ("term", "normal", True),
        ("normal", "kill", True),
    ],
)
def test_real_owned_sessions_report_only_actual_escalation(tmp_path, parent_mode, child_mode, forced):
    process = subprocess.Popen(
        [sys.executable, "-c", PARENT, str(tmp_path), parent_mode, child_mode, CHILD],
        start_new_session=True,
    )
    tree = layout_batch.OwnedProcessTree(process)
    child = None
    try:
        deadline = time.monotonic() + 5
        while True:
            pid_file = tmp_path / "child-pid"
            if pid_file.exists():
                child = int(pid_file.read_text())
            if child in tree.known and (tmp_path / "child-ready").exists():
                break
            if time.monotonic() >= deadline:
                pytest.fail("child session was not ready and captured")
            time.sleep(0.01)
        result = tree.close(0.2)
        assert result["forced"] is forced
        assert result["main_cleanup"]["forced"] is (parent_mode != "normal")
        assert result["detached_cleanup"] == [child]
        assert len(result["detached_results"]) == 1
        detached = result["detached_results"][0]
        assert detached["pid"] == detached["pgid"] == child
        assert detached["forced"] is (child_mode != "normal")
        assert detached["signals"][0]["signal"] == "SIGINT"
        assert all(stage["sent"] for stage in detached["signals"])
        assert not layout_batch._group_alive(child)
        if child_mode == "normal":
            assert (tmp_path / "child-exited").read_text() == "clean"
        if parent_mode == "normal":
            assert process.returncode == 0
    finally:
        tree._done.set()
        tree._thread.join(timeout=2)
        for pid in (child, process.pid):
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait(timeout=5)


def test_group_exiting_before_escalation_signal_is_not_forced(monkeypatch):
    alive = iter((True, True, False))
    monkeypatch.setattr(layout_batch, "_process_row", lambda pid: None)
    monkeypatch.setattr(layout_batch, "_group_alive", lambda pgid: next(alive))

    def vanished(*args):
        raise ProcessLookupError()

    monkeypatch.setattr(layout_batch.os, "killpg", vanished)
    assert layout_batch._stop_group(123, 0.1, lambda: None)["forced"] is False
