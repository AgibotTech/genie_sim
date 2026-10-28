# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Process-group teardown extracted from RLinf's Newton subprocess path."""

import os
import signal
import subprocess
import time

_TEARDOWN_LADDER = (
    (signal.SIGINT, 10.0),  # let the engine shut ROS down cleanly
    (signal.SIGTERM, 5.0),
    (signal.SIGKILL, 5.0),
)


def _process_group_gone(proc: subprocess.Popen, pgid: int) -> bool:
    """Is EVERY process in ``pgid`` gone?

    ``proc`` is the ``ros2 run`` launcher, not the engine — the engine is its
    child. ``proc.poll()`` first so an already-exited launcher is reaped:
    an unreaped zombie still counts as a live group member for ``killpg``,
    which would make this report "alive" forever.
    """
    proc.poll()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Tear down the engine launcher AND the engine it spawned.

    We spawn with ``start_new_session=True``, so ``proc.pid`` is also the
    process-group id and one ``killpg`` reaches both.

    Escalation is driven by whether the whole process GROUP is gone, not by
    whether the launcher exited: ``proc`` is only ``ros2 run``, so waiting on
    it can succeed while the engine that ignored SIGTERM survives as an
    orphan, still holding the reset subscription and the state publisher and
    competing with the new engine for the same topics.
    """
    pgid = proc.pid
    for sig, grace in _TEARDOWN_LADDER:
        if _process_group_gone(proc, pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if _process_group_gone(proc, pgid):
                return
            time.sleep(0.05)
    if not _process_group_gone(proc, pgid):
        raise RuntimeError(f"Engine process group {pgid} survived teardown")
