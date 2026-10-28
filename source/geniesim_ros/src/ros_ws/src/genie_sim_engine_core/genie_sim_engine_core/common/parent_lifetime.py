# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in: end this node when the server that launched it is gone.

The counterpart of ``rlinf_geniesim.child_lifetime``, which describes the launch
chain. Nodes started without its environment variables are unaffected.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys

#: Set to "1" by rlinf_geniesim.child_lifetime; keep the names in sync.
ENV_FLAG = "GENIESIM_EXIT_WITH_PARENT"
ENV_SERVER_PID = "GENIESIM_PARENT_SERVER_PID"
_PR_SET_PDEATHSIG = 1


def exit_with_parent() -> bool:
    """Arm SIGKILL on parent death when the launcher asked for it; return whether armed.

    Exits immediately if the launching server already died before the signal
    was armed, which the kernel would otherwise never report."""
    if os.environ.get(ENV_FLAG) != "1" or not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(err)}")
    server_pid = int(os.environ.get(ENV_SERVER_PID, "0") or 0)
    if server_pid > 0 and not _alive(server_pid):
        print(f"[exit-with-parent] launching server {server_pid} is gone; exiting", flush=True)
        os._exit(1)
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
