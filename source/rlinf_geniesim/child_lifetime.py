# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Make a backend child process end when this server process dies.

A server killed with SIGKILL runs no ``close()``. Its children run in their own
session (so ``close()`` can ``killpg`` them), which ``geniesim post-train server
stop`` does not reach, and they would keep their GPU memory and shared memory.

``bind_to_server`` starts the child's wrapper (``/isaac-sim/python.sh`` or
``ros2 run``) under ``setpriv --pdeathsig KILL``. The node under that wrapper is
a separate fork and does not inherit the setting, so the environment asks it to
arm the same signal itself (``genie_sim_engine_core.common.parent_lifetime``).

Linux delivers the signal when the *thread* that forked the child exits, so the
child is bound only when it is launched from the main thread.
"""

from __future__ import annotations

import os
import shutil
import threading

#: Read by genie_sim_engine_core.common.parent_lifetime; keep the names in sync.
ENV_FLAG = "GENIESIM_EXIT_WITH_PARENT"
ENV_SERVER_PID = "GENIESIM_PARENT_SERVER_PID"


def bind_to_server(argv: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` that launch ``argv`` bound to this process's lifetime.

    Returns the inputs unchanged, with a warning, when ``setpriv`` is missing or
    the caller is not the main thread."""
    setpriv = shutil.which("setpriv")
    if setpriv is None:
        print("[child-lifetime] WARNING: setpriv not found; this child can outlive a killed server", flush=True)
        return list(argv), dict(env)
    if threading.current_thread() is not threading.main_thread():
        print(
            f"[child-lifetime] WARNING: {argv[0]} launched from thread "
            f"{threading.current_thread().name!r}; not binding it, because parent-death "
            f"signals follow the launching thread",
            flush=True,
        )
        return list(argv), dict(env)
    bound_env = dict(env)
    bound_env[ENV_FLAG] = "1"
    bound_env[ENV_SERVER_PID] = str(os.getpid())
    return [setpriv, "--pdeathsig", "KILL", "--", *argv], bound_env
