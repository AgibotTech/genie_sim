# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Give the standalone engine one orderly, non-reentrant shutdown path."""

from contextlib import contextmanager
import os
import signal


@contextmanager
def shutdown_signals():
    """Unwind on the first termination signal; let resource cleanup finish.

    A worker launched by a background shell can inherit SIGINT=SIG_IGN.
    rclpy chains that disposition and only shuts down its own context, while
    Newton's independent C++ ROS context remains valid. Explicit Python
    handlers make both termination signals reach the entry point's existing
    KeyboardInterrupt/finally path. Install before rclpy initializes so its
    native handler chains to this handler rather than to inherited SIG_IGN.
    """
    previous = {}
    stopping = False

    def request_shutdown(signum, frame):
        nonlocal stopping
        first = not stopping
        stopping = True
        name = signal.Signals(signum).name
        message = (
            f"[genie_sim_engine_newton] {name}: entering orderly resource cleanup\n"
            if first
            else f"[genie_sim_engine_newton] additional {name} ignored while resources finish closing\n"
        )
        try:
            # A signal can interrupt a buffered logger write. Use one direct
            # stderr write so diagnostics cannot re-enter that stream's lock.
            os.write(2, message.encode())
        except OSError:
            pass
        if first:
            raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_shutdown)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
