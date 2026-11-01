# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import threading
import time
import traceback


class MainLoopWatchdog:
    """Hard-exit the process when one main-loop world.step() never returns.

    world.step() can block forever inside Kit's native app.update(). No Python
    runs on the main thread then, so the fatal_error / shutdown path in app.py
    never executes and the process idles until something external kills it.
    This thread notices the stalled step, dumps every Python thread's stack and
    calls os._exit(), the only exit that does not need the main thread.
    The main thread's stack ends at the Python call that entered native code;
    locating the native frame needs py-spy --native or gdb before the exit.
    """

    def __init__(
        self,
        timeout_s,
        *,
        exit_code=1,
        poll_s=1.0,
        clock=time.monotonic,
        exit_fn=os._exit,
        stream=None,
    ):
        self._timeout_s = float(timeout_s)
        self._exit_code = int(exit_code)
        self._poll_s = float(poll_s)
        self._clock = clock
        self._exit_fn = exit_fn
        self._stream = stream
        self._lock = threading.Lock()
        self._step_started_at = None
        self._stopped = threading.Event()
        self._thread = None

    @property
    def enabled(self):
        return self._timeout_s > 0

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="geniesim-main-loop-watchdog", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_s * 2)

    def step_started(self):
        with self._lock:
            self._step_started_at = self._clock()

    def step_finished(self):
        with self._lock:
            self._step_started_at = None

    def stalled_for(self):
        """Seconds the in-flight step has been running, or None between steps."""
        with self._lock:
            started_at = self._step_started_at
        return None if started_at is None else self._clock() - started_at

    def check(self):
        """Exit the process if the in-flight step exceeded the timeout."""
        stalled_for = self.stalled_for()
        if stalled_for is None or stalled_for < self._timeout_s:
            return False
        stream = self._stream or sys.stderr
        print(
            f"[MainLoopWatchdog] world.step() has not returned for {stalled_for:.0f}s "
            f"(limit {self._timeout_s:.0f}s); dumping threads and exiting with code {self._exit_code}",
            file=stream,
            flush=True,
        )
        try:
            _dump_threads(stream)
        except Exception as error:
            print(f"[MainLoopWatchdog] thread dump failed: {error!r}", file=stream, flush=True)
        try:
            sys.stdout.flush()
            stream.flush()
        except Exception:
            pass
        self._exit_fn(self._exit_code)
        return True

    def _run(self):
        while not self._stopped.wait(self._poll_s):
            if self.check():
                return


def _dump_threads(stream):
    # faulthandler needs a real file descriptor; Kit replaces sys.stderr with
    # its own writer, so format the frames through the stream instead.
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        print(f"Thread {names.get(ident, '?')} ({ident:#x}), most recent call last:", file=stream)
        print("".join(traceback.format_stack(frame)), file=stream)
    stream.flush()
