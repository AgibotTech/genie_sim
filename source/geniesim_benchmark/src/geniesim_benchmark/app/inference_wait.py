# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in headless mitigation: avoid Kit updates during inference waits."""
import time


class InferenceWaitGate:
    def __init__(self, enabled, emit=print, sleep=time.sleep, clock=time.monotonic):
        self.enabled = enabled
        self.emit = emit
        self.sleep = sleep
        self.clock = clock
        self.started = None

    def wait(self, api):
        if not self.enabled:
            return False
        if api.physics_paused and not api.exit:
            if self.started is None:
                self.started = self.clock()
                self.emit("[inference-wait] Kit updates suspended")
            # Do not call Kit, timeline, world or camera APIs while waiting.
            # The benchmark worker owns inference and clears physics_paused.
            self.sleep(0.02)
            return True
        if self.started is not None:
            self.emit(f"[inference-wait] Kit updates released after {self.clock() - self.started:.3f}s; exit={api.exit}")
            self.started = None
        return False
