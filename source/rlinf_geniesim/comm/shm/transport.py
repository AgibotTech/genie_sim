# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""SHM server transport: owns the serve loop, dispatches verbs to the handler,
wakes via a SemPair instead of busy-polling."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Optional
import numpy as np

from rlinf_geniesim.comm import register_transport
from rlinf_geniesim.comm.base import SimHandler, SimTransport
from rlinf_geniesim.comm.shm.layout import (
    RequestPhase,
    ResponsePhase,
    BootstrapInfo,
    allocate_step_shm,
    allocate_ctrl_shm,
    allocate_frame_shm,
)
from rlinf_geniesim.comm.shm.result import make_shm_result_buffer
from rlinf_geniesim.comm.shm.sem import SemPair
from posix_ipc import BusyError, SignalError


@register_transport("shm")
class ShmTransport(SimTransport):
    def __init__(self, *, bootstrap_out: str, shm_suffix: str = "") -> None:
        self._bootstrap_out = bootstrap_out
        self._suffix = shm_suffix or f"rlinf_{os.getpid()}"
        self._step = self._ctrl = self._frame = None
        self._sem: Optional[SemPair] = None
        self._stop = False

    def serve(self, handler: SimHandler) -> None:
        dims, schema = handler.handshake()
        s = self._suffix
        self._step = allocate_step_shm(f"/step_{s}", dims)
        self._ctrl = allocate_ctrl_shm(f"/ctrl_{s}", dims)
        self._frame = allocate_frame_shm(f"/frame_{s}", dims)
        req_name, resp_name = f"/sem_req_{s}", f"/sem_resp_{s}"
        self._sem = SemPair(req_name, resp_name, create=True)
        out = make_shm_result_buffer(self._ctrl, self._frame)
        # Wire the handler's renderer (if any) to the SAME Frame SHM the
        # ResultBuffer views, and bind the scene schema — both required before
        # the first render. No-op for handlers without a separate renderer.
        handler.on_serve_start(frame_region=self._frame)
        # Warm up BEFORE announcing 'serving' so the first client step doesn't
        # absorb one-time bring-up costs. Physics/eval first (checker
        # construction + the engine's first-step initialisation — measured
        # >60s at 8 envs, which blew straight through the client's 60s step
        # timeout), then the render node's own bring-up (30-120s). After this,
        # 'serving' == BOTH sides initialized. Optional hooks (no-op for
        # handlers without them).
        _warm_phys = getattr(handler, "warmup_physics", None)
        if _warm_phys is not None:
            _warm_phys()
        _warm = getattr(handler, "warmup_render", None)
        if _warm is not None:
            _warm()

        boot = BootstrapInfo(
            sim_dims=dims,
            step_shm_name=self._step.name,
            ctrl_shm_name=self._ctrl.name,
            frame_shm_name=self._frame.name,
        ).as_dict()
        boot["sem_req_name"] = req_name
        boot["sem_resp_name"] = resp_name
        boot["schema"] = schema
        Path(self._bootstrap_out).write_text(json.dumps(boot))
        print(f"[shm] serving; bootstrap → {self._bootstrap_out}", flush=True)

        # Poll the renderer's liveness while idle so closing the Isaac Sim GUI
        # (render subprocess exit) tears the whole sim down instead of hanging
        # here forever. 1s timeout: no spin while stepping (a real request wakes
        # us immediately), one cheap wakeup/sec when idle.
        _alive = getattr(handler, "renderer_alive", None)
        while not self._stop:
            try:
                self._sem.wait_request(timeout=1.0)  # blocking, no spin
            except (BusyError, SignalError):
                # BusyError: 1s idle timeout. SignalError: Ctrl+C / SIGTERM
                # interrupted the wait — the signal handler set _stop. In both
                # cases fall through to the liveness/stop checks below so the
                # loop exits cleanly (no uncaught-exception traceback).
                if self._stop:
                    break
                if _alive is not None and not _alive():
                    print("[shm] render subprocess exited (GUI closed?) — " "stopping serve loop", flush=True)
                    break
                continue
            verb = int(self._step.field("request_phase"))
            tick = int(self._step.field("request_tick"))
            try:
                if verb == int(RequestPhase.STEP):
                    handler.step(self._ctrl.field("actions_buf"), out)
                    self._respond(ResponsePhase.STEP_DONE, tick)
                elif verb == int(RequestPhase.RESET):
                    handler.reset(
                        self._ctrl.field("reset_mask_buf"),
                        out,
                        seeds=self._ctrl.field("reset_seed_buf"),
                        instances=self._ctrl.field("reset_instance_buf"),
                        instructions=self._ctrl.field("reset_instruction_buf"),
                    )
                    self._respond(ResponsePhase.RESET_DONE, tick)
                elif verb == int(RequestPhase.SHUTDOWN):
                    handler.close()
                    self._respond(ResponsePhase.SHUTDOWN_ACK, tick)
                    self._sem.post_response()
                    break
                else:
                    self._respond(ResponsePhase.ERROR, tick, code=2, msg=f"unhandled verb {verb}")
            except Exception as e:  # noqa: BLE001
                self._respond(ResponsePhase.ERROR, tick, code=1, msg=repr(e))
            self._sem.post_response()  # wake the client

    def _respond(self, phase: ResponsePhase, tick: int, code: int = 0, msg: str = "") -> None:
        self._step.field("response_tick").fill(tick)
        self._step.field("error_code").fill(code)
        if msg:
            buf = self._step.field("error_msg")
            data = msg.encode("utf-8")[: buf.shape[0] - 1]
            buf[: len(data)] = np.frombuffer(data, dtype=np.uint8)
            buf[len(data)] = 0
        self._step.field("response_phase").fill(int(phase))  # written last

    def close(self) -> None:
        self._stop = True
        if self._sem is not None:
            self._sem.close(unlink=True)
            self._sem = None
        for shm in (self._step, self._ctrl, self._frame):
            if shm is not None:
                try:
                    shm.close(unlink=True)
                except Exception:
                    pass
        self._step = self._ctrl = self._frame = None


__all__ = ["ShmTransport"]
