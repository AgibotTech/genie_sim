# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# comm/shm/client.py
"""SHM trainer-side client: blocking step/reset over SHM + SemPair."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np

from rlinf_geniesim.comm import register_client
from rlinf_geniesim.comm.base import SimClient, SimDims, ResultBuffer
from rlinf_geniesim.comm.shm.layout import (
    RequestPhase,
    ResponsePhase,
    attach_step_shm,
    attach_ctrl_shm,
    attach_frame_shm,
)
from rlinf_geniesim.comm.shm.result import make_shm_result_buffer
from rlinf_geniesim.comm.shm.sem import SemPair


@register_client("shm")
class ShmClient(SimClient):
    def __init__(self, *, bootstrap_path: str, timeout_s: float = 30.0) -> None:
        boot = json.loads(Path(bootstrap_path).read_text())
        self._dims = SimDims(**boot["sim_dims"])
        self._schema = boot.get("schema", {})
        self._timeout = timeout_s
        self._step = attach_step_shm(boot["step_shm_name"], self._dims)
        self._ctrl = attach_ctrl_shm(boot["ctrl_shm_name"], self._dims)
        self._frame = attach_frame_shm(boot["frame_shm_name"], self._dims)
        self._sem = SemPair(boot["sem_req_name"], boot["sem_resp_name"], create=False)
        self._out = make_shm_result_buffer(self._ctrl, self._frame)
        self._tick = int(self._step.field("request_tick"))

    def handshake(self) -> tuple[SimDims, dict]:
        return self._dims, self._schema

    def init(self, task_init: dict, timeout_s: float = 300.0):
        """Reject task initialization because SHM dimensions are fixed at bootstrap.

        A task change requires new regions and client attachment. Use ZMQ for
        initialization over an existing connection.
        """
        raise NotImplementedError(
            "ShmClient.init() is unsupported by design: SHM's fixed-"
            "length layout is sized from dims at bootstrap, so an init "
            "that changes dims needs new SHM regions + a fresh bootstrap "
            "handshake — i.e. a process restart in substance. Use the zmq "
            "backend (comm.zmq.client.ZmqClient.init) for a live init, "
            "or restart the SHM server for a new task."
        )

    def reload(self, task_init: dict, timeout_s: float = 300.0):
        """Reject task reloads for the same fixed-layout constraint as ``init``."""
        raise NotImplementedError(
            "ShmClient.reload() is unsupported by design: SHM's fixed-"
            "length layout is sized from dims at bootstrap, so a reload "
            "that changes dims needs new SHM regions + a fresh bootstrap "
            "handshake — i.e. a process restart in substance. Use the zmq "
            "backend (comm.zmq.client.ZmqClient.reload) for a live reload, "
            "or restart the SHM server for a new task."
        )

    def task_metadata(self, sub_task_name: Optional[str] = None) -> dict:
        """Reject metadata queries; SHM has no variable-length JSON channel."""
        raise NotImplementedError(
            "ShmClient.task_metadata() is unsupported by design: SHM's "
            "fixed-length, dims-shaped layout (comm/shm/layout.py) has no "
            "channel for an arbitrary variable-length JSON metadata blob "
            "— same structural reason ShmClient.reload() is unsupported. "
            "Use the zmq backend (comm.zmq.client.ZmqClient.task_metadata) "
            "instead."
        )

    def randomization_readback(self) -> dict:
        """Unsupported by design — the same structural reason
        ``task_metadata()`` above is unsupported: SHM's control region is a
        FIXED-size, per-env numeric array layout sized from ``dims`` at
        bootstrap (``comm/shm/layout.py``'s FieldSpecs), with no verb
        dispatch or variable-length channel to carry an arbitrary
        {requested, resolved} JSON structure over. Returning ``{}`` here
        would be worse than raising: the caller could not tell "this
        transport cannot answer" apart from "the server genuinely has no
        randomization configured", which is exactly the ambiguity this
        readback interface exists to eliminate (see
        docs/RANDOMIZATION_READBACK_API.md sec. 4). Use the zmq backend
        (comm.zmq.client.ZmqClient.randomization_readback) for a live
        query."""
        raise NotImplementedError(
            "ShmClient.randomization_readback() is unsupported by design: "
            "SHM's fixed-length, dims-shaped layout (comm/shm/layout.py) has "
            "no channel for an arbitrary variable-length JSON reply — same "
            "structural reason ShmClient.task_metadata() is unsupported. Use "
            "the zmq backend (comm.zmq.client.ZmqClient.randomization_readback) "
            "instead."
        )

    def set_backends(self, backends: dict) -> None:
        raise NotImplementedError(
            "shm transport has no init/task_metadata verb (its control block is "
            "sized from dims at startup), so backends cannot be declared at "
            "runtime — use the zmq transport"
        )

    def _wait_response(self, tick: int) -> None:
        deadline = time.monotonic() + self._timeout
        while True:
            self._sem.wait_response(timeout=max(0.0, deadline - time.monotonic()))
            if int(self._step.field("response_tick")) == tick:
                break
        rphase = int(self._step.field("response_phase"))
        if rphase == int(ResponsePhase.ERROR):
            n = int(self._step.field("error_code"))
            raw = bytes(self._step.field("error_msg"))
            msg = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
            raise RuntimeError(f"sim error[{n}]: {msg}")

    def _await_pending(self) -> None:
        if self._step is None:
            raise RuntimeError("SHM client is closed")
        self._tick = int(self._step.field("request_tick"))
        if int(self._step.field("response_tick")) != self._tick:
            # A timed-out request still owns the shared input buffers, even
            # if another client has since attached to this server.
            self._wait_response(self._tick)

    def _roundtrip(self, phase: RequestPhase) -> ResultBuffer:
        self._await_pending()
        self._tick += 1
        self._step.field("request_tick").fill(self._tick)
        self._step.field("request_phase").fill(int(phase))  # written last
        self._sem.post_request()
        self._wait_response(self._tick)
        return self._out

    def step(self, actions: np.ndarray, need_images: bool = True) -> ResultBuffer:
        # The renderer writes directly to Frame SHM; need_images is ignored.
        self._await_pending()
        self._ctrl.field("actions_buf")[:] = actions
        return self._roundtrip(RequestPhase.STEP)

    def reset(self, mask: np.ndarray, seeds=None, instances=None, instructions=None) -> ResultBuffer:
        """Reset the masked envs; every one of them comes back ready to act
        (see ``SimHandler.reset``)."""
        self._await_pending()
        self._ctrl.field("reset_mask_buf")[:] = mask
        sd = self._ctrl.field("reset_seed_buf")
        sd[:] = -1 if seeds is None else np.asarray(seeds, dtype=np.int64)
        it = self._ctrl.field("reset_instance_buf")
        it[:] = -1 if instances is None else np.asarray(instances, dtype=np.int64)
        ib = self._ctrl.field("reset_instruction_buf")
        ib[:] = -1 if instructions is None else np.asarray(instructions, dtype=np.int64)
        return self._roundtrip(RequestPhase.RESET)

    def close(self) -> None:
        """Detach this client without stopping the server or its pending request."""
        if self._sem is not None:
            self._sem.close(unlink=False)
            self._sem = None
        self._out = None
        for shm in (self._step, self._ctrl, self._frame):
            try:
                shm.close(unlink=False)
            except Exception:
                pass
        self._step = self._ctrl = self._frame = None


__all__ = ["ShmClient"]
