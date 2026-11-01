# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ZMQ client (trainer side) — same SimClient surface as the SHM client.

Addressing: pass ``address="tcp://host:5555"`` directly, or ``bootstrap_path``
pointing at the JSON the ZMQ transport wrote ({"address": ...}) — the latter
keeps VecEnv's config identical across backends on the same machine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np

from rlinf_geniesim.comm import register_client
from rlinf_geniesim.comm.base import SimClient, SimDims
from rlinf_geniesim.comm.zmq._wire import pack_request, unpack_result

# What the client would guess when the caller declared nothing.
#
# This is NOT sent by ``env.init()``: without a ``set_backends`` call the
# payload carries no engine section at all, so the server keeps its own
# configuration. Sending a guess would be rejected by the backends lock. On a
# server's very first init there is nothing to inherit and ``set_backends`` is
# mandatory.
#
# Keep this declaration aligned with ``init_task.backends_from_args``.
# ``num_envs`` is a workload property supplied by ``env.init()``.
DEFAULT_BACKENDS = {
    "physics": "isaac_fused",
    # The server does NOT read this key; the solver is decided by the backend
    # class. It still has to be DECLARED, and identically on both sides: the
    # backends lock (``sim_server.check_backends_locked``) compares every key
    # except ``num_envs``, so a key one caller sends and another omits makes
    # the two mutually exclusive against one server. ``init_task`` declares
    # it, so this declaration must match.
    "physics_solver": "physx",
    "physics_solver_substep": 4,
    "render": "isaac_fused",
    "device": "cuda:0",
}


@register_client("zmq")
class ZmqClient(SimClient):
    def __init__(self, *, address: str = "", bootstrap_path: str = "", timeout_s: float = 60.0) -> None:
        if not address:
            if not bootstrap_path:
                raise ValueError("ZmqClient needs address= or bootstrap_path=")
            boot = json.loads(Path(bootstrap_path).read_text())
            address = boot["address"]
        # A server binding 0.0.0.0 is reachable locally via 127.0.0.1.
        self._address = address.replace("//0.0.0.0:", "//127.0.0.1:")
        self._timeout_ms = int(float(timeout_s) * 1000)
        import zmq  # noqa: PLC0415

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._make_sock()
        self._dims: Optional[SimDims] = None
        self._schema: dict = {}

    def _make_sock(self):
        s = self._ctx.socket(self._zmq.REQ)
        s.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        s.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        s.setsockopt(self._zmq.LINGER, 0)
        s.connect(self._address)
        return s

    def _roundtrip(self, verb: str, payload=None, need_images: bool = True) -> list[bytes]:
        try:
            self._sock.send_multipart(pack_request(verb, payload, need_images=need_images))
            frames = self._sock.recv_multipart()
        except self._zmq.error.Again:
            # REQ sockets wedge after a missed reply — rebuild for the caller.
            self._sock.close(0)
            self._sock = self._make_sock()
            raise TimeoutError(f"zmq {verb} timed out after {self._timeout_ms}ms " f"({self._address})")
        status = json.loads(frames[0].decode())
        if not status.get("ok"):
            raise RuntimeError(f"sim error[{verb}]: {status.get('err')}")
        return frames[1:]

    # ------------------------------------------------------------------

    def handshake(self):
        # handshake carries its data in the STATUS frame itself.
        try:
            self._sock.send_multipart(pack_request("handshake", None))
            reply = json.loads(self._sock.recv_multipart()[0].decode())
        except self._zmq.error.Again:
            self._sock.close(0)
            self._sock = self._make_sock()
            raise TimeoutError(f"zmq handshake timed out ({self._address})")
        if not reply.get("ok"):
            raise RuntimeError(f"sim error[handshake]: {reply.get('err')}")
        self._dims = SimDims(**reply["dims"])
        self._schema = reply["schema"]
        return self._dims, self._schema

    def task_metadata(self, sub_task_name: Optional[str] = None) -> dict:
        """Query task and suite metadata without requiring an active handler.

        When sub_task_name is supplied, request its switchable instance IDs using
        the declared physics backend. Convert JSON instance/instruction keys back
        to integers. Unnamed queries avoid the backend's superset scale gate."""
        # ``physics``: which instances are switchable depends on the ENGINE,
        # not on the task alone -- the superset scale gate that decides it
        # describes mujoco-warp and does not apply to PhysX. The server cannot
        # work that out by itself at this point: ``set_backends`` is local
        # client state that sends nothing, and callers query metadata BEFORE
        # ``init`` (RLinf's geniesim_env does set_backends -> task_metadata ->
        # init), so no engine declaration has reached the server yet. This side
        # is the only one that knows, so it says so.
        #
        # Sent ONLY when set_backends was actually called. Never declared means
        # the caller has stated nothing, and the server answers conservatively
        # on an absent key -- which is both today's behaviour and what an older
        # client looks like to a newer server. ``get_backends()`` is not used
        # here precisely because it substitutes DEFAULT_BACKENDS for "unset",
        # erasing that distinction.
        payload = {"sub_task_name": sub_task_name} if sub_task_name else None
        declared = getattr(self, "_backends", None)
        if payload is not None and declared:
            physics = str(declared.get("physics") or "").strip()
            if physics:
                payload["physics"] = physics
        try:
            self._sock.send_multipart(pack_request("task_metadata", payload))
            reply = json.loads(self._sock.recv_multipart()[0].decode())
        except self._zmq.error.Again:
            self._sock.close(0)
            self._sock = self._make_sock()
            raise TimeoutError(f"zmq task_metadata timed out ({self._address})")
        if not reply.get("ok"):
            raise RuntimeError(f"sim error[task_metadata]: {reply.get('err')}")
        md = dict(reply["task_metadata"])
        sub_tasks = md.get("sub_tasks")
        if isinstance(sub_tasks, dict):
            fixed_sub_tasks = {}
            for name, entry in sub_tasks.items():
                entry = dict(entry)
                ibi = entry.get("instructions_by_instance")
                if isinstance(ibi, dict):
                    entry["instructions_by_instance"] = {int(k): v for k, v in ibi.items()}
                fixed_sub_tasks[name] = entry
            md["sub_tasks"] = fixed_sub_tasks
        return md

    def randomization_readback(self) -> dict:
        """See ``SimClient.randomization_readback``. One-shot query, same
        request/reply shape as ``handshake()``/``task_metadata()`` above (one
        send, one recv, no payload -- the server answers from whatever
        config it currently holds, there is nothing for this client to
        tell it)."""
        try:
            self._sock.send_multipart(pack_request("randomization_readback", None))
            reply = json.loads(self._sock.recv_multipart()[0].decode())
        except self._zmq.error.Again:
            self._sock.close(0)
            self._sock = self._make_sock()
            raise TimeoutError(f"zmq randomization_readback timed out ({self._address})")
        if not reply.get("ok"):
            raise RuntimeError(f"sim error[randomization_readback]: {reply.get('err')}")
        return dict(reply["randomization_readback"])

    def set_backends(self, backends: dict) -> None:
        """See ``SimClient.set_backends``. Local client state; sends nothing."""
        if not isinstance(backends, Mapping):
            raise TypeError(f"set_backends expects a mapping, got {type(backends).__name__}")
        if "num_envs" in backends:
            raise ValueError(
                "set_backends must not carry num_envs: parallelism is a "
                "workload property, given in env.init({..., 'num_envs': N}). "
                "The server keeps it under backends for historical reasons "
                "and env.init merges it in when assembling the payload."
            )
        if not backends:
            raise ValueError(
                "set_backends({}) is meaningless — an empty declaration would "
                "reach the server with no 'physics' key and come back as the "
                "confusing sim error[init]: 'physics'. To ask for the built-in "
                "defaults, don't call set_backends() at all."
            )
        self._backends = dict(backends)

    def get_backends(self) -> dict:
        # Return a copy, so a caller mutating the result cannot affect this.
        # `is not None` rather than truthiness: once set_backends has been
        # called the value must be reflected as given, even if it is falsy.
        # "never set" and "set to something falsy" are different.
        backends = getattr(self, "_backends", None)
        return dict(backends if backends is not None else DEFAULT_BACKENDS)

    def init(self, task_init: dict, timeout_s: float = 300.0):
        """Replace the server task and return the new ``(dims, schema)``.

        The server rebuilds its handler while retaining this connection. Callers
        must replace cached dimensions and schema with the returned values.
        The wire verb remains ``reload`` for compatibility with older servers;
        current servers accept both ``init`` and ``reload``.

        ``timeout_s`` applies to the rebuild response. The normal receive timeout
        is restored afterward; a timeout recreates the REQ socket so it can send
        another request.
        """
        ms = int(float(timeout_s) * 1000)
        try:
            self._sock.setsockopt(self._zmq.RCVTIMEO, ms)
            self._sock.send_multipart(pack_request("reload", task_init))
            frames = self._sock.recv_multipart()
        except self._zmq.error.Again:
            # REQ sockets wedge after a missed reply — rebuild for the
            # caller, same as _roundtrip's timeout handling.
            self._sock.close(0)
            self._sock = self._make_sock()
            raise TimeoutError(f"zmq init timed out after {timeout_s}s ({self._address})")
        else:
            self._sock.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        reply = json.loads(frames[0].decode())
        if not reply.get("ok"):
            raise RuntimeError(f"sim error[init]: {reply.get('err')}")
        self._dims = SimDims(**reply["dims"])
        self._schema = reply["schema"]
        return self._dims, self._schema

    def reload(self, task_init: dict, timeout_s: float = 300.0):
        """Compatibility alias for :meth:`init`."""
        return self.init(task_init, timeout_s=timeout_s)

    def step(self, actions: np.ndarray, need_images=True):
        """``need_images`` may be a bool or a per-env mask of length N.

        With a mask the server encodes and returns only those env rows; the
        rest are zero and ``result.image_envs`` reports which are real.
        """
        frames = self._roundtrip("step", np.asarray(actions, dtype=np.float32), need_images=need_images)
        return unpack_result(frames[0], frames[1:])

    def step_chunk(self, actions: np.ndarray, need_images=True):
        """Run actions shaped (chunk, N, A) in one request.

        Only the final substep returns images and large state arrays. The reply's
        chunk_scalars contains rewards, terminations, truncations and success
        with shape (4, chunk, N). The server does not auto-reset midway through
        the chunk."""
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"step_chunk actions must be (chunk, N, A), got {a.shape}")
        frames = self._roundtrip("step_chunk", a, need_images=need_images)
        return unpack_result(frames[0], frames[1:])

    def reset(self, mask: np.ndarray, seeds=None, instances=None, instructions=None):
        """Reset the masked envs; every one of them comes back ready to act
        (see ``SimHandler.reset``)."""
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        sd = np.full_like(m, -1.0) if seeds is None else np.asarray(seeds, dtype=np.float64).reshape(-1)
        it = np.full_like(m, -1.0) if instances is None else np.asarray(instances, dtype=np.float64).reshape(-1)
        ib = np.full_like(m, -1.0) if instructions is None else np.asarray(instructions, dtype=np.float64).reshape(-1)
        # payload (4, N): [mask; seeds; instance ids; instruction ids]
        # (float64 holds int values exactly up to 2^53).
        frames = self._roundtrip("reset", np.stack([m, sd, it, ib]))
        return unpack_result(frames[0], frames[1:])

    def shutdown(self) -> None:
        try:
            self._roundtrip("shutdown")
        except Exception:  # noqa: BLE001
            pass
        self._sock.close(0)

    def close(self) -> None:
        """Detach WITHOUT stopping the server (SimClient ABC). Use
        ``shutdown()`` to also stop the sim."""
        self._sock.close(0)


__all__ = ["ZmqClient"]
