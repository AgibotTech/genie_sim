# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ZMQ server transport with deferred init/reload replies.

The REP socket permits one pending request. serve returns an init payload
without replying, allowing the caller to rebuild the handler while keeping
the socket alive. The next serve completes the pending request with the new
handshake; fail_pending_reload sends an error if construction fails.

serve_idle accepts metadata, init and shutdown before any handler exists.
Step and reset receive NO_ACTIVE_TASK until initialization succeeds.
SHM regions provide staging buffers and a renderer frame sink; the remote
client receives serialized copies through ZMQ."""

from __future__ import annotations

import json
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rlinf_geniesim.comm import register_transport
from rlinf_geniesim.comm.base import SimHandler, SimTransport
from rlinf_geniesim.comm.shm.layout import (
    allocate_step_shm,
    allocate_ctrl_shm,
    allocate_frame_shm,
)
from rlinf_geniesim.comm.shm.result import make_shm_result_buffer
from rlinf_geniesim.comm.zmq._wire import (
    NO_ACTIVE_TASK,
    pack_result,
    pack_result_no_images,
    pack_images_only,
    request_image_envs,
    request_needs_images,
    unpack_request,
)


_LOGGER = logging.getLogger(__name__)


@register_transport("zmq")
class ZmqTransport(SimTransport):
    def __init__(
        self, *, bind: str = "tcp://0.0.0.0:5555", jpeg_quality: int = 0, bootstrap_out: str = "", shm_suffix: str = ""
    ) -> None:
        self._bind = bind
        self._jpeg = int(jpeg_quality)
        self._bootstrap_out = bootstrap_out
        self._suffix = shm_suffix or f"rlinf_zmq_{os.getpid()}"
        self._stop = False
        # Reload state (see module docstring): the REP socket is created
        # once and RETAINED across ``serve()`` calls so a reload can answer
        # the client from a later call — never re-bound (EADDRINUSE) and
        # never dropped mid-reload (would orphan the client's in-flight REQ).
        self._sock = None
        self._pending_reload = False  # a reload reply is owed on next entry
        self._serve_gen = 0  # bumped per serve() call (see below)
        # Metadata is independent of the currently loaded handler.
        self._task_metadata_provider = None
        # Randomization readback (docs/RANDOMIZATION_READBACK_API.md): set by
        # sim_server.py's ``_attach_backends_lock`` right after this
        # transport is built (see ``set_randomization_readback_provider``
        # below). Unlike ``_task_metadata_provider`` this one DOES read
        # "what's currently active" — it is closed over the same ``_active``
        # state ``_init_validator`` maintains, so it lives in that same
        # attach function rather than a standalone one.
        self._randomization_readback_provider = None
        # init-payload validator injected by sim_server. Raising rejects the
        # init and keeps serving the current task. None means no validation.
        self._init_validator = None
        # Commit accepted settings only once the rebuilt handler is serving.
        self._init_committer = None

    def _bind_socket(self) -> bool:
        """Bind the retained REP socket if needed; return whether it was created.

        Both active and idle serving may be the first caller.
        """
        if self._sock is not None:
            return False
        import zmq  # noqa: PLC0415

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REP)
        self._sock.bind(self._bind)
        return True

    def set_task_metadata_provider(self, provider) -> None:
        """Install ``provider(sub_task_name: str | None) -> dict``.

        The provider is invoked for each metadata request in active or idle mode.
        It must remain independent of the loaded handler so it stays valid across
        task rebuilds.
        """
        self._task_metadata_provider = provider

    def set_randomization_readback_provider(self, provider) -> None:
        """Wire the ``randomization_readback`` verb to ``provider``:
        ``provider() -> dict``, no arguments (unlike
        ``set_task_metadata_provider``'s ``sub_task_name``-parameterized
        one — this query has nothing analogous to thread through, it is
        always "the whole current answer"). Called once by
        ``sim_server.py`` right after this transport is built (see
        ``_attach_backends_lock``).

        Safe to call whether or not ``serve()``/``serve_idle()`` has started
        yet (just stores the callable; the verb handler below invokes it
        lazily on each request)."""
        self._randomization_readback_provider = provider

    def _note_step_prof(self, t0: float, t1: float, t2: float, need_images: bool, blobs, n_img_envs: int = -1) -> None:
        """Log a server-side step breakdown at DEBUG roughly every 5 seconds:
        handler, packing, send.

        The handler profile covers only ``handler.step()``; a reply also pays
        ``pack_result`` (JPEG encoding) and ``send_multipart``, which are
        otherwise unmeasured. Frames with and without images are accumulated
        separately, since their costs differ by an order of magnitude.
        """
        p = getattr(self, "_step_prof", None)
        if p is None:
            p = self._step_prof = {
                # handler, pack, send, count, bytes, envs wanting images
                "t_log": t2,
                "img": [0.0, 0.0, 0.0, 0, 0, 0],
                "noimg": [0.0, 0.0, 0.0, 0, 0, 0],
            }
        row = p["img"] if need_images else p["noimg"]
        row[0] += t1 - t0
        row[1] += t2 - t1
        row[2] += time.monotonic() - t2
        row[3] += 1
        row[4] += sum(len(b) for b in blobs)
        # How many envs wanted an image this step. Rendering produces one
        # frame for the whole batch while fetching is per env, so the two
        # numbers together show the ratio. -1 means the caller did not report.
        if need_images and n_img_envs >= 0:
            row[5] += n_img_envs
        if t2 - p["t_log"] < 5.0:
            return
        parts = []
        for label, r in (("with images", p["img"]), ("no images", p["noimg"])):
            if not r[3]:
                continue
            n = r[3]
            envs = f" avg {r[5]/n:.1f} envs wanted images" if label == "with images" and r[5] else ""
            parts.append(
                f"{label} {n} steps: handler={r[0]/n*1000:.1f} "
                f"pack={r[1]/n*1000:.1f} send={r[2]/n*1000:.1f}ms "
                f"({r[4]/n/1e6:.2f}MB{envs})"
            )
            r[:] = [0.0, 0.0, 0.0, 0, 0, 0]
        if parts:
            _LOGGER.debug("[zmq-prof] %s", " | ".join(parts))
        p["t_log"] = t2

    def serve(self, handler: SimHandler) -> Optional[dict]:
        """Serve ``handler`` until shutdown or init/reload.

        Returns ``None`` on normal shutdown (the ``shutdown`` verb, or the
        render subprocess dying, or an external ``stop()``/``close()``) —
        same as before. Returns the received task_init **dict** when the
        ``init``/``reload`` verb arrives instead (same code path, see the
        module docstring) — the socket is kept open and the
        caller (``sim_server.py``) is expected to tear the old handler down,
        build a new one from that dict, and call ``serve(new_handler)``
        again on THIS SAME transport instance (see module docstring for why
        the deferred reply is safe on a strict REP socket).
        """
        import zmq  # noqa: PLC0415

        first_time = self._bind_socket()
        sock = self._sock

        # Private staging SHM: a fresh name per serve() generation so a
        # reload's reallocation (dims may differ; the old regions were
        # already close()d by the PREVIOUS call's finally block, but POSIX
        # shm names otherwise collide until this process exits) never races
        # a not-yet-unlinked segment. Generation 0 keeps the original
        # (pre-reload) naming untouched.
        s = self._suffix if self._serve_gen == 0 else f"{self._suffix}_g{self._serve_gen}"
        self._serve_gen += 1

        # On failure, release the regions this generation already allocated
        # rather than leaking them, then re-raise. ``_pending_reload`` is left
        # as it was, for the caller to settle via fail_pending_reload().
        regions: list = []
        try:
            dims, schema = handler.handshake()
            step_parameters = inspect.signature(handler.step).parameters
            supports_pre_copy = "pre_copy_callback" in step_parameters or any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in step_parameters.values()
            )
            step = allocate_step_shm(f"/step_{s}", dims)  # staging (private)
            regions.append(step)
            ctrl = allocate_ctrl_shm(f"/ctrl_{s}", dims)
            regions.append(ctrl)
            frame = allocate_frame_shm(f"/frame_{s}", dims)
            regions.append(frame)
            out = make_shm_result_buffer(ctrl, frame)
            handler.on_serve_start(frame_region=frame)
            # Physics warm-up before render warm-up: it advances physics one
            # step, so the render pushes post-step poses and the first frame
            # matches what the client first sees.
            _warm_phys = getattr(handler, "warmup_physics", None)
            if _warm_phys is not None:
                _warm_phys()
            _warm = getattr(handler, "warmup_render", None)
            if _warm is not None:
                _warm()

            from dataclasses import asdict  # noqa: PLC0415

            dims_dict = asdict(dims)
            schema_dict = schema if isinstance(schema, dict) else json.loads(schema)

            if first_time:
                if self._bootstrap_out:
                    Path(self._bootstrap_out).write_text(
                        json.dumps({"backend": "zmq", "address": self._bind, "jpeg_quality": self._jpeg})
                    )
                print(f"[zmq] serving; bootstrap → {self._bind} " f"(jpeg_quality={self._jpeg or 'off'})", flush=True)
            elif self._pending_reload:
                # This IS the boot-waited (handshake/on_serve_start/warmup
                # done above) new handler the previous serve() call's reload
                # request is still blocked on — answer it now, before any
                # new recv, so the REP socket's owed send happens exactly
                # once, in order.
                sock.send_multipart(
                    [
                        json.dumps(
                            {"ok": True, "dims": dims_dict, "schema": schema_dict, "jpeg_quality": self._jpeg}
                        ).encode()
                    ]
                )
                if self._init_committer is not None:
                    self._init_committer()
                self._pending_reload = False
                print("[zmq] reload complete; resuming serve", flush=True)
        except Exception:
            for region in regions:
                try:
                    region.close(unlink=True)
                except Exception:  # noqa: BLE001
                    pass
            raise

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        _alive = getattr(handler, "renderer_alive", None)
        reload_cfg: Optional[dict] = None
        try:
            while not self._stop:
                if not poller.poll(1000):  # 1s idle heartbeat
                    if _alive is not None and not _alive():
                        print("[zmq] render subprocess exited — stopping", flush=True)
                        break
                    continue
                frames = sock.recv_multipart()
                try:
                    verb, payload = unpack_request(frames)
                    if verb == "handshake":
                        sock.send_multipart(
                            [
                                json.dumps(
                                    {"ok": True, "dims": dims_dict, "schema": schema_dict, "jpeg_quality": self._jpeg}
                                ).encode()
                            ]
                        )
                    elif verb == "task_metadata":
                        # Metadata uses one JSON request and reply without rebuilding a handler.
                        if self._task_metadata_provider is None:
                            raise RuntimeError(
                                "task_metadata provider not set (see " "sim_server.py's _attach_task_metadata_provider)"
                            )
                        sub = (payload or {}).get("sub_task_name") if isinstance(payload, dict) else None
                        # ``physics``: the caller's engine declaration. The
                        # provider needs it because switchability depends on
                        # the engine, and nothing else on this connection has
                        # carried it yet (set_backends is client-local and
                        # init has not happened). Absent -> "" = not declared,
                        # which the provider answers conservatively; that is
                        # what an older client sends.
                        phys = (payload or {}).get("physics") or "" if isinstance(payload, dict) else ""
                        sock.send_multipart(
                            [
                                json.dumps(
                                    {
                                        "ok": True,
                                        "task_metadata": self._task_metadata_provider(sub, phys),
                                    }
                                ).encode()
                            ]
                        )
                    elif verb == "randomization_readback":
                        # Same one-request/one-reply shape as task_metadata
                        # above -- no payload (see set_randomization_readback_
                        # provider's docstring for why there's nothing to
                        # thread through).
                        if self._randomization_readback_provider is None:
                            raise RuntimeError(
                                "randomization_readback provider not set " "(see sim_server.py's _attach_backends_lock)"
                            )
                        sock.send_multipart(
                            [
                                json.dumps(
                                    {
                                        "ok": True,
                                        "randomization_readback": self._randomization_readback_provider(),
                                    }
                                ).encode()
                            ]
                        )
                    elif verb == "step":
                        actions = np.asarray(payload, dtype=np.float32)
                        if actions.shape != (dims.num_envs, dims.action_dim):
                            raise ValueError(
                                f"step payload must have shape {(dims.num_envs, dims.action_dim)}, "
                                f"got {actions.shape}"
                            )
                        need_images = request_needs_images(frames)
                        image_envs = request_image_envs(frames)
                        _p0 = time.monotonic()
                        pre_pack = {}

                        def _pre_copy_pack(_pp=pre_pack, _out=out):
                            _pp["meta"], _pp["blobs"] = pack_result_no_images(_out)

                        if supports_pre_copy:
                            handler.step(
                                actions,
                                out,
                                need_images=need_images,
                                pre_copy_callback=_pre_copy_pack,
                            )
                        else:
                            handler.step(actions, out, need_images=need_images)
                        _p1 = time.monotonic()
                        if pre_pack:
                            meta = pre_pack["meta"]
                            blobs = pre_pack["blobs"]
                            if need_images:
                                img_specs, img_blobs = pack_images_only(out, self._jpeg, image_envs=image_envs)
                                meta["fields"].extend(img_specs)
                                blobs.extend(img_blobs)
                            head = json.dumps(meta).encode()
                        else:
                            head, blobs = pack_result(
                                out, self._jpeg, include_images=need_images, image_envs=image_envs
                            )
                        _p2 = time.monotonic()
                        sock.send_multipart([json.dumps({"ok": True}).encode(), head, *blobs])
                        self._note_step_prof(
                            _p0,
                            _p1,
                            _p2,
                            need_images,
                            blobs,
                            (
                                len(image_envs)
                                if image_envs is not None
                                else (
                                    int(out.images.shape[0])
                                    if need_images and getattr(out, "images", None) is not None
                                    else -1
                                )
                            ),
                        )
                    elif verb == "step_chunk":
                        # Batch substeps to avoid one network round trip per action.
                        # Auto-reset remains client-owned because its seed, instance
                        # and instruction selection are local to the rollout cursor.
                        acts = np.asarray(payload, dtype=np.float32)
                        if acts.ndim != 3 or acts.shape[0] == 0 or acts.shape[1:] != (dims.num_envs, dims.action_dim):
                            raise ValueError(
                                f"step_chunk payload must be (chunk > 0, {dims.num_envs}, {dims.action_dim}), "
                                f"got shape {acts.shape}"
                            )
                        need_images = request_needs_images(frames)
                        image_envs = request_image_envs(frames)
                        nsub = int(acts.shape[0])
                        # Per-substep scalars: [rewards; terminations; truncations; success]
                        scal = np.zeros((4, nsub, int(acts.shape[1])), dtype=np.float32)
                        _p0 = time.monotonic()
                        for _i in range(nsub):
                            # Only the last substep renders. The substeps in
                            # between still step physics and still get
                            # scored as normal, they just don't produce an
                            # image -- the same meaning need_images has in
                            # the per-substep path.
                            handler.step(
                                np.ascontiguousarray(acts[_i]), out, need_images=(need_images and _i == nsub - 1)
                            )
                            scal[0, _i] = np.asarray(out.rewards).reshape(-1)
                            scal[1, _i] = np.asarray(out.terminations).reshape(-1)
                            scal[2, _i] = np.asarray(out.truncations).reshape(-1)
                            _succ = getattr(out, "success", None)
                            if _succ is not None:
                                scal[3, _i] = np.asarray(_succ).reshape(-1)
                        _p1 = time.monotonic()
                        head, blobs = pack_result(out, self._jpeg, include_images=need_images, image_envs=image_envs)
                        # The per-substep scalars are attached as one extra
                        # field. ``unpack_result`` does a name-based setattr
                        # over ``meta["fields"]``, so it doesn't need to know
                        # this field name ahead of time; an old client never
                        # sends this verb, so it never sees this field either.
                        meta = json.loads(head.decode())
                        meta["fields"].append(
                            {
                                "name": "chunk_scalars",
                                "enc": "raw",
                                "shape": list(scal.shape),
                                "dtype": str(scal.dtype),
                            }
                        )
                        blobs = [*blobs, scal.tobytes()]
                        _p2 = time.monotonic()
                        sock.send_multipart([json.dumps({"ok": True}).encode(), json.dumps(meta).encode(), *blobs])
                        self._note_step_prof(
                            _p0,
                            _p1,
                            _p2,
                            need_images,
                            blobs,
                            (
                                len(image_envs)
                                if image_envs is not None
                                else (
                                    int(out.images.shape[0])
                                    if need_images and getattr(out, "images", None) is not None
                                    else -1
                                )
                            ),
                        )
                    elif verb == "reset":
                        arr = np.asarray(payload, dtype=np.float64)
                        if not (
                            arr.shape == (dims.num_envs,)
                            or (arr.ndim == 2 and 1 <= arr.shape[0] <= 4 and arr.shape[1] == dims.num_envs)
                        ):
                            raise ValueError(
                                f"reset payload must be ({dims.num_envs},) or (1..4, {dims.num_envs}), "
                                f"got shape {arr.shape}"
                            )
                        seeds = insts = instrs = None
                        if arr.ndim == 2:  # (rows, N): [mask; seeds; instances; instructions]
                            mask = arr[0].astype(np.float32)
                            if arr.shape[0] > 1:
                                seeds = arr[1].astype(np.int64)
                            if arr.shape[0] > 2:
                                insts = arr[2].astype(np.int64)
                            if arr.shape[0] > 3:
                                instrs = arr[3].astype(np.int64)
                        else:  # legacy: mask only
                            mask = arr.astype(np.float32)
                        handler.reset(mask, out, seeds=seeds, instances=insts, instructions=instrs)
                        head, blobs = pack_result(out, self._jpeg)
                        sock.send_multipart([json.dumps({"ok": True}).encode(), head, *blobs])
                    elif verb in ("init", "reload"):
                        # Both request names use the same initialization path.
                        #
                        # This request is not answered here: the reply goes
                        # out from the next serve() once the rebuild finished,
                        # or from fail_pending_reload() if it failed.
                        if not isinstance(payload, dict):
                            sock.send_multipart(
                                [
                                    json.dumps(
                                        {"ok": False, "err": "init/reload requires a task_init " "dict payload"}
                                    ).encode()
                                ]
                            )
                            continue
                        if self._init_validator is not None:
                            try:
                                checked = self._init_validator(payload)
                            except Exception as _e:  # noqa: BLE001
                                sock.send_multipart([json.dumps({"ok": False, "err": str(_e)}).encode()])
                                continue
                            # The validator may return a completed payload
                            # (the client's minimal init merged onto the
                            # active configuration); None means validate only.
                            # The completed payload is the one that must go
                            # on, or the rebuild gate sees the raw one.
                            if isinstance(checked, dict):
                                payload = checked
                        self._pending_reload = True
                        reload_cfg = payload
                        break
                    elif verb == "shutdown":
                        handler.close()
                        sock.send_multipart([json.dumps({"ok": True}).encode()])
                        break
                    else:
                        sock.send_multipart([json.dumps({"ok": False, "err": f"unknown verb {verb}"}).encode()])
                except Exception as e:  # noqa: BLE001
                    # REP must always reply once per request.
                    try:
                        sock.send_multipart([json.dumps({"ok": False, "err": repr(e)}).encode()])
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            # These staging regions belong to this serve generation, even
            # when the embedding process continues after server shutdown.
            for region in (step, ctrl, frame):
                try:
                    region.close(unlink=True)
                except Exception:  # noqa: BLE001
                    pass
            if reload_cfg is None:
                # Real shutdown (shutdown verb / dead renderer / external
                # stop()) — close the socket for good, unlike a reload exit.
                try:
                    sock.close(0)
                except Exception:  # noqa: BLE001
                    pass
                self._sock = None
        return reload_cfg

    def serve_idle(self) -> Optional[dict]:
        """Serve metadata, init and shutdown while no handler is active.

        Return an accepted init payload with its reply pending for handler rebuild.
        Reject step/reset with NO_ACTIVE_TASK. Return None on shutdown. The socket
        remains owned by this transport across successive handler generations."""
        import zmq  # noqa: PLC0415

        first_time = self._bind_socket()
        if first_time:
            if self._bootstrap_out:
                Path(self._bootstrap_out).write_text(
                    json.dumps({"backend": "zmq", "address": self._bind, "jpeg_quality": self._jpeg})
                )
            print(f"[zmq] serving (idle — no active task) → {self._bind}", flush=True)
        sock = self._sock

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        shutting_down = False
        try:
            while not self._stop:
                if not poller.poll(1000):  # 1s idle heartbeat
                    continue
                frames = sock.recv_multipart()
                try:
                    verb, payload = unpack_request(frames)
                    if verb in ("init", "reload"):
                        # Same two names, same code, as serve()'s branch
                        # above — see that branch's comment for why.
                        if not isinstance(payload, dict):
                            sock.send_multipart(
                                [
                                    json.dumps(
                                        {"ok": False, "err": "init/reload requires a task_init " "dict payload"}
                                    ).encode()
                                ]
                            )
                            continue
                        if self._init_validator is not None:
                            try:
                                checked = self._init_validator(payload)
                            except Exception as _e:  # noqa: BLE001
                                sock.send_multipart([json.dumps({"ok": False, "err": str(_e)}).encode()])
                                continue
                            # Same contract as serve()'s init branch: the
                            # completed payload is the one that goes on.
                            if isinstance(checked, dict):
                                payload = checked
                        self._pending_reload = True
                        return payload
                    elif verb == "task_metadata":
                        # Metadata remains available before task initialization.
                        if self._task_metadata_provider is None:
                            raise RuntimeError(
                                "task_metadata provider not set (see " "sim_server.py's _attach_task_metadata_provider)"
                            )
                        sub = (payload or {}).get("sub_task_name") if isinstance(payload, dict) else None
                        # ``physics``: the caller's engine declaration. The
                        # provider needs it because switchability depends on
                        # the engine, and nothing else on this connection has
                        # carried it yet (set_backends is client-local and
                        # init has not happened). Absent -> "" = not declared,
                        # which the provider answers conservatively; that is
                        # what an older client sends.
                        phys = (payload or {}).get("physics") or "" if isinstance(payload, dict) else ""
                        sock.send_multipart(
                            [
                                json.dumps(
                                    {
                                        "ok": True,
                                        "task_metadata": self._task_metadata_provider(sub, phys),
                                    }
                                ).encode()
                            ]
                        )
                    elif verb == "randomization_readback":
                        # Answerable with NO active handler too, same reason
                        # as task_metadata above: it reads _active["cfg"]
                        # (the last ACCEPTED init payload, which is set
                        # before the handler rebuild even attempts), not the
                        # handler itself.
                        if self._randomization_readback_provider is None:
                            raise RuntimeError(
                                "randomization_readback provider not set " "(see sim_server.py's _attach_backends_lock)"
                            )
                        sock.send_multipart(
                            [
                                json.dumps(
                                    {
                                        "ok": True,
                                        "randomization_readback": self._randomization_readback_provider(),
                                    }
                                ).encode()
                            ]
                        )
                    elif verb == "shutdown":
                        sock.send_multipart([json.dumps({"ok": True}).encode()])
                        shutting_down = True
                        break
                    else:
                        # handshake/step/reset/anything else: no active task.
                        sock.send_multipart([json.dumps({"ok": False, "err": NO_ACTIVE_TASK}).encode()])
                except Exception as e:  # noqa: BLE001
                    try:
                        sock.send_multipart([json.dumps({"ok": False, "err": repr(e)}).encode()])
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            if shutting_down or self._stop:
                try:
                    sock.close(0)
                except Exception:  # noqa: BLE001
                    pass
                self._sock = None
        return None

    def fail_pending_reload(self, error_msg: str) -> None:
        """Send the error reply owed after a failed handler rebuild.

        Do nothing when no reply is pending or no socket exists.
        """
        if not self._pending_reload or self._sock is None:
            return
        try:
            self._sock.send_multipart([json.dumps({"ok": False, "err": str(error_msg)}).encode()])
        except Exception:  # noqa: BLE001
            pass
        self._pending_reload = False

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        """SimTransport ABC: request the serve loop to exit."""
        self._stop = True


__all__ = ["ZmqTransport"]
