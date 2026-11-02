# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Owned observations copied only from a completed physics frame.

The engine owns source/epoch identity and calls ``on_tick`` on its physics
thread. ROS callbacks only validate plain requests and enqueue them. This local
implementation adapts the on-demand capture contract from the historical Newton
IPC helper; body readback already performs the authored-root/WXYZ conversion.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
import uuid

import numpy as np


class SnapshotCaptureError(RuntimeError):
    """An owner readback failure, with terminal replies ready for transport."""

    def __init__(self, cause, responses):
        super().__init__(f"snapshot capture failed: {cause}")
        self.responses = responses


def capture_state(engine, request_id: str, source_id: str | None = None) -> dict:
    """Copy one committed engine frame; call only on the physics owner thread.

    ``bodies`` contains world authored-root poses in xyz+wxyz order. USD matrices
    use row vectors; scoring matrices use column vectors. Raw SDK body arrays
    belong to the engine's separate lossless archive and are not converted here.
    """
    identity = (engine._source_id, int(engine._epoch), int(engine._tick), float(engine._sim_time))
    if source_id is not None and source_id != identity[0]:
        raise ValueError("snapshot source identity does not match engine")
    if not math.isfinite(identity[3]):
        raise ValueError("non-finite snapshot simulation time")
    state = engine._state_0
    q = np.array(state.joint_q.numpy(), dtype=float, copy=True)
    particles = np.array(state.particle_q.numpy(), dtype=float, copy=True)
    # This method applies mesh-to-authored-root offsets and converts xyzw to
    # wxyz already. A second composition would move/rotate authored props twice.
    poses, paths = engine.get_body_transforms()
    poses = np.array(poses, dtype=float, copy=True)
    paths = list(paths)
    if q.ndim != 1 or particles.ndim != 2 or particles.shape[1] != 3:
        raise ValueError("invalid snapshot joint or particle array shape")
    if poses.shape != (len(paths), 7):
        raise ValueError("snapshot body paths and positions differ")
    if not all(np.isfinite(array).all() for array in (q, poses, particles)):
        raise ValueError("non-finite physics snapshot")
    joints = {}
    for name in engine._joint_names:
        scalar = engine._scalar.get(name)
        if scalar is not None:
            index = scalar.q_start
            if not 0 <= index < len(q):
                raise ValueError(f"snapshot joint index is out of range: {name}")
            value = float(q[index])
        elif name in engine._static_joint_q:
            value = float(engine._static_joint_q[name])
        else:
            raise ValueError(f"snapshot missing joint coordinate: {name}")
        if not math.isfinite(value):
            raise ValueError(f"non-finite snapshot joint: {name}")
        joints[name] = value
    bodies = {}
    for path, pose in zip(paths, poses):
        if path in bodies:
            raise ValueError(f"duplicate snapshot body path: {path}")
        bodies[path] = pose.tolist()
    deformables = []
    for mesh_id, obj in enumerate(engine._scene.objects):
        if obj["kind"] not in ("cloth_mesh", "tet_mesh"):
            continue
        path = obj["mesh_prim_path"]
        bounds = engine._scene._ranges.get(path)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"snapshot missing particle range: {path}")
        start, count = bounds
        if (
            not isinstance(start, (int, np.integer))
            or isinstance(start, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or isinstance(count, (bool, np.bool_))
            or start < 0
            or count <= 0
            or start + count > len(particles)
        ):
            raise ValueError(f"invalid snapshot particle range: {path}")
        deformables.append(
            {
                "mesh_id": mesh_id,
                "name": obj["name"],
                "prim_path": path,
                "points": particles[start : start + count].tolist(),
            }
        )
    if state is not engine._state_0 or identity != (
        engine._source_id,
        int(engine._epoch),
        int(engine._tick),
        float(engine._sim_time),
    ):
        raise ValueError("snapshot committed frame changed during capture")
    return {
        "version": 1,
        "snapshot_id": str(uuid.uuid4()),
        "request_id": request_id,
        "source_id": identity[0],
        "epoch": identity[1],
        "tick": identity[2],
        "cloth_tick": identity[2],
        "sim_time": identity[3],
        "captured_wall_time_ns": time.time_ns(),
        "pose_schema": "xyz+wxyz",
        "matrix_conventions": {"usd": "row-vector", "scoring": "column-vector"},
        "joints": joints,
        "bodies": bodies,
        "deformables": deformables,
    }


class SnapshotRequests:
    """Thread-safe plain request queue with an immutable per-epoch result cache.

    Completed snapshots remain available for the full epoch, including after a
    failure/close. A reset terminalizes pending requests with their old identity,
    then isolates all future request IDs in the new epoch. Only ``on_tick`` reads
    engine state, so initial-state and paused-state requests use the same owner
    thread boundary as ordinary completed ticks.
    """

    def __init__(self, engine):
        self.engine = engine
        self.source_id = engine._source_id
        self.epoch = int(engine._epoch)
        self._lock = threading.Lock()
        self._pending = {}
        self._payloads = {}
        self._cache = {}
        self._terminal_reason = None
        self._closed = False

    def _error(self, request_id, reason):
        return {
            "version": 1,
            "request_id": request_id,
            "source_id": self.source_id,
            "epoch": self.epoch,
            "error": str(reason),
        }

    def request(self, payload) -> dict | None:
        """Validate and enqueue owned JSON data; return an error or cached reply."""
        with self._lock:
            request_id = None
            try:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if not isinstance(payload, dict):
                    raise ValueError("snapshot request must be a JSON object")
                request_id = payload.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("snapshot request_id must be a nonempty string")
                if not isinstance(payload.get("source_id"), str) or not payload["source_id"]:
                    raise ValueError("snapshot source_id is required")
                epoch = payload.get("epoch")
                if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
                    raise ValueError("snapshot epoch must be a nonnegative integer")
                if payload["source_id"] != self.source_id:
                    raise ValueError("stale snapshot source_id")
                if epoch != self.epoch:
                    raise ValueError("stale snapshot epoch")
                # Canonical JSON both owns the payload and defines exact retry
                # identity, including any optional client correlation fields.
                canonical = json.dumps(payload, sort_keys=True, allow_nan=False)
            except (ValueError, TypeError, RecursionError) as exc:
                # Parser/encoder recursion is malformed client data. Keep this
                # boundary separate from owner capture and transport failures.
                safe_id = request_id if isinstance(request_id, str) else None
                return self._error(safe_id, exc)
            if request_id in self._payloads:
                if self._payloads[request_id] != canonical:
                    return self._error(request_id, "snapshot request_id payload conflict")
                return copy.deepcopy(self._cache.get(request_id))
            self._payloads[request_id] = canonical
            if self._terminal_reason is not None:
                response = self._error(request_id, self._terminal_reason)
                self._cache[request_id] = response
                return copy.deepcopy(response)
            self._pending[request_id] = None
            return None

    def on_tick(self) -> list[dict]:
        """Resolve every queued request from the current committed physics frame."""
        responses = []
        with self._lock:
            for request_id in list(self._pending):
                try:
                    if self.engine._source_id != self.source_id or self.engine._epoch != self.epoch:
                        raise ValueError("snapshot source/epoch changed without reset_epoch")
                    response = capture_state(self.engine, request_id, self.source_id)
                except Exception as exc:
                    # Invalid client requests never reach this owner boundary.
                    # A corrupt committed state or failed SDK readback must
                    # latch the engine, rather than silently permit another
                    # hardware transaction after returning an error response.
                    responses.extend(self._cancel(f"snapshot capture failed: {exc}"))
                    raise SnapshotCaptureError(exc, responses) from exc
                self._cache[request_id] = response
                responses.append(copy.deepcopy(response))
                del self._pending[request_id]
            self._pending.clear()
        return responses

    def _cancel(self, reason) -> list[dict]:
        self._terminal_reason = str(reason)
        responses = []
        for request_id in self._pending:
            response = self._error(request_id, self._terminal_reason)
            self._cache[request_id] = response
            responses.append(copy.deepcopy(response))
        self._pending.clear()
        return responses

    def cancel(self, reason) -> list[dict]:
        """Reject pending/new requests while preserving every completed result."""
        with self._lock:
            return self._cancel(reason)

    def reset_epoch(self, new_epoch: int) -> list[dict]:
        """Cancel old pending requests, then begin an isolated engine epoch."""
        with self._lock:
            if self._closed:
                raise RuntimeError("snapshot source is closed")
            if not isinstance(new_epoch, int) or isinstance(new_epoch, bool) or new_epoch <= self.epoch:
                raise ValueError("snapshot reset_epoch must advance the epoch")
            if self.engine._source_id != self.source_id or self.engine._epoch != new_epoch:
                raise ValueError("snapshot reset_epoch must match the committed engine identity")
            responses = self._cancel("snapshot request cancelled by epoch reset")
            self._payloads.clear()
            self._cache.clear()
            self.epoch = new_epoch
            self._terminal_reason = None
            return responses

    def close(self, reason="snapshot source closed") -> list[dict]:
        """Terminalize pending requests once without dropping completed snapshots."""
        with self._lock:
            if self._closed:
                return []
            self._closed = True
            return self._cancel(reason)


class ObservationSnapshotSource:
    """ROS transport attached exclusively to the engine's existing node."""

    def __init__(self, engine, node, topic: str, *, requests: SnapshotRequests | None = None):
        if node is None or not isinstance(topic, str) or not topic.startswith("/") or topic == "/":
            raise ValueError("snapshot source requires an existing ROS node and absolute topic prefix")
        from std_msgs.msg import String
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self.engine = engine
        self._requests = requests if requests is not None else SnapshotRequests(engine)
        if self._requests.engine is not engine:
            raise ValueError("snapshot queue belongs to another engine")
        self._node = node
        self._closed = False
        self._transport_lock = threading.RLock()
        # A renderer may finish GPU warmup after the physical snapshot was
        # captured. Retain the latest immutable state until its subscriber joins;
        # requests and action commands keep their existing volatile semantics.
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = node.create_publisher(String, topic + "/snapshot", qos)
        try:
            self.subscription = node.create_subscription(String, topic + "/request", self.request, 10)
        except BaseException:
            node.destroy_publisher(self.publisher)
            raise

    @property
    def source_id(self):
        return self._requests.source_id

    @property
    def epoch(self):
        return self._requests.epoch

    def _publish(self, responses):
        from std_msgs.msg import String

        with self._transport_lock:
            if self._closed:
                return responses
            try:
                for response in responses:
                    self.publisher.publish(String(data=json.dumps(response, allow_nan=False)))
            except Exception as exc:
                report = getattr(self.engine, "report_transport_failure", None)
                if report is not None:
                    report(exc)
                raise
        return responses

    def request(self, msg):
        with self._transport_lock:
            if self._closed:
                return None
            response = self._requests.request(msg.data)
            if response is not None:
                try:
                    self._publish([response])
                except Exception:
                    if getattr(self.engine, "report_transport_failure", None) is None:
                        raise
            return response

    def on_tick(self):
        try:
            return self._publish(self._requests.on_tick())
        except SnapshotCaptureError as exc:
            self._publish(exc.responses)
            raise

    def cancel(self, reason):
        return self._publish(self._requests.cancel(reason))

    def reset_epoch(self, new_epoch):
        return self._publish(self._requests.reset_epoch(new_epoch))

    def close(self, reason="snapshot source closed"):
        with self._transport_lock:
            if self._closed:
                return []
            responses = self._requests.close(reason)
            try:
                self._publish(responses)
            finally:
                self._closed = True
                try:
                    self._node.destroy_subscription(self.subscription)
                finally:
                    self._node.destroy_publisher(self.publisher)
            return responses
