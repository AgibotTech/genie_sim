# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Attach to the public committed-action and synchronized-observation services.

This client owns transport resources only. The caller starts the existing engine
and renderer and owns episode shutdown; this module never starts a physics world.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import threading
import time
from types import MappingProxyType
import uuid

import yaml


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _positive(value, name):
    if not _finite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _names(value):
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(name, str) and bool(name) for name in value)
        and len(set(value)) == len(value)
    )


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def packet_dict(value):
    """Return an independent JSON-compatible copy of an immutable packet."""
    if isinstance(value, Mapping):
        return {key: packet_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [packet_dict(item) for item in value]
    return value


@dataclass(frozen=True)
class IPCClientConfig:
    scene_yaml: Path
    launcher_yaml: Path
    action_topic: str
    observation_topic: str
    physics_hz: float
    cpu_steps_per_ipc: int | None

    @classmethod
    def from_files(cls, scene_yaml, launcher_yaml):
        scene_path = Path(scene_yaml).expanduser().resolve(strict=True)
        launcher_path = Path(launcher_yaml).expanduser().resolve(strict=True)
        scene = yaml.safe_load(scene_path.read_text())
        launcher = yaml.safe_load(launcher_path.read_text())
        try:
            settings = scene["libuipc"]
            hz = _positive(launcher["genie_sim_engine"]["ros__parameters"]["physics_hz"], "physics_hz")
            topics = [settings[name]["topic"] for name in ("action_sequence", "observation_snapshot")]
        except (KeyError, TypeError) as exc:
            raise ValueError("IPC client requires explicit launcher physics_hz and scene protocol topics") from exc
        if (
            any(
                not isinstance(topic, str) or not topic.startswith("/") or topic == "/" or topic.endswith("/")
                for topic in topics
            )
            or topics[0] == topics[1]
        ):
            raise ValueError("IPC topics must be distinct absolute prefixes without trailing slashes")
        cpu = settings.get("cpu_mujoco")
        ratio = None
        if cpu is not None:
            cpu_hz = _positive(cpu.get("physics_hz"), "cpu_mujoco.physics_hz")
            ratio = cpu.get("sync_substeps")
            if not _integer(ratio, 1) or not math.isclose(cpu_hz / hz, ratio, abs_tol=1e-8, rel_tol=0):
                raise ValueError("CPU barrier substeps must match the YAML CPU/IPC rates")
        return cls(scene_path, launcher_path, *topics, hz, ratio)


_IDENTITY_KEYS = ("source_id", "epoch", "request_id", "snapshot_id", "tick", "cloth_tick", "sim_time")


class IPCClient:
    """One serialized, fail-closed IPC session using engine-discovered names.

    ``step`` returns a correlated completion, never a claim that a target was
    reached. An uncertain send/capture poisons this object; create a fresh client
    and episode instead of replaying it. There is deliberately no reset method.
    """

    def __init__(self, config, *, ready_timeout_s=600.0, wait_timeout_s=120.0, transport=None, event=None):
        self.config = config
        self.ready_timeout_s = _positive(ready_timeout_s, "ready_timeout_s")
        self.wait_timeout_s = _positive(wait_timeout_s, "wait_timeout_s")
        self._transport = transport
        self._event = event if event is not None else lambda packet: None
        self._identity = None
        self._last_tick = 0
        self._last_sim_time = 0.0
        self._last_cpu_total = 0
        self._clock_offset = None
        self._failure = None
        self._started = self._closed = False
        self._lock = threading.Lock()

    @property
    def identity(self):
        self._healthy()
        return self._identity

    def _healthy(self):
        if self._failure is not None:
            raise RuntimeError(f"IPC session failed; close and create a fresh session: {self._failure}")
        if self._closed or not self._started:
            raise RuntimeError("IPC session is not started or is closed")

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent IPC client calls are unsupported")

    def _identity_message(self, packet):
        if (
            not isinstance(packet, dict)
            or type(packet.get("version")) is not int
            or packet.get("version") != 1
            or not isinstance(packet.get("source_id"), str)
            or not packet["source_id"]
            or not _integer(packet.get("epoch"))
            or not _integer(packet.get("tick"))
            or not _finite(packet.get("sim_time"))
            or packet["sim_time"] < 0
            or not _names(packet.get("joint_names"))
            or not _names(packet.get("reported_joint_names"))
            or not set(packet["joint_names"]).issubset(packet["reported_joint_names"])
            or not _finite(packet.get("physics_hz"))
            or not math.isclose(packet["physics_hz"], self.config.physics_hz, rel_tol=0, abs_tol=1e-9)
            or packet.get("status") not in ("ready", "failed", "closed")
            or packet.get("ready") is not (packet.get("status") == "ready")
        ):
            raise ValueError("invalid or incompatible IPC identity")
        if packet["status"] != "ready":
            raise RuntimeError(f"IPC source is {packet['status']}")
        if self._identity is not None:
            for key in ("source_id", "epoch", "joint_names", "reported_joint_names"):
                if packet_dict(self._identity[key]) != packet[key]:
                    raise RuntimeError(f"IPC source identity changed: {key}; start a fresh session")
        if self._clock_offset is None:
            self._clock_offset = packet["sim_time"] - packet["tick"] / self.config.physics_hz
        self._validate_clock(packet)
        self._identity = _freeze(packet)
        self._last_tick = max(self._last_tick, packet["tick"])
        self._last_sim_time = max(self._last_sim_time, packet["sim_time"])

    def _validate_clock(self, packet):
        expected = self._clock_offset + packet["tick"] / self.config.physics_hz
        if not math.isclose(packet["sim_time"], expected, abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError("IPC tick and simulation time differ from the latched epoch clock")

    def _next(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("IPC protocol wait timed out; command is never replayed")
        kind, packet = self._transport.next(remaining)
        self._received(kind, packet)
        return kind, packet

    def _received(self, kind, packet):
        self._event(
            {
                "event": "received",
                "channel": kind,
                "packet": packet_dict(packet),
                "raw": getattr(self._transport, "last_raw", None),
                "received_wall_time_ns": time.time_ns(),
            }
        )
        if kind == "identity":
            self._identity_message(packet)

    def _drain(self):
        # Consume already delivered terminal identity updates before publishing.
        # This is a nonblocking queue read, never another physics request.
        while True:
            try:
                kind, packet = self._transport.next(0)
            except TimeoutError:
                return
            except BaseException as exc:
                self._failure = exc
                raise
            try:
                self._received(kind, packet)
            except BaseException as exc:
                self._failure = exc
                raise

    def start(self):
        self._enter()
        try:
            if self._started or self._closed:
                raise RuntimeError("IPC client can only start once")
            if self._transport is None:
                from ._ipc_bridge import IPCROSBridge

                self._transport = IPCROSBridge(self.config, discovery_timeout_s=self.wait_timeout_s)
            self._transport.start()
            deadline = time.monotonic() + self.ready_timeout_s
            while self._identity is None:
                self._next(deadline)
            self._started = True
            return self
        except BaseException as exc:
            self._failure = exc
            if self._transport is not None:
                self._transport.close()
            self._closed = True
            raise
        finally:
            self._lock.release()

    def _send(self, kind, packet):
        self._event(
            {"event": "request", "channel": kind, "packet": packet_dict(packet), "sent_wall_time_ns": time.time_ns()}
        )
        self._transport.publish(kind, packet)

    def _request(self):
        return {
            "request_id": uuid.uuid4().hex,
            "source_id": self._identity["source_id"],
            "epoch": self._identity["epoch"],
        }

    @staticmethod
    def _matches(packet, request):
        return isinstance(packet, dict) and all(
            type(packet.get(key)) is type(value) and packet[key] == value for key, value in request.items()
        )

    def step(self, positions, duration_s, *, timeout_s=None, on_accepted=None):
        """Publish exactly one full named target and validate its committed counts."""
        self._enter()
        sent = False
        try:
            self._healthy()
            self._drain()
            names = self._identity["joint_names"]
            if (
                not isinstance(positions, Mapping)
                or set(positions) != set(names)
                or not all(_finite(v) for v in positions.values())
            ):
                raise ValueError("action requires the entire finite controlled joint set")
            duration = _positive(duration_s, "duration_s")
            ticks = duration * self.config.physics_hz
            if ticks < 1 or not math.isclose(ticks, round(ticks), abs_tol=1e-7, rel_tol=0):
                raise ValueError("action duration must contain an integer number of IPC steps")
            ticks = round(ticks)
            timeout = self.wait_timeout_s if timeout_s is None else _positive(timeout_s, "timeout_s")
            request = self._request()
            packet = dict(
                **request, names=list(names), positions=[float(positions[n]) for n in names], duration_s=duration
            )
            sent = True
            self._send("action", packet)
            accepted = False
            deadline = time.monotonic() + timeout
            while True:
                kind, response = self._next(deadline)
                if kind != "done" or not self._matches(response, request):
                    continue
                status = response.get("status")
                if status in ("rejected", "cancelled"):
                    raise RuntimeError(f"IPC action {status}: {response.get('error', '')}")
                if (
                    status not in ("accepted", "completed")
                    or response.get("ipc_steps") != ticks
                    or type(response.get("ipc_steps")) is not int
                ):
                    raise ValueError("invalid action acknowledgement status or IPC count")
                if status == "accepted":
                    if not accepted:
                        accepted = True
                        if on_accepted is not None:
                            on_accepted(_freeze(response))
                    continue
                if (
                    not _finite(response.get("executed_sim_time_s"))
                    or not math.isclose(
                        response["executed_sim_time_s"], ticks / self.config.physics_hz, abs_tol=1e-9, rel_tol=0
                    )
                    or not _integer(response.get("tick"), self._last_tick + ticks)
                    or not _finite(response.get("sim_time"))
                    or response["sim_time"] + 1e-9 < self._last_sim_time + ticks / self.config.physics_hz
                ):
                    raise ValueError("invalid committed IPC completion time or tick")
                if self.config.cpu_steps_per_ipc is not None:
                    cpu = response.get("cpu_steps")
                    if (
                        not _integer(cpu)
                        or cpu != ticks * self.config.cpu_steps_per_ipc
                        or not _integer(response.get("cpu_total_steps"), self._last_cpu_total + cpu)
                    ):
                        raise ValueError("invalid committed CPU completion counts")
                    self._last_cpu_total = response["cpu_total_steps"]
                self._validate_clock(response)
                self._last_tick = response["tick"]
                self._last_sim_time = response["sim_time"]
                if not accepted and on_accepted is not None:
                    on_accepted(_freeze(response))
                return _freeze(response)
        except BaseException as exc:
            if sent:
                self._failure = exc
            raise
        finally:
            self._lock.release()

    def _validate_snapshot(self, packet):
        if (
            type(packet.get("version")) is not int
            or packet.get("version") != 1
            or not isinstance(packet.get("snapshot_id"), str)
            or not packet["snapshot_id"]
            or not _integer(packet.get("tick"))
            or packet.get("cloth_tick") != packet["tick"]
            or type(packet.get("cloth_tick")) is not int
            or not _finite(packet.get("sim_time"))
            or packet["sim_time"] < 0
            or not _integer(packet.get("captured_wall_time_ns"), 1)
            or packet.get("pose_schema") != "xyz+wxyz"
        ):
            raise ValueError("invalid physical snapshot identity or pose schema")
        self._validate_clock(packet)
        joints, bodies, meshes = packet.get("joints"), packet.get("bodies"), packet.get("deformables")
        if (
            not isinstance(joints, dict)
            or set(joints) != set(self._identity["reported_joint_names"])
            or not all(_finite(v) for v in joints.values())
            or not isinstance(bodies, dict)
            or not bodies
        ):
            raise ValueError("invalid physical snapshot joints/bodies")
        for path, pose in bodies.items():
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or not isinstance(pose, (list, tuple))
                or len(pose) != 7
                or not all(_finite(v) for v in pose)
                or sum(float(v) ** 2 for v in pose[3:]) < 1e-12
            ):
                raise ValueError("invalid absolute authored-root WXYZ pose")
        if not isinstance(meshes, list):
            raise ValueError("invalid snapshot deformables")
        seen = set()
        for mesh in meshes:
            if not isinstance(mesh, dict):
                raise ValueError("invalid snapshot mesh")
            key = mesh.get("mesh_id")
            points = mesh.get("points")
            if (
                not _integer(key)
                or key in seen
                or not isinstance(mesh.get("name"), str)
                or not mesh["name"]
                or not isinstance(mesh.get("prim_path"), str)
                or not mesh["prim_path"].startswith("/")
                or not isinstance(points, list)
                or not points
                or any(
                    not isinstance(point, list) or len(point) != 3 or not all(_finite(v) for v in point)
                    for point in points
                )
            ):
                raise ValueError("invalid snapshot mesh identity/points")
            seen.add(key)

    def _validate_observation(self, response, state):
        if any(
            type(response.get(key)) is not type(state[key]) or response[key] != state[key] for key in _IDENTITY_KEYS
        ):
            raise ValueError("renderer identity conflicts with physical snapshot")
        if response.get("snapshot") != state:
            raise ValueError("renderer embedded snapshot conflicts with physical snapshot")
        images = response.get("images")
        reference = response.get("render_reference_time")
        if (
            not isinstance(images, dict)
            or set(images) != {"head", "hand_left", "hand_right"}
            or not _integer(response.get("render_frame"), 1)
            or not isinstance(reference, (list, tuple))
            or len(reference) != 2
            or not _integer(reference[0])
            or not _integer(reference[1], 1)
            or not _integer(response.get("render_complete_wall_time_ns"), 1)
        ):
            raise ValueError("partial or invalid synchronized renderer batch")
        for image in images.values():
            if (
                not isinstance(image, dict)
                or image.get("encoding") != "JPEG"
                or not _integer(image.get("width"), 1)
                or not _integer(image.get("height"), 1)
                or not isinstance(image.get("image_data"), str)
            ):
                raise ValueError("invalid synchronized JPEG")
            try:
                if not base64.b64decode(image["image_data"], validate=True):
                    raise ValueError("empty synchronized JPEG")
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid synchronized JPEG base64") from exc

    def _capture(self, render, timeout_s):
        self._enter()
        sent = False
        try:
            self._healthy()
            self._drain()
            timeout = self.wait_timeout_s if timeout_s is None else _positive(timeout_s, "timeout_s")
            deadline = time.monotonic() + timeout
            wait_ready = getattr(self._transport, "wait_observation_ready", None) if render else None
            if wait_ready is not None:
                try:
                    # The renderer advertises observations after RGB warmup.
                    # Capture new physical state afterward, while physics keeps running.
                    wait_ready(timeout, on_progress=self._drain)
                    self._drain()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("IPC observation readiness exhausted the capture timeout")
                except BaseException as exc:
                    self._failure = exc
                    raise
            request = self._request()
            sent = True
            self._send("snapshot", request)
            state = rendered = None
            while True:
                kind, packet = self._next(deadline)
                if kind not in ("snapshot", "observation") or not self._matches(packet, request):
                    continue
                if packet.get("error"):
                    raise RuntimeError(f"IPC capture failed: {packet['error']}")
                if kind == "snapshot":
                    self._validate_snapshot(packet)
                    if packet["tick"] < self._last_tick or packet["sim_time"] + 1e-9 < self._last_sim_time:
                        raise ValueError("snapshot physical clock moved backwards")
                    self._last_tick = packet["tick"]
                    self._last_sim_time = packet["sim_time"]
                    if state is not None and state != packet:
                        raise ValueError("conflicting duplicate physical snapshot")
                    state = packet
                else:
                    if rendered is not None and rendered != packet:
                        raise ValueError("conflicting duplicate rendered observation")
                    rendered = packet
                if state is not None and (not render or rendered is not None):
                    if render:
                        self._validate_observation(rendered, state)
                    return _freeze(rendered if render else state)
        except BaseException as exc:
            if sent:
                self._failure = exc
            raise
        finally:
            self._lock.release()

    def snapshot(self, *, timeout_s=None):
        return self._capture(False, timeout_s)

    def observe(self, *, timeout_s=None):
        return self._capture(True, timeout_s)

    def close(self):
        self._enter()
        try:
            if not self._closed and self._transport is not None:
                self._transport.close()
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
