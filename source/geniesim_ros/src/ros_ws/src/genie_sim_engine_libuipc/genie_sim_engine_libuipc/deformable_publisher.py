# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in live channel for IPC's solved deformable (cloth/tet) particle state.

Neither separate-process render backend this repo has (the C++ render_node's
``/tf_scene`` + ``/tf``, or the Isaac Sim native renderer's tf-driven Fabric
writes) carries anything but rigid-body transforms -- confirmed by reading
both, not assumed; see the migration plan, section 5.1.1. This is the
backend-agnostic deformable frame channel that section proposed: a topology
message sent once, and a per-tick flat numeric message every consumer can
decode without a custom .msg build, following the same "flat array reduces
serialization overhead for TF-shaped data" precedent already used by RLinf's
own ``/rlinf/pose_paths`` (latched JSON) + ``/rlinf/pose_flat`` (Float64
tick+poses) pair.

Wiring follows ``JointFeedbackConfig``/``IPCDriver.start_feedback`` exactly:
config is scene-YAML-only and opt-in (absence disables everything), and the
publisher attaches to the *existing* lifecycle node passed in by the engine
after ``start_listener()`` -- no second rclpy context, no owned executor
thread. Publishing needs no spin: only subscriptions/timers/services do.

Wire format
-----------
``<topic>_topology`` (``std_msgs/String``, JSON, published once, latched via
transient-local QoS so a late subscriber still gets it):

    {"generation": 1, "objects": [
        {"mesh_id": 0, "name": "tshirt",
         "prim_path": "/World/Objects/tshirt/Aligned/body", "n_particles": 7547}
    ]}

``<topic>_flat`` (``std_msgs/Float32MultiArray``, every publish tick):

    [generation, tick, mesh_id_0, n_0, x0,y0,z0, x1,y1,z1, ..., mesh_id_1, n_1, ...]

One message covers every declared deformable object in the scene (concatenated,
not one publish per object), matching how ``/rlinf/pose_flat`` covers every
body in one message rather than one per body. ``generation`` stays 1 for the
lifetime of a scene plugin instance (no scene rebuilds happen without a new
process in this codebase's IPC path today); it is on the wire now so a future
mid-run topology change (were one ever added) does not need a wire-format
version bump, only a generation increment a consumer already knows to watch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeformablePublishConfig:
    """Opt-in live deformable-state publish, configured exclusively by the scene YAML."""

    topic: str = "/rlinf/deformable"
    publish_hz: float = 0.0  # 0.0 -> every physics tick

    @classmethod
    def from_scene(cls, scene: dict) -> "DeformablePublishConfig | None":
        """Validate ``libuipc.deformable_publish``; absence disables the publisher entirely."""
        raw = (scene.get("libuipc") or {}).get("deformable_publish")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("libuipc.deformable_publish must be a mapping")
        if not raw.get("enabled", False):
            return None
        topic = raw.get("topic", "/rlinf/deformable")
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise ValueError("libuipc.deformable_publish.topic must be an absolute topic")
        publish_hz = float(raw.get("publish_hz", 0.0))
        if publish_hz < 0.0:
            raise ValueError("libuipc.deformable_publish.publish_hz must be >= 0 (0 = every tick)")
        return cls(topic=topic, publish_hz=publish_hz)


class DeformableParticlePublisher:
    """Publishes solved particle positions for every declared deformable object.

    ``objects`` is ``[(mesh_id, name, prim_path, n_particles), ...]`` — fixed at
    construction, one entry per ``cloth_mesh``/``tet_mesh`` scene-yaml object.
    Topology publishes once, on the first ``publish_frame`` call (lazily, so it
    rides the same "first call after the node exists" moment as the flat
    stream rather than needing a separate call site).
    """

    _GENERATION = 1

    def __init__(self, config: DeformablePublishConfig, objects: list[tuple[int, str, str, int]], logger: Any) -> None:
        self._config = config
        self._objects = list(objects)
        self._logger = logger
        self._node: Any = None
        self._topology_pub: Any = None
        self._flat_pub: Any = None
        self._topology_sent = False
        self._last_publish_sim_time: float | None = None

    def start(self, node: Any) -> None:
        """Attach publishers to the existing lifecycle node. Idempotent."""
        if self._node is not None or node is None:
            return
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32MultiArray, String

        self._node = node
        self._topology_pub = node.create_publisher(
            String,
            self._config.topic + "_topology",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latched: late subscribers still get it
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._flat_pub = node.create_publisher(
            Float32MultiArray,
            self._config.topic + "_flat",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST),
        )
        names = ", ".join(f"{name}({n})" for _, name, _, n in self._objects)
        self._logger.info(
            f"[deformable-publish] topic={self._config.topic!r} objects=[{names}] "
            f"publish_hz={self._config.publish_hz:g}"
        )

    def publish_frame(
        self, tick: int, sim_time: float, particle_q: np.ndarray, ranges: dict[str, tuple[int, int]]
    ) -> None:
        """Publish one frame's worth of particle positions for every declared object.

        ``ranges`` maps ``prim_path -> (start, count)`` into ``particle_q``, as
        built by ``IPCScene.on_build`` (``self._ranges``). ``particle_q`` is the
        full model array for this tick — sliced per object here, not by the
        caller, so a throttled call (``publish_hz``) still reads the current
        tick's positions rather than a stale slice taken earlier.
        """
        if self._node is None:
            return
        if self._config.publish_hz > 0.0:
            period = 1.0 / self._config.publish_hz
            if self._last_publish_sim_time is not None and sim_time - self._last_publish_sim_time < period - 1e-9:
                return
        self._last_publish_sim_time = sim_time

        from std_msgs.msg import Float32MultiArray, String

        if not self._topology_sent:
            payload = {
                "generation": self._GENERATION,
                "objects": [
                    {"mesh_id": mesh_id, "name": name, "prim_path": prim_path, "n_particles": n}
                    for mesh_id, name, prim_path, n in self._objects
                ],
            }
            self._topology_pub.publish(String(data=json.dumps(payload)))
            self._topology_sent = True
            self._logger.info(f"[deformable-publish] topology sent: {len(self._objects)} object(s)")

        flat: list[float] = [float(self._GENERATION), float(tick)]
        for mesh_id, name, prim_path, n_declared in self._objects:
            start, count = ranges.get(prim_path, (None, None))
            if start is None:
                raise RuntimeError(f"deformable-publish: no particle range for {prim_path!r} (declared n={n_declared})")
            if count != n_declared:
                raise RuntimeError(
                    f"deformable-publish: {prim_path!r} range count {count} != declared n_particles {n_declared}"
                )
            pts = particle_q[start : start + count]
            if not np.isfinite(pts).all():
                raise RuntimeError(f"deformable-publish: non-finite particle_q for {name!r} at tick {tick}")
            flat.append(float(mesh_id))
            flat.append(float(count))
            flat.extend(pts.astype(np.float32).reshape(-1).tolist())
        msg = Float32MultiArray()
        msg.data = flat
        self._flat_pub.publish(msg)
