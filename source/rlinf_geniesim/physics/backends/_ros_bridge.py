# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS subscriptions and joint-command publication for upstream physics nodes.

Cache joint state, body transforms, clock, scene joints and latched AABB tables.
The local-AABB subscription uses TRANSIENT_LOCAL durability to match its writer.
rclpy is imported only when the bridge starts."""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import numpy as np


# Wire contract for the reset_epoch causal credential (see js_reset_epoch()
# and _on_joint_states below): the engine stamps /joint_states.header.frame_id
# with this prefix + a monotonic int, bumped only AFTER a drained reset batch
# fully applies (state writes + randomization + sync_state_into_solver).
_RESET_EPOCH_FRAME_PREFIX = "reset_epoch:"


# ---------------------------------------------------------------------------
# Lazy ROS imports
# ---------------------------------------------------------------------------


def _import_ros():
    """Import rclpy + msg types lazily.

    Raises ImportError with a friendly message if ROS isn't sourced.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile,
            ReliabilityPolicy,
            HistoryPolicy,
            DurabilityPolicy,
        )
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        from std_msgs.msg import Float32MultiArray, Int32MultiArray, String

        return (
            rclpy,
            Node,
            QoSProfile,
            ReliabilityPolicy,
            HistoryPolicy,
            DurabilityPolicy,
            JointState,
            TFMessage,
            Float32MultiArray,
            Int32MultiArray,
            String,
        )
    except ImportError as e:
        raise ImportError(
            "rclpy / standard ROS messages not importable. Inside the "
            "geniesim_ros container, source ROS first:\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  source /workspace/devel/setup.bash\n"
            f"original error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# RosPhysicsBridge
# ---------------------------------------------------------------------------


class RosPhysicsBridge:
    """Own a ROS node and spinner thread for upstream state and action topics.

    send_action publishes once per call. Multi-world state names are retained for
    the physics backend to reshape and align with its per-environment schema."""

    def __init__(
        self,
        node_name: str = "rlinf_physics_bridge",
        ns: str = "",
        joint_names: Optional[list[str]] = None,
    ) -> None:
        self._node_name = node_name
        # Optional ROS namespace for the bridge node.
        self._ns = ns.rstrip("/")

        # ``joint_names`` orders the ``/joint_command`` payload. Filled
        # in by ``set_joint_layout`` after the upstream emits the URDF.
        self._joint_names: list[str] = list(joint_names or [])

        # Latest cached state (written by callbacks, read by getters).
        self._lock = threading.Lock()
        self._static_prop_aabb = None
        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_vel: Optional[np.ndarray] = None
        # Real joint names from /joint_states (so the backend can use them as
        # the /joint_command layout instead of generic 'joint_%d'; for
        # multi-world these are world-indexed 'env_<w>/<name>').
        self._latest_joint_names: Optional[list[str]] = None
        self._latest_body_xyzwxyz: Optional[np.ndarray] = None  # (B, 7)
        self._latest_body_names: Optional[list[str]] = None
        # /joint_states_scene — the v2 channel for articulated SCENE degrees
        # of freedom (door hinge / drawer slide). They are deliberately absent
        # from /joint_states: not robot-controllable, so including them would
        # inflate the RL action surface. ``name[i]`` is the bare compiled joint
        # name; nothing on the wire says which OBJECT owns it, so two instances
        # of one asset in a scene are indistinguishable here (see
        # ``_upstream_base.get_scene_joint_states``).
        self._scene_joint_names: Optional[list[str]] = None
        self._scene_joint_pos: Optional[np.ndarray] = None
        # /rlinf/body_aabb_local — the engine's STATIC per-body local AABB
        # table, latched, one message for the life of the stage.
        self._body_aabb_local_names: Optional[list[str]] = None
        self._body_aabb_local_table: Optional[np.ndarray] = None  # (B, 6)
        self._latest_sim_time: float = 0.0
        # Monotonic ARRIVAL counter for /joint_states. The lockstep engine
        # publishes exactly ONE message per K-tick burst, so "js_seq advanced"
        # == "the commanded burst finished" — unlike sim_time freshness, which
        # mid-burst /clock signals could satisfy EARLY (steps then read state
        # 1-3 ticks stale).
        self._js_seq: int = 0
        # Causal credential parsed from /joint_states.header.frame_id (see
        # js_reset_epoch()). -1 until an epoch has been seen: either an
        # engine build without the feature, or no /joint_states parsed yet.
        # ``_js_reset_epoch_raw`` caches the last RAW frame_id string so the
        # hot-path callback (~every 8ms) only re-parses the int when the
        # string actually changed (reset ticks only, not every ordinary
        # publish).
        self._js_reset_epoch: int = -1
        self._js_reset_epoch_raw: str = ""
        # Engine wall-clock publish stamp (seconds, ``header.stamp``) of the
        # /joint_states message that committed the CURRENT _js_reset_epoch
        # value (see _on_joint_states). 0.0 until an epoch has ever been
        # seen. Paired with ``_tf_stamp`` below for the reset causality wait
        # in _upstream_base.py's reset() — see js_epoch_stamp()/tf_stamp()
        # getters for why comparing these two is valid despite neither
        # being literal simulated time.
        self._js_epoch_stamp: float = 0.0
        self._reset_seeded_pub = None
        self._Int64MultiArrayMsg = None
        self._expected_envs = None
        self._tf_seq: int = 0
        # Engine wall-clock publish stamp (seconds) of the latest /tf_render
        # message that carried real engine body frames (see _on_tf_render's
        # early-return for render-bridge-only messages, which does NOT
        # update this — kept-last semantics match _latest_body_xyzwxyz).
        # 0.0 until such a message has been seen.
        self._tf_stamp: float = 0.0

        # Populated in ``start()``.
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spinner: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._joint_command_pub = None
        self._reset_pub = None
        self._JointStateMsg = None
        self._TFMessageMsg = None
        self._Int32MultiArrayMsg = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        (
            rclpy,
            Node,
            QoSProfile,
            ReliabilityPolicy,
            HistoryPolicy,
            DurabilityPolicy,
            JointState,
            TFMessage,
            _,
            Int32MultiArray,
            String,
        ) = _import_ros()

        # Init rclpy if no other component already did. ``rclpy.init``
        # is idempotent in practice but raises if called twice; guard.
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy

        ns = self._ns or None
        self._node = Node(self._node_name, namespace=ns)
        self._JointStateMsg = JointState
        self._TFMessageMsg = TFMessage

        # QoS — each topic must match what the upstream publishes:
        #   /joint_states : SensorDataQoS (BEST_EFFORT + KEEP_LAST(5))
        #   /tf_render    : RELIABLE      + KEEP_LAST(100)
        #   /joint_command: RELIABLE      (we are the publisher)
        # Subscriber-publisher reliability mismatch silently drops all
        # messages, so each gets its own QoS below.
        from rclpy.qos import qos_profile_sensor_data

        cmd_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        joint_states_qos = qos_profile_sensor_data
        tf_render_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        self._node.create_subscription(JointState, "joint_states", self._on_joint_states, joint_states_qos)
        self._node.create_subscription(JointState, "joint_states_scene", self._on_joint_states_scene, joint_states_qos)
        # The engine's body-pose channel. ``use_tf_render=true`` makes the
        # engine publish everything, robot links included, on ``/tf_render``
        # rather than a scene-only ``/tf_scene``. This is a hard pair with the
        # backend switch: changing one alone subscribes to a topic nobody
        # publishes, which raises nothing and returns empty poses.
        #
        # Keep this engine topic distinct from the render bridge's
        # ``/rlinf/tf_render`` namespace.
        self._node.create_subscription(TFMessage, "tf_render", self._on_tf_render, tf_render_qos)

        # /rlinf/body_aabb_local — STATIC per-body local AABB table, published
        # exactly once at build time and LATCHED so a bridge that attaches
        # later still gets it.
        #
        # ⚠ TRANSIENT_LOCAL on BOTH sides or nothing arrives. Durability is a
        # writer-side promise in ROS 2, but it is also part of the QoS
        # compatibility contract: a VOLATILE reader is not compatible with a
        # TRANSIENT_LOCAL writer, so DDS never pairs them. There is no error
        # and no timeout — ``ros2 topic list`` shows the topic, ``ros2 topic
        # echo`` (which defaults to VOLATILE) shows nothing, and
        # ``get_body_aabbs`` silently returns zero boxes forever.
        aabb_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._node.create_subscription(String, "/rlinf/body_aabb_local", self._on_body_aabb_local, aabb_qos)
        self._node.create_subscription(String, "/rlinf/static_prop_aabb", self._on_static_prop_aabb, aabb_qos)

        self._joint_command_pub = self._node.create_publisher(JointState, "joint_command", cmd_qos)

        # /reset_envs — one-shot per-world reset trigger. RELIABLE (must not
        # drop: a lost reset = a world left un-reset); the engine's reset
        # listener subscribes RELIABLE to match.
        self._Int32MultiArrayMsg = Int32MultiArray
        reset_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self._reset_pub = self._node.create_publisher(Int32MultiArray, "/rlinf/reset_envs", reset_qos)

        # Spin in a background thread so the rlinf main loop can call
        # ``send_action`` / ``get_joint_pos`` synchronously.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinner = threading.Thread(target=self._spin_loop, name=f"{self._node_name}-spin", daemon=True)
        self._spinner.start()

    def _spin_loop(self) -> None:
        # See renderer-side _ros_image_bridge.py for why we use spin()
        # instead of polled spin_once.
        try:
            self._executor.spin()
        except Exception:  # noqa: BLE001
            return

    def close(self) -> None:
        self._stop.set()
        if self._spinner and self._spinner.is_alive():
            self._spinner.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        # rclpy.shutdown() is left to the parent process (multiple
        # bridges share one rclpy runtime).
        self._node = None
        self._executor = None
        self._spinner = None

    # ------------------------------------------------------------------
    # Layout (set after we learn the upstream's joint order)
    # ------------------------------------------------------------------

    def set_joint_layout(self, joint_names: list[str]) -> None:
        """Tell the bridge what joint order to use when publishing
        ``/joint_command``. Typically called once after the upstream
        loads the URDF and we read ``SceneSchema.joint_names``."""
        self._joint_names = list(joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    # ------------------------------------------------------------------
    # Subscription callbacks (run on rclpy executor thread)
    # ------------------------------------------------------------------

    def _on_joint_states(self, msg) -> None:
        with self._lock:
            self._js_seq += 1
        # msg.position / msg.velocity are array-like; len(msg.name) == #joints
        try:
            pos = np.asarray(msg.position, dtype=np.float32)
            vel = (
                np.asarray(msg.velocity, dtype=np.float32)
                if len(msg.velocity) == len(msg.position)
                else np.zeros_like(pos)
            )
            t = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
            names = list(msg.name) if len(msg.name) == len(pos) else None

            # reset_epoch causal credential: the epoch and the joint state
            # travel in this same message and are written together under one
            # lock, so the instant a caller sees the epoch advance the cached
            # state is already post-reset. There is no window where the epoch
            # arrived but the state is stale.
            #
            # The string-equality gate comes first: frame_id is constant
            # across ordinary ticks, so this avoids re-parsing on the hot
            # path.
            frame_id = getattr(msg.header, "frame_id", "") or ""
            new_epoch = None
            if frame_id != self._js_reset_epoch_raw:
                self._js_reset_epoch_raw = frame_id
                if frame_id.startswith(_RESET_EPOCH_FRAME_PREFIX):
                    try:
                        new_epoch = int(frame_id[len(_RESET_EPOCH_FRAME_PREFIX) :])
                    except ValueError:
                        new_epoch = None  # malformed suffix — keep whatever we had

            with self._lock:
                self._latest_joint_pos = pos
                self._latest_joint_vel = vel
                if names is not None:
                    self._latest_joint_names = names
                self._latest_sim_time = t
                if new_epoch is not None:
                    self._js_reset_epoch = new_epoch
                    # Record this message's stamp as the epoch-
                    # confirming tick's timestamp, under the SAME lock
                    # acquisition as the epoch bump itself (and the
                    # joint-state mirror write above) — same causal
                    # guarantee js_reset_epoch() already relies on.
                    self._js_epoch_stamp = t
        except Exception:  # noqa: BLE001
            return

    def _on_tf_render(self, msg) -> None:
        try:
            # Cache only the engine's own body frames. The two producers are
            # on separate topics now, and this filter is cheap insurance
            # against anyone merging them back: frames the render bridge
            # republished would break per-world grouping and starve the render
            # barrier.
            #
            # Everything else is an engine frame: bare ``env_<w>/...`` when
            # N>1, an absolute prim path when N==1. Dropping everything with a
            # leading "/" would discard the N==1 dialect entirely.
            transforms = [
                t for t in (msg.transforms or []) if not t.child_frame_id.startswith(("/World/envs/", "/RenderOVRTX/"))
            ]
            if not transforms:
                return  # this message carried only render-bridge frames — keep last
            n = len(transforms)
            arr = np.zeros((n, 7), dtype=np.float32)
            names = []
            for i, t in enumerate(transforms):
                tr = t.transform.translation
                rot = t.transform.rotation
                arr[i, 0] = tr.x
                arr[i, 1] = tr.y
                arr[i, 2] = tr.z
                # ROS quaternion order is (x, y, z, w); we use
                # (qw, qx, qy, qz).
                arr[i, 3] = rot.w
                arr[i, 4] = rot.x
                arr[i, 5] = rot.y
                arr[i, 6] = rot.z
                names.append(t.child_frame_id)
            # Engine wall-clock publish stamp of this batch (all
            # transforms in one /tf_render message share one stamp — the
            # engine sets it once per publish_body_tf_render call, see
            # realtime_ros_node.cpp). Read BEFORE the lock (stamp fields are
            # plain ints, no sharing concern) so the lock hold stays tiny.
            stamp = transforms[0].header.stamp
            tf_stamp = float(stamp.sec) + stamp.nanosec * 1e-9
            with self._lock:
                self._latest_body_xyzwxyz = arr
                self._latest_body_names = names
                self._tf_seq += 1
                self._tf_stamp = tf_stamp
        except Exception:  # noqa: BLE001
            return

    def send_action(self, joint_positions: np.ndarray) -> None:
        """Publish one ``/joint_command`` message.

        ``joint_positions`` is expected to be shape ``(num_joints,)`` in
        the order given by ``set_joint_layout``.
        """
        if self._joint_command_pub is None:
            raise RuntimeError("RosPhysicsBridge.start() must be called first")
        if not self._joint_names:
            raise RuntimeError(
                "set_joint_layout(...) must be called before send_action; " "the bridge has no joint order to use"
            )
        if joint_positions.shape != (len(self._joint_names),):
            raise ValueError(
                f"joint_positions shape {joint_positions.shape} != expected " f"({len(self._joint_names)},)"
            )
        msg = self._JointStateMsg()
        msg.name = list(self._joint_names)
        msg.position = [float(x) for x in joint_positions]
        # velocity / effort empty — upstream PD computes from position
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._joint_command_pub.publish(msg)

    def send_reset(self, env_ids, seeds=None, instances=None) -> None:
        """Publish one reset message. Plain: ``/reset_envs``
        (Int32MultiArray of world ids; empty = all). Extended: when ``seeds``
        or ``instances`` carries any valid (>=0) entry, publish
        ``/reset_envs_seeded`` (Int64MultiArray of flat [env, seed, instance]
        TRIPLES; -1 = no-op for that slot) so the engine applies deterministic
        per-env object randomization and/or in-group instance switching after
        the restore."""
        if self._reset_pub is None:
            raise RuntimeError("RosPhysicsBridge.start() must be called first")
        ids = [int(i) for i in env_ids]
        has_sd = seeds is not None and any(int(x) >= 0 for x in seeds)
        has_it = instances is not None and any(int(x) >= 0 for x in instances)
        if has_sd or has_it:
            if self._reset_seeded_pub is None:
                from std_msgs.msg import Int64MultiArray  # noqa: PLC0415

                # Must match the engine's ``topic + "_seeded"`` exactly. A
                # mismatch raises nothing: the publisher publishes, the
                # subscriber waits, and reset requests are silently dropped.
                self._reset_seeded_pub = self._node.create_publisher(Int64MultiArray, "/rlinf/reset_envs_seeded", 10)
                self._Int64MultiArrayMsg = Int64MultiArray
            use_ids = ids if ids else list(range(int(self._expected_envs or 1)))
            msg = self._Int64MultiArrayMsg()
            flat: list[int] = []
            for i, e in enumerate(use_ids):
                sd = int(seeds[i]) if seeds is not None and i < len(seeds) else -1
                it = int(instances[i]) if instances is not None and i < len(instances) else -1
                flat += [int(e), sd, it]
            msg.data = flat
            self._reset_seeded_pub.publish(msg)
            return
        msg = self._Int32MultiArrayMsg()
        msg.data = ids
        self._reset_pub.publish(msg)

    # ------------------------------------------------------------------
    # State accessors (snapshot the lock-protected cache)
    # ------------------------------------------------------------------

    def latest_joint_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_joint_pos is None else self._latest_joint_pos.copy()

    def latest_joint_vel(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_joint_vel is None else self._latest_joint_vel.copy()

    def latest_joint_names(self) -> Optional[list[str]]:
        with self._lock:
            return None if self._latest_joint_names is None else list(self._latest_joint_names)

    def latest_body_poses(self) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
        """Return ``(xyzwxyz, names)`` — both ``None`` if no message yet."""
        with self._lock:
            poses = None if self._latest_body_xyzwxyz is None else self._latest_body_xyzwxyz.copy()
            names = None if self._latest_body_names is None else list(self._latest_body_names)
            return poses, names

    def _on_joint_states_scene(self, msg) -> None:
        """Latch the scene-DoF snapshot. Best-effort: a malformed message is
        dropped rather than allowed to take down the executor thread, matching
        every other callback here."""
        try:
            pos = np.asarray(msg.position, dtype=np.float32)
            names = list(msg.name) if len(msg.name) == len(pos) else None
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._scene_joint_pos = pos
            if names is not None:
                self._scene_joint_names = names

    def latest_scene_joint_states(self) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
        """Return the latest scene-joint names and positions, or (None, None).

        Names are bare compiled joint names aligned with positions; the topic carries
        no owning-object path. Scenes without articulated props need not publish it."""
        with self._lock:
            names = None if self._scene_joint_names is None else list(self._scene_joint_names)
            pos = None if self._scene_joint_pos is None else self._scene_joint_pos.copy()
            return names, pos

    def _on_body_aabb_local(self, msg) -> None:
        """Cache a complete, validated local-AABB table.

        Payload names align with [cx, cy, cz, hx, hy, hz] rows. Reject the entire table
        on malformed data rather than pairing a body's name with another body's box."""
        try:
            payload = json.loads(msg.data)
            names = list(payload.get("names") or [])
            table = np.asarray(payload.get("aabb_local") or [], dtype=np.float32)
            if table.size == 0:
                table = table.reshape(0, 6)
            if table.ndim != 2 or table.shape[1] != 6 or table.shape[0] != len(names):
                return
            with self._lock:
                self._body_aabb_local_names = names
                self._body_aabb_local_table = table
        except Exception:  # noqa: BLE001
            return

    def latest_body_aabb_local(self) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
        """Return ``(names, (B, 6) centre+half-extent table)`` from the latched
        ``/rlinf/body_aabb_local`` — both ``None`` until it arrives."""
        with self._lock:
            names = None if self._body_aabb_local_names is None else list(self._body_aabb_local_names)
            table = None if self._body_aabb_local_table is None else self._body_aabb_local_table.copy()
            return names, table

    def tf_seq(self) -> int:
        """Arrival count of cached /tf_render body-frame messages — the version
        key for get_body_poses memoization (poses only change when a new burst
        publishes)."""
        with self._lock:
            return self._tf_seq

    def js_seq(self) -> int:
        """Arrival count of /joint_states messages (see _js_seq)."""
        with self._lock:
            return self._js_seq

    def js_reset_epoch(self) -> int:
        """Causal credential parsed from /joint_states.header.frame_id
        (``"reset_epoch:<n>"``). Returns -1 until an epoch has been seen —
        either an engine build without the feature, or no /joint_states
        parsed yet. See ``_on_joint_states`` for why "epoch advanced" implies
        the cached joint state is already post-reset."""
        with self._lock:
            return self._js_reset_epoch

    def js_epoch_stamp(self) -> float:
        """Return the wall-clock publish stamp of the current reset epoch, or 0.0.

        The upstream node stamps joint messages with its own clock, not simulation time."""
        with self._lock:
            return self._js_epoch_stamp

    def tf_stamp(self) -> float:
        """Return the wall-clock publish stamp of the latest engine body TF.

        Render-only frames do not update it; return 0 before engine TF arrives.
        Joint states and body TF share the engine clock. Comparing this stamp with
        ``js_epoch_stamp()`` establishes whether body poses were published at or
        after the joint-state message confirming a reset epoch.
        """
        with self._lock:
            return self._tf_stamp

    def latest_sim_time(self) -> float:
        with self._lock:
            return self._latest_sim_time

    def has_state(self) -> bool:
        with self._lock:
            return self._latest_joint_pos is not None

    def _on_static_prop_aabb(self, msg) -> None:
        """Cache the latched static-prop world AABBs by checker-facing path.

        Payload names align with [min_x, min_y, min_z, max_x, max_y, max_z] rows."""
        try:
            payload = json.loads(msg.data)
            names = [str(x) for x in (payload.get("names") or [])]
            table = np.asarray(payload.get("aabb_world") or [], dtype=np.float32)
            if table.size == 0:
                table = table.reshape(0, 6)
            if table.ndim != 2 or table.shape[1] != 6 or table.shape[0] != len(names):
                return  # malformed payload — keep whatever we had (or nothing)
            if not np.isfinite(table).all() or np.any(table[:, :3] > table[:, 3:]):
                return
            with self._lock:
                self._static_prop_aabb = {nm: table[i].copy() for i, nm in enumerate(names)}
        except Exception:  # noqa: BLE001
            return

    def latest_static_prop_aabb(self) -> Optional[dict]:
        """Return ``{prim_path: [min_x,min_y,min_z,max_x,max_y,max_z]}`` from
        the latched ``/rlinf/static_prop_aabb`` message, or ``None`` if it
        hasn't arrived (this task named no props, or an older engine that
        doesn't publish the topic). Boxes are WORLD-frame and constant."""
        with self._lock:
            if self._static_prop_aabb is None:
                return None
            return {k: v.copy() for k, v in self._static_prop_aabb.items()}


__all__ = ["RosPhysicsBridge"]
