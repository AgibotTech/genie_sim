# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Application-neutral ROS transport extracted from RLinf's Newton path.

joint_states and tf_render cache joint/body state independently. joint_command
carries full position targets. /rlinf/reset_envs[_seeded] are engine wire names
retained for compatibility. ROS is imported only when start() is called.
ClientConfig and SimulationClient, rather than application SHM or task code,
consume this bridge.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

# The engine stamps /joint_states.header.frame_id with this prefix + a monotonic
# int, bumped only AFTER a drained reset batch fully applies (state writes +
# randomization + sync_state_into_solver). _on_joint_states caches it atomically.
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
        )
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        from std_msgs.msg import Int32MultiArray

        return (
            rclpy,
            Node,
            QoSProfile,
            ReliabilityPolicy,
            HistoryPolicy,
            JointState,
            TFMessage,
            Int32MultiArray,
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
# RosEngineBridge
# ---------------------------------------------------------------------------


class RosEngineBridge:
    """Bridge node that owns one rclpy.Node and runs a spinner thread.

    Transport extracted from origin/geniesim4 RLinf; application-independent.
    Absolute /rlinf/* names are the existing ENGINE wire protocol, not imports.
    A session requires exclusive ownership of its ROS domain/topic set.
    """

    def __init__(
        self,
        node_name: str = "genie_sim_client_bridge",
        ns: str = "",
        joint_names: Optional[list[str]] = None,
    ) -> None:
        self._node_name = node_name
        # Client uses the default namespace; this does not isolate absolute reset topics.
        self._ns = ns.rstrip("/")

        # ``joint_names`` orders the ``/joint_command`` payload. Filled
        # in by ``set_joint_layout`` after the upstream emits the URDF.
        self._joint_names: list[str] = list(joint_names or [])

        # Latest cached state (written by callbacks, copied by read_state).
        self._lock = threading.Lock()
        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_vel: Optional[np.ndarray] = None
        # Real joint names from /joint_states (so the backend can use them as
        # the /joint_command layout instead of generic 'joint_%d'; for
        # multi-world these are world-indexed 'env_<w>/<name>').
        self._latest_joint_names: Optional[list[str]] = None
        self._latest_body_xyzwxyz: Optional[np.ndarray] = None  # (B, 7)
        self._latest_body_names: Optional[list[str]] = None
        self._latest_sim_time: float = 0.0
        # Monotonic arrival counter, advanced atomically with the cache.
        # No command ID exists on this wire: a new sequence is NOT proof
        # that this client's command committed. See arrival_v1 in DESIGN.md.
        self._js_seq: int = 0
        # Causal credential parsed from /joint_states.header.frame_id.
        # -1 until an epoch has been seen: either an engine build without the
        # feature, or no /joint_states parsed yet.
        # ``_js_reset_epoch_raw`` caches the last RAW frame_id string so the
        # hot-path callback (~every 8ms) only re-parses the int when the
        # string actually changed (reset ticks only, not every ordinary
        # publish).
        self._js_reset_epoch: int = -1
        self._js_reset_epoch_raw: str = ""
        # Engine wall-clock publish stamp (seconds, ``header.stamp``) of the
        # /joint_states message that committed the CURRENT _js_reset_epoch
        # value (see _on_joint_states). 0.0 until an epoch has ever been
        # seen. Paired with ``_tf_stamp`` below for the T4b causality wait
        # in SimulationClient.reset(). Both topics use the same engine-node
        # clock, with joint states published before body transforms each tick.
        # Neither timestamp is simulated time.
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
        self._spin_error = None
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spinner: Optional[threading.Thread] = None
        self._joint_command_pub = None
        self._reset_pub = None
        self._JointStateMsg = None
        self._Int32MultiArrayMsg = None
        self._lifecycle_clients = {}
        self._EmptyRequest = None
        self._GetStateRequest = None

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
            JointState,
            TFMessage,
            Int32MultiArray,
        ) = _import_ros()

        # Init rclpy if no other component already did. ``rclpy.init``
        # is idempotent in practice but raises if called twice; guard.
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy

        ns = self._ns or None
        self._node = Node(self._node_name, namespace=ns)
        self._JointStateMsg = JointState

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
        # The engine's body-pose channel. ``use_tf_render=true`` makes the
        # engine publish everything, robot links included, on ``/tf_render``
        # rather than a scene-only ``/tf_scene``. This is a hard pair with the
        # backend switch: changing one alone subscribes to a topic nobody
        # publishes, which raises nothing and returns empty poses.
        # Keep this distinct from the render bridge's /rlinf/tf_render.
        self._node.create_subscription(TFMessage, "tf_render", self._on_tf_render, tf_render_qos)

        self._joint_command_pub = self._node.create_publisher(JointState, "joint_command", cmd_qos)

        # /rlinf/reset_envs — one-shot per-world reset trigger. RELIABLE (must not
        # drop: a lost reset = a world left un-reset); the engine's reset
        # listener subscribes RELIABLE to match.
        self._Int32MultiArrayMsg = Int32MultiArray
        reset_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self._reset_pub = self._node.create_publisher(Int32MultiArray, "/rlinf/reset_envs", reset_qos)

        # Spin in a background thread while the client sends and waits.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinner = threading.Thread(target=self._spin_loop, name=f"{self._node_name}-spin", daemon=True)
        self._spinner.start()

    def _spin_loop(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:
            self._spin_error = exc

    def close(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if self._spinner and self._spinner.is_alive():
            self._spinner.join(timeout=2.0)
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
        self._lifecycle_clients.clear()

    def _create_lifecycle_clients(self) -> None:
        # Service imports stay lazy; importing the package needs no ROS runtime.
        from lifecycle_msgs.srv import GetState
        from std_srvs.srv import Empty

        self._EmptyRequest = Empty.Request
        self._GetStateRequest = GetState.Request
        # The C++ bridge is created by Newton's session with this node name;
        # its init_ros uses argc=0, so launch's Python-node remap does not apply.
        for verb in ("pause", "resume"):
            self._lifecycle_clients[verb] = self._node.create_client(Empty, f"/genie_sim_engine_newton/{verb}")
        self._lifecycle_clients["get_state"] = self._node.create_client(GetState, "/genie_sim_engine_newton/get_state")

    def set_paused(self, paused: bool, timeout_s: float) -> None:
        """Wait for the engine loop to apply pause/resume at a tick boundary.

        Empty service responses only acknowledge enqueueing. GetState is the
        existing engine-side acknowledgement that the loop applied the verb.
        This does not add a command-id or a fresh snapshot/render credential.
        """
        if self._node is None:
            raise RuntimeError("ROS bridge is closed")
        if not self._lifecycle_clients:
            self._create_lifecycle_clients()
        deadline = time.monotonic() + timeout_s
        verb = "pause" if paused else "resume"
        command = self._lifecycle_clients[verb]
        get_state = self._lifecycle_clients["get_state"]

        def wait(predicate, description):
            while True:
                self.check_health()
                if predicate():
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"engine {verb}: {description} timed out after {timeout_s:g}s")
                time.sleep(0.001)

        def call(client, request):
            future = client.call_async(request)
            try:
                wait(future.done, "service response")
                result = future.result()
                if result is None:
                    raise RuntimeError(f"engine {verb}: service returned no response")
                return result
            finally:
                if not future.done():
                    future.cancel()

        wait(lambda: command.service_is_ready() and get_state.service_is_ready(), "service discovery")
        call(command, self._EmptyRequest())
        expected = 2 if paused else 3  # lifecycle_msgs/State: INACTIVE / ACTIVE
        while call(get_state, self._GetStateRequest()).current_state.id != expected:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"engine {verb}: state confirmation timed out after {timeout_s:g}s")
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Layout (set after we learn the upstream's joint order)
    # ------------------------------------------------------------------

    def set_joint_layout(self, joint_names: list[str]) -> None:
        """Tell the bridge what joint order to use when publishing
        ``/joint_command``. Typically called once after the upstream
        loads the URDF and we read ``SceneSchema.joint_names``."""
        self._joint_names = list(joint_names)

    # ------------------------------------------------------------------
    # Subscription callbacks (run on rclpy executor thread)
    # ------------------------------------------------------------------

    def _on_joint_states(self, msg) -> None:
        # msg.position / msg.velocity are array-like; len(msg.name) == #joints
        try:
            if not msg.name or len(msg.name) != len(msg.position):
                return
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
                self._js_seq += 1
                self._latest_joint_pos = pos.copy()
                self._latest_joint_vel = vel.copy()
                if names is not None:
                    self._latest_joint_names = names
                self._latest_sim_time = t
                if new_epoch is not None:
                    self._js_reset_epoch = new_epoch
                    # T4b: record THIS message's stamp as the epoch-
                    # confirming tick's timestamp, under the SAME lock
                    # acquisition as the epoch bump itself (and the
                    # joint-state mirror write above) — same causal
                    # guarantee read_state() relies on.
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
            # T4b: engine wall-clock publish stamp of this batch (all
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
            raise RuntimeError("RosEngineBridge.start() must be called first")
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
        """Publish one reset message. Plain: ``/rlinf/reset_envs``
        (Int32MultiArray of world ids; empty = all). Extended: when ``seeds``
        or ``instances`` carries any valid (>=0) entry, publish
        ``/rlinf/reset_envs_seeded`` (Int64MultiArray of flat [env, seed, instance]
        TRIPLES; -1 = no-op for that slot) so the engine applies deterministic
        per-env object randomization and/or in-group instance switching after
        the restore."""
        if self._reset_pub is None:
            raise RuntimeError("RosEngineBridge.start() must be called first")
        ids = [int(i) for i in env_ids]
        has_sd = seeds is not None and any(int(x) >= 0 for x in seeds)
        has_it = instances is not None and any(int(x) >= 0 for x in instances)
        if has_sd or has_it:
            self.prepare_reset(True)
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

    def read_state(self) -> dict:
        """Copy all mirrors and credentials in one critical section.

        Joint/TF timestamps are publisher wall time, NOT simulation time.
        This is a coherent cache read, not an engine commit barrier.
        """
        with self._lock:
            return {
                "positions": None if self._latest_joint_pos is None else self._latest_joint_pos.copy(),
                "velocities": None if self._latest_joint_vel is None else self._latest_joint_vel.copy(),
                "joint_names": tuple(self._latest_joint_names or ()),
                "body_poses": None if self._latest_body_xyzwxyz is None else self._latest_body_xyzwxyz.copy(),
                "body_names": tuple(self._latest_body_names or ()),
                "joint_sequence": self._js_seq,
                "body_sequence": self._tf_seq,
                "joint_stamp": self._latest_sim_time,
                "body_stamp": self._tf_stamp,
                "reset_epoch": self._js_reset_epoch,
                "reset_stamp": self._js_epoch_stamp,
            }

    def check_health(self) -> None:
        if self._spin_error is not None:
            raise RuntimeError("ROS bridge executor failed") from self._spin_error
        if self._rclpy is not None and not self._rclpy.ok():
            raise RuntimeError("ROS context was shut down")

    def command_readers(self) -> int:
        return self._joint_command_pub.get_subscription_count()

    def prepare_reset(self, seeded: bool) -> None:
        if seeded and self._reset_seeded_pub is None:
            from std_msgs.msg import Int64MultiArray

            self._reset_seeded_pub = self._node.create_publisher(Int64MultiArray, "/rlinf/reset_envs_seeded", 10)
            self._Int64MultiArrayMsg = Int64MultiArray

    def reset_readers(self, seeded: bool = False) -> int:
        publisher = self._reset_seeded_pub if seeded else self._reset_pub
        return publisher.get_subscription_count()
