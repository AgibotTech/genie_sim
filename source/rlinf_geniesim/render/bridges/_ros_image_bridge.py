# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS bridge for the render side.

Upstream renderer (``genie_sim_render_node`` / ``isaacsim_render.py``):
  * subscribes to ``/tf_render``      — body poses to apply to USD prims
  * publishes ``<topic>/image_raw``   — sensor_msgs/Image, one per camera
  * publishes ``<topic>/camera_info`` — sensor_msgs/CameraInfo

This bridge:
  * publishes ``/tf_render`` (so multiple subscribers — both the
    renderer and any rviz instance — see the same poses).
  * subscribes to each ``<topic>/image_raw`` and copies the frame into
    the corresponding slot of the Frame SHM ``rgb_buf`` view.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from rlinf_geniesim.sim_clock import encode_tick_to_stamp


def _import_ros():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        from tf2_msgs.msg import TFMessage
        from geometry_msgs.msg import TransformStamped

        return rclpy, Node, QoSProfile, ReliabilityPolicy, HistoryPolicy, Image, TFMessage, TransformStamped
    except ImportError as e:
        raise ImportError(
            "rclpy / standard ROS messages not importable. Inside the " "geniesim_ros container, source ROS first."
        ) from e


class RosRenderBridge:
    """Own ROS pose publishers and RGB image subscriptions for the renderer."""

    def __init__(
        self,
        node_name: str = "rlinf_render_bridge",
        ns: str = "",
    ) -> None:
        self._node_name = node_name
        self._ns = ns.rstrip("/")

        # Filled in by start().
        self._rclpy = None
        self._node = None
        self._executor = None
        self._spinner: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._tf_pub = None
        self._image_subs: list = []

        # Latest frame caches (per-camera). Key = camera topic; value =
        # (np.ndarray (H, W, 3) uint8, sim_time).
        self._lock = threading.Lock()
        self._latest_frames: dict[str, tuple[np.ndarray, float]] = {}

        # Lazily-resolved msg classes:
        self._TFMessageMsg = None
        self._TransformStampedMsg = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, image_topics: list[str]) -> None:
        """Bring up the rclpy node and subscribe to each camera topic.

        ``image_topics`` are the renderer-side topic names, e.g.
        ``["/cam_front/image_raw", "/cam_wrist/image_raw"]``.
        """
        rclpy, Node, QoSProfile, ReliabilityPolicy, HistoryPolicy, Image, TFMessage, TransformStamped = _import_ros()

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy

        ns = self._ns or None
        self._node = Node(self._node_name, namespace=ns)
        self._TFMessageMsg = TFMessage
        self._TransformStampedMsg = TransformStamped

        cmd_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        # Upstream image publisher uses RELIABLE + KEEP_LAST(10) (verified
        # via `ros2 topic info -v`). Match exactly.
        img_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        # Keep render pose commands separate from upstream physics state topics.
        self._tf_pub = self._node.create_publisher(TFMessage, "/rlinf/tf_render", cmd_qos)

        # FLAT pose channel (perf): one Float64MultiArray per frame instead of
        # ~500 nested TransformStamped — rclpy deserialization of the TFMessage
        # cost the render node ~140ms/frame at 8 envs. Path table goes once on a
        # LATCHED topic (late-joining node still gets it); frames carry
        # [tick, n, n*7 floats] in path-table order.
        from std_msgs.msg import Float64MultiArray as _Flat, String as _Str  # noqa: PLC0415
        from rclpy.qos import DurabilityPolicy as _Dur  # noqa: PLC0415

        self._FlatMsg, self._StrMsg = _Flat, _Str
        self._flat_pub = self._node.create_publisher(_Flat, "rlinf/pose_flat", cmd_qos)
        self._paths_pub = self._node.create_publisher(
            _Str,
            "rlinf/pose_paths",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=_Dur.TRANSIENT_LOCAL,
            ),
        )
        self._last_flat_names: Optional[list] = None

        for topic in image_topics:
            sub = self._node.create_subscription(Image, topic, self._make_image_cb(topic), img_qos)
            self._image_subs.append(sub)

        from rclpy.executors import MultiThreadedExecutor

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spinner = threading.Thread(target=self._spin_loop, name=f"{self._node_name}-spin", daemon=True)
        self._spinner.start()

    def _spin_loop(self) -> None:
        # Use spin() rather than a polled spin_once loop. spin() blocks
        # on the underlying wait set so the kernel wakes us only when
        # there's actually work — and it does NOT race with the other
        # bridge's executor (rclpy's executors are isolated by the
        # nodes they own). The 0.05s spin_once polling we had before
        # was missing image callbacks on busy systems.
        try:
            self._executor.spin()
        except Exception:  # noqa: BLE001
            # Executor shutdown raises during close(); swallow.
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
        self._node = None
        self._executor = None
        self._spinner = None
        self._image_subs = []

    # ------------------------------------------------------------------
    # Outgoing /tf_render
    # ------------------------------------------------------------------

    def publish_tf_render(
        self,
        body_names: list[str],
        body_xyzwxyz: np.ndarray,
        tick: int,
        parent_frame: str = "world",
    ) -> None:
        """Publish all body poses on ``/tf_render`` in one TFMessage.

        ``body_xyzwxyz`` shape ``(B, 7)`` with quat order
        ``[qw, qx, qy, qz]``. The global ``tick`` (int64) is encoded into
        every transform's stamp via ``encode_tick_to_stamp`` so a rendered
        frame can be labelled with the tick of the poses it used.
        """
        if self._tf_pub is None:
            raise RuntimeError("RosRenderBridge.start() must be called first")
        if body_xyzwxyz.shape[0] != len(body_names):
            raise ValueError("body_names / poses length mismatch")

        TFMessage = self._TFMessageMsg
        TransformStamped = self._TransformStampedMsg

        msg = TFMessage()
        sec, nsec = encode_tick_to_stamp(tick)

        for name, p in zip(body_names, body_xyzwxyz):
            t = TransformStamped()
            t.header.stamp.sec = int(sec)
            t.header.stamp.nanosec = int(nsec)
            t.header.frame_id = parent_frame
            t.child_frame_id = name
            t.transform.translation.x = float(p[0])
            t.transform.translation.y = float(p[1])
            t.transform.translation.z = float(p[2])
            # rlinf order is (qw, qx, qy, qz); ROS order is (x, y, z, w).
            t.transform.rotation.w = float(p[3])
            t.transform.rotation.x = float(p[4])
            t.transform.rotation.y = float(p[5])
            t.transform.rotation.z = float(p[6])
            msg.transforms.append(t)

        self._tf_pub.publish(msg)

    def publish_flat_poses(
        self,
        body_names: list[str],
        body_xyzwxyz: np.ndarray,
        tick: int,
    ) -> None:
        """Publish poses on the FLAT channel: ``/rlinf/pose_flat`` carries
        ``[tick, n] + n*7`` float64 in path-table order; the path table itself
        goes on latched ``/rlinf/pose_paths`` (JSON) when it changes, or when
        it has been explicitly invalidated — see ``invalidate_flat_path_table``,
        which ``IsaacTiledBackend.reload_stage()`` calls after every successful
        hot-swap so a superset name list (unchanged across reload) still gets
        republished for the fresh render-side driver.
        ~100x cheaper to deserialize than the TFMessage equivalent."""
        if self._flat_pub is None:
            raise RuntimeError("RosRenderBridge.start() must be called first")
        if body_names != self._last_flat_names:
            import json as _json  # noqa: PLC0415

            sm = self._StrMsg()
            sm.data = _json.dumps(list(body_names))
            self._paths_pub.publish(sm)
            self._last_flat_names = list(body_names)
        fm = self._FlatMsg()
        buf = np.empty(2 + body_xyzwxyz.size, dtype=np.float64)
        buf[0] = float(int(tick))
        buf[1] = float(len(body_names))
        buf[2:] = body_xyzwxyz.reshape(-1)
        import array as _array  # noqa: PLC0415

        a = _array.array("d")
        a.frombytes(buf.tobytes())
        fm.data = a
        self._flat_pub.publish(fm)

    def invalidate_flat_path_table(self) -> None:
        """Republish the latched flat-pose path table on the next pose update.

        A reloaded render driver has no previous path-index mapping."""
        self._last_flat_names = None

    # ------------------------------------------------------------------
    # Incoming images
    # ------------------------------------------------------------------

    def _make_image_cb(self, topic: str):
        def cb(msg) -> None:
            try:
                # Supported encodings (OVRTX publishes ``rgba8`` by default;
                # CPU paths sometimes use rgb8 / bgr8).
                enc = msg.encoding.lower()
                h, w = int(msg.height), int(msg.width)
                # ``msg.data`` is array.array on rclpy; numpy frombuffer
                # gives a zero-copy view we then reshape.
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                if enc in ("rgb8", "8uc3"):
                    arr = buf.reshape(h, w, 3)
                elif enc == "bgr8":
                    arr = buf.reshape(h, w, 3)[..., ::-1]  # BGR → RGB
                elif enc in ("rgba8", "8uc4"):
                    arr = buf.reshape(h, w, 4)[..., :3]  # drop alpha
                elif enc == "bgra8":
                    rgba = buf.reshape(h, w, 4)
                    arr = rgba[..., [2, 1, 0]]  # BGR → RGB, drop A
                else:
                    return
                t = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
                with self._lock:
                    # Cache a *copy* — the underlying msg.data buffer is
                    # invalidated once the callback returns.
                    self._latest_frames[topic] = (arr.copy(), t)
            except Exception:  # noqa: BLE001
                return

        return cb

    def latest_frame(self, topic: str) -> Optional[tuple[np.ndarray, float]]:
        with self._lock:
            v = self._latest_frames.get(topic)
            if v is None:
                return None
            return v[0].copy(), v[1]

    def has_all_frames(self, topics: list[str]) -> bool:
        with self._lock:
            return all(t in self._latest_frames for t in topics)


__all__ = ["RosRenderBridge"]
