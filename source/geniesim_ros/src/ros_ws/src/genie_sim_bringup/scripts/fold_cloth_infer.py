#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Closed-loop VLA inference against a live RT Engine scene, over ROS only.

Talks to a CoRobot-protocol inference server (msgpack over WebSocket) and drives
the robot through ``/joint_command``. Deliberately imports nothing from
``geniesim_benchmark``: that package pulls in ``geniesim_assets`` and
``isaacsim``, neither of which is importable outside the benchmark's own
Isaac-hosted interpreter. Everything here comes from ROS plus numpy / cv2 /
msgpack / websockets.

Run with ``python3`` after sourcing the ROS overlay -- NOT with ``omni_python``.
``omni_python`` is CPython 3.11 while ROS jazzy's ``rclpy`` extension is built
for 3.12, so importing rclpy there fails outright.

    # terminal 1 -- the scene. The launcher MUST be one that starts a render
    # backend, or no camera images are published and this script waits forever:
    # launcher_libuipc has ``renders: []`` (physics only). Use the _ovrtx or
    # _isaacsim variant instead.
    ros2 launch genie_sim_bringup app.launch.py \
        scene:=scene_flat_g2_wuji_ipc_fold_cloth \
        launcher_config:=launcher_libuipc_ovrtx headless:=false

    # terminal 2 -- this script
    source /opt/ros/jazzy/setup.bash
    python3 fold_cloth_infer.py --infer-host 192.168.1.100:8999

Protocol mirrors ``geniesim_benchmark.benchmark.policy.corobotpolicy``: request
``{"method": "infer", "params": {...}}``, response ``{"result": {...}}`` with
per-channel ``{"kind", "values"}`` chunks.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time

import numpy as np

try:
    import cv2
    import msgpack
    import websockets.sync.client
except ImportError as exc:  # pragma: no cover - environment probe
    sys.exit(f"missing dependency: {exc}. Expected inside the Genie Sim container.")

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32MultiArray

# ---------------------------------------------------------------------------
# Joint layout for G2_crsB_wuji. Copied from
# geniesim_benchmark/utils/name_utils.py (ROBOT_CONFIGS["G2_crsB_wuji"]) because
# that module cannot be imported here; the payload's channel order is part of
# the wire contract, so these lists must stay in step with it.
# ---------------------------------------------------------------------------

# Spelled out rather than generated: the idx prefixes are not a formula (they are
# the robot's own DoF numbering) and a silent mismatch here would be commanded to
# the wrong joint.
ARM_JOINTS = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]
WAIST_JOINTS = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]
HEAD_JOINTS = [
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
]
# 5 fingers x 4 joints per hand, thumb->pinky, left then right: the 40-DoF order
# WUJI_AJ_NAMES declares.
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GRIPPER_JOINTS = [f"hand_{s}_{f}_joint{j}" for s in ("l", "r") for f in _FINGERS for j in range(1, 5)]

# Every G2 announces itself as g2a_sim. Note the benchmark's own client only
# *logs* eef_type and never puts it on the wire, so the server is expected to
# infer the five-finger hand from the 40-wide gripper channel.
ROBOT_TYPE = "g2a_sim"

# Camera topic PREFIXES from config/scene_flat_g2_wuji_ipc_fold_cloth.yaml, keyed
# by the payload's own camera names. The scene yaml's ``topic.rgb`` is a prefix,
# not a full topic: both publishers append "/image_raw" (render_node's
# RosImagePublisherPlugin does `topic + "/image_raw"`, and isaacsim_render.py
# sets the same on its CamHelper nodes), so subscribing to the bare prefix
# receives nothing.
CAMERA_TOPIC_PREFIXES = {
    "head": "/genie_sim/head_camera_rgb",
    "hand_left": "/genie_sim/hand_left_camera_rgb",
    "hand_right": "/genie_sim/hand_right_camera_rgb",
}
IMAGE_SUFFIX = "/image_raw"
CAMERA_TOPICS = {cam: prefix + IMAGE_SUFFIX for cam, prefix in CAMERA_TOPIC_PREFIXES.items()}

_OPEN_TIMEOUT_SEC = 30
_PING_INTERVAL_SEC = 20
_PING_TIMEOUT_SEC = 60


def _ws_connect_compat(uri: str, **kwargs):
    """``websockets.sync.client.connect``, minus kwargs this version rejects.

    Mirrors ``geniesim_benchmark.utils.comm.websocket_client.ws_connect_compat``
    (inlined, since that package is not importable here). websockets 12's sync
    client takes neither ``ping_interval`` nor ``ping_timeout`` and has no
    ``**kwargs`` to swallow them, so passing them raises TypeError; 15 accepts
    both. Filter against the actual signature rather than version-sniffing.
    """
    try:
        params = inspect.signature(websockets.sync.client.connect).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        dropped = sorted(set(kwargs) - set(params))
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        if dropped:
            print(f"[fold_cloth_infer] websockets {websockets.__version__}: ignoring {', '.join(dropped)}")
    return websockets.sync.client.connect(uri, **kwargs)


def imgmsg_to_rgb(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image to an HxWx3 RGB array.

    Hand-rolled rather than via cv_bridge: this keeps the script runnable on any
    interpreter that has rclpy, and the encodings the render node emits are a
    short, known list. ``step`` is honoured so row padding cannot skew the image.
    """
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    rows = np.frombuffer(msg.data, dtype=np.uint8)
    rows = rows.reshape(msg.height, msg.step)[:, : msg.width * channels]
    img = rows.reshape(msg.height, msg.width, channels)
    if msg.encoding in ("bgr8", "bgra8"):
        return np.ascontiguousarray(img[:, :, 2::-1])
    if msg.encoding == "mono8":
        return np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img[:, :, :3])


def encode_image_jpeg(rgb: np.ndarray, quality: int = 95) -> dict:
    """JPEG-encode one frame into the payload's image dict."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return {"encoding": "JPEG", "image_data": buf.tobytes(), "height": rgb.shape[0], "width": rgb.shape[1]}


class SimBridge(Node):
    """Latest-frame cache for the observation topics, plus the command publishers."""

    def __init__(self) -> None:
        super().__init__("fold_cloth_infer")
        # BEST_EFFORT readers accept both BEST_EFFORT and RELIABLE writers, so
        # this matches the sensor streams without assuming their QoS.
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        reliable_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        self.images: dict[str, np.ndarray] = {}
        self.joints: dict[str, float] = {}
        self._decode_warned: set[str] = set()

        for cam, topic in CAMERA_TOPICS.items():
            self.create_subscription(Image, topic, self._make_image_cb(cam), sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, sensor_qos)

        # /joint_command is BEST_EFFORT on the engine side (see
        # genie_sim_engine_core/common/lifecycle_ros.py).
        self.cmd_pub = self.create_publisher(JointState, "/joint_command", sensor_qos)
        self.reset_pub = self.create_publisher(Int32MultiArray, "/rlinf/reset_envs", reliable_qos)

    def _make_image_cb(self, cam: str):
        def cb(msg: Image) -> None:
            try:
                self.images[cam] = imgmsg_to_rgb(msg)
            except ValueError as exc:
                if cam not in self._decode_warned:
                    self._decode_warned.add(cam)
                    self.get_logger().error(f"{cam}: {exc}")

        return cb

    def _joint_cb(self, msg: JointState) -> None:
        self.joints = dict(zip(msg.name, msg.position))

    def spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=max(0.0, deadline - time.monotonic()))

    def wait_for_observation(self, timeout: float) -> None:
        """Block until every camera and /joint_states has produced a frame.

        On timeout, report each missing topic's publisher count. Zero publishers
        on the cameras while /joint_states is live is the signature of a launcher
        whose ``renders:`` list is empty (e.g. launcher_libuipc): physics runs,
        but no render node exists to publish images.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.images) == len(CAMERA_TOPICS) and self.joints:
                return

        pending = [(CAMERA_TOPICS[c], c) for c in CAMERA_TOPICS if c not in self.images]
        if not self.joints:
            pending.append(("/joint_states", "joint states"))
        lines = []
        for topic, label in pending:
            n = self.count_publishers(topic)
            lines.append(f"  {label}: {topic} ({n} publisher{'' if n == 1 else 's'})")
        detail = "\n".join(lines)

        cams_dead = all(self.count_publishers(CAMERA_TOPICS[c]) == 0 for c in CAMERA_TOPICS)
        hint = (
            "\nNo publisher on any camera topic. If the scene is up, its launcher has no "
            "render backend: launcher_libuipc sets `renders: []`, so nothing publishes "
            "images. Relaunch with launcher_libuipc_ovrtx (or _isaacsim)."
            if cams_dead
            else "\nCheck that the scene is running and that ROS_DOMAIN_ID matches."
        )
        raise TimeoutError(f"no observations after {timeout:.0f}s:\n{detail}{hint}")

    def read_joints(self, names: list[str]) -> list[float]:
        """Positions for `names`, 0.0 where the sim does not publish that joint.

        The fold_cloth scene welds body and head (``convert_joints_to_fixed``),
        so those DoFs are legitimately absent from /joint_states.
        """
        return [float(self.joints.get(n, 0.0)) for n in names]

    def send_command(self, targets: dict[str, float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(targets)
        msg.position = [float(v) for v in targets.values()]
        self.cmd_pub.publish(msg)

    def request_reset(self) -> None:
        """Reset every world. Empty payload is the wire's 'all worlds'."""
        self.reset_pub.publish(Int32MultiArray(data=[]))


class InferenceClient:
    """CoRobot-protocol client: msgpack frames over a synchronous WebSocket."""

    def __init__(self, host: str, port: int | None) -> None:
        self.uri = f"ws://{host}:{port}" if port is not None else f"ws://{host}"
        self._ws = None

    def connect(self) -> None:
        if self._ws is not None:
            return
        self._ws = _ws_connect_compat(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=_OPEN_TIMEOUT_SEC,
            ping_interval=_PING_INTERVAL_SEC,
            ping_timeout=_PING_TIMEOUT_SEC,
        )

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def infer(self, payload: dict) -> dict:
        self.connect()
        self._ws.send(msgpack.packb(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"server error: {response}")
        result = msgpack.unpackb(response, raw=False)
        if result.get("error"):
            raise RuntimeError(f"server returned error: {result['error']}")
        return result["result"]


def build_payload(
    bridge: SimBridge,
    prompt: str,
    task_name: str,
    episode_idx: int,
    history: list[dict],
    hist_interval: int,
) -> dict:
    ts_ns = time.time_ns()
    payload = {
        "method": "infer",
        "params": {
            "timestamps": {"head": ts_ns, "states": ts_ns},
            "images": {cam: encode_image_jpeg(bridge.images[cam]) for cam in CAMERA_TOPICS},
            "states": {
                "head_joint_states": bridge.read_joints(HEAD_JOINTS),
                "arm_joint_states": bridge.read_joints(ARM_JOINTS),
                "waist_joint_states": bridge.read_joints(WAIST_JOINTS),
                "gripper_states": bridge.read_joints(GRIPPER_JOINTS),
                # FK-derived end_pose needs the benchmark's IK/FK solver, which
                # is not importable here. Servers that key off end_pose instead
                # of joint states will need the benchmark runner.
                "end_pose": None,
            },
            "prompt": prompt,
            "robot_type": ROBOT_TYPE,
            "task_name": task_name,
            "episode_idx": episode_idx,
            # No task evaluator outside the benchmark, so progress stays empty
            # and done is only ever reported at the step cap.
            "episode_done": False,
            "task_progress": [],
        },
    }
    if hist_interval > 0:
        payload["params"]["history"] = {"interval": hist_interval, "images": history}
    return payload


def parse_result(result: dict) -> tuple[list[dict], int]:
    """Flatten one response into (per-step target dicts, hist_frame_interval).

    Only JOINT_ABS is handled: EEF_ABS would need the IK solver from
    geniesim_benchmark. ``filter_abs_joint`` there returns the action unchanged,
    so JOINT_ABS values are applied verbatim -- as is wuji's gripper channel,
    whose post-process is a passthrough with a 0.0 offset.

    The server opts into history observations per response: a positive
    ``hist_frame_interval`` enables capture during the chunk about to be
    replayed and sets its sampling stride; 0 or missing disables it.
    """
    left, right = result.get("left_arm", {}), result.get("right_arm", {})
    kinds = {left.get("kind", "JOINT_ABS"), right.get("kind", "JOINT_ABS")}
    if kinds != {"JOINT_ABS"}:
        raise ValueError(f"only JOINT_ABS is supported here, server sent {sorted(kinds)} (EEF_ABS needs the IK solver)")

    left_vals, right_vals = np.asarray(left["values"]), np.asarray(right["values"])
    left_eff, right_eff = np.asarray(result["left_effector"]), np.asarray(result["right_effector"])

    def optional(key: str) -> np.ndarray | None:
        raw = result.get(key)
        if raw is None or (isinstance(raw, dict) and not raw.get("values")):
            return None
        arr = np.asarray(raw["values"] if isinstance(raw, dict) else raw)
        return arr if arr.size else None

    waist_vals = optional("waist")
    head_vals = optional("head")

    steps = []
    for i in range(left_vals.shape[0]):
        entry = {
            "arm": np.concatenate([left_vals[i], right_vals[i]]),
            "gripper": np.concatenate([left_eff[i], right_eff[i]]),
        }
        if waist_vals is not None:
            entry["waist"] = waist_vals[i]
        if head_vals is not None:
            entry["head"] = head_vals[i]
        steps.append(entry)

    try:
        hist_interval = max(int(result.get("hist_frame_interval", 0) or 0), 0)
    except (TypeError, ValueError):
        hist_interval = 0
    return steps, hist_interval


def to_targets(step: dict, drivable: set[str], skipped: set[str], log) -> dict[str, float]:
    """Zip an action chunk step onto joint names, dropping welded/absent DoFs."""
    targets: dict[str, float] = {}
    for key, names in (
        ("arm", ARM_JOINTS),
        ("gripper", GRIPPER_JOINTS),
        ("waist", WAIST_JOINTS),
        ("head", HEAD_JOINTS),
    ):
        values = step.get(key)
        if values is None:
            continue
        if len(values) != len(names):
            raise ValueError(f"{key}: server sent {len(values)} values, robot has {len(names)} joints")
        for name, value in zip(names, values):
            if name in drivable:
                targets[name] = value
            elif name not in skipped:
                skipped.add(name)
                log.warning(f"{name} absent from /joint_states (welded in this scene?) -- not commanding it")
    return targets


def run_episode(bridge: SimBridge, client: InferenceClient, args, episode_idx: int) -> int:
    log = bridge.get_logger()
    drivable = set(bridge.joints)
    skipped: set[str] = set()
    history: list[dict] = []
    hist_interval = 0
    step_period = 1.0 / args.fps
    step = 0

    while step < args.max_steps:
        payload = build_payload(bridge, args.instruction, args.task_name, episode_idx, history, hist_interval)
        history = []
        t0 = time.monotonic()
        # The interval applies to the chunk we are about to replay, so it takes
        # effect immediately rather than on the next response.
        chunk, hist_interval = parse_result(client.infer(payload))
        log.info(f"step {step}: chunk of {len(chunk)} in {time.monotonic() - t0:.2f}s")

        for offset, entry in enumerate(chunk):
            if step >= args.max_steps:
                break
            bridge.send_command(to_targets(entry, drivable, skipped, log))
            bridge.spin(step_period)
            step += 1
            # Sample the head camera on the server's requested cadence so the
            # frames ride along with the next payload.
            if hist_interval > 0 and (offset + 1) % hist_interval == 0 and "head" in bridge.images:
                history.append({"head": encode_image_jpeg(bridge.images["head"])})

    return step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--infer-host", required=True, metavar="HOST[:PORT]", help="inference server, e.g. 10.0.0.5:8999"
    )
    parser.add_argument("--instruction", default="Fold the cloth.", help="prompt sent to the policy")
    parser.add_argument("--task-name", default="cloth_fold_sim", help="task_name field in the payload")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600, help="per-episode step cap (no task evaluator here)")
    parser.add_argument("--fps", type=float, default=30.0, help="rate at which chunk steps are published")
    parser.add_argument(
        "--reset-between-episodes", action="store_true", help="publish /rlinf/reset_envs between episodes"
    )
    parser.add_argument("--wait-timeout", type=float, default=60.0, help="seconds to wait for the first observation")
    args = parser.parse_args()

    host, _, port_str = args.infer_host.rpartition(":")
    if host and port_str.isdigit():
        port = int(port_str)
    else:
        host, port = args.infer_host, None

    rclpy.init()
    bridge = SimBridge()
    client = InferenceClient(host, port)
    try:
        bridge.get_logger().info(f"waiting for observations (up to {args.wait_timeout:.0f}s)...")
        bridge.wait_for_observation(args.wait_timeout)
        shapes = ", ".join(f"{c}={bridge.images[c].shape}" for c in CAMERA_TOPICS)
        bridge.get_logger().info(f"observations up: {shapes}, {len(bridge.joints)} joints")

        client.connect()
        bridge.get_logger().info(f"connected to {client.uri}")

        for episode in range(args.episodes):
            if episode and args.reset_between_episodes:
                bridge.request_reset()
                bridge.spin(2.0)
            bridge.get_logger().info(f"=== episode {episode + 1}/{args.episodes} ===")
            steps = run_episode(bridge, client, args, episode)
            bridge.get_logger().info(f"episode {episode + 1} finished after {steps} steps")
    except KeyboardInterrupt:
        bridge.get_logger().info("interrupted")
    except (TimeoutError, RuntimeError, ValueError) as exc:
        bridge.get_logger().error(str(exc))
        return 1
    finally:
        client.close()
        bridge.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
