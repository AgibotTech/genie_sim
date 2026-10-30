#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Publish the 20 movable joints per Wuji hand from ROS retargeting or SDK input.

Standalone output is /joint_command; combined teleop uses wuji.bridge.
Each glove has an independent retarget session. Two-glove publishing waits
for both first frames and orders the left hand before the right."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Mapping, Sequence

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
JOINT_COUNT = 4


def wuji_joint_name_map() -> dict[str, str]:
    """Return the complete external-Wuji -> GenieSim name mapping."""
    return {
        f"{side}_finger{finger_id}_joint{joint_id}": f"hand_{side[0]}_{finger_name}_joint{joint_id}"
        for side in ("left", "right")
        for finger_id, finger_name in enumerate(FINGER_NAMES, start=1)
        for joint_id in range(1, JOINT_COUNT + 1)
    }


JOINT_NAME_MAP = wuji_joint_name_map()
_SIDE_CODE = {"left": "l", "right": "r"}


def translate_joint_state(names, positions):
    """Translate a partial JointState, returning only known Wuji finger DOFs."""
    translated_names = []
    translated_positions = []
    for name, position in zip(names, positions):
        target = JOINT_NAME_MAP.get(name)
        if target is None:
            continue
        translated_names.append(target)
        translated_positions.append(float(position))
    return translated_names, translated_positions


def canonical_side_joints(side: str) -> tuple[str, ...]:
    """Return the 20 GenieSim Wuji joints in SDK firmware order."""
    normalized = str(side).lower()
    try:
        side_code = _SIDE_CODE[normalized]
    except KeyError as exc:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}") from exc
    return tuple(
        f"hand_{side_code}_{finger}_joint{joint}" for finger in FINGER_NAMES for joint in range(1, JOINT_COUNT + 1)
    )


def sdk_joint_command(
    qpos_by_side: Mapping[str, Sequence[float]],
) -> tuple[list[str], list[float]]:
    """Validate and merge SDK 20-vectors into one canonical command."""
    names: list[str] = []
    positions: list[float] = []
    for side in ("left", "right"):
        if side not in qpos_by_side:
            continue
        side_names = canonical_side_joints(side)
        values = list(qpos_by_side[side])
        if len(values) != len(side_names):
            raise ValueError(f"SDK retarget output for {side} must contain 20 values, got {len(values)}")
        values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"SDK retarget output for {side} contains a non-finite value")
        names.extend(side_names)
        positions.extend(values)
    if not names:
        raise ValueError("SDK command contains no hand data")
    return names, positions


def keypoints_from_skeleton(frame: Any) -> list[tuple[float, float, float]]:
    """Extract and validate one SDK hand skeleton as 21 xyz points."""
    try:
        points = [tuple(float(axis) for axis in joint.pose.position) for joint in frame.joints]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("SDK hand_skeleton has an invalid joint payload") from exc
    if len(points) != 21 or any(len(point) != 3 for point in points):
        raise ValueError(f"SDK hand_skeleton must have shape (21, 3), got ({len(points)}, ?)")
    if not all(math.isfinite(axis) for point in points for axis in point):
        raise ValueError("SDK hand_skeleton contains a non-finite value")
    return points


def drain_latest(subscription: Any) -> Any | None:
    """Drain a non-blocking SDK subscription and keep only its newest frame."""
    latest = None
    while True:
        frame = subscription.recv()
        if frame is None:
            return latest
        latest = frame


class SdkHandInput:
    """One glove subscription paired with an independent retarget session."""

    def __init__(self, side: str, subscription: Any, session: Any) -> None:
        self.side = side
        self.subscription = subscription
        self.session = session

    def poll_latest(self) -> list[float] | None:
        frame = drain_latest(self.subscription)
        if frame is None:
            return None
        import numpy as np

        keypoints = np.asarray(keypoints_from_skeleton(frame), dtype=np.float32)
        qpos = self.session.step(keypoints)
        # Reuse the final command validator before accepting a held value.
        _names, values = sdk_joint_command({self.side: qpos})
        return values

    def close(self) -> None:
        close = getattr(self.subscription, "close", None)
        if callable(close):
            close()


def _select_sdk_user(manager: Any, selector: str) -> tuple[Mapping[str, Any], bool]:
    """Switch SDK user by exact ID/display name and return prior state."""
    previous = manager.current_user()
    selector = str(selector).strip()
    if not selector:
        return previous, False
    matches = [
        user for user in manager.list_users() if user.get("user_id") == selector or user.get("display_name") == selector
    ]
    if not matches:
        raise ValueError(f"SDK user not found: {selector!r}")
    if len(matches) > 1:
        raise ValueError(f"SDK user name is ambiguous; pass its user_id instead: {selector!r}")
    manager.switch_user(matches[0]["user_id"])
    return previous, matches[0].get("user_id") != previous.get("user_id")


class WujiSdkBridge:
    """Native SDK input, retargeting and merged ROS publishing in one node."""

    def __init__(
        self,
        *,
        left_sn: str,
        right_sn: str,
        sdk_user: str,
        output_topic: str,
        rate_hz: float,
    ) -> None:
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState

        try:
            from wuji_sdk import HandModel, Handedness, RetargetSession, SdkManager
        except ImportError as exc:
            raise RuntimeError("wuji_sdk with RetargetSession is required; install/upgrade wuji-sdk first") from exc

        self.node = Node("wuji_sdk_bridge")
        self._joint_state_type = JointState
        try:
            self._manager = SdkManager.instance()
            self._previous_user, self._switched_user = _select_sdk_user(self._manager, sdk_user)
            current_user = self._manager.current_user()
            if current_user.get("is_default"):
                self.node.get_logger().warning(
                    "SDK Default user uses the generic hand model; pass --sdk-user "
                    "with a calibrated user for best tracking"
                )

            requested = {side: sn for side, sn in (("left", left_sn), ("right", right_sn)) if sn}
            self._inputs: dict[str, SdkHandInput] = {}
            self._latest: dict[str, list[float]] = {}
            for side, serial_number in requested.items():
                device = self._manager.connect(sn=serial_number, device_name=f"geniesim_glove_{side}")
                reported_side = str(device.hand_side().get()).strip().lower()
                if reported_side != side:
                    raise RuntimeError(
                        f"glove {serial_number} reports side={reported_side!r}, " f"but was passed as --{side}-sn"
                    )
                handedness = Handedness.Left if side == "left" else Handedness.Right
                session = RetargetSession.for_hand(HandModel.WujiHand, side=handedness)
                self._inputs[side] = SdkHandInput(side, device.hand_skeleton().subscribe(), session)

            qos = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self.publisher = self.node.create_publisher(JointState, output_topic, qos)
            self._started = False
            self._last_error: dict[str, float] = {}
            self.timer = self.node.create_timer(1.0 / rate_hz, self._on_timer)
            self.node.get_logger().info(
                f"native SDK bridge sides={tuple(self._inputs)} -> {output_topic} at {rate_hz:g} Hz"
            )
        except Exception:
            try:
                self.close()
            finally:
                self.node.destroy_node()
            raise

    def _on_timer(self) -> None:
        for side, source in self._inputs.items():
            try:
                qpos = source.poll_latest()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_error.get(side, 0.0) >= 2.0:
                    self.node.get_logger().error(f"{side} SDK frame rejected: {exc}")
                    self._last_error[side] = now
                continue
            if qpos is not None:
                self._latest[side] = qpos

        # Bimanual publishing begins only after both gloves have valid data.
        if any(side not in self._latest for side in self._inputs):
            return
        names, positions = sdk_joint_command(self._latest)
        message = self._joint_state_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = names
        message.position = positions
        self.publisher.publish(message)
        if not self._started:
            self.node.get_logger().info(
                f"publishing one merged command after first frame from {', '.join(self._inputs)}"
            )
            self._started = True

    def close(self) -> None:
        errors = []
        for source in getattr(self, "_inputs", {}).values():
            try:
                source.close()
            except Exception as exc:
                errors.append(exc)
        manager = getattr(self, "_manager", None)
        if manager is not None:
            try:
                manager.disconnect_all()
            except Exception as exc:
                errors.append(exc)
            if getattr(self, "_switched_user", False):
                try:
                    manager.switch_user(self._previous_user["user_id"])
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError(f"Wuji SDK cleanup failed: {errors}") from errors[0]


class WujiJointBridge:
    """Translate retargeter joint names for the standalone hand-control entry point."""

    def __init__(self, input_topic: str, output_topic: str, warn_period: float = 2.0):
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState

        self.node = Node("wuji_joint_bridge")
        qos = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.node.create_publisher(JointState, output_topic, qos)
        self.warn_period = max(0.1, warn_period)
        self.last_warn = 0.0

        def callback(msg):
            names, positions = translate_joint_state(msg.name, msg.position)
            if names:
                out = JointState()
                out.header = msg.header
                out.name = names
                out.position = positions
                self.publisher.publish(out)
            unknown = [name for name in msg.name if name not in JOINT_NAME_MAP]
            now = time.monotonic()
            if unknown and now - self.last_warn >= self.warn_period:
                self.node.get_logger().warning(f"Ignoring non-Wuji or fixed joint(s): {', '.join(unknown)}")
                self.last_warn = now

        self.subscription = self.node.create_subscription(JointState, input_topic, callback, qos)


def main() -> None:
    import rclpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("retargeting", "sdk"),
        default="retargeting",
        help="Input source (default: retargeting ROS JointState)",
    )
    parser.add_argument("--input-topic", default="/wuji_hand/joint_command")
    parser.add_argument("--output-topic", default="/joint_command")
    parser.add_argument("--left-sn", default="", help="Left glove serial number (SDK source)")
    parser.add_argument("--right-sn", default="", help="Right glove serial number (SDK source)")
    parser.add_argument(
        "--sdk-user",
        default="",
        help="Calibrated SDK user_id or exact display name (default: current user)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=500.0,
        help="SDK merged publish rate in Hz (default: 500)",
    )
    args = parser.parse_args()

    if args.source == "sdk":
        if not args.left_sn and not args.right_sn:
            parser.error("SDK source requires at least one of --left-sn or --right-sn")
        if not math.isfinite(args.rate) or args.rate <= 0:
            parser.error("--rate must be finite and positive")
    elif args.left_sn or args.right_sn or args.sdk_user:
        parser.error("--left-sn/--right-sn/--sdk-user require --source sdk")

    # Custom CLI options were already consumed by argparse; don't hand them
    # to rclpy as unknown ROS arguments.
    rclpy.init(args=[])
    bridge = None
    try:
        if args.source == "sdk":
            bridge = WujiSdkBridge(
                left_sn=args.left_sn,
                right_sn=args.right_sn,
                sdk_user=args.sdk_user,
                output_topic=args.output_topic,
                rate_hz=args.rate,
            )
        else:
            bridge = WujiJointBridge(args.input_topic, args.output_topic)
        rclpy.spin(bridge.node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if bridge is not None:
                try:
                    close = getattr(bridge, "close", None)
                    if callable(close):
                        close()
                finally:
                    bridge.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
