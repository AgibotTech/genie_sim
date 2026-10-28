#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Test driver for the parallel-jaw grippers via /joint_command.

Publishes JointState setpoints on the leader joints (one per side); the
follower / support joints are mimicked at the simulator's command-apply
stage (see the engine's mimic broadcast).

Robot-agnostic by construction: this script does NOT assume the leader is
prismatic-in-meters or revolute-in-radians. It just pushes whatever
setpoint you configure below to /joint_command. Per-robot tuning lives in
the constants block at the top of the file — set ``OPEN_POS`` /
``CLOSE_POS`` to the values your robot's gripper leader joint expects in
its native unit (m for prismatic, rad for revolute).

Examples
--------
    # toggle both grippers open<->closed forever
    ros2 run genie_sim_bringup gripper_cmds.py both toggle

    # open just the left gripper
    ros2 run genie_sim_bringup gripper_cmds.py left open

    # close both
    ros2 run genie_sim_bringup gripper_cmds.py both close

    # send a custom raw setpoint (no clamping beyond the safety bounds below)
    ros2 run genie_sim_bringup gripper_cmds.py right set --pos -0.85

    # one-shot: send the command once and exit (default re-publishes at 50 Hz)
    ros2 run genie_sim_bringup gripper_cmds.py both open --once
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# Per-robot tuning. Pick a preset with ``ACTIVE_ROBOT`` below.
#
# The joint NAMES must match the leader names exposed by the staged robot USD.
# URDF→USD imports prefix every joint with ``idxNN_`` (ascending along the
# kinematic tree); legacy hand-authored USDs do not. Open the console log right
# after bringup — the "[genie_sim_engine] joints …" line lists the exact names.
#
# These used to be six bare module constants, each assigned three times in a
# row; Python kept only the LAST assignment, so the file silently shipped the
# ``gripper_active_master_joint`` preset. That joint does not exist on the G2
# swiftpicker/omnipicker URDFs, so every command was dropped as `unknown` and
# the jaws never moved. Presets are now a dict — adding a robot can't shadow
# another one.
#
# open / close are in the leader joint's native unit:
#   * prismatic gripper → meters  (e.g. open=0.0, close=0.024)
#   * revolute  gripper → radians (e.g. open=0.0, close=-0.85)
# ``limits`` should match the leader joint's URDF lower/upper; order doesn't
# matter (the clamp takes min/max).
#
# On ``close`` past first contact: commanding all the way to the kinematic
# limit is DELIBERATE for grasping. The swiftpicker master runs kp=2000 with
# actuator force capped at ±50 N·m, so it saturates at only 0.025 rad of
# position error. A 19 mm object blocks the jaws around -0.733; commanding
# -0.91 leaves ~0.18 rad of standing error, which pins the actuator at its
# 50 N·m ceiling — that saturation IS the grip force. Command only to
# first-contact and you get a limp hold.
_PRESETS = {
    # G2 with swiftpicker or omnipicker (URDF→USD, idxNN_ prefixed).
    # Master URDF limits: lower=-0.91, upper=0.0 → 0.0 is fully open.
    # Measured on the live mjxpbd chef scene (tf2_echo between
    # gripper_r_left_support_link and gripper_r_right_support_link):
    #     command  0.0   → pad-link separation 0.141 m  (jaws OPEN)
    #     command -0.91  → pad-link separation 0.051 m  (jaws CLOSED)
    # NOTE the /joint_states readback for the master reports +0.91 after a
    # -0.91 command: the staged USD's joint frame is mirrored relative to the
    # URDF axis. The mechanism closes correctly — don't "fix" the sign here
    # based on the readback alone, measure the pads.
    "g2": {
        "left": "idx31_gripper_l_inner_joint1",
        "right": "idx71_gripper_r_inner_joint1",
        "open": 0.0,
        "close": -0.91,
        "limits": (-0.91, 0.0),
    },
    # Hand-authored USD naming, no idxNN_ prefix, prismatic jaws in meters.
    "legacy_active_joint": {
        "left": "gripper_l_active_joint",
        "right": "gripper_r_active_joint",
        "open": 0.024,
        "close": 0.0,
        "limits": (0.0, 0.024),
    },
    # Single shared master joint name for both sides.
    "legacy_shared_master": {
        "left": "gripper_active_master_joint",
        "right": "gripper_active_master_joint",
        "open": 0.8,
        "close": 0.0,
        "limits": (0.0, 0.8),
    },
}

ACTIVE_ROBOT = "g2"

_P = _PRESETS[ACTIVE_ROBOT]
LEFT_LEADER = _P["left"]
RIGHT_LEADER = _P["right"]
OPEN_POS = _P["open"]
CLOSE_POS = _P["close"]
LIMIT_A, LIMIT_B = _P["limits"]

DEFAULT_RATE_HZ = 50.0
TOGGLE_PERIOD_SEC = 2.0
# ---------------------------------------------------------------------------


def _sides(which: str) -> list[str]:
    if which == "left":
        return [LEFT_LEADER]
    if which == "right":
        return [RIGHT_LEADER]
    return [LEFT_LEADER, RIGHT_LEADER]


def _clamp(x: float) -> float:
    # Order-agnostic clamp: works whether LIMIT_A < LIMIT_B (prismatic
    # 0.0..0.044 m) or LIMIT_A > LIMIT_B (revolute 0.0..-1.0 rad). Using
    # min/max instead of assuming the constants are pre-sorted prevents
    # the silent "everything clamps to one endpoint" failure mode that
    # broke the previous version (LIMIT_LOWER=0.0, LIMIT_UPPER=-1.0 →
    # max(0, min(-1, x)) ≡ 0.0 for all x).
    lo = min(LIMIT_A, LIMIT_B)
    hi = max(LIMIT_A, LIMIT_B)
    return max(lo, min(hi, x))


class GripperCommander(Node):
    def __init__(self, joints: list[str], rate_hz: float):
        super().__init__("gripper_cmds_test")
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, "/joint_command", qos)
        self.joints = joints
        self.period = 1.0 / max(1.0, rate_hz)
        self._positions: list[float] = [OPEN_POS] * len(joints)

    def set_positions(self, positions: list[float]) -> None:
        assert len(positions) == len(self.joints)
        self._positions = [_clamp(p) for p in positions]

    def publish_once(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joints)
        msg.position = list(self._positions)
        self.pub.publish(msg)

    def run_static(self, hold_sec: float) -> None:
        """Publish the current positions repeatedly for ``hold_sec`` seconds.

        We re-publish even though the receiver buffer is sticky — at a fresh
        bringup the publisher might race the subscriber's discovery, so a
        single one-shot can be silently dropped.
        """
        deadline = time.monotonic() + hold_sec
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_once()
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)

    def run_toggle(self, period_sec: float) -> None:
        """Alternate between OPEN and CLOSE every ``period_sec``.

        Order matters: set positions first, log the action that's about to
        be commanded, then publish. Logging after publishing would describe
        the *previous* state (the bug present in the first version of this
        script).
        """
        closing = False
        while rclpy.ok():
            target = CLOSE_POS if closing else OPEN_POS
            self.set_positions([target] * len(self.joints))
            self._log_state("closing" if closing else "opening")
            self.run_static(period_sec)
            closing = not closing

    def _log_state(self, action: str) -> None:
        self.get_logger().info(f"{action} {', '.join(self.joints)} -> {self._positions}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["left", "right", "both"], help="which gripper to drive")
    p.add_argument(
        "action",
        choices=["open", "close", "toggle", "set"],
        help="open/close=hold endpoint, toggle=alternate, set=use --pos",
    )
    p.add_argument(
        "--pos",
        type=float,
        default=None,
        help=(
            f"raw setpoint for 'set' (clamped to the safety bounds "
            f"[{min(LIMIT_A, LIMIT_B)}, {max(LIMIT_A, LIMIT_B)}] in the leader "
            f"joint's native unit). OPEN_POS={OPEN_POS}, CLOSE_POS={CLOSE_POS}."
        ),
    )
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="publish rate (Hz)")
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold open/close/set before exiting")
    p.add_argument("--toggle-period", type=float, default=TOGGLE_PERIOD_SEC, help="seconds per toggle phase")
    p.add_argument("--once", action="store_true", help="publish a single message and exit (no re-publish loop)")
    args = p.parse_args(argv)
    if args.action == "set" and args.pos is None:
        p.error("--pos is required when action=set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    joints = _sides(args.which)
    node = GripperCommander(joints, args.rate)

    if args.action == "open":
        node.set_positions([OPEN_POS] * len(joints))
    elif args.action == "close":
        node.set_positions([CLOSE_POS] * len(joints))
    elif args.action == "set":
        node.set_positions([float(args.pos)] * len(joints))

    try:
        if args.once:
            node.publish_once()
            time.sleep(0.05)  # let the middleware actually flush
        elif args.action == "toggle":
            node.run_toggle(args.toggle_period)
        else:
            node.run_static(args.hold)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
