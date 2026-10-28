#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Whole-body controller smoke-test: random goal sequences on /joint_command.

Subscribes to /robot_description to discover joints and their URDF limits,
tracks /joint_states, and drives the robot through a sequence of random goals.
A new goal is generated only after every joint reaches the current one
(|state - goal| < tol), or after --goal-timeout seconds.

Joint groups and default excursion ranges
-----------------------------------------
  body     ±3 deg   (conservative — heavy torso)
  head     ±15 deg  (light, sensor head)
  arm      ±20 deg  (arm links)
  hand     ±20 deg  (Wuji hand_l_* / hand_r_* fingers, independently actuated)
  gripper  full [lower, upper] — non-mimic joints only (mimic followers skipped at parse)
  chassis  ±30 deg  absolute steering range from neutral; --no-with-chassis to skip

Command modes (``--cmd-mode``)
------------------------------
  step   (default) Publish the goal directly.  Step input excites the
                   PD's full response — used for sim PD tuning to
                   measure overshoot, settling time, and oscillation
                   margins under instantaneous setpoint changes.
  ramp             Publish a velocity-limited interpolated setpoint per
                   tick.  Each joint's published command advances
                   toward the random goal at most ``vel_<group>``
                   rad/s per tick, anchored on the previously-
                   published value (continuous in command space).
                   Used for production-style smooth-trajectory
                   testing where the sim's PD never sees > ``sat_err``
                   of correction in one tick → no actuator saturation,
                   no induced overshoot.

Usage
-----
    ros2 run genie_sim_bringup wbc_cmds.py                              # default: step input
    ros2 run genie_sim_bringup wbc_cmds.py --cmd-mode ramp               # smooth trajectory
    ros2 run genie_sim_bringup wbc_cmds.py --cmd-mode ramp --vel-arm 15  # gentler arm ramp
    ros2 run genie_sim_bringup wbc_cmds.py --body-deg 3 --arm-deg 30
    ros2 run genie_sim_bringup wbc_cmds.py --chassis-deg 15
    ros2 run genie_sim_bringup wbc_cmds.py --no-with-chassis
    ros2 run genie_sim_bringup wbc_cmds.py --tol 0.03 --goal-timeout 8
    ros2 run genie_sim_bringup wbc_cmds.py --exclude head
    ros2 run genie_sim_bringup wbc_cmds.py --duration 60 --seed 42
    ros2 run genie_sim_bringup wbc_cmds.py --joints hand_l_ hand_r_ --cmd-mode ramp --hand-deg 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# ---------------------------------------------------------------------------
_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
_BEST_EFFORT_QOS = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)

# Name-substring → group key.  First match wins; order matters.
# Gripper classification is handled separately in _classify() to split
# active (commanded) from passive (follower) joints.
_GROUP_PATTERNS: List[Tuple[str, str]] = [
    ("chassis", "chassis"),
    ("arm", "arm"),
    ("head", "head"),
    ("body", "body"),
]

DEFAULT_BODY_DEG = 3.0
DEFAULT_HEAD_DEG = 15.0
DEFAULT_ARM_DEG = 20.0
DEFAULT_HAND_DEG = 20.0
DEFAULT_CHASSIS_DEG = 30.0
DEFAULT_GUARD = 0.005  # rad/m kept inside each limit
DEFAULT_TOL = 0.05  # rad/m — "goal reached" threshold per joint
DEFAULT_RATE_HZ = 20.0
DEFAULT_TIMEOUT = 5.0  # s — advance to next goal if not all converge
LOG_HZ = 1.0

# Per-group max joint velocity for ramp mode (deg/s).  Defaults sized
# so the slowest groups (body, arm) stay within their sim PD's
# saturation envelope at typical kp / max_effort settings.
#
# Saturation envelope = (max_effort / kp).  For the current
# physics_params.yaml::articulation_view_runtime values (Option α
# tuning — lowered arm kp to fit URDF effort caps cleanly):
#
#   * body      kp=1e5  / max_effort=1200 → sat_err = 0.012 rad = 0.69°
#   * arm_sh    kp=6000 / max_effort=108  → sat_err = 0.018 rad = 1.03°
#   * arm_mid   kp=2000 / max_effort=35   → sat_err = 0.0175 rad = 1.00°
#   * arm_wr    kp=1000 / max_effort=18   → sat_err = 0.018 rad = 1.03°
#   * head      kp=1e3  / max_effort=1200 → sat_err > 1 rad (effectively no cap)
#
# At pub rate 20 Hz (dt=50ms), per-tick command step = vel * 50ms.
# For arm joints (~1° envelope), 20°/s gives exactly one full
# envelope per tick — saturation fires on every step.  Keep
# vel ≤ 10°/s so per-tick step (0.5°) stays well INSIDE the
# envelope and PD never saturates.  Body / head / chassis have
# much wider envelopes and can run faster without saturating.
#
# Override any of these via --vel-<group> deg/s on the command line.
DEFAULT_VEL_BODY_DEG = 5.0  # heavy torso — slow even with big saturation envelope
DEFAULT_VEL_HEAD_DEG = 30.0  # light, no effective saturation
DEFAULT_VEL_ARM_DEG = 10.0  # main throttle — pick based on arm_shoulder sat envelope
DEFAULT_VEL_CHASSIS_DEG = 40.0  # steering wheel — fast snap to setpoint, kd=200 settles
DEFAULT_VEL_GRIPPER_DEG = 60.0  # gripper close/open — light master, mimic followers brake naturally
DEFAULT_VEL_HAND_DEG = 30.0  # independent Wuji finger joints
# ---------------------------------------------------------------------------


def _parse_urdf_joints(
    urdf_xml: str,
    exclude_patterns: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Return {name: (lower, upper)} for every revolute/prismatic non-mimic joint."""
    try:
        root = ET.fromstring(urdf_xml)
    except ET.ParseError:
        return {}
    joints: Dict[str, Tuple[float, float]] = {}
    for joint in root.iter("joint"):
        if joint.attrib.get("type", "") not in ("revolute", "prismatic"):
            continue
        name = joint.attrib.get("name", "")
        if not name or any(p in name for p in exclude_patterns):
            continue
        if joint.find("mimic") is not None:  # passive follower — skip
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            lower = float(limit.attrib.get("lower", "0"))
            upper = float(limit.attrib.get("upper", "0"))
        except ValueError:
            continue
        if lower < upper:
            joints[name] = (lower, upper)
    return joints


def _classify(name: str) -> str:
    """Return the group key for a joint name.

    Parallel grippers use full range; independently actuated Wuji fingers
    use the hand excursion. Mimic followers were excluded during URDF parsing.
    """
    if name.startswith(("hand_l_", "hand_r_")):
        return "hand"
    if "gripper" in name:
        return "gripper"
    for pat, group in _GROUP_PATTERNS:
        if pat in name:
            return group
    return "other"


class WbcCommander(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("wbc_cmds")

        self._guard: float = DEFAULT_GUARD
        self._tol: float = args.tol
        self._rate_hz: float = args.rate
        self._goal_timeout: float = args.goal_timeout
        self._duration: Optional[float] = args.duration
        self._start: float = time.monotonic()

        # Trajectory mode + per-group velocity limit (ramp mode only).
        # Ramp mode publishes a smoothed setpoint that advances at most
        # ``vel_<group> * dt`` toward the random goal each tick, anchored
        # on the previously published value so the trajectory is
        # continuous in command space across goal changes.  Step mode
        # publishes the goal directly (legacy behaviour).
        self._cmd_mode: str = args.cmd_mode
        self._vel: Dict[str, float] = {
            "body": math.radians(args.vel_body),
            "head": math.radians(args.vel_head),
            "arm": math.radians(args.vel_arm),
            "chassis": math.radians(args.vel_chassis),
            "gripper": math.radians(args.vel_gripper),
            "hand": math.radians(args.vel_hand),
            # "other" — fall through to body's slower velocity for
            # safety against unclassified joints.
            "other": math.radians(args.vel_body),
        }
        # Per-joint smoothed setpoint that lives across goal changes.
        # Initialized lazily from state when a joint first receives a
        # goal; thereafter advances by velocity-limited rate toward
        # the current goal each tick.  Reset on --cmd-mode step (we
        # just publish the goal verbatim).
        self._smoothed: Dict[str, float] = {}

        # Build exclude list: user patterns + chassis when disabled
        exclude = list(args.exclude or [])
        if not args.with_chassis:
            exclude.append("chassis")
        self._exclude: List[str] = exclude

        self._filter: List[str] = args.joints or []

        # per-group excursion in radians (None = full range)
        self._range: Dict[str, Optional[float]] = {
            "body": math.radians(args.body_deg),
            "head": math.radians(args.head_deg),
            "arm": math.radians(args.arm_deg),
            "chassis": math.radians(args.chassis_deg),
            "gripper": None,
            "hand": math.radians(args.hand_deg),
            "other": math.radians(args.body_deg),
        }

        self._joints: Dict[str, Tuple[float, float]] = {}  # name → (lower, upper)
        self._states: Dict[str, float] = {}  # name → latest position
        self._init_states: Dict[str, float] = {}  # name → position at first goal (anchor)
        self._goals: Dict[str, float] = {}  # name → current goal
        self._goal_count: int = 0
        self._goal_start: float = 0.0
        self._description_received: bool = False

        self._desc_sub = self.create_subscription(String, "/robot_description", self._on_description, _LATCHED_QOS)
        self._state_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_states, _BEST_EFFORT_QOS)
        self._cmd_pub = self.create_publisher(JointState, "/joint_command", _BEST_EFFORT_QOS)

        self.create_timer(1.0 / max(1.0, self._rate_hz), self._step)
        self.create_timer(1.0 / LOG_HZ, self._log_status)

    # ------------------------------------------------------------------
    def _on_description(self, msg: String) -> None:
        joints = _parse_urdf_joints(msg.data, self._exclude)
        if self._filter:
            joints = {k: v for k, v in joints.items() if any(f in k for f in self._filter)}
        self._joints = joints
        self._description_received = True
        groups = self._group_counts()
        by_group: Dict[str, List[str]] = {}
        for name, (lo, hi) in joints.items():
            g = _classify(name)
            by_group.setdefault(g, []).append(f"{name}[{lo:.3f},{hi:.3f}]")
        self.get_logger().info(json.dumps({"event": "description_received", "joints": len(joints), "groups": groups}))
        if not joints:
            self.get_logger().error(
                json.dumps(
                    {
                        "event": "no_matching_joints",
                        "filters": self._filter,
                        "exclude": self._exclude,
                        "hint": "Wuji fingers use --joints hand_l_ hand_r_; parallel jaws use gripper",
                    }
                )
            )
        # Log the active command mode + per-group velocity table so
        # the operator can verify what's running without re-reading
        # the CLI.  In step mode the velocity table is informational
        # only (not applied).
        self.get_logger().info(
            json.dumps(
                {
                    "event": "cmd_mode_active",
                    "mode": self._cmd_mode,
                    "vel_deg_per_s_in_ramp": {g: round(math.degrees(v), 2) for g, v in self._vel.items()},
                    "note": (
                        "step mode: PD sees instantaneous setpoint jump → measures overshoot/settling"
                        if self._cmd_mode == "step"
                        else "ramp mode: velocity-limited interpolation per tick"
                    ),
                }
            )
        )
        for g, names in sorted(by_group.items()):
            self.get_logger().info(json.dumps({"event": "joint_trace", "group": g, "joints": names}))

    def _on_joint_states(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._states[name] = pos

    # ------------------------------------------------------------------
    def _step(self) -> None:
        if self._duration is not None and (time.monotonic() - self._start) >= self._duration:
            self.get_logger().info(json.dumps({"event": "done", "goals_completed": self._goal_count}))
            rclpy.shutdown()
            return

        if not self._joints or any(name not in self._states for name in self._joints):
            return

        advance = False
        if not self._goals:
            advance = True
        elif self._all_reached():
            advance = True
        elif time.monotonic() - self._goal_start >= self._goal_timeout:
            stuck = [n for n, g in self._goals.items() if n in self._states and abs(self._states[n] - g) >= self._tol]
            stuck_detail = {
                n: {
                    "goal": self._goals[n],
                    "command": self._goals[n] if self._cmd_mode == "step" else self._smoothed.get(n, self._states[n]),
                    "state": self._states[n],
                    "error": self._goals[n] - self._states[n],
                }
                for n in stuck
            }
            self.get_logger().warn(
                json.dumps(
                    {
                        "event": "goal_timeout",
                        "goal": self._goal_count,
                        "stuck": stuck,
                        "stuck_detail": stuck_detail,
                        "timeout_s": self._goal_timeout,
                        "tolerance": self._tol,
                        "cmd_mode": self._cmd_mode,
                    }
                )
            )
            advance = True

        if advance:
            if not self._init_states:
                self._init_states = dict(self._states)
            self._goals = self._new_goals()
            self._goal_count += 1
            self._goal_start = time.monotonic()

        self._publish_goals()

    def _all_reached(self) -> bool:
        return bool(self._goals) and all(
            n in self._states and abs(self._states[n] - g) < self._tol for n, g in self._goals.items()
        )

    def _new_goals(self) -> Dict[str, float]:
        goals: Dict[str, float] = {}
        for name, (lower, upper) in self._joints.items():
            group = _classify(name)
            excursion = self._range.get(group)
            current = self._init_states.get(name, (lower + upper) * 0.5)
            lo_lim = lower + self._guard
            hi_lim = upper - self._guard
            if lo_lim >= hi_lim:
                lo_lim, hi_lim = lower, upper

            if group == "chassis":
                # absolute range from neutral (0), not relative to current position
                lo = max(lo_lim, -excursion)
                hi = min(hi_lim, excursion)
                goals[name] = random.uniform(lo, hi) if lo < hi else 0.0
            elif excursion is None:
                goals[name] = random.uniform(lo_lim, hi_lim)
            else:
                lo = max(lo_lim, current - excursion)
                hi = min(hi_lim, current + excursion)
                goals[name] = random.uniform(lo, hi) if lo < hi else current
        return goals

    def _publish_goals(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self._cmd_mode == "step":
            # Step mode — publish the final goal directly so the
            # simulator's PD sees an instantaneous setpoint jump.
            # Useful for measuring overshoot, settling time, and
            # oscillation margins.  Smoothed-setpoint state is not
            # used here; we reset it so a later switch back to
            # ramp mode starts from current state, not stale memory.
            self._smoothed.clear()
            msg.name = list(self._goals.keys())
            msg.position = list(self._goals.values())
        else:
            # Ramp mode — advance each joint's published command at
            # most ``vel_<group> * dt`` toward the goal.  Starting
            # value for a joint we've never seen: current measured
            # state if available, else the goal itself (no smoothing
            # possible without an anchor).  Continuous across goal
            # changes: subsequent ticks keep advancing from the LAST
            # PUBLISHED value, not from current state — this avoids
            # the published trajectory jumping backwards when the
            # measured state lags the command.
            dt = 1.0 / max(1.0, self._rate_hz)
            names: List[str] = []
            positions: List[float] = []
            for name, goal in self._goals.items():
                anchor = self._smoothed.get(name)
                if anchor is None:
                    anchor = self._states.get(name, goal)
                group = _classify(name)
                max_step = self._vel.get(group, self._vel["other"]) * dt
                err = goal - anchor
                if abs(err) <= max_step:
                    new = goal
                else:
                    new = anchor + math.copysign(max_step, err)
                self._smoothed[name] = new
                names.append(name)
                positions.append(new)
            msg.name = names
            msg.position = positions
        self._cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    def _log_status(self) -> None:
        elapsed = time.monotonic() - self._start

        if not self._description_received:
            status = "WAITING_FOR_DESCRIPTION"
        elif not self._joints:
            status = "NO_MATCHING_JOINTS"
        elif any(name not in self._states for name in self._joints):
            status = "WAITING_FOR_STATES"
        elif not self._goals:
            status = "INITIALIZING"
        else:
            status = "MOVING"

        by_group: Dict[str, List[int]] = {}
        for name, goal in self._goals.items():
            g = _classify(name)
            by_group.setdefault(g, [0, 0])
            by_group[g][1] += 1
            if name in self._states and abs(self._states[name] - goal) < self._tol:
                by_group[g][0] += 1
        group_summary = {g: f"{v[0]}/{v[1]}" for g, v in by_group.items()}

        total = len(self._goals)
        reached = sum(1 for n, g in self._goals.items() if n in self._states and abs(self._states[n] - g) < self._tol)

        record = {
            "t": round(elapsed, 2),
            "status": status,
            "goal": self._goal_count,
            "progress": f"{reached}/{total}",
            "missing_feedback": [n for n in self._joints if n not in self._states],
            "on_goal_s": round(time.monotonic() - self._goal_start, 1),
            **group_summary,
        }

        chassis_detail = {
            n: {
                "goal": round(g, 3),
                "state": round(self._states[n], 3) if n in self._states else "untracked",
            }
            for n, g in self._goals.items()
            if _classify(n) == "chassis"
        }
        if chassis_detail:
            record["chassis_detail"] = chassis_detail
        self.get_logger().info(json.dumps(record))

    def _group_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name in self._joints:
            g = _classify(name)
            counts[g] = counts.get(g, 0) + 1
        return counts


# ---------------------------------------------------------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--body-deg",
        type=float,
        default=DEFAULT_BODY_DEG,
        help=f"body joint excursion ±deg (default {DEFAULT_BODY_DEG})",
    )
    p.add_argument(
        "--head-deg",
        type=float,
        default=DEFAULT_HEAD_DEG,
        help=f"head joint excursion ±deg (default {DEFAULT_HEAD_DEG})",
    )
    p.add_argument(
        "--arm-deg", type=float, default=DEFAULT_ARM_DEG, help=f"arm joint excursion ±deg (default {DEFAULT_ARM_DEG})"
    )
    p.add_argument(
        "--hand-deg",
        type=float,
        default=DEFAULT_HAND_DEG,
        help=f"Wuji finger excursion ±deg from initial measured pose (default {DEFAULT_HAND_DEG})",
    )
    p.add_argument(
        "--chassis-deg",
        type=float,
        default=DEFAULT_CHASSIS_DEG,
        help=f"chassis absolute steering half-range ±deg from neutral (default {DEFAULT_CHASSIS_DEG})",
    )
    p.add_argument(
        "--with-chassis",
        dest="with_chassis",
        action="store_true",
        default=True,
        help="include chassis steering joints (default: on)",
    )
    p.add_argument(
        "--no-with-chassis", dest="with_chassis", action="store_false", help="exclude chassis steering joints"
    )
    p.add_argument(
        "--tol", type=float, default=DEFAULT_TOL, help=f"goal-reached tolerance rad/m (default {DEFAULT_TOL})"
    )
    p.add_argument(
        "--goal-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"advance after this many seconds even if not all joints converge (default {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ, help=f"command publish rate Hz (default {DEFAULT_RATE_HZ})"
    )
    p.add_argument(
        "--cmd-mode",
        choices=("step", "ramp"),
        default="step",
        help=(
            "trajectory style.  'step' (default) publishes the goal "
            "directly so the PD sees an instantaneous setpoint jump — "
            "used to characterise overshoot, settling, and ringing "
            "margins (the typical sim-tuning case).  'ramp' publishes "
            "a velocity-limited interpolated setpoint each tick, "
            "anchored on the previously published value — gentle on "
            "the sim's PD, no actuator saturation, matches a "
            "real-robot trajectory follower."
        ),
    )
    p.add_argument(
        "--vel-body",
        type=float,
        default=DEFAULT_VEL_BODY_DEG,
        help=(
            f"ramp-mode body joint velocity ceiling (deg/s, default " f"{DEFAULT_VEL_BODY_DEG}).  Ignored in step mode."
        ),
    )
    p.add_argument(
        "--vel-head",
        type=float,
        default=DEFAULT_VEL_HEAD_DEG,
        help=f"ramp-mode head joint velocity ceiling (deg/s, default {DEFAULT_VEL_HEAD_DEG})",
    )
    p.add_argument(
        "--vel-arm",
        type=float,
        default=DEFAULT_VEL_ARM_DEG,
        help=(
            f"ramp-mode arm joint velocity ceiling (deg/s, default "
            f"{DEFAULT_VEL_ARM_DEG}).  Bound by arm_shoulder's "
            f"saturation envelope at sim pub rate × deg/s — see "
            f"DEFAULT_VEL_ARM_DEG comment in source for full math."
        ),
    )
    p.add_argument(
        "--vel-chassis",
        type=float,
        default=DEFAULT_VEL_CHASSIS_DEG,
        help=f"ramp-mode chassis steering velocity ceiling (deg/s, default {DEFAULT_VEL_CHASSIS_DEG})",
    )
    p.add_argument(
        "--vel-gripper",
        type=float,
        default=DEFAULT_VEL_GRIPPER_DEG,
        help=f"ramp-mode gripper velocity ceiling (deg/s, default {DEFAULT_VEL_GRIPPER_DEG})",
    )
    p.add_argument(
        "--vel-hand",
        type=float,
        default=DEFAULT_VEL_HAND_DEG,
        help=f"ramp-mode Wuji finger velocity ceiling (deg/s, default {DEFAULT_VEL_HAND_DEG})",
    )
    p.add_argument("--duration", type=float, default=None, help="auto-stop after N seconds")
    p.add_argument("--exclude", nargs="*", metavar="PATTERN", help="skip joints whose name contains any of these")
    p.add_argument(
        "--joints", nargs="*", metavar="PATTERN", help="only command joints whose name contains one of these"
    )
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible sequences")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.seed is not None:
        random.seed(args.seed)

    rclpy.init()
    node = WbcCommander(args)
    try:
        rclpy.spin(node)
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
