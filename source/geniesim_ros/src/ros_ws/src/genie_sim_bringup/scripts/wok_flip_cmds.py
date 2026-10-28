#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Wok-flip ("颠锅") motion choreography for the right arm via MoveIt IK.

Ports the OPEN-LOOP portion of newton's
``newton/examples/flex/chow_mein_flip_fem.py`` wok controller -- the
GATHER lead-in, the TOSS, and the return-to-home -- into a pure-numpy
Cartesian pose curve for the wok (held in the right hand), solves it to
right-arm (simple_arm_r / arm_r_end_link, 7 DoF) joint waypoints with
MoveIt's /compute_ik service, and replays them on /joint_command.

The closed-loop CATCH stage is intentionally dropped: it depends on the
noodle sensor feedback that only exists inside the newton sim, which this
ROS-side script has no access to.

Two-stage design
-----------------
  Stage A (offline solve, before any motion):
    1. Read /joint_states, /compute_fk the current arm_r_end_link pose ->
       the wok "home" pose. The flip curve's (delta-position, delta-pitch)
       is layered on top of it, ROTATED ABOUT THE WOK FOOD POINT (--pivot)
       so the food tips in place (demo-style scoop), not about the wrist.
    2. Sample the flip time-curve at the --rate PUBLISH SCHEDULE (one sample
       per publish frame) -> Cartesian pose targets in base_link.
    3. /compute_ik each target (previous solution seeds the next, so the
       redundant 7-DoF solutions stay continuous) -> one joint waypoint per
       publish frame.
    4. Validate: solve rate, joint-limit excursions, max |dq| between
       adjacent waypoints. --dry-run stops here and prints the report.

  Stage B (execute):
    Publish the pre-solved waypoints 1:1 at --rate Hz. There is NO joint-space
    interpolation: each frame already sits on the Cartesian arc (solved at this
    same schedule), so streaming them reproduces the arc exactly. A per-tick
    velocity clamp (VEL_ARM_CLAMP) only guards against a pathological IK jump.

Why one density, not two
------------------------
  Solving one IK per publish frame (instead of solving sparse corners and
  lerping between them in JOINT space) is what removes the toss-reversal
  "spiral": a joint-space chord between two sparse corners bows the wok-center
  off the Cartesian arc -- sideways in Y and down in Z -- worst at the reversal
  where the arm reconfigures fastest. --rate now sets BOTH the publish rate and
  the solve density; higher = smoother arc + more IK calls at solve time only.

Usage
-----
    # 1. bring up move_group + controllers + sim bridge first:
    ros2 launch genie_sim_moveit wbc.launch.py

    # 2. run the flip with the tuned defaults (no args needed):
    ros2 run genie_sim_bringup wok_flip_cmds.py

    # 3. reachability dry-run (solve + validate only, no motion):
    ros2 run genie_sim_bringup wok_flip_cmds.py --dry-run

    # tune the flip vigor / tempo (everything has a sensible default):
    ros2 run genie_sim_bringup wok_flip_cmds.py --toss-pitch 0.12 --time-scale 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionIK

# ---------------------------------------------------------------------------
# Right-arm planning group (see genie.srdf / kinematics.yaml).
# NOTE: the planning frame is the SRDF model root "odom" (planar_joint's
# parent_frame), NOT base_link. move_group's /compute_fk and /compute_ik
# return error -21 (FRAME_TRANSFORM_FAILURE) when asked to express poses in
# base_link, but succeed in "odom" (verified). The chassis sits at the origin
# so the numeric pose is identical either way; only the frame label differs.
GROUP = "simple_arm_r"
TIP_LINK = "arm_r_end_link"
BASE_FRAME = "odom"
ARM_JOINTS = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

_BEST_EFFORT_QOS = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)

# ---------------------------------------------------------------------------
# Flip controller knobs (ported verbatim from chow_mein_flip_fem.py:82-115).
# Times are in the newton sim's "natural" wok timescale; --time-scale stretches
# them so the arm PD can track. Only the OPEN-LOOP gather->toss->return phases
# are used here.
TOSS_T = 0.40
DWELL_T = 0.35

TOSS_DIP = 0.010
TOSS_LIFT = 0.075
TOSS_FWD = 0.010
TOSS_BACK = 0.015
TOSS_PITCH = 0.38

GATHER_T = 0.7
GATHER_PITCH = 0.34
GATHER_LIFT = 0.012

# ---------------------------------------------------------------------------
# Scene-fixed constants for the welded chef wok (no longer CLI args — they are
# determined by the scene geometry / weld, not tuned per run):
#   FLIP_AXIS  : the wok sits LEVEL with its handle along base -X, so base +Y is
#                its true lateral axis -> a pure fore/aft tip with ZERO roll.
#   PITCH_SIGN : -1 tips the pan so food is tossed up-and-back toward the cook
#                (the correct direction for how the wok is welded in the hand).
#   FOOD_PIVOT : base/odom point the flip rotates about = the wok FOOD CONTACT
#                point (home (0.9949,0,1.090) minus food depth 0.090). The wok
#                tips around this so food stays put and the wrist traces the arc.
#                Change this only if the wok spawn / geometry moves.
#   VEL_ARM_CLAMP : per-tick joint-velocity SAFETY clamp (deg/s). Frames are
#                solved one-per-publish-tick on the Cartesian arc, so a normal
#                flip needs <=~7 deg/tick; this only catches a pathological IK
#                redundant-branch jump. NOT a tuning knob (lowering it re-creates
#                the toss-reversal "spiral").
FLIP_AXIS = np.array([0.0, 1.0, 0.0])
PITCH_SIGN = -1.0
FOOD_PIVOT = np.array([0.9949, 0.0, 1.000])
VEL_ARM_CLAMP = 500.0


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw), ported from chow_mein_flip_fem.py:121-164.
# ---------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector `v` by xyzw quaternion `q`. Ported from chow_mein_flip_fem.py:132."""
    x, y, z, w = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_about_vec(axis: np.ndarray, theta: float) -> np.ndarray:
    """xyzw quaternion for rotation `theta` about an arbitrary world axis vector."""
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.asarray(axis, dtype=float) / n
    s = math.sin(0.5 * theta)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(0.5 * theta)])


def smootherstep(s: float) -> float:
    """Perlin's 5th-order smoothstep: 6s^5 - 15s^4 + 10s^3.

    Like smoothstep its FIRST derivative is 0 at both ends, but its SECOND
    derivative is ALSO 0 at both ends. So when two of these segments meet
    (as in _toss_pose's p=0.40 / p=0.70 seams) both velocity AND acceleration
    are continuous across the seam (C2), instead of smoothstep's C1 (velocity
    continuous, acceleration jumps). The acceleration jump at p=0.70 was an
    infinite-jerk impulse that the real arm executed as a ~5deg wok yaw / roll
    twist at the toss reversal (the residual "spiral"); the demo never saw it
    because its wok is kinematic (no arm dynamics). C2 seams remove that jerk.
    """
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


# ---------------------------------------------------------------------------
# Open-loop flip curve: f(t) -> (dpos[3], dpitch) relative to the wok home.
# Built from three phases laid end to end:
#   gather  (GATHER_T): pitch 0 -> -GATHER_PITCH, dz 0 -> GATHER_LIFT
#   toss    (TOSS_T)  : the chow_mein_flip_fem.py:672-688 _toss_pose curve
#   return  (DWELL_T) : dx/dz/pitch smoothly back to 0
# dz is "up" (maps to base +z); dx is the in-plane fore/aft (maps to base +x).
# Pitch sign / axis are scene-fixed constants (PITCH_SIGN / FLIP_AXIS).
# ---------------------------------------------------------------------------
def _toss_pose(p: float) -> Tuple[float, float, float]:
    """Return (dx, dz, pitch) for toss phase relative position p in [0,1].
    Ported from chow_mein_flip_fem.py Controller._toss_pose (:672-688), but with
    smootherstep instead of smoothstep so the p=0.40 / p=0.70 seams are C2
    (acceleration-continuous). The demo's smoothstep leaves an acceleration jump
    at p=0.70 (+34.7 -> -13.3) = an infinite-jerk impulse; harmless for the
    demo's kinematic wok but the real arm executes it as a ~5deg yaw/roll twist
    at the reversal. See smootherstep()."""
    if p < 0.40:
        s = smootherstep(p / 0.40)
        dx = -TOSS_FWD * s
        dz = GATHER_LIFT - TOSS_DIP * s
        pitch = -GATHER_PITCH
    elif p < 0.70:
        s = smootherstep((p - 0.40) / 0.30)
        dx = -TOSS_FWD + (TOSS_FWD + TOSS_BACK) * s
        dz = (GATHER_LIFT - TOSS_DIP) + (TOSS_DIP - GATHER_LIFT + TOSS_LIFT) * s
        pitch = -GATHER_PITCH + (GATHER_PITCH + TOSS_PITCH) * s
    else:
        s = smootherstep((p - 0.70) / 0.30)
        dx = TOSS_BACK * (1.0 - s)
        dz = TOSS_LIFT * (1.0 - s)
        pitch = TOSS_PITCH * (1.0 - s)
    return dx, dz, pitch


class FlipCurve:
    """One open-loop flip.

    The PATH is parametrized by normalized progress ``u`` in [0,1]
    (:meth:`sample_norm`); the per-phase tempo lives ONLY in the t<->u map
    (:meth:`time_to_u`). Stage A walks the publish schedule in real time and
    converts each tick to ``u`` via :meth:`time_to_u`, so the fast toss is
    sampled at the same publish density as the rest -- denser in u where it is
    quick, exactly matching how the frames are streamed. Each phase's u-share is
    its NATURAL (unscaled) duration fraction."""

    def __init__(
        self,
        gather_scale: float,
        toss_scale: float,
        return_scale: float,
        toss_lift: float,
        toss_pitch: float,
        gather_pitch: float,
        toss_fwd: float = 0.0,
    ):
        global TOSS_LIFT, TOSS_PITCH, GATHER_PITCH
        TOSS_LIFT = toss_lift
        TOSS_PITCH = toss_pitch
        GATHER_PITCH = gather_pitch
        self.toss_fwd = toss_fwd  # extra fore/aft lunge during the throw (m); <0 = pull BACK toward the cook
        self.t_gather = GATHER_T * gather_scale
        self.t_toss = TOSS_T * toss_scale
        self.t_return = DWELL_T * return_scale
        self.duration = self.t_gather + self.t_toss + self.t_return

    @staticmethod
    def _ushares() -> Tuple[float, float]:
        tot = GATHER_T + TOSS_T + DWELL_T
        return GATHER_T / tot, TOSS_T / tot  # (gather share, toss share)

    @staticmethod
    def _lunge_shape(p: float) -> float:
        """0 -> 1 over the launch window (toss p in [0.4,0.7]) then 1 -> 0 in the
        return (p in [0.7,1.0]). Scaled by toss_fwd and added to the toss dx, so a
        negative toss_fwd gives a BACKWARD (-X, toward the cook) velocity right at
        the launch — the real '颠锅' pull-back that throws food toward center.
        smootherstep (C2) so it adds no acceleration jump at the seams either."""
        if p < 0.40:
            return 0.0
        if p < 0.70:
            return smootherstep((p - 0.40) / 0.30)
        return 1.0 - smootherstep((p - 0.70) / 0.30)

    def sample_norm(self, u: float) -> Tuple[float, float, float]:
        """(dx, dz, dpitch) at normalized flip progress u in [0,1]."""
        ug, ut = self._ushares()
        if u < ug:
            s = smootherstep(u / ug if ug > 0 else 1.0)  # C2 gather lead-in
            return 0.0, GATHER_LIFT * s, -GATHER_PITCH * s
        if u < ug + ut:
            p = (u - ug) / ut if ut > 0 else 1.0
            dx, dz, pitch = _toss_pose(p)
            dx += self.toss_fwd * self._lunge_shape(p)  # pull-back lunge (see _lunge_shape)
            return dx, dz, pitch
        # return phase: _toss_pose already lands at (0,0,0) at p=1, so hold home
        return 0.0, 0.0, 0.0

    def time_to_u(self, t: float) -> float:
        """Map real time t in [0, duration] to normalized progress u in [0,1]
        using the per-phase (scaled) durations — this is where the tempo lives."""
        ug, ut = self._ushares()
        if t < self.t_gather:
            return ug * (t / self.t_gather if self.t_gather > 0 else 1.0)
        t -= self.t_gather
        if t < self.t_toss:
            return ug + ut * (t / self.t_toss if self.t_toss > 0 else 1.0)
        t -= self.t_toss
        f = min(t / self.t_return, 1.0) if self.t_return > 0 else 1.0
        return (ug + ut) + (1.0 - ug - ut) * f


class WokFlipCommander(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("wok_flip_cmds")
        self.args = args
        self._states: Dict[str, float] = {}

        self._state_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_states, _BEST_EFFORT_QOS)
        self._cmd_pub = self.create_publisher(JointState, "/joint_command", _BEST_EFFORT_QOS)
        self._fk_cli = self.create_client(GetPositionFK, "/compute_fk")
        self._ik_cli = self.create_client(GetPositionIK, "/compute_ik")

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._states[name] = pos

    def _wait_for_states(self, timeout: float = 10.0) -> bool:
        t0 = time.monotonic()
        while rclpy.ok() and time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(j in self._states for j in ARM_JOINTS):
                return True
        return False

    def _arm_seed(self) -> List[float]:
        return [self._states.get(j, 0.0) for j in ARM_JOINTS]

    # --- service helpers ------------------------------------------------
    def _call(self, client, request, timeout: float = 5.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def _compute_fk(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names = [TIP_LINK]
        req.robot_state.joint_state.name = list(ARM_JOINTS)
        req.robot_state.joint_state.position = self._arm_seed()
        res = self._call(self._fk_cli, req)
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            return None
        p = res.pose_stamped[0].pose
        pos = np.array([p.position.x, p.position.y, p.position.z])
        quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        return pos, quat

    def _compute_ik(self, pos: np.ndarray, quat: np.ndarray, seed: List[float]) -> Optional[List[float]]:
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = GROUP
        r.ik_link_name = TIP_LINK
        r.avoid_collisions = True
        r.timeout.sec = 0
        r.timeout.nanosec = int(0.05 * 1e9)
        r.robot_state.joint_state.name = list(ARM_JOINTS)
        r.robot_state.joint_state.position = list(seed)
        ps = r.pose_stamped
        ps.header.frame_id = BASE_FRAME
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        res = self._call(self._ik_cli, req)
        if res is None or res.error_code.val != 1:
            return None
        name_to_pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if not all(j in name_to_pos for j in ARM_JOINTS):
            return None
        return [name_to_pos[j] for j in ARM_JOINTS]

    def _make_curve(self) -> "FlipCurve":
        """Build the flip curve. --toss-time-scale (when >0) makes ONLY the toss
        phase faster than gather/return (which use --time-scale), so the toss can
        be a real snap while the lead-in/return stay smooth and trackable."""
        a = self.args
        toss_ts = a.toss_time_scale if a.toss_time_scale and a.toss_time_scale > 0 else a.time_scale
        return FlipCurve(a.time_scale, toss_ts, a.time_scale, a.toss_lift, a.toss_pitch, a.gather_pitch, a.toss_fwd)

    # --- Stage A: offline solve ----------------------------------------
    def solve(self) -> Optional[List[List[float]]]:
        """Solve the whole flip (--cycles flips) to joint waypoints.
        Returns the waypoint list, or None on a fatal failure."""
        if not self._fk_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/compute_fk service unavailable")
            return None
        if not self._ik_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/compute_ik service unavailable")
            return None
        if not self._wait_for_states():
            self.get_logger().error("no /joint_states for the right arm")
            return None

        home = self._compute_fk()
        if home is None:
            self.get_logger().error("/compute_fk failed for the home pose")
            return None
        p0, q0 = home
        self.get_logger().info(
            json.dumps(
                {
                    "event": "home_pose",
                    "pos": [round(float(v), 4) for v in p0],
                    "quat": [round(float(v), 4) for v in q0],
                }
            )
        )

        curve = self._make_curve()
        seed = self._arm_seed()
        # Pivot the flip about a point between the wrist (p0) and the wok FOOD POINT
        # (F0=FOOD_PIVOT), set by --pivot-frac: 1.0 = full food pivot (demo-style
        # scoop, biggest wrist arc -> most IK redundancy/sway), 0.0 = rotate about
        # the wrist in place (smooth, no sway, but no scoop). d0 is the wrist's lever
        # arm about the chosen pivot; rotating it by q_d makes the wrist trace the
        # compensating arc. At home (q_d=identity, dx=dz=0) target_pos == p0 for ANY
        # frac, so the start pose is unchanged.
        F0 = FOOD_PIVOT
        frac = float(self.args.pivot_frac)
        pivot = p0 + frac * (F0 - p0)
        d0 = p0 - pivot  # == frac * (p0 - F0)
        self.get_logger().info(
            json.dumps(
                {
                    "event": "pivot",
                    "F0": [round(float(v), 4) for v in F0],
                    "pivot_frac": round(frac, 3),
                    "pivot_eff": [round(float(v), 4) for v in pivot],
                    "radius_m": round(float(np.linalg.norm(d0)), 4),
                }
            )
        )
        # Flip basis: rotate about base +Y (FLIP_AXIS). The wok sits LEVEL with its
        # handle along base -X, so base +Y is its true lateral axis -> the pan tips
        # purely fore/aft with ZERO sideways roll. Only the ROTATION uses this axis;
        # the fore/aft TRANSLATION below stays on base +X so the pull-back is straight
        # toward the robot, not diagonal.
        flip_axis = FLIP_AXIS
        self.get_logger().info(
            json.dumps(
                {
                    "event": "flip_basis",
                    "flip_axis": [round(float(v), 3) for v in flip_axis],
                }
            )
        )
        # Solve ONE IK per publish frame, sampling the Cartesian flip arc directly
        # at the execute() publish schedule. The frame sequence IS the publish
        # sequence (no later joint-space interpolation), so every published pose
        # sits exactly on the Cartesian arc -- this is what kills the toss-reversal
        # "spiral": joint-space lerping between sparse corners used to bow the
        # wok-center sideways (Y) and down (Z) off the arc. Frames are dense and
        # adjacent (small Cartesian step), so the seeded IK stays continuous.
        rate = max(1.0, self.args.rate)
        dt = 1.0 / rate
        curve_dur = curve.duration
        cycles = max(1, self.args.cycles)
        n_ticks = int(math.ceil(curve_dur * cycles * rate))
        waypoints: List[List[float]] = []
        fails = 0
        for k in range(n_ticks + 1):
            t = k * dt
            cyc = min(int(t // curve_dur) if curve_dur > 0 else 0, cycles - 1)
            un = curve.time_to_u(t - cyc * curve_dur)  # normalized progress within this flip
            dx, dz, dpitch = curve.sample_norm(un)
            dpitch *= PITCH_SIGN  # tip the pan so food is tossed up-and-back toward the cook
            q_d = quat_about_vec(flip_axis, dpitch)
            # Rotate about the handle-aligned flip_axis (no sideways ROLL), but
            # TRANSLATE in pure base axes: dx -> +X (fore/aft, straight toward/away
            # from the robot), dz -> +Z (up). Translating along the yawed handle
            # instead would add an unwanted left/right component to the pull-back.
            center = pivot + np.array([dx, 0.0, dz])
            target_pos = center + quat_rotate(q_d, d0)
            target_quat = quat_mul(q_d, q0)
            sol = self._compute_ik(target_pos, target_quat, seed)
            if sol is None:
                fails += 1
                self.get_logger().warn(
                    json.dumps(
                        {
                            "event": "ik_fail",
                            "cycle": cyc,
                            "frame": k,
                            "t": round(t, 3),
                            "dpitch": round(dpitch, 3),
                        }
                    )
                )
                if waypoints:
                    waypoints.append(list(waypoints[-1]))  # hold last good
                continue
            waypoints.append(sol)
            seed = sol  # continuity: seed next IK from this solution

        total = n_ticks + 1
        ok = total - fails
        self._report(waypoints, ok, total)
        return waypoints if waypoints else None

    def _report(self, waypoints: List[List[float]], ok: int, total: int) -> None:
        if not waypoints:
            self.get_logger().error(json.dumps({"event": "solve_report", "solved": f"0/{total}"}))
            return
        wp = np.asarray(waypoints)
        dq = np.abs(np.diff(wp, axis=0)) if len(wp) > 1 else np.zeros((1, len(ARM_JOINTS)))
        self.get_logger().info(
            json.dumps(
                {
                    "event": "solve_report",
                    "solved": f"{ok}/{total}",
                    "waypoints": len(waypoints),
                    "max_dq_rad": round(float(dq.max()), 4),
                    "max_dq_deg": round(float(math.degrees(dq.max())), 2),
                    "per_joint_range_deg": [
                        round(float(math.degrees(wp[:, k].max() - wp[:, k].min())), 1) for k in range(len(ARM_JOINTS))
                    ],
                }
            )
        )

    # --- Stage B: execute ----------------------------------------------
    def execute(self, waypoints: List[List[float]]) -> None:
        """Publish the pre-solved waypoints 1:1 at --rate Hz.

        Each waypoint is one IK solution sampled directly on the Cartesian arc
        at this same publish schedule (Stage A), so there is NO joint-space
        interpolation here -- we just stream the frames. Interpolating between
        sparse corners used to bow the wok-center off the arc at the toss
        reversal (the "spiral"); solving one frame per publish tick removes the
        gap entirely. VEL_ARM_CLAMP survives only as a per-tick safety clamp
        against a pathological IK jump; it does not shape the (already smooth) path.
        """
        rate = max(1.0, self.args.rate)
        dt = 1.0 / rate
        max_step = math.radians(VEL_ARM_CLAMP) * dt  # rad/tick safety clamp only

        wp = np.asarray(waypoints)
        n_wp = len(wp)
        published = np.array(self._arm_seed(), dtype=float)
        self.get_logger().info(
            json.dumps(
                {
                    "event": "execute_start",
                    "frames": n_wp,
                    "rate_hz": rate,
                    "duration_s": round(n_wp * dt, 2),
                    "vel_arm_deg_s": VEL_ARM_CLAMP,
                }
            )
        )
        for k in range(n_wp):
            if not rclpy.ok():
                break
            target = wp[k]
            # Per-tick safety clamp: frames are dense so |err| is tiny in normal
            # operation; this only catches a pathological IK redundant-branch jump.
            err = target - published
            step = np.clip(err, -max_step, max_step)
            published = published + step
            self._publish(published)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        self.get_logger().info(json.dumps({"event": "execute_done"}))

    def _publish(self, positions: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [float(v) for v in positions]
        self._cmd_pub.publish(msg)


# ---------------------------------------------------------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cycles", type=int, default=3, help="number of flips to perform (default 3)")
    p.add_argument(
        "--time-scale",
        type=float,
        default=2.0,
        help="stretch the gather/return tempo by this factor (default 2.0; >1 = slower, easier to track)",
    )
    p.add_argument(
        "--toss-time-scale",
        type=float,
        default=2.5,
        help="time-scale for the TOSS phase ONLY (0 = use --time-scale; default 2.5). Set this SMALLER "
        "than --time-scale to keep a smooth slow gather/return but make the toss a faster snap — "
        "the toss is what launches food (the wok must move fast to toss). Peak wok angular velocity "
        "(what flings food) scales as (gather_pitch + toss_pitch) / toss_time_scale.",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="publish rate Hz (default 60). Also the IK SOLVE density: solve() computes one IK per "
        "publish frame directly on the Cartesian arc, so the path is smooth with no "
        "joint-space interpolation. Higher rate = denser frames = smoother arc + more "
        "IK calls at solve time (zero extra runtime cost).",
    )
    p.add_argument("--toss-lift", type=float, default=0.07, help="toss vertical throw (m, default 0.07)")
    p.add_argument("--toss-pitch", type=float, default=0.08, help="toss flip pitch (rad, default 0.08)")
    p.add_argument(
        "--toss-fwd",
        type=float,
        default=-0.04,
        help="fore/aft LUNGE during the throw (m, default -0.04). NEGATIVE = pull the wok BACK toward "
        "the cook (base -X) right at the launch -> throws food up-and-BACK toward the wok "
        "center (the real 颠锅 pull-back), instead of letting it sail forward over the far "
        "rim. POSITIVE = lunge forward. Stay >= -0.12; beyond that the forward return-snap "
        "dominates and re-flings food.",
    )
    p.add_argument(
        "--gather-pitch",
        type=float,
        default=0.24,
        help="how far the pan tilts BACK during gather (rad, default 0.24). The toss swings the pan "
        "from +gather_pitch through to -toss_pitch, so the throw's total swing (and thus the "
        "peak angular velocity that flings food) is gather_pitch+toss_pitch. LOWER this to make "
        "the whole flip gentler; a big gather-pitch holds the pan tilted and slides food forward.",
    )
    p.add_argument(
        "--pivot-frac",
        type=float,
        default=1.0,
        help="how far the rotation pivot sits from the WRIST toward the wok FOOD point "
        "(0..1, default 1.0). 1.0 = full food pivot (demo scoop: bowl tips in place, food "
        "returns to center, straight pull-back); 0.0 = rotate about the wrist in place (bowl "
        "sweeps like a shovel, flings food forward). Dial DOWN to trade scoop vigor for less sway.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage A only: solve all IK + print the validation report, publish nothing",
    )
    p.add_argument(
        "--save",
        type=str,
        default="",
        help="after solving, write the joint waypoints + timing to this JSON file so you can "
        "--replay it later WITHOUT move_group / IK. Pairs well with --dry-run (solve + save, "
        "no motion).",
    )
    p.add_argument(
        "--replay",
        type=str,
        default="",
        help="skip IK entirely: load joint waypoints from this JSON (saved with --save) and "
        "execute them. Needs only the running sim (/joint_states + /joint_command) — no "
        "move_group / wbc.launch, so no /joint_command publisher conflict. Edit the file's "
        '"timing".rate to re-pace the replay without re-solving.',
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Waypoint cache: solve IK once, save, then --replay forever without move_group.
# The "timing" block records the knobs the run was solved with (for provenance);
# only "rate" affects a replay (it paces the 1:1 frame stream). The other keys are
# solve-time shape params already baked into the absolute joint waypoints, kept
# here so a saved file documents how it was generated. Edit "rate" to re-pace a
# replay without re-solving.
# ---------------------------------------------------------------------------
_TIMING_KEYS = (
    "time_scale",
    "toss_time_scale",
    "cycles",
    "toss_lift",
    "toss_pitch",
    "toss_fwd",
    "gather_pitch",
    "pivot_frac",
    "rate",
)


def _save_waypoints(path: str, waypoints: List[List[float]], args: argparse.Namespace) -> None:
    data = {
        "joint_names": list(ARM_JOINTS),
        "tip_link": TIP_LINK,
        "waypoints": [[float(v) for v in wp] for wp in waypoints],
        "timing": {k: getattr(args, k) for k in _TIMING_KEYS},
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _load_waypoints(path: str, args: argparse.Namespace) -> Optional[List[List[float]]]:
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[wok_flip_cmds] cannot read replay file {path!r}: {exc}", file=sys.stderr)
        return None
    names = data.get("joint_names", [])
    if list(names) != list(ARM_JOINTS):
        print(
            f"[wok_flip_cmds] WARNING: saved joint_names {names} != expected {list(ARM_JOINTS)}; "
            f"replaying in the saved order anyway — re-solve if the arm is wrong.",
            file=sys.stderr,
        )
    # Restore the timing knobs from the file onto args. Only "rate" changes a
    # replay (it paces the 1:1 frame stream); the rest are solve-time shape params
    # already baked into the waypoints and are restored purely so the values match
    # what the run was solved with. Edit the file's "timing".rate to re-pace.
    timing = data.get("timing", {}) or {}
    for k in _TIMING_KEYS:
        if k in timing:
            setattr(args, k, timing[k])
    return data.get("waypoints")


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = WokFlipCommander(args)
    try:
        # --- replay: no IK, no move_group, just the sim ---
        if args.replay:
            waypoints = _load_waypoints(args.replay, args)
            if waypoints is None:
                node.get_logger().error(f"failed to load replay file: {args.replay}")
                return 1
            node.get_logger().info(
                json.dumps(
                    {
                        "event": "replay_load",
                        "file": args.replay,
                        "waypoints": len(waypoints),
                        "time_scale": args.time_scale,
                        "rate": args.rate,
                    }
                )
            )
            if not node._wait_for_states():
                node.get_logger().error("no /joint_states for the right arm — is the sim running?")
                return 1
            node.execute(waypoints)
            return 0

        # --- solve (Stage A), optionally save, optionally execute ---
        waypoints = node.solve()
        if waypoints is None:
            node.get_logger().error("solve failed; aborting")
            return 1
        if args.save:
            _save_waypoints(args.save, waypoints, args)
            node.get_logger().info(json.dumps({"event": "saved", "file": args.save, "waypoints": len(waypoints)}))
        if args.dry_run:
            node.get_logger().info(json.dumps({"event": "dry_run_complete", "waypoints": len(waypoints)}))
            return 0
        node.execute(waypoints)
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
