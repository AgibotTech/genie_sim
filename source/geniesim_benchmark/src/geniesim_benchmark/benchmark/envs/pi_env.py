# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
import os
import logging
import numpy as np

from .dummy_env import DummyEnv
from .chassis_pose import SwervePosePlanner

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

# Chassis tracking diagnostics ([chassis-pose] per-step, [chassis-steer]
# per-step) are emitted at DEBUG level so they stay silent in normal runs.
# Set CHASSIS_DEBUG=1 to surface them for tuning/analysis.
if os.environ.get("CHASSIS_DEBUG"):
    logging.getLogger("ColoredLogger").setLevel(logging.DEBUG)

try:
    import osqp
    from scipy import sparse as _sp_sparse

    _OSQP_AVAILABLE = True
except Exception as _osqp_err:
    osqp = None
    _sp_sparse = None
    _OSQP_AVAILABLE = False
    logger.warning(f"osqp not available, swerve QP disabled: {_osqp_err}")

from geniesim_benchmark.benchmark.tasks.llm_task import LLMTask
from geniesim_benchmark.utils.ikfk_utils import ik_frame, arm_base_transform_from_poses

_STEER_JOINT_NAMES = [
    "idx111_chassis_lwheel_front_joint1",
    "idx121_chassis_lwheel_rear_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx141_chassis_rwheel_rear_joint1",
]
_DRIVE_JOINT_NAMES = [
    "idx112_chassis_lwheel_front_joint2",
    "idx122_chassis_lwheel_rear_joint2",
    "idx132_chassis_rwheel_front_joint2",
    "idx142_chassis_rwheel_rear_joint2",
]
_WHEEL_POSITIONS = np.array(
    [
        [0.23, 0.22],
        [-0.23, 0.22],
        [0.23, -0.22],
        [-0.23, -0.22],
    ]
)
_WHEEL_RADIUS = 0.07
# Real chassis translation limit (m/s), applied to the planar vector norm.
# Wheel acceleration/parking are enforced separately by the asset filter.
_MAX_LINEAR_SPEED = 1.0
_MAX_ANGULAR_SPEED = 1.57
_CHASSIS_OMEGA_SCALE = 1.0

_CHUNK_STEPS = 30
_STEP_DT = 1.0 / _CHUNK_STEPS

# POSE_DELTA uses chassis_pose.SwervePosePlanner with live actuator feedback.

# Closed-loop position hold on the in-chunk linear drift. _compute_step_correction
# measures (expected_dxy − actual_dxy) each tick and injects a corrective vx/vy
# = KP * err / horizon_dt. The policy commands ~0 base velocity during the pick,
# but the arm reaching forward/down to grasp pushes the free-rolling chassis
# backward (~10 cm/chunk, a pure physics arm-base reaction — the QP/wheels are
# commanded to 0, see [chassis-qp] v_cmd≈0 yet actual dx≈-0.11 m). With KP=0
# the velocity controller has no position feedback to resist it. KP=0.5 pushes
# the base back toward its anchor; bounded by the ±_CHASSIS_MAX_LINEAR_CORR
# (0.15 m/s) clamp + 5 mm deadband + correction LPF so it can't shove the base
# into the object mid-grasp. Dial down if it overshoots/disturbs the pick.
_CHASSIS_KP_LINEAR = 0.5
# Closed-loop catch-up on the in-chunk yaw deficit. _compute_step_correction
# measures (expected_dyaw − actual_dyaw) each tick and injects a corrective
# omega = KP * err / horizon_dt, so when the base lags its commanded heading
# (the swerve wheels are slew-limited and can't deliver yaw instantly) the
# controller actively spins faster to close the gap. KP=1.0 is deadbeat over
# the 6-step horizon; the real safety is the ±_CHASSIS_MAX_ANGULAR_CORR clamp
# (0.3 rad/s) + the 1° deadband + the correction LPF. Dial down toward ~0.5 if
# the yaw overshoots/oscillates on turn entry.
_CHASSIS_KP_ANGULAR = 1.0
_CHASSIS_CORR_HORIZON_STEPS = 6
# Raised 0.15→0.35 so the linear position-hold can actually counter the
# pick-phase backslide. With KP_LINEAR=0.5 the correction saturates at 0.15 m/s
# once the drift exceeds ~6 cm, which capped recovery and left ~10-20 cm of
# locked-in backslide when the arm reaches to grasp. 0.35 m/s lets it cancel the
# impulse within the chunk. Navigation is unaffected: its tracking lag stays
# <6 cm, below where the old 0.15 clamp ever bound.
_CHASSIS_MAX_LINEAR_CORR = 0.35
_CHASSIS_MAX_ANGULAR_CORR = 0.3
_CHASSIS_CORR_LPF_ALPHA = 0.3
_CHASSIS_ERR_DEADBAND_LINEAR = 0.005
_CHASSIS_ERR_DEADBAND_ANGULAR = np.deg2rad(1.0)
# One-pole IIR on the policy's (vx, vy, omega) before the QP. The policy
# emits a 30 Hz chunk where individual steps swing through e.g. 0 → 0.5
# rad/s → 0; without smoothing, _direct_cmd's per-step cmd_θ swings ±45° at
# 15 Hz, which the steer joints can't track cleanly and the QP's kinematic
# equality ends up forcing vy=wz=0 — every commanded rotation collapses
# into forward translation.
#
# Both channels use α=0.3 (τ≈100 ms, ~3 ticks). An earlier build set α_ang
# lower (0.15) on the assumption that steer joints were ~40,000× over-damped
# and needed the extra settling window — that was a misread of USD vs PhysX
# unit conventions; the asset's true damping (≈5000) yields τ≈50 ms, well
# inside one tick, so the larger α_ang only added pure latency.
_CHASSIS_LIN_LPF_ALPHA = 0.3
_CHASSIS_ANG_LPF_ALPHA = 0.3

# Mirror g2sim/genie_sim_chassis_controller (servo_4ws.yaml + servo_node.cpp).
# axis_distance / wheel_distance / wheel_radius /
# max_drive_speed match the g2sim production config exactly.
# Steering angle limits come from each loaded articulation joint, in radians.
# max_steer_speed and max_drive_accel set the per-tick QP slew
# (= limit * dt at 30 Hz). Raised above g2sim's 100 Hz values (12 rad/s,
# 20 rad/s²) to shorten the chunk-start startup delay: the base can't rotate
# until the wheels BOTH reorient (steer slew) AND spin up (drive accel), and
# the within-chunk residual profile showed the base barely moving for the
# first ~12 ticks. 18 rad/s steer → max_dθ 0.6 rad/tick (reorient 90° in
# ~3 ticks).
# max_drive_accel 90 rad/s² → max_dv 0.21 m/s/tick: reaches a 0.36 m/s nav
# cruise in ~2 ticks and the 1.0 m/s cap in ~5 ticks, halving the linear
# spin-up dead-time that left large-translation chunks ~30% short (chunk#1/#2/#6
# only covered ~70% of a 25 cm command). This is the QP's raw-command slew;
# when wheelParking is enabled, the asset filter subsequently enforces its
# own asset-defined maxWheelAcceleration at every physics step.
_QP_AXIS_DISTANCE = 0.46
_QP_WHEEL_DISTANCE = 0.44
_QP_MAX_STEER_SPEED = 24.0
_QP_MAX_DRIVE_SPEED = 1.0
_QP_MAX_DRIVE_ACCEL = 90.0
_QP_TWIST_WEIGHT = 10.0
_QP_CMD_WEIGHT = 1.0
# Penalty weight on the kinematic-constraint slack (see _OptimalSwerveSolver).
# The 8 kinematic rows used to be HARD equalities; at 30 Hz, on cold-start
# rotation with mid-reconfiguration (incoherent) steer angles and small
# mixed-sign drive speeds, the linearized equality became incompatible with
# the steer slew box → OSQP primal-infeasible (status -3). The fallback then
# froze the last (bad) wheel command, so v_prev never changed and every
# subsequent solve re-hit the same infeasible linearization — a permanent
# lock that froze the wheels and stopped the base from moving at all.
# Making the kinematic rows SOFT (J·x − s = b, with ‖s‖² penalized at this
# weight) keeps the QP always primal-feasible: it degrades to a best-effort
# twist instead of failing. 1e3 ≫ w_twist(10)/w_cmd(1) so kinematic
# consistency still dominates — residual stays ~1e-3 in normal operation.
_QP_SLACK_WEIGHT = 1.0e3
_QP_ALPHA_DIR_THRESH = float(np.cos(np.deg2rad(5e-2)))
# v_tilde bootstrap floor (m/s). Used only when |v_prev[i]| is below this AND
# u_ref demands non-trivial motion — keeps the kinematic Jacobian's θ columns
# (proportional to v_tilde) full-rank so the linearization can express the
# desired twist on cold-start or zero-crossings. g2sim runs at 100 Hz where the
# per-tick slew is small enough to avoid this corner; at 30 Hz larger jumps
# make the bootstrap necessary.
_QP_V_TILDE_FLOOR = 1e-2

# The USD asset caps the steer joints at ~4.08 rad/s (see [chassis-drive]
# props dump), well below the QP's software slew (_QP_MAX_STEER_SPEED 12 rad/s
# → max_dθ 0.4 rad/tick). At 30 Hz that physical cap means re-steering ~90° to
# enter a turn takes 1.57/4.08 ≈ 0.38 s ≈ 11 ticks (~1/3 of a chunk), which is
# the dominant source of yaw tracking lag — the wheels simply can't reorient as
# fast as the QP commands. Lift the physical cap to match the software limit so
# the wheels re-steer at the rate the QP already plans for.
_STEER_MAX_VELOCITY = 24.0
# The steer joints ship with maxEffort=5000. The realized re-steer rate caps
# at ~3-4 rad/s even on a joint whose maxVelocity was raised to 12 — i.e. the
# velocity cap is NOT the binding limit; the joint is effort/torque-saturated
# (huge position-drive stiffness vs the 5000 effort ceiling). Raise the effort
# ceiling so the steer drive can actually accelerate the wheel to the planned
# rate. At 5e4 the realized rate reached ~9.7 rad/s; raised to 2e5 alongside
# the higher QP steer slew (24 rad/s) to shorten the chunk-start reorientation.
_STEER_MAX_EFFORT = 2.0e5
# g2a_sim's steer joints ship with drive stiffness=0.0 (see the [chassis-drive]
# props dump: props=(...,maxVelocity,maxEffort,stiffness,damping) with
# stiffness=0.0) -- a position-drive with zero stiffness generates NO
# corrective torque toward a commanded joint_positions target regardless of
# _STEER_MAX_VELOCITY/_STEER_MAX_EFFORT, since those only raise the ceiling on
# a term that's multiplied by zero. That's why θ_cmd (this module's own
# steer target) swings ±0.3-0.8 rad every tick while θ_act (measured) stays
# essentially frozen: raising the effort/velocity ceiling assumed the OTHER
# robot's "huge position-drive stiffness" documented above, which isn't true
# here. Set explicitly, once, alongside that ceiling -- large enough that a
# few-degree error already saturates _STEER_MAX_EFFORT (matching the same
# saturate-the-ceiling behavior the ceiling override was tuned for).
_STEER_STIFFNESS = 1.0e6
# Damping to author alongside _STEER_STIFFNESS. MUST be a fixed constant, not
# read back from `articulation.dof_properties` and re-written -- that array
# reports damping in PhysX's per-RADIAN convention, while the USD DriveAPI
# attribute this gets written to (set_robot_joint_drive_gains ->
# UsdPhysics.DriveAPI) is authored in per-DEGREE convention (same rad<->deg
# mismatch already documented above for _CHASSIS_ANG_LPF_ALPHA). Reading the
# radian-convention number back and writing it into the degree-convention
# attribute multiplies the authored value by (180/pi) ~= 57.3 on every write;
# doing that once per episode reset compounds it exponentially (observed:
# 1,880,909,568 after one reset -> 107,768,184,832 after the next -- a 57.3x
# jump each time). The asset's real shipped damping, decoded the same way: the
# very first (pre-override) dof_properties dump read 572957.8125 rad-
# convention, i.e. 572957.8125 / 57.29578 = 10000.0 in the degree convention
# this constant is in.
_STEER_DAMPING = 1.0e4


def _limit_planar_speed(vx, vy):
    """Preserve direction while respecting the real chassis's 1 m/s limit."""
    scale = min(1.0, _MAX_LINEAR_SPEED / max(float(np.hypot(vx, vy)), 1e-12))
    return float(vx * scale), float(vy * scale)


def _normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _quat_wxyz_to_yaw(quat_wxyz):
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _swerve_forward(vx, vy, omega):
    vx = float(np.clip(vx, -_MAX_LINEAR_SPEED, _MAX_LINEAR_SPEED))
    vy = float(np.clip(vy, -_MAX_LINEAR_SPEED, _MAX_LINEAR_SPEED))
    omega = float(np.clip(omega, -_MAX_ANGULAR_SPEED, _MAX_ANGULAR_SPEED))

    steer_angles = np.zeros(4)
    drive_velocities = np.zeros(4)
    for i in range(4):
        wx, wy = _WHEEL_POSITIONS[i]
        vx_w = vx - omega * wy
        vy_w = vy + omega * wx
        speed = np.hypot(vx_w, vy_w)
        if speed > 1e-6:
            angle = np.arctan2(vy_w, vx_w)
            if abs(_normalize_angle(angle)) > np.pi / 2:
                steer_angles[i] = _normalize_angle(angle + np.pi)
                drive_velocities[i] = -speed / _WHEEL_RADIUS
            else:
                steer_angles[i] = angle
                drive_velocities[i] = speed / _WHEEL_RADIUS
    return steer_angles, drive_velocities


_WHEEL_RXS = np.array(
    [
        +0.5 * _QP_AXIS_DISTANCE,
        -0.5 * _QP_AXIS_DISTANCE,
        +0.5 * _QP_AXIS_DISTANCE,
        -0.5 * _QP_AXIS_DISTANCE,
    ]
)
_WHEEL_RYS = np.array(
    [
        +0.5 * _QP_WHEEL_DISTANCE,
        +0.5 * _QP_WHEEL_DISTANCE,
        -0.5 * _QP_WHEEL_DISTANCE,
        -0.5 * _QP_WHEEL_DISTANCE,
    ]
)


def _nearest_wheel_direction(vx, vy, previous, lower, upper):
    """Choose the least joint travel among legal equivalent rolling directions.

    Directions separated by pi are equivalent if rolling speed is reversed.
    Compare unwrapped joint coordinates: a bounded joint cannot cross its hard
    stop just because the wrapped angular distance is shorter.
    """
    speed = float(np.hypot(vx, vy))
    if speed < 1e-9:
        return float(np.clip(previous, lower, upper)), 0.0
    heading = float(np.arctan2(vy, vx))
    first = np.ceil((lower - heading) / np.pi)
    last = np.floor((upper - heading) / np.pi)
    if first <= last:
        turns = int(np.clip(np.rint((previous - heading) / np.pi), first, last))
        return heading + turns * np.pi, speed if turns % 2 == 0 else -speed
    # A joint with less than pi of travel may have no exact rolling direction.
    # Choose the boundary with least lateral mismatch, then least joint travel.
    theta = min((lower, upper), key=lambda t: (abs(np.sin(t - heading)), abs(t - previous)))
    return float(theta), float(vx * np.cos(theta) + vy * np.sin(theta))


def _direct_cmd(u_ref, steer_prev, drive_prev_lin, steer_limits):
    vx, vy, wz = float(u_ref[0]), float(u_ref[1]), float(u_ref[2])
    wz_abs = abs(wz)
    cmd_theta = np.zeros(4)
    cmd_v = np.zeros(4)
    alpha = 1.0
    for i in range(4):
        rx = _WHEEL_RXS[i]
        ry = _WHEEL_RYS[i]
        vxi = vx if wz_abs < 1e-9 else vx - wz * ry
        vyi = vy if wz_abs < 1e-9 else vy + wz * rx
        theta_prev = float(steer_prev[i])
        theta, v = _nearest_wheel_direction(vxi, vyi, theta_prev, *steer_limits[i])

        v = float(np.clip(v, -_QP_MAX_DRIVE_SPEED, _QP_MAX_DRIVE_SPEED))
        cmd_theta[i] = theta
        cmd_v[i] = v

        if wz_abs < 1e-2 or abs(float(drive_prev_lin[i])) > 1e-2:
            continue
        d_vec = np.array([np.cos(theta_prev), np.sin(theta_prev)])
        t_vec = np.array([np.cos(theta), np.sin(theta)])
        alpha_i = float(np.clip(d_vec @ t_vec, 0.0, 1.0))
        alpha_i = alpha_i if alpha_i > _QP_ALPHA_DIR_THRESH else 0.0
        alpha = min(alpha, alpha_i)

    cmd_v *= alpha
    return cmd_theta, cmd_v


def _limit_array_sync(cmd, prev, max_delta):
    """Synchronous proportional clamp on (cmd − prev), like g2sim limitArray."""
    cmd_arr = np.asarray(cmd, dtype=np.float64).copy()
    prev_arr = np.asarray(prev, dtype=np.float64)
    if max_delta <= 0.0 or cmd_arr.size == 0:
        return cmd_arr
    deltas = np.abs(cmd_arr - prev_arr)
    max_ratio = float(np.max(deltas)) / float(max_delta)
    if max_ratio > 1.0:
        cmd_arr = prev_arr + (cmd_arr - prev_arr) / max_ratio
    return cmd_arr


def _fmt_arr(a, fmt="{:+.3f}"):
    return "[" + ",".join(fmt.format(float(v)) for v in np.asarray(a).reshape(-1)) + "]"


class _OptimalSwerveSolver:
    """Single-step QP servo, mirrors g2sim::OptimalServo.

    Variables x ∈ R^19 = [θ_1..4, v_1..4, vx, vy, wz, s_1..8]; v_i is wheel
    linear speed (m/s). Convert to articulation rad/s with v / wheel_radius.
    s_1..8 are slacks on the kinematic rows (see below) — they are not part
    of the wheel command and are sliced off after the solve.

    Costs (weighted ‖·‖²): cmd-tracking on x[0:8] vs cmd_ref (from
    _direct_cmd, weight 1); twist-tracking on x[8:11] vs u_ref (weight 10);
    kinematic slack ‖s‖² (weight _QP_SLACK_WEIGHT, large).
    SOFT kinematic constraint (8 rows): linearized kinematic constraint
      v_i·cosθ_i − (vx − wz·ry_i) − s_2i   = 0
      v_i·sinθ_i − (vy + wz·rx_i) − s_2i+1 = 0
    linearized at (θ_actual, v_prev_lin). The slack makes these rows always
    satisfiable, so the QP can never be primal-infeasible; the large slack
    weight keeps the kinematic residual ~0 whenever a consistent solution
    exists.
    Hard inequality (11 rows = box on θ/v/twist only): θ ∈ [joint_lower, joint_upper]
    ∩ [θ_actual ± ω·dt]; v ∈ [-v_max, v_max] ∩ [v_prev ± a·dt]; twist box.
    Slacks are unconstrained (no box rows).
    """

    def __init__(self, dt, steer_limits):
        self.dt = float(dt)
        self.steer_limits = np.asarray(steer_limits, dtype=np.float64).reshape(4, 2).copy()
        self.max_delta_steer = _QP_MAX_STEER_SPEED * self.dt
        # max_drive_accel is angular (rad/s² at the wheel) per g2sim's
        # servo_node convention; convert to wheel linear m/s for the QP box.
        self.max_delta_drive = _QP_MAX_DRIVE_ACCEL * _WHEEL_RADIUS * self.dt
        self._n = 19
        self._m = 19
        self._w_cmd = _QP_CMD_WEIGHT
        self._w_twist = _QP_TWIST_WEIGHT
        self._w_slack = _QP_SLACK_WEIGHT

        # Cost Hessian P is diagonal and constant across ticks.
        self._P_csc = _sp_sparse.diags(
            np.array(
                [self._w_cmd] * 8 + [self._w_twist] * 3 + [self._w_slack] * 8,
                dtype=np.float64,
            ),
            format="csc",
        )

        # Build CSC sparsity pattern of A. Column-major order:
        #   cols 0..3 (θ_i)  : rows {2i, 2i+1, 8+i}
        #   cols 4..7 (v_i)  : rows {2i, 2i+1, 12+i}
        #   col  8    (vx)   : rows {0, 2, 4, 6, 16}
        #   col  9    (vy)   : rows {1, 3, 5, 7, 17}
        #   col 10    (wz)   : rows {0..7, 18}
        #   cols 11..18 (s_j): row {j}  (one -1 entry, slack on kinematic row j)
        a_indices = []
        a_indptr = [0]
        for i in range(4):
            a_indices.extend([2 * i, 2 * i + 1, 8 + i])
            a_indptr.append(len(a_indices))
        for i in range(4):
            a_indices.extend([2 * i, 2 * i + 1, 12 + i])
            a_indptr.append(len(a_indices))
        a_indices.extend([0, 2, 4, 6, 16])
        a_indptr.append(len(a_indices))
        a_indices.extend([1, 3, 5, 7, 17])
        a_indptr.append(len(a_indices))
        a_indices.extend([0, 1, 2, 3, 4, 5, 6, 7, 18])
        a_indptr.append(len(a_indices))
        for j in range(8):
            a_indices.append(j)
            a_indptr.append(len(a_indices))
        self._A_indices = np.asarray(a_indices, dtype=np.int64)
        self._A_indptr = np.asarray(a_indptr, dtype=np.int64)
        self._A_nnz = self._A_indices.size

        self._prob = None
        self._setup_failed = False
        self._first_solve_logged = False
        self.last_info = {}

    def _kin_residual_max(self, x):
        theta = x[0:4]
        v = x[4:8]
        vx, vy, wz = float(x[8]), float(x[9]), float(x[10])
        c = np.cos(theta)
        s = np.sin(theta)
        r1 = v * c - (vx - wz * _WHEEL_RYS)
        r2 = v * s - (vy + wz * _WHEEL_RXS)
        return float(max(np.max(np.abs(r1)), np.max(np.abs(r2))))

    def _build_A_data(self, theta_tilde, v_tilde):
        c = np.cos(theta_tilde)
        s = np.sin(theta_tilde)
        rxs = _WHEEL_RXS
        rys = _WHEEL_RYS
        d = np.empty(self._A_nnz, dtype=np.float64)
        k = 0
        for i in range(4):
            d[k] = -v_tilde[i] * s[i]
            k += 1
            d[k] = v_tilde[i] * c[i]
            k += 1
            d[k] = 1.0
            k += 1
        for i in range(4):
            d[k] = c[i]
            k += 1
            d[k] = s[i]
            k += 1
            d[k] = 1.0
            k += 1
        for _ in range(4):
            d[k] = -1.0
            k += 1
        d[k] = 1.0
        k += 1
        for _ in range(4):
            d[k] = -1.0
            k += 1
        d[k] = 1.0
        k += 1
        for i in range(4):
            d[k] = rys[i]
            k += 1
            d[k] = -rxs[i]
            k += 1
        d[k] = 1.0
        k += 1
        # slack columns: one -1 per kinematic row
        for _ in range(8):
            d[k] = -1.0
            k += 1
        return d

    def _build_q(self, cmd_ref, u_ref):
        q = np.zeros(self._n, dtype=np.float64)
        q[0:8] = -self._w_cmd * cmd_ref
        q[8:11] = -self._w_twist * u_ref
        # slacks (q[11:19]) have no linear term; cost is pure w_slack·‖s‖²
        return q

    def _build_lu(self, theta_tilde, v_tilde, theta_actual, v_prev):
        l = np.empty(self._m, dtype=np.float64)
        u = np.empty(self._m, dtype=np.float64)
        # Equality rows: J·x̃ − r(x̃) collapses to ∓v·sin(θ)·θ / +v·cos(θ)·θ.
        s = np.sin(theta_tilde)
        c = np.cos(theta_tilde)
        for i in range(4):
            b0 = -v_tilde[i] * s[i] * theta_tilde[i]
            b1 = v_tilde[i] * c[i] * theta_tilde[i]
            l[2 * i] = u[2 * i] = b0
            l[2 * i + 1] = u[2 * i + 1] = b1
        # θ box: hard limit ∩ slew about measured angle.
        for i in range(4):
            lower, upper = self.steer_limits[i]
            lo = max(lower, theta_actual[i] - self.max_delta_steer)
            hi = min(upper, theta_actual[i] + self.max_delta_steer)
            if lo > hi:
                # Recover an out-of-limit measured state to a legal boundary;
                # averaging the disjoint intervals can command an illegal angle.
                lo = hi = float(np.clip(theta_actual[i], lower, upper))
            l[8 + i] = lo
            u[8 + i] = hi
        # v box: hard limit ∩ slew about previous commanded drive speed.
        for i in range(4):
            lo = max(-_QP_MAX_DRIVE_SPEED, v_prev[i] - self.max_delta_drive)
            hi = min(+_QP_MAX_DRIVE_SPEED, v_prev[i] + self.max_delta_drive)
            if lo > hi:
                lo = hi = 0.5 * (lo + hi)
            l[12 + i] = lo
            u[12 + i] = hi
        # Twist box.
        l[16] = -_MAX_LINEAR_SPEED
        u[16] = +_MAX_LINEAR_SPEED
        l[17] = -_MAX_LINEAR_SPEED
        u[17] = +_MAX_LINEAR_SPEED
        l[18] = -_MAX_ANGULAR_SPEED
        u[18] = +_MAX_ANGULAR_SPEED
        return l, u

    def solve(self, u_ref, steer_actual, drive_prev_lin):
        if not _OSQP_AVAILABLE or self._setup_failed:
            return None

        theta_actual = np.asarray(steer_actual, dtype=np.float64).reshape(4)
        v_prev = np.asarray(drive_prev_lin, dtype=np.float64).reshape(4)
        u_ref_arr = np.asarray(u_ref, dtype=np.float64).reshape(3)

        theta_tilde = theta_actual.copy()

        cmd_theta, cmd_v = _direct_cmd(u_ref_arr, theta_tilde, v_prev, self.steer_limits)
        cmd_ref = np.concatenate([cmd_theta, cmd_v])

        # Linearize the kinematic equality at v_tilde. The Jacobian's θ_i
        # columns scale with v_tilde, so v_tilde[i]=0 collapses them and — when
        # the four steer angles differ — the 8 equality rows over-determine
        # the remaining 7 variables, pinning the QP to x=0. g2sim avoids this
        # in practice because at 100 Hz the per-tick slew can't push v_prev
        # to 0 in normal operation; at 30 Hz a sign reversal or rate-limit
        # corner can. Bootstrap from the un-damped per-wheel target speed
        # (signed by the current steer direction) so the linearization stays
        # full-rank whenever non-trivial motion is requested.
        v_tilde = v_prev.copy()
        if float(np.linalg.norm(u_ref_arr)) > 1e-3:
            for i in range(4):
                if abs(v_tilde[i]) >= _QP_V_TILDE_FLOOR:
                    continue
                vxi = u_ref_arr[0] - u_ref_arr[2] * _WHEEL_RYS[i]
                vyi = u_ref_arr[1] + u_ref_arr[2] * _WHEEL_RXS[i]
                mag = float(np.hypot(vxi, vyi))
                if mag < _QP_V_TILDE_FLOOR:
                    continue
                ci = float(np.cos(theta_tilde[i]))
                si = float(np.sin(theta_tilde[i]))
                proj = ci * vxi + si * vyi
                sgn = 1.0 if proj >= 0.0 else -1.0
                v_tilde[i] = sgn * max(mag, _QP_V_TILDE_FLOOR)

        A_data = self._build_A_data(theta_tilde, v_tilde)
        q = self._build_q(cmd_ref, u_ref_arr)
        l, u = self._build_lu(theta_tilde, v_tilde, theta_actual, v_prev)

        try:
            if self._prob is None:
                A_csc = _sp_sparse.csc_matrix(
                    (A_data, self._A_indices, self._A_indptr),
                    shape=(self._m, self._n),
                )
                prob = osqp.OSQP()
                # Mirror g2sim's OSQPSolver settings: rho=0.1, default eps,
                # no polish, no cross-tick warm start. polish + warm_start in
                # the previous build was the secondary failure mode — once
                # OSQP polished a solution to x≈0 (see degenerate
                # linearization above), the cached primal/dual seeded the
                # next solve back at 0 and the QP never recovered.
                prob.setup(
                    P=self._P_csc,
                    q=q,
                    A=A_csc,
                    l=l,
                    u=u,
                    verbose=False,
                    rho=0.1,
                    eps_abs=1e-3,
                    eps_rel=1e-3,
                    max_iter=4000,
                    polish=False,
                    warm_start=False,
                )
                self._prob = prob
                logger.info(
                    f"[chassis-qp] OSQP setup OK | n={self._n} m={self._m} "
                    f"nnz_A={self._A_nnz} dt={self.dt:.4f} "
                    f"max_dθ={self.max_delta_steer:.3f}rad max_dv={self.max_delta_drive:.3f}m/s"
                )
            else:
                self._prob.update(q=q, l=l, u=u, Ax=A_data)
            res = self._prob.solve()
        except Exception as e:
            logger.warning(f"swerve QP solve failed: {e}")
            self._prob = None
            self._setup_failed = True
            self.last_info = {"error": str(e)}
            return None

        if res is None or res.info.status_val not in (1, 2):
            lo_box = l[8:12]
            hi_box = u[8:12]
            self.last_info = {
                "status_val": int(res.info.status_val) if res is not None else None,
                "iter": int(res.info.iter) if res is not None else None,
                "theta_actual": tuple(float(v) for v in theta_actual),
                "v_prev": tuple(float(v) for v in v_prev),
                "v_tilde": tuple(float(v) for v in v_tilde),
                "cmd_theta": tuple(float(v) for v in cmd_theta),
                "theta_lo": tuple(float(v) for v in lo_box),
                "theta_hi": tuple(float(v) for v in hi_box),
            }
            return None
        x = getattr(res, "x", None)
        if x is None or not np.all(np.isfinite(x)):
            self.last_info = {"status_val": int(res.info.status_val), "x_nan": True}
            return None
        x = np.asarray(x, dtype=np.float64)
        self.last_info = {
            "status_val": int(res.info.status_val),
            "iter": int(res.info.iter),
            "cost": float(res.info.obj_val),
            "kin_residual_max": self._kin_residual_max(x),
            "u_ref": tuple(float(v) for v in u_ref_arr),
            "theta_actual": tuple(float(v) for v in theta_actual),
            "v_prev": tuple(float(v) for v in v_prev),
            "v_tilde": tuple(float(v) for v in v_tilde),
            "theta_solved": tuple(float(v) for v in x[0:4]),
            "v_lin_solved": tuple(float(v) for v in x[4:8]),
            "twist_solved": (float(x[8]), float(x[9]), float(x[10])),
            "cmd_ref_theta": tuple(float(v) for v in cmd_theta),
            "cmd_ref_v": tuple(float(v) for v in cmd_v),
        }
        if not self._first_solve_logged:
            self._first_solve_logged = True
            logger.info(
                f"[chassis-qp] first solve OK | status={self.last_info['status_val']} "
                f"iter={self.last_info['iter']} cost={self.last_info['cost']:.4f} "
                f"kinres={self.last_info['kin_residual_max']:.2e}"
            )
        return x[0:4].copy(), x[4:8].copy(), x[8:11].copy()


class PiEnv(DummyEnv):
    def __init__(self, api_core, task_file: str, init_task_config, need_setup=True):
        super().__init__(api_core, task_file, init_task_config, need_setup)
        self.load_task_setup()
        self._drive_props_logged = False
        self._robot_base_z = None
        self._reset_chassis_loop_state()
        # WBC tasks run the fixed-base USD but release the world anchor once the
        # robot is posed, so the chassis can drive without tipping during reset.
        self._free_base = bool(init_task_config.get("robot", {}).get("free_base", False))
        self._task_base_pose = self._pose_from_config(init_task_config.get("robot", {}).get("robot_init_pose"))
        if self._free_base and self._task_base_pose is None:
            logger.warning("free_base task has no robot.robot_init_pose; base will not be recentered between episodes")

    @staticmethod
    def _pose_from_config(pose):
        """(position, quaternion) out of a robot_init_pose dict, or None."""
        if not isinstance(pose, dict) or "position" not in pose or "quaternion" not in pose:
            return None
        return [float(v) for v in pose["position"]], [float(v) for v in pose["quaternion"]]

    def _reset_chassis_joints(self):
        """Clear the previous episode's steering and drive targets before release."""
        try:
            steer_idx = [self.robot_joint_indices[n] for n in _STEER_JOINT_NAMES]
            drive_idx = [self.robot_joint_indices[n] for n in _DRIVE_JOINT_NAMES]
        except KeyError:
            return
        articulation = self.api_core._get_articulation()
        if articulation is None:
            return
        self._configure_chassis_steering(articulation, steer_idx)
        self._apply_chassis_drive_overrides(articulation, steer_idx)
        positions = np.full(articulation.num_dof, np.nan)
        velocities = np.full(articulation.num_dof, np.nan)
        for i in range(4):
            positions[steer_idx[i]] = np.clip(0.0, *self._steer_limits[i])
            velocities[drive_idx[i]] = 0.0
        self.api_core.apply_chassis_action(positions, velocities)
        logger.info("chassis steer/drive targets cleared for reset")

    def _configure_chassis_steering(self, articulation, steer_idx):
        """Read effective asset limits once per reset/articulation replacement."""
        if self._steer_limits_articulation is articulation:
            return
        # Isaac's DOF properties are radians, unlike angular USD attributes.
        props = articulation.dof_properties
        limits = np.column_stack((props["lower"][steer_idx], props["upper"][steer_idx]))
        if (
            limits.shape != (4, 2)
            or np.any(np.isnan(limits))
            or np.any(limits[:, 0] > limits[:, 1])
            or np.any(np.isposinf(limits[:, 0]))
            or np.any(np.isneginf(limits[:, 1]))
        ):
            raise ValueError(f"Invalid chassis steering joint limits: {limits}")
        self._steer_limits = np.asarray(limits, dtype=np.float64)
        self._pose_planner = None
        self._swerve_qp = _OptimalSwerveSolver(_STEP_DT, self._steer_limits) if _OSQP_AVAILABLE else None
        self._steer_limits_articulation = articulation
        logger.info(f"[chassis-steer] asset limits (deg): {np.degrees(self._steer_limits).tolist()}")

    def _apply_chassis_drive_overrides(self, articulation, steer_idx):
        """Steer-joint stiffness + maxVelocity/maxEffort override (idempotent,
        safe to call repeatedly). See _STEER_STIFFNESS/_STEER_DAMPING/
        _STEER_MAX_VELOCITY/_STEER_MAX_EFFORT for why each is needed. Called
        from _reset_chassis_joints (every reset, robust to articulation
        recreation) and, as a fallback for non-free-base swerve robots that
        never go through that path, lazily once from _apply_chassis_velocity."""
        try:
            joint_gains = {}
            for n in _STEER_JOINT_NAMES:
                if self.robot_joint_indices.get(n) is None:
                    continue
                joint_gains[n] = (_STEER_STIFFNESS, _STEER_DAMPING)
            if joint_gains:
                self.api_core.set_robot_joint_drive_gains(joint_gains)
                logger.debug(f"[chassis-drive] steer stiffness->{_STEER_STIFFNESS:.0f} for {joint_gains}")
            else:
                logger.warning("[chassis-drive] steer stiffness override skipped -- no joint indices resolved")
        except Exception as e:
            logger.warning(f"[chassis-drive] steer stiffness override failed: {e}")

        try:
            view = getattr(articulation, "_articulation_view", None)
            if view is not None and hasattr(view, "set_max_joint_velocities"):
                n_steer = len(steer_idx)
                view.set_max_joint_velocities(
                    np.full(n_steer, _STEER_MAX_VELOCITY, dtype=np.float32),
                    joint_indices=np.asarray(steer_idx),
                )
                if hasattr(view, "set_max_efforts"):
                    view.set_max_efforts(
                        np.full(n_steer, _STEER_MAX_EFFORT, dtype=np.float32),
                        joint_indices=np.asarray(steer_idx),
                    )
                logger.debug(
                    f"[chassis-drive] steer maxVelocity->{_STEER_MAX_VELOCITY:.1f} rad/s "
                    f"maxEffort->{_STEER_MAX_EFFORT:.0f} for {n_steer} joints (by index)"
                )
            else:
                logger.warning(
                    "[chassis-drive] set_max_joint_velocities unavailable; " "steer speed cap left at asset default"
                )
        except Exception as e:
            logger.warning(f"[chassis-drive] steer drive-param override failed: {e}")

    def _episode_base_pose(self):
        """Where the base must stand at episode start.

        The init_base generalization overrides the task config when it hands us
        a concrete pose; its threshold-only form carries no pose and is applied
        once per episode file by apply_generalization, so ignore it here.
        """
        pose = self._pose_from_config(self.get_robot_init_pose()) or self._task_base_pose
        if pose is None or self._robot_base_z is None:
            return pose
        # The scene-generation pipeline places the robot on a 2D floor plan
        # and (for this robot at least) always hands back z=0 for "the
        # floor" -- fine for a point on paper, but base_link's origin isn't
        # at ground level for a wheeled robot (it's roughly at axle height),
        # so teleporting straight to z=0 buries the wheels _WHEEL_RADIUS deep
        # in the floor mesh. PhysX's collision response to that interpenetration
        # is what was showing up as the chassis getting violently kicked
        # right after reset (see _capture_robot_base_z). Keep the source
        # pose's x/y (that part IS the intended spawn point) but always use
        # the robot's own real standing height for z.
        position, quat = pose
        return [position[0], position[1], self._robot_base_z], quat

    def _capture_robot_base_z(self):
        """Cache the base-link reset height once.

        Use the observed articulation height for the episode reset.
        """
        if self._robot_base_z is not None:
            return
        try:
            articulation = self.api_core._get_articulation()
            if articulation is None:
                return
            position, _ = articulation.get_world_pose()
            self._robot_base_z = float(position[2])
            logger.info(f"[chassis-drive] captured robot standing height z={self._robot_base_z:.4f}m")
        except Exception as e:
            logger.warning(f"failed to capture robot standing height: {e}")

    def _reset_chassis_loop_state(self):
        # Set from the steer joints in get_observation; robots without a swerve
        # base stay False and never report/consume chassis state.
        self._has_chassis = False
        self._chunk_anchor_xy = None
        self._chunk_anchor_yaw = None
        self._chunk_expected_dx_world = 0.0
        self._chunk_expected_dy_world = 0.0
        self._chunk_expected_dyaw = 0.0
        self._last_base_xy = None
        self._last_base_yaw = None
        self._chassis_motion = None
        self._pose_planner = None
        self._chassis_corr_prev = (0.0, 0.0, 0.0)
        self._twist_lpf = np.zeros(3)
        self._chunk_index = 0
        self._last_steer_actual = np.zeros(4)
        self._prev_steer_actual_log = None  # for realized steer-rate diagnostic
        self._last_steer_cmd = np.zeros(4)
        self._last_drive_cmd_lin = np.zeros(4)
        self._chassis_log_counter = 0
        self._last_qp_status_logged = None
        self._swerve_qp = None
        self._steer_limits = None
        self._steer_limits_articulation = None
        # --- pose-target chassis mode ---
        self._chassis_anchor = None  # (x1_xy, x1_yaw), re-locked at every chunk head
        self._chassis_prev_target = None  # Last world target, used for boundary error reporting.
        self._servo_delta = None
        self._chassis_pose_delta = None
        # Length (in env-steps) of the chassis chunk currently anchored, and
        # how many steps into it we are. Latched from the policy's actual
        # chunk length at each chunk_start (see _begin_chassis_chunk) instead
        # of assuming the _CHUNK_STEPS fallback, so a real chunk size that
        # differs from 30 doesn't desync the anchor mid-chunk.
        self._chassis_active_chunk_len = _CHUNK_STEPS
        self._chassis_step_in_chunk = 0

    def load_task_setup(self):
        self.task = LLMTask(self)

    def get_observation(self, fetch_images=True):
        # C: single physics-loop round-trip for image + depth + joint_state,
        # vs serial run_on_physics_loop waits before. B: when fetch_images is
        # False (chunk replay between inferences) the policy won't consume
        # images, so we skip them.
        bundle = self.data_courier.get_obs_bundle(fetch_images=fetch_images)
        self._update_chassis_motion(bundle.get("chassis_motion"))
        images = bundle.get("images", {})
        depth = bundle.get("depth") or {}
        full_joint_states = bundle["joint_state"]

        def joint_values(names):
            return [full_joint_states[name] for name in names]

        steer_names = _STEER_JOINT_NAMES
        try:
            self._last_steer_actual = np.array(
                [float(full_joint_states[n]) for n in steer_names],
                dtype=np.float64,
            )
            self._has_chassis = bool(steer_names)
        except (KeyError, TypeError):
            self._has_chassis = False

        waist_joints = self.cfg["waist_joints"]
        gripper_values = joint_values(self.cfg["gripper_joints"])
        # Diagnostic companion to [gripper-cmd]: the measured joint, so a command
        # that never closes can be told apart from one the joint refuses to
        # follow. Same throttle, so the two lines interleave per step.
        _g_prev = getattr(self, "_gripper_obs_log", None)
        if (
            _g_prev is None
            or max(abs(a - b) for a, b in zip(gripper_values, _g_prev)) > 0.01
            or self.current_step % 90 == 0
        ):
            logger.info(
                f"[gripper-obs] step={self.current_step} " f"actual={[round(float(v), 4) for v in gripper_values]}"
            )
            self._gripper_obs_log = list(gripper_values)
        # gripper_joints is [left..., right...]; split in half so multi-DOF
        # (dexterous) hands map their whole finger set per side, and the 2-value
        # parallel gripper still yields 1 per side.
        n_half = len(gripper_values) // 2

        # Keyed states; policies flatten as left_arm + right_arm + left/right
        # gripper + waist + head (same order as the legacy flat list).
        states = {
            "left_arm": joint_values(self.cfg["left_arm_joints"]),
            "right_arm": joint_values(self.cfg["right_arm_joints"]),
            "left_gripper": gripper_values[:n_half],
            "right_gripper": gripper_values[n_half : 2 * n_half],
            "waist": joint_values(waist_joints),
            "head": joint_values(self.cfg["head_joints"]),
        }
        if getattr(self, "_has_chassis", False):
            # Chassis pose is reported relative to the chunk-start anchor, so
            # the current-state entry is always the zero delta.
            states["chassis"] = [0.0, 0.0, 0.0]
        self.cur_arm = states["left_arm"] + states["right_arm"]

        obs = {"images": images, "states": states}
        want_depth = bool(getattr(self, "need_depth", False))
        obs["depth"] = depth if want_depth else None
        if want_depth:
            obs["calibration"] = bundle.get("calibration") or {}
        obs["eef"] = self.ikfk_solver.compute_eef(self.cur_arm)
        if ik_frame(getattr(self, "robot_cfg", "")) == "base_link":
            # This embodiment's IK/FK already solves in base_link, so the two
            # declared EEF frames name the same frame. Pin the transform to
            # identity: arx has no arm_base_link link. Other embodiments
            # use their actual arm_base_link and measured torso transform.
            obs["arm_base_transform"] = np.eye(4)
        else:
            try:
                base_pos, base_rot = self.api_core._get_link_world_pose("base_link")
                arm_pos, arm_rot = self.api_core._get_link_world_pose("arm_base_link")
                obs["arm_base_transform"] = arm_base_transform_from_poses(base_pos, base_rot, arm_pos, arm_rot)
                if self._chassis_motion is None:
                    self._last_base_xy = (float(base_pos[0]), float(base_pos[1]))
                    self._last_base_yaw = _quat_wxyz_to_yaw(base_rot)
            except Exception as e:
                logger.warning(f"arm_base_transform unavailable: {e}")
        return obs

    def reset(self):
        self.api_core.run_on_physics_loop(self.api_core.set_chassis_servo_callback, None)
        self._capture_robot_base_z()
        self._followed_objects = set()
        self._picked_objects = set()
        self.last_update_time = time.time()
        self.has_done = False
        self._reset_chassis_loop_state()
        self.task.reset(self)
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()

        init_gripper = self.init_gripper_target()

        if self._free_base:
            # The chassis parks wherever the last episode left it, so drive the
            # base home first — then re-weld, because on a free base the torso
            # snap from the joint targets below would topple the robot.
            base_pose = self._episode_base_pose()
            if base_pose is not None:
                # Release before moving: the USD ships the weld enabled, so on
                # the first episode it would drag the base straight back off the
                # pose we just set.
                self.api_core.set_robot_base_anchored(False)
                self.api_core.reset_robot_base(*base_pose)
            self.api_core.set_robot_base_anchored(True)
            # Do this while the base is welded so the transient is absorbed.
            self._reset_chassis_joints()

        # Apply all initial joint targets in one physics-loop round-trip.
        # is_trajectory=False is a direct snap-to-target, so a single apply
        # is enough — replaying it (as the previous nested loop did) is a
        # no-op for the controller and just burns physics ticks.
        self.api_core.set_joint_positions_batched(
            [
                (self.init_arm, [self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], False),
                (self.init_waist, [self.robot_joint_indices[v] for v in self.cfg["waist_joints"]], False),
                (self.init_head, [self.robot_joint_indices[v] for v in self.cfg["head_joints"]], False),
                (init_gripper, [self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], False),
            ]
        )

        # Poll initial targets with early exit on convergence.
        steer_names = _STEER_JOINT_NAMES
        eps = 1e-2
        init_arm_arr = np.array(self.init_arm)
        init_waist_arr = np.array(self.init_waist)
        init_head_arr = np.array(self.init_head)
        init_gripper_arr = np.array(init_gripper)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            time.sleep(0.02)
            js = self.data_courier.get_joint_state_dict()
            if not js:
                continue
            try:
                arm = np.array([js[n] for n in self.cfg["arm_joints"]])
                waist = np.array([js[n] for n in self.cfg["waist_joints"]])
                gripper = np.array([js[n] for n in self.cfg["gripper_joints"]])
                head = np.array([])
            except KeyError:
                continue
            arm_ok = np.max(np.abs(arm - init_arm_arr)) < eps if arm.size else True
            waist_ok = np.max(np.abs(waist - init_waist_arr)) < eps if waist.size else True
            head_ok = np.max(np.abs(head - init_head_arr)) < eps if head.size else True
            gripper_ok = np.max(np.abs(gripper - init_gripper_arr)) < eps if gripper.size else True
            # Steer joints were just commanded to 0 (_reset_chassis_joints)
            # while still welded; wait for the measured position before
            # releasing the anchor. A non-swerve base has no steer joints ->
            # KeyError -> treat as converged (nothing to wait for). G2_90d
            # must report every configured steer joint.
            try:
                steer = np.array([js[n] for n in steer_names])
                steer_ok = np.max(np.abs(steer)) < eps if steer.size else True
            except KeyError:
                steer_ok = True
            if arm_ok and waist_ok and head_ok and gripper_ok and steer_ok:
                break
        logger.info("Finish reset robot...")

        self.api_core.reset_env()

        if self._free_base:
            # Belt-and-suspenders alongside the steer-convergence wait above:
            # that wait has a 1s deadline and can still time out short of
            # eps, and reset_env() just above may itself perturb joint state.
            # Force the drive target to match wherever every joint actually
            # is, right before unweld, so there is no residual error left for
            # a real (nonzero) drive stiffness to correct against a now-free
            # base -- see sync_drive_targets_to_current's docstring.
            self.api_core.sync_drive_targets_to_current(log_label="pre-unweld reset")
            self.api_core.set_robot_base_anchored(False)

        try:
            obs = self.get_observation()
        except Exception as e:
            logger.warning("get_observation failed after reset (%s) — retrying same-world read once", e)
            time.sleep(1.0)
            obs = self.get_observation()
        logger.info("Finish reset env...")
        return obs

    def apply_chassis_target(self, chassis, *, chunk_start=False):
        """Apply one benchmark chassis pose target."""
        self.api_core.check_chassis_servo()
        return self.api_core.run_on_physics_loop(self._submit_chassis_target, chassis, chunk_start)

    def set_policy_step_info(self, *, chunk_started=False, chunk_len=None):
        self._policy_chunk_started = bool(chunk_started)
        if chunk_len is not None and int(chunk_len) > 0:
            self._policy_chunk_len_steps = int(chunk_len)

    def step(self, action):
        if action is None:
            self.has_done = True
            return self.get_observation(), self.has_done, False, self.task.task_progress

        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        arm = action.get("arm")
        gripper = action.get("gripper")
        if gripper is not None:
            g = [float(v) for v in gripper]
            prev = getattr(self, "_gripper_cmd_log", None)
            if prev is None or max(abs(a - b) for a, b in zip(g, prev)) > 0.01 or self.current_step % 90 == 0:
                logger.info(f"[gripper-cmd] step={self.current_step} cmd={[round(v, 4) for v in g]}")
                self._gripper_cmd_log = g
        waist = action.get("waist")
        head = action.get("head")
        chassis = action.get("chassis")
        if chassis is not None and action.get("chassis_kind", "POSE_DELTA") != "POSE_DELTA":
            raise ValueError(f"Unsupported chassis kind: {action['chassis_kind']}")

        if hasattr(self, "_policy_chunk_len_steps"):
            chunk_start = self._policy_chunk_started
        else:
            chunk_start = (self.current_step - 1) % _CHUNK_STEPS == 0
        batch = []
        if arm is not None:
            batch.append(([float(v) for v in arm], [self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], True))
        if gripper is not None:
            batch.append(
                ([float(v) for v in gripper], [self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], True)
            )
        if waist is not None:
            names = self.cfg["waist_joints"][: len(waist)]
            batch.append(([float(v) for v in waist], [self.robot_joint_indices[v] for v in names], True))
        if head is not None:
            names = self.cfg["head_joints"][: len(head)]
            batch.append(([float(v) for v in head], [self.robot_joint_indices[v] for v in names], True))
        if batch:
            self.api_core.set_joint_positions_batched(batch)
        if chassis is not None and len(chassis) >= 3:
            self.apply_chassis_target(chassis, chunk_start=chunk_start)

        if getattr(self, "settle_due", self.need_infer):
            if self._free_base:
                self._wait_robot_settled(arm=arm, waist=waist, head=head, gripper=gripper)
            elif arm is not None:
                self._wait_arm_settled([float(v) for v in arm])
        next_obs = self.get_observation(fetch_images=self.need_infer or self.has_done)
        return next_obs, self.has_done, need_update, self.task.task_progress

    def _wait_arm_settled(self, target_arm):
        """Poll until arm joints converge to target (or timeout). Mirrors the
        convergence loop in reset(); physics free-runs so the arm keeps moving
        between polls."""
        eps = self.cfg.get("infer_settle_eps", 1e-2)
        timeout = self.cfg.get("infer_settle_timeout", 1.0)
        names = self.cfg["arm_joints"]
        target = np.array(target_arm)
        deadline = time.time() + timeout
        while time.time() < deadline:
            js = self.data_courier.get_joint_state_dict()
            if js:
                cur = np.array([js[n] for n in names])
                if cur.size and np.max(np.abs(cur - target)) < eps:
                    return
            time.sleep(0.01)
        logger.warning(f"arm did not settle within {timeout:.2f}s before observation")

    def _wait_robot_settled(self, *, arm=None, waist=None, head=None, gripper=None):
        """Require whole-body convergence for free-base WBC tasks before planning.

        Grippers may stop on an object before their free-space target; require
        stability instead of position convergence for these joints. The chassis
        servo remains active throughout this wait and subsequent image capture.
        """
        eps = float(self.cfg.get("infer_settle_eps", 1e-2))
        timeout = float(self.cfg.get("infer_settle_timeout", 2.0))
        stable_for = float(self.cfg.get("infer_settle_stable_time", 0.1))
        targets = {}
        for group, values in (("arm", arm), ("waist", waist), ("head", head)):
            if values is not None:
                targets.update(zip(self.cfg[group + "_joints"], map(float, values)))
        fingers = self.cfg["gripper_joints"] if gripper is not None else []
        previous = None
        stable_since = None
        # Timeout is simulation time; a slow renderer must not consume the
        # robot's entire convergence budget. Wall time is only a watchdog.
        deadline = time.monotonic() + float(self.cfg.get("infer_settle_wall_timeout", 30.0))
        start_sim = None
        while time.monotonic() < deadline:
            self.api_core.check_chassis_servo()
            bundle = self.data_courier.get_obs_bundle(fetch_images=False, link_names=("base_link",))
            js = bundle.get("joint_state") or {}
            now = float(bundle["simulation_time"])
            if start_sim is None:
                start_sim = now
            if now - start_sim >= timeout:
                break
            names = list(targets) + list(fingers)
            ready = all(n in js and np.isfinite(js[n]) for n in names)
            if ready:
                joint_errors = (abs(js[n] - target) for n, target in targets.items())
                ready = all(e < eps for e in joint_errors)
                current = np.array([js[n] for n in names])
                # Require several stable samples, not a single target crossing.
                ready = ready and previous is not None and np.all(np.abs(current - previous) < eps * 0.1)
                previous = current
            motion = bundle.get("chassis_motion")
            target = self._chassis_prev_target
            if target is not None:
                if motion is None:
                    ready = False
                else:
                    xy = np.linalg.norm(np.asarray(target[:2]) - motion["position"][:2])
                    yaw = abs(_normalize_angle(target[2] - _quat_wxyz_to_yaw(motion["orientation"])))
                    ready = ready and xy < 0.002 and yaw < np.deg2rad(0.1)
                    ready = ready and np.linalg.norm(motion["linear_velocity"][:2]) < 0.005
                    ready = ready and abs(motion["angular_velocity"][2]) < 0.005
            stable_since = (stable_since if stable_since is not None else now) if ready else None
            if stable_since is not None and now - stable_since >= stable_for:
                return True
            time.sleep(0.01)
        logger.warning(f"robot did not settle within {timeout:.2f} simulation seconds before observation")
        return False

    def _submit_chassis_target(self, delta, chunk_start):
        """Install a command atomically on the physics thread."""
        self._servo_motion = self.api_core._get_chassis_motion_state()
        self._update_chassis_motion(self._servo_motion)
        if chunk_start or getattr(self, "_servo_delta", None) is None:
            self._lock_chassis_anchor(physics_thread=True)
            self._servo_elapsed = 0.0
            self._servo_duration = self._chassis_active_chunk_len * _STEP_DT
            self._servo_until_control = 0.0
            self._servo_skip_tick = True
        self._servo_delta = np.asarray(delta[:3], dtype=float).copy()
        # First solve now; subsequent waypoint changes are consumed on the fixed
        # simulation cadence, independent of worker/observation wall time.
        if self._servo_until_control <= 0:
            self._apply_chassis_pose_target(
                self._servo_delta, remaining=self._servo_duration - self._servo_elapsed, physics_thread=True
            )
            self._servo_until_control = _STEP_DT
        self.api_core.set_chassis_servo_callback(self._chassis_servo_tick)

    def _chassis_servo_tick(self, dt):
        if self._servo_skip_tick:
            self._servo_skip_tick = False
            return
        self._servo_elapsed += dt
        self._servo_until_control -= dt
        if self._servo_until_control > 1e-9:
            return
        self._servo_until_control += _STEP_DT
        remaining = self._servo_duration - self._servo_elapsed
        try:
            self._servo_motion = self.api_core._get_chassis_motion_state()
            self._update_chassis_motion(self._servo_motion)
            self._apply_chassis_pose_target(self._servo_delta, remaining=remaining, physics_thread=True)
        except Exception:
            # Never leave the last nonzero command latched after a servo fault.
            articulation = self.api_core._get_articulation()
            q = np.full(articulation.num_dof, np.nan)
            v = np.full(articulation.num_dof, np.nan)
            v[[self.robot_joint_indices[n] for n in _DRIVE_JOINT_NAMES]] = 0.0
            self.api_core._apply_chassis_action(q, v)
            raise

    def _begin_chassis_chunk(self):
        if (
            self._chunk_anchor_xy is not None
            and self._chunk_anchor_yaw is not None
            and self._last_base_xy is not None
            and self._last_base_yaw is not None
        ):
            actual_dx_w = self._last_base_xy[0] - self._chunk_anchor_xy[0]
            actual_dy_w = self._last_base_xy[1] - self._chunk_anchor_xy[1]
            actual_dyaw = _normalize_angle(self._last_base_yaw - self._chunk_anchor_yaw)

            yaw0 = self._chunk_anchor_yaw
            cos0 = float(np.cos(yaw0))
            sin0 = float(np.sin(yaw0))
            exp_dx_b = cos0 * self._chunk_expected_dx_world + sin0 * self._chunk_expected_dy_world
            exp_dy_b = -sin0 * self._chunk_expected_dx_world + cos0 * self._chunk_expected_dy_world
            act_dx_b = cos0 * actual_dx_w + sin0 * actual_dy_w
            act_dy_b = -sin0 * actual_dx_w + cos0 * actual_dy_w

            err_dx_b = exp_dx_b - act_dx_b
            err_dy_b = exp_dy_b - act_dy_b
            err_dyaw = _normalize_angle(self._chunk_expected_dyaw - actual_dyaw)

            logger.info(
                f"[chassis-loop] chunk#{self._chunk_index} summary | "
                f"expected(base)=(dx={exp_dx_b:+.4f}m, dy={exp_dy_b:+.4f}m, "
                f"dyaw={np.degrees(self._chunk_expected_dyaw):+.2f}°) | "
                f"actual(base)=(dx={act_dx_b:+.4f}m, dy={act_dy_b:+.4f}m, "
                f"dyaw={np.degrees(actual_dyaw):+.2f}°) | "
                f"err(base)=(dx={err_dx_b:+.4f}m, dy={err_dy_b:+.4f}m, "
                f"dyaw={np.degrees(err_dyaw):+.2f}°)"
            )

        self._chunk_index += 1
        if self._last_base_xy is not None and self._last_base_yaw is not None:
            self._chunk_anchor_xy = self._last_base_xy
            self._chunk_anchor_yaw = self._last_base_yaw
        else:
            self._chunk_anchor_xy = None
            self._chunk_anchor_yaw = None
        self._chunk_expected_dx_world = 0.0
        self._chunk_expected_dy_world = 0.0
        self._chunk_expected_dyaw = 0.0
        self._chassis_corr_prev = (0.0, 0.0, 0.0)
        # Latch this chunk's length for _apply_chassis_pose_target's feedback
        # horizon (see set_policy_step_info) -- fixed for the chunk's
        # lifetime even if a later chunk reports a different length.
        self._chassis_active_chunk_len = getattr(self, "_policy_chunk_len_steps", _CHUNK_STEPS)
        self._chassis_step_in_chunk = 0

    def _compute_step_correction(self):
        if (
            self._chunk_anchor_xy is None
            or self._chunk_anchor_yaw is None
            or self._last_base_xy is None
            or self._last_base_yaw is None
        ):
            self._chassis_corr_prev = (0.0, 0.0, 0.0)
            return 0.0, 0.0, 0.0

        actual_dx_w = self._last_base_xy[0] - self._chunk_anchor_xy[0]
        actual_dy_w = self._last_base_xy[1] - self._chunk_anchor_xy[1]
        actual_dyaw = _normalize_angle(self._last_base_yaw - self._chunk_anchor_yaw)

        err_dx_w = self._chunk_expected_dx_world - actual_dx_w
        err_dy_w = self._chunk_expected_dy_world - actual_dy_w
        err_dyaw = _normalize_angle(self._chunk_expected_dyaw - actual_dyaw)

        yaw_now = self._last_base_yaw
        cos_y = float(np.cos(yaw_now))
        sin_y = float(np.sin(yaw_now))
        err_dx_b = cos_y * err_dx_w + sin_y * err_dy_w
        err_dy_b = -sin_y * err_dx_w + cos_y * err_dy_w

        if abs(err_dx_b) < _CHASSIS_ERR_DEADBAND_LINEAR:
            err_dx_b = 0.0
        if abs(err_dy_b) < _CHASSIS_ERR_DEADBAND_LINEAR:
            err_dy_b = 0.0
        if abs(err_dyaw) < _CHASSIS_ERR_DEADBAND_ANGULAR:
            err_dyaw = 0.0

        horizon_dt = _CHASSIS_CORR_HORIZON_STEPS * _STEP_DT
        corr_vx_target = _CHASSIS_KP_LINEAR * err_dx_b / horizon_dt
        corr_vy_target = _CHASSIS_KP_LINEAR * err_dy_b / horizon_dt
        corr_omega_target = _CHASSIS_KP_ANGULAR * err_dyaw / horizon_dt

        corr_vx_target = float(np.clip(corr_vx_target, -_CHASSIS_MAX_LINEAR_CORR, _CHASSIS_MAX_LINEAR_CORR))
        corr_vy_target = float(np.clip(corr_vy_target, -_CHASSIS_MAX_LINEAR_CORR, _CHASSIS_MAX_LINEAR_CORR))
        corr_omega_target = float(np.clip(corr_omega_target, -_CHASSIS_MAX_ANGULAR_CORR, _CHASSIS_MAX_ANGULAR_CORR))

        prev_vx, prev_vy, prev_omega = self._chassis_corr_prev
        a = _CHASSIS_CORR_LPF_ALPHA
        corr_vx = prev_vx + a * (corr_vx_target - prev_vx)
        corr_vy = prev_vy + a * (corr_vy_target - prev_vy)
        corr_omega = prev_omega + a * (corr_omega_target - prev_omega)

        self._chassis_corr_prev = (corr_vx, corr_vy, corr_omega)
        return corr_vx, corr_vy, corr_omega

    def _update_chassis_motion(self, motion):
        """Use pose and actuator feedback from the same physics snapshot."""
        self._chassis_motion = motion
        if motion is not None:
            self._last_base_xy = tuple(float(v) for v in motion["position"][:2])
            self._last_base_yaw = _quat_wxyz_to_yaw(motion["orientation"])

    def _refresh_base_pose(self):
        """Refresh worker-side pose without images.

        The persistent servo samples its own feedback on the physics thread;
        it must not use this potentially older observation cache for control.
        """
        try:
            bundle = self.data_courier.get_obs_bundle(fetch_images=False, link_names=("base_link",))
            self._update_chassis_motion(bundle.get("chassis_motion"))
            lp = bundle.get("link_poses") or {}
            if self._chassis_motion is None and "base_link" in lp:
                base_pos, base_rot = lp["base_link"]
                self._last_base_xy = (float(base_pos[0]), float(base_pos[1]))
                self._last_base_yaw = _quat_wxyz_to_yaw(base_rot)
        except Exception as e:
            logger.warning(f"refresh base pose for anchor failed: {e}")

    def _lock_chassis_anchor(self, physics_thread=False):
        """Lock x1 = current base pose at the start of each action chunk.
        All per-step Δ in this chunk are defined relative to this anchor."""
        if physics_thread:
            motion = self._servo_motion
            xy = tuple(float(v) for v in motion["position"][:2]) if motion is not None else None
            yaw = _quat_wxyz_to_yaw(motion["orientation"]) if motion is not None else None
        else:
            self._refresh_base_pose()
            xy, yaw = self._last_base_xy, self._last_base_yaw
        # This observation may follow inference/arm-settling delays. Report a
        # boundary residual, not a timed 1-second PASS based on a 3cm/3deg box.
        if self._chassis_prev_target is not None and self._last_base_xy is not None and self._last_base_yaw is not None:
            tx, ty, tyaw = self._chassis_prev_target
            ex = tx - self._last_base_xy[0]
            ey = ty - self._last_base_xy[1]
            exy = float(np.hypot(ex, ey))
            eyaw = _normalize_angle(tyaw - self._last_base_yaw)
            delta = self._chassis_pose_delta
            distance = np.linalg.norm(delta[:2]) if delta is not None else 0.0
            turn = abs(delta[2]) if delta is not None else 0.0
            xy_relative = f"{100 * exy / distance:.2f}%" if distance > 1e-9 else "n/a"
            yaw_relative = f"{100 * abs(eyaw) / turn:.2f}%" if turn > 1e-9 else "n/a"
            logger.info(
                f"[chassis-track] chunk#{self._chunk_index} boundary residual | "
                f"err_xy={exy*1000:.2f}mm ({xy_relative}) "
                f"err_yaw={np.degrees(eyaw):+.3f}° ({yaw_relative})"
            )
        if xy is not None and yaw is not None:
            self._chassis_anchor = (xy, yaw)
        else:
            self._chassis_anchor = None
        # The next pose command supplies this chunk's diagnostic target.
        self._chassis_prev_target = None

    def _apply_chassis_pose_target(self, delta, *, remaining=None, physics_thread=False):
        self._chassis_pose_delta = np.asarray(delta[:3], dtype=float).copy()
        # Worker-side observations may update their cache concurrently. Use the
        # physics thread's own immutable snapshot for pose AND actuator feedback.
        motion = self._servo_motion if physics_thread else self._chassis_motion
        required_joints = _STEER_JOINT_NAMES + _DRIVE_JOINT_NAMES
        missing = [name for name in required_joints if name not in self.robot_joint_indices]
        if missing:
            raise ValueError(f"Predictive chassis pose control is missing required joints: {missing}")
        if motion is None:
            raise RuntimeError(
                "Predictive chassis pose control requires initialized chassis feedback with "
                "wheelParking:enabled=true and a positive wheelParking:maxWheelAcceleration"
            )
        missing_feedback = [name for name in required_joints if name not in motion["joint_names"]]
        if missing_feedback:
            raise ValueError(f"Predictive chassis feedback is missing required joints: {missing_feedback}")
        if self._chassis_anchor is None:
            return
        articulation = self.api_core._get_articulation()
        steer_idx = [self.robot_joint_indices[n] for n in _STEER_JOINT_NAMES]
        drive_idx = [self.robot_joint_indices[n] for n in _DRIVE_JOINT_NAMES]
        self._configure_chassis_steering(articulation, steer_idx)
        measured_steer = [motion["joint_names"].index(n) for n in _STEER_JOINT_NAMES]
        measured_drive = [motion["joint_names"].index(n) for n in _DRIVE_JOINT_NAMES]
        stiffness = motion["drive_stiffness"][measured_steer]
        damping = motion["drive_damping"][measured_steer]
        if (
            not np.all(np.isfinite(stiffness))
            or not np.all(np.isfinite(damping))
            or np.any(stiffness <= 0)
            or np.any(damping <= 0)
        ):
            raise ValueError("Pose control requires driven steering joints")
        # Read live tensor gains: authored USD overrides are not necessarily
        # reflected by an already-created PhysX articulation.
        steer_tau = damping / stiffness
        if self._pose_planner is None:
            self._pose_planner = SwervePosePlanner(
                self._steer_limits,
                _WHEEL_POSITIONS,
                _WHEEL_RADIUS,
                _STEER_MAX_VELOCITY,
                steer_tau,
                dt=_STEP_DT,
                max_speed=_MAX_LINEAR_SPEED,
                max_yaw_rate=_MAX_ANGULAR_SPEED,
            )
        else:
            self._pose_planner.steer_tau = steer_tau
        (ax, ay), ayaw = self._chassis_anchor
        ca, sa = np.cos(ayaw), np.sin(ayaw)
        target = np.array(
            [ax + ca * delta[0] - sa * delta[1], ay + sa * delta[0] + ca * delta[1], _normalize_angle(ayaw + delta[2])]
        )
        xy = motion["position"][:2] if physics_thread else self._last_base_xy
        yaw = _quat_wxyz_to_yaw(motion["orientation"]) if physics_thread else self._last_base_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_body = np.array([[c, s], [-s, c]])
        error = target - np.array([*xy, yaw])
        error[:2] = world_to_body @ error[:2]
        error[2] = _normalize_angle(error[2])
        velocity = world_to_body @ motion["linear_velocity"][:2]
        state = {
            "steer": motion["joint_positions"][measured_steer],
            "drive": motion["joint_velocities"][measured_drive],
            "filtered": motion["filtered_velocities"][measured_drive],
            "twist": np.array([*velocity, motion["angular_velocity"][2]]),
        }
        # Equal relative XY/yaw errors should have equal pose cost on mixed
        # commands. Zero components retain the geometric weighting; their
        # tracking error is reported as absolute drift, not a percentage.
        distance, turn = np.linalg.norm(delta[:2]), abs(float(delta[2]))
        self._pose_planner.yaw_weight = distance / turn if distance > 1e-6 and turn > 1e-6 else 0.35
        if remaining is None:
            # Standalone deterministic controller tests supply one call per tick.
            remaining = (self._chassis_active_chunk_len - self._chassis_step_in_chunk) * _STEP_DT
            self._chassis_step_in_chunk += 1
        angles, wheel_velocities = self._pose_planner.solve(error, state, remaining, motion["wheel_acceleration"])
        positions = np.full(articulation.num_dof, np.nan)
        velocities = np.full(articulation.num_dof, np.nan)
        positions[steer_idx] = angles
        velocities[drive_idx] = wheel_velocities
        # The asset filter remains the sole physical-step ramp/parking owner.
        if physics_thread:
            self.api_core._apply_chassis_action(positions, velocities)
        else:
            self.api_core.apply_chassis_action(positions, velocities)
        self._last_steer_cmd = angles.copy()
        self._last_drive_cmd_lin = wheel_velocities * _WHEEL_RADIUS
        self._chassis_prev_target = tuple(target)

    def _apply_chassis_velocity(self, vx, vy, omega, closed_loop=True, smooth=True):
        try:
            steer_idx = [self.robot_joint_indices[n] for n in _STEER_JOINT_NAMES]
            drive_idx = [self.robot_joint_indices[n] for n in _DRIVE_JOINT_NAMES]
        except KeyError:
            return
        articulation = self.api_core._get_articulation()
        self._configure_chassis_steering(articulation, steer_idx)

        vx_raw = float(vx)
        vy_raw = float(vy)
        omega_raw = float(omega) * _CHASSIS_OMEGA_SCALE

        # Smooth the policy twist before it reaches _direct_cmd / QP. Steer
        # joints in the asset are over-damped (stiffness 5e3, damping 5.7e6)
        # and can't track a 30 Hz cmd_θ that swings ±20° per tick. Without
        # this filter the wheels stay near 0, the kinematic equality forces
        # vy=wz=0, and rotation commands collapse into forward translation
        # — visible as chunk#8 expected dyaw=+10.83° / actual=+1.14° in the
        # previous run. Use the smoothed value for both the QP feed and the
        # chunk-expected accumulator so the diagnostic stays apples-to-apples.
        # Different α per channel: omega's smoother (α_ω=0.15) gives the
        # over-damped steer joints ~2× longer to settle on each cmd_θ; vx/vy
        # keep α=0.3 because the drive side tracks velocity targets cleanly.
        if smooth:
            a_lin = _CHASSIS_LIN_LPF_ALPHA
            a_ang = _CHASSIS_ANG_LPF_ALPHA
            self._twist_lpf[0] += a_lin * (vx_raw - self._twist_lpf[0])
            self._twist_lpf[1] += a_lin * (vy_raw - self._twist_lpf[1])
            self._twist_lpf[2] += a_ang * (omega_raw - self._twist_lpf[2])
            vx_lp = float(self._twist_lpf[0])
            vy_lp = float(self._twist_lpf[1])
            omega_lp = float(self._twist_lpf[2])
        else:
            vx_lp, vy_lp, omega_lp = vx_raw, vy_raw, omega_raw

        if closed_loop:
            corr_vx, corr_vy, corr_omega = self._compute_step_correction()
        else:
            corr_vx = corr_vy = corr_omega = 0.0

        vx_cmd, vy_cmd = _limit_planar_speed(vx_lp + corr_vx, vy_lp + corr_vy)
        omega_cmd = omega_lp + corr_omega

        if self._last_base_yaw is not None:
            yaw = self._last_base_yaw
            cos_y = float(np.cos(yaw))
            sin_y = float(np.sin(yaw))
            self._chunk_expected_dx_world += (cos_y * vx_lp - sin_y * vy_lp) * _STEP_DT
            self._chunk_expected_dy_world += (sin_y * vx_lp + cos_y * vy_lp) * _STEP_DT
        self._chunk_expected_dyaw = _normalize_angle(self._chunk_expected_dyaw + omega_lp * _STEP_DT)

        qp_result = None
        if self._swerve_qp is not None:
            qp_result = self._swerve_qp.solve(
                (vx_cmd, vy_cmd, omega_cmd),
                self._last_steer_actual,
                self._last_drive_cmd_lin,
            )

        if qp_result is not None:
            theta_new, v_lin_new, _ = qp_result
            theta_new = _limit_array_sync(theta_new, self._last_steer_cmd, self._swerve_qp.max_delta_steer)
            theta_new = np.clip(theta_new, self._steer_limits[:, 0], self._steer_limits[:, 1])
            v_lin_new = _limit_array_sync(v_lin_new, self._last_drive_cmd_lin, self._swerve_qp.max_delta_drive)
            # OSQP permits small constraint residuals; keep the wheel-speed
            # command within the physical limit even at solver tolerance.
            v_lin_new = np.clip(v_lin_new, -_QP_MAX_DRIVE_SPEED, _QP_MAX_DRIVE_SPEED)
            steer_angles = theta_new
            drive_velocities = v_lin_new / _WHEEL_RADIUS
            self._last_steer_cmd = np.asarray(theta_new, dtype=np.float64)
            self._last_drive_cmd_lin = np.asarray(v_lin_new, dtype=np.float64)
        else:
            # QP failed (should be rare now the kinematic rows are soft and
            # the problem is always primal-feasible — only a true solver error
            # or NaN reaches here). Decay the drive toward 0 and update the
            # cached command so v_prev actually changes next tick. The old
            # code re-published the previous command verbatim WITHOUT updating
            # the cache, which — once a bad command was latched — fed the
            # identical degenerate linearization every tick and locked the QP
            # in infeasibility forever. Decaying breaks any such lock.
            v_lin_new = 0.5 * np.asarray(self._last_drive_cmd_lin, dtype=np.float64)
            steer_angles = np.clip(self._last_steer_cmd, self._steer_limits[:, 0], self._steer_limits[:, 1])
            self._last_steer_cmd = steer_angles.copy()
            self._last_drive_cmd_lin = v_lin_new
            drive_velocities = v_lin_new / _WHEEL_RADIUS

        num_dof = articulation.num_dof

        if not self._drive_props_logged:
            self._drive_props_logged = True
            try:
                props = getattr(articulation, "dof_properties", None)
                names = getattr(articulation, "dof_names", None)
                for label, joint_names in (("steer", _STEER_JOINT_NAMES), ("drive", _DRIVE_JOINT_NAMES)):
                    for n in joint_names:
                        i = self.robot_joint_indices.get(n)
                        if i is None:
                            logger.info(f"[chassis-drive] {label} {n} -> dof index missing")
                            continue
                        prop = props[i] if props is not None and i < len(props) else None
                        actual_name = names[i] if names is not None and i < len(names) else "?"
                        logger.info(f"[chassis-drive] {label} {n} idx={i} dof_name={actual_name} props={prop}")
            except Exception as e:
                logger.warning(f"[chassis-drive] dof properties dump failed: {e}")

            # Fallback for a swerve robot that never goes through
            # _reset_chassis_joints (non-free-base). Free-base robots (this
            # one included) get this from there instead, every reset -- see
            # that method's comment for why a per-episode re-apply matters.
            self._apply_chassis_drive_overrides(articulation, steer_idx)

        positions = np.full(num_dof, np.nan)
        velocities = np.full(num_dof, np.nan)
        for i in range(4):
            positions[steer_idx[i]] = steer_angles[i]
            velocities[drive_idx[i]] = drive_velocities[i]

        applied = self.api_core.apply_chassis_action(
            positions,
            velocities,
        )
        if applied is False:
            return False

        # --- steer-tracking diagnostic ---
        # θ_cmd = position target sent to the steer joints (post-slew);
        # θ_act = measured steer angle; realized rate = Δθ_act/_STEP_DT. If
        # |rate| caps well below the QP slew (12 rad/s) the steer joint, not
        # the controller, is the bottleneck for turn-entry. Logged only while
        # actively reorienting (max |θ_cmd−θ_act| > 0.05 rad).
        steer_act = np.asarray(self._last_steer_actual, dtype=np.float64).reshape(4)
        steer_err = np.asarray(steer_angles, dtype=np.float64).reshape(4) - steer_act
        if self._prev_steer_actual_log is not None:
            rate = (steer_act - self._prev_steer_actual_log) / _STEP_DT
        else:
            rate = np.zeros(4)
        self._prev_steer_actual_log = steer_act.copy()
        if float(np.max(np.abs(steer_err))) > 0.05:
            logger.debug(
                f"[chassis-steer] θ_cmd={_fmt_arr(np.asarray(steer_angles))} "
                f"θ_act={_fmt_arr(steer_act)} "
                f"err={_fmt_arr(steer_err)} "
                f"rate(rad/s)={_fmt_arr(rate)} "
                f"|rate|max={float(np.max(np.abs(rate))):.2f}"
            )

        self._chassis_log_counter += 1
        qp_info = self._swerve_qp.last_info if self._swerve_qp else {}
        qp_status = qp_info.get("status_val")
        bad_status = qp_status not in (None, 1)
        status_changed = qp_status != self._last_qp_status_logged
        self._last_qp_status_logged = qp_status
        should_log = (
            (self._chassis_log_counter % 150) == 0
            or (bad_status and status_changed)
            or (qp_result is None and self._swerve_qp is not None)
        )
        if should_log:
            if qp_result is not None:
                twist_sol = qp_info.get("twist_solved", (0.0, 0.0, 0.0))
                logger.info(
                    f"[chassis-qp] tick#{self._chassis_log_counter} "
                    f"status={qp_status} iter={qp_info.get('iter')} "
                    f"cost={qp_info.get('cost', float('nan')):.4f} "
                    f"kinres={qp_info.get('kin_residual_max', float('nan')):.2e} | "
                    f"u_ref=({vx_cmd:+.3f},{vy_cmd:+.3f},{omega_cmd:+.3f}) "
                    f"twist_sol=({twist_sol[0]:+.3f},{twist_sol[1]:+.3f},{twist_sol[2]:+.3f}) | "
                    f"θ_act={_fmt_arr(self._last_steer_actual)} "
                    f"θ_cmd={_fmt_arr(self._last_steer_cmd)} "
                    f"v_cmd_m/s={_fmt_arr(self._last_drive_cmd_lin)}"
                )
            else:
                logger.info(
                    f"[chassis-qp] tick#{self._chassis_log_counter} FALLBACK "
                    f"qp_status={qp_status} | "
                    f"u_ref=({vx_cmd:+.3f},{vy_cmd:+.3f},{omega_cmd:+.3f}) "
                    f"θ_act={_fmt_arr(np.asarray(qp_info.get('theta_actual', self._last_steer_actual)))} "
                    f"v_prev={_fmt_arr(np.asarray(qp_info.get('v_prev', self._last_drive_cmd_lin)))} "
                    f"v_tilde={_fmt_arr(np.asarray(qp_info.get('v_tilde', [0,0,0,0])))} "
                    f"cmd_θ={_fmt_arr(np.asarray(qp_info.get('cmd_theta', [0,0,0,0])))} "
                    f"θ_box=[{_fmt_arr(np.asarray(qp_info.get('theta_lo',[0,0,0,0])))},"
                    f"{_fmt_arr(np.asarray(qp_info.get('theta_hi',[0,0,0,0])))}]"
                )
        return True
