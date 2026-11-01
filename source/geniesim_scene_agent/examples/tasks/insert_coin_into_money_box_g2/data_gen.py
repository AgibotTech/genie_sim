# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

# Insert a coin into the money box: synthesize face grasps from the settled
# pose, lift, align with the live slot axes, descend, release and retreat.
# Custom grasp frames and the coin/slot clearance calculations below use the
# asset's Z-up entity frame.

COIN_OID = "benchmark_coin_w000"     # Scale variant *1.03, synchronized with LLM_RESULT.py.
COIN_BOX_OID = "benchmark_coin_box_000"
MONEY_BOX_OID = "benchmark_money_box_000"

# Money-box geometry in the Z-up entity asset frame.
# Top relative to center: 187/2=93.5mm, approximately 0.0936; slot half-length=54.5/2mm.
# Coin geometry must match LLM_RESULT.py COIN_SCALE. The framework does not scale client quantities:
# robot_api.py:3390 hardcodes object_info["scale"]=1.0. This task uses neither grasp_pose.pkl nor
# get_object_size_in_world_frame, so only these geometry constants require adjustment. Silent mismatch
# corrupts both pad-contact length and fingertip gates; _check_coin_radius is therefore essential.
# COIN_SCALE must equal its LLM_RESULT.py value.
# Circumscribed collision-hull radius: bbox half-width is smaller by 6 micrometers; do not use it.
# Thickness; tray-origin-to-rim and tray-origin-to-cell-bottom (below origin) distances.
# The rim constant must equal the same-named LLM_RESULT value.
MB_HALF_H = 0.0936      #
SLOT_HALF_L = 0.02725   #
COIN_SCALE = 1.03                        #
COIN_RADIUS = 0.0187794 * COIN_SCALE     #
COIN_THICKNESS = 0.002 * COIN_SCALE      #
CB_RIM_ABOVE_ORIGIN = 0.017540           #
CB_FLOOR_ABOVE_ORIGIN = -0.005103        #

# Hover above the slot, then descend with the coin center above the opening
# so the lower edge enters before release. Keep fingertips clear of the lid.
# Validate IK at the configured hover height when changing the robot posture.
HOVER_HEIGHT = 0.10     #
INSERT_DROP = 0.010

# Synthesize grasps from the settled coin pose and use GraspConfig world poses.
# Custom commanded frames use col0=grip, col1=third axis and col2=approach;
# annotated poses use a different axis convention. Grip the two disc faces,
# use the configured approach-elevation range and keep closed fingertips above
# the tray rim. Contact height and depth jointly determine fingertip clearance.
COIN_CONTACT_HEIGHTS = (0.010, 0.012, 0.014, 0.016, 0.018)
GRASP_ELEV_DEG = tuple(float(x) for x in np.arange(45.0, 60.1, 2.5))
COIN_GRASP_DEPTHS = (0.009, 0.012, 0.015)
assert min(COIN_GRASP_DEPTHS) >= 0.006, "depth<6mm measured 0/14 successes; do not restore it"
assert max(COIN_GRASP_DEPTHS) <= 0.015, "depth>15mm puts the commanded point below the rim and grips the partition exterior"
assert min(COIN_CONTACT_HEIGHTS) >= 0.010, "With hc too low, the closed-fingertip-above-rim gate rejects every deep candidate"
assert max(COIN_CONTACT_HEIGHTS) <= 0.018, "hc above 18mm approaches the 18.78mm top edge, putting the pads on the narrowing arc"

# Pad span relative to commanded TCP along approach. Open and closed fingertips differ by 17mm:
# q=0 fully open: tips 17.18mm BEHIND TCP, pad gap=89.7mm.
# q=-0.91 closed: tips 0.25mm AHEAD of TCP, pad gap=0.
# During closure, tips sweep from 17.2mm behind to TCP. Select the offset for the current jaw state.
# Open offset measured in hang_cups:415: 0.0172m backward; closed offset is 0.25mm forward.
# Pad length=85.7-17.2=68.5mm, swinging forward to TCP on closing.
# Minimum coin pad-contact length is the intersection of the disc chord along a with PAD_SPAN_CLOSED.
# Its proximal endpoint is TCP, so all contact lies behind TCP (higher). Keeping TCP above the tray
# rim therefore keeps the entire contact above it as well.
PAD_SPAN = (-0.0857, -0.0172)          #
FINGERTIP_OFFSET = -PAD_SPAN[1]        #
CLOSED_TIP_OFFSET = 0.00025            #
PAD_SPAN_CLOSED = (-0.0685, 0.0)       #
MIN_PAD_CONTACT = 0.008
# Filter approach elevation and closed-fingertip clearance against the tray rim.
# Use the same clearance threshold for candidate selection and post-pick checks.
GRASP_ELEV_MIN = np.deg2rad(45.0)   #
GRASP_ELEV_MAX = np.deg2rad(60.0)   #
GRASP_GRIP_AXIS_MAX_TILT = 0.15
FINGERTIP_RIM_MARGIN = 0.002
PRE_GRASP_DISTANCE = 0.06           #
PICK_UP_DISTANCE = 0.15
PRE_GRASP_RIM_MARGIN = 0.030        #

# Share final-check thresholds with offline validation tools. Check achieved
# slot entry and post-release containment using the live coin and slot geometry.
MB_INSIDE_MAX_DZ = 0.0
MB_INSIDE_MAX_RADIAL = 0.0674



def _coin_disc_frame(coin_quat_wxyz):
    """Return orthogonal disc directions (n, u, v) from the live entity frame.

    n is entity col1, the disc normal. u is world Z projected into the disc
    plane and v completes the orthogonal frame. Use these directions for
    face-grasp synthesis in the commanded gripper convention.
    """
    R = quat2mat_wxyz(np.asarray(coin_quat_wxyz, dtype=float))
    n = np.asarray(R[:, 1], dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    u = z - np.dot(z, n) * n
    if np.linalg.norm(u) < 1e-6:
        # If the coin lies flat (vertical normal), its disc has no upward in-plane direction. Choose any
        # orthogonal basis and let the later elevation filter reject it instead of crashing here.
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, z)
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return n, u, v


def _check_coin_radius(coin_z, rim_z):
    """Infer the upright coin radius from tray-relative height.

    Return (radius, consistent) by comparing the measured value with COIN_RADIUS.
    """
    wall = CB_RIM_ABOVE_ORIGIN - CB_FLOOR_ABOVE_ORIGIN
    r_meas = (float(coin_z) - float(rim_z)) + wall
    # Use asymmetric tolerance because error directions have different causes and risks.
    # Measured inferred-radius precision=0.024mm (three k=1 episodes). A 3% scale step changes radius
    # 0.56mm, but spawn seating rounds height from 14.312 to 15mm, adding 0.688mm before settlement,
    # more than that scale step.
    # * Too small, r_meas<R -0.25mm: hard failure. A coin cannot sit lower than its radius without mesh
    #   penetration, so this means mismatch. An oversized constant overestimates exposure/contact and
    #   commands grasps that miss the coin. It rejected k=1 scene with k=1.03 constants (18.76 vs19.34mm).
    # * Too large, r_meas>R +0.25mm: warn only, since incomplete settlement is indistinguishable.
    #   Hard-fail only beyond 1.2mm, exceeding any seating-rounding discrepancy.
    # The original symmetric 1.5mm gate admitted k=1 scenes with k=1.03 constants. A hard filter must
    # actually constrain the quantity it is intended to check.
    d = r_meas - COIN_RADIUS
    ok = (d >= -0.00025) and (d <= 0.0012)
    if ok and d > 0.00025:
        print(f"[coin] Note: inferred radius {r_meas*1000:.2f}mm exceeds the constant by {d*1000:.2f}mm — "
              f"the 0.69mm seating-rounding gap may not have settled yet (expected +0.07mm after settling)."
              f"If it stays here, COIN_SCALE is out of sync.")
    if not ok:
        print(f"[coin] ⚠⚠ Coin radius mismatch (difference {d*1000:+.2f}mm): inferred measurement {r_meas*1000:.2f}mm, "
              f"data_gen COIN_RADIUS={COIN_RADIUS*1000:.2f}mm (COIN_SCALE={COIN_SCALE})")
        # Report coin height relative to the tray rim together with tray and table pose
        # to distinguish a displaced coin from a displaced support.
        if abs(d) > 0.05:
            print(f"[coin] This is not a scale issue: difference {d*1000:+.0f}mm; scale explains at most ±1mm."
                  f"The coin is not in the cell: disc center z={float(coin_z):.3f}, rim z={float(rim_z):.3f}.")
            print(f"[coin] Disc center far below the rim ⇒ the coin likely has no collider and is in free fall. Check the asset first:"
                  f"Aligned.usda must have a self-contained @./Aligned.usd@ payload; "
                  f"cross-directory references to other assets may fail to resolve silently on workers.")
        else:
            print(f"[coin] LLM_RESULT.py COIN_SCALE is almost certainly out of sync with this value — "
                  f"the scene uses the former. Synchronize before rerunning, or pad contact and fingertip gates will both shift "
                  f"without an error.")
    return r_meas, ok


def _check_disc_normal(coin_quat_wxyz, cb_quat_wxyz):
    """Independently verify the chosen disc-normal column; True means passed.

    Physics supplies the criterion: the coin is confined between two ribs, so its normal must
    align with rib arrangement, tray local_y. In the entity frame this is col2, because
    entity col2=-shape col1 after Rx(90), and shape col1 is local_y.
    A wrong baked-frame derivation empties the whole candidate pool while resembling a layout
    failure. This check reports coordinate drift directly.
    """
    n = _coin_disc_frame(coin_quat_wxyz)[0]
    ly = quat2mat_wxyz(np.asarray(cb_quat_wxyz, dtype=float))[:, 2]
    dot = abs(float(n @ (ly / np.linalg.norm(ly))))
    if dot < 0.95:
        print(f"[coin] ⚠⚠ Disc-normal verification failed: |n̂·tray local_y| = {dot:.4f} (requires >0.95)."
              f"n̂={np.round(n, 4)} tray local_y={np.round(ly, 4)}")
        print(f"[coin] This almost certainly means the wrong column index was selected (see _coin_disc_frame), "
              f"rather than a scene issue. Continuing produces a ring of grasp points outside the coin, "
              f"with arm yaw ≈90° rejected by its own gate.")
        return False
    return True
    z = np.array([0.0, 0.0, 1.0])
    u = z - np.dot(z, n) * n
    if np.linalg.norm(u) < 1e-6:
        # If the coin lies flat (vertical normal), its disc has no upward in-plane direction. Choose any
        # orthogonal basis and let the later elevation filter reject it instead of crashing here.
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, z)
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return n, u, v


def _synthesize_coin_grasps(coin_pos, coin_quat_wxyz):
    """Return face-grasp world poses with shape (N, 4, 4).

    Use the commanded frame convention: col0 is grip, col1 completes the frame
    and col2 is approach. Generate candidates from the live disc frame and
    configured contact heights, elevations and depths.
    """
    centre = np.asarray(coin_pos, dtype=float)
    n, u, v = _coin_disc_frame(coin_quat_wxyz)

    mats = []
    for hc in COIN_CONTACT_HEIGHTS:
        C = centre + hc * u
        for e_deg in GRASP_ELEV_DEG:
            e = np.radians(e_deg)
            # Do not directly use cos(E)*v - sin(E)*u: u is vertical only for a perfectly upright coin.
            # At coin tilt t, u_z=cos(t), so actual elevation=asin(sin(E)*cos(t))<E. A measured 0.05-degree tilt
            # made the 45-degree rung read 44.95 degrees, causing its own gate to reject 120 candidates.
            # v=normalize(n cross world-Z) is exactly horizontal, v_z=0. Solve c*u_z=sin(E) instead to obtain
            # exact requested elevation independent of coin tilt.
            c_u = np.sin(e) / max(abs(float(u[2])), 1e-9)
            if c_u >= 1.0:
                continue        # Coin tilt makes this elevation impossible within the disc plane; let the hard filter report the rejection.
            c_v = np.sqrt(max(1.0 - c_u ** 2, 0.0))
            for s_v in (1.0, -1.0):
                a_hat = s_v * c_v * v - c_u * u
                a_hat = a_hat / np.linalg.norm(a_hat)
                for depth in COIN_GRASP_DEPTHS:
                    p = C + depth * a_hat
                    for s_g in (1.0, -1.0):
                        g = s_g * n
                        M = np.eye(4)
                        M[:3, 0] = g                      # col0 = grip axis.
                        M[:3, 2] = a_hat                  # col2 = approach
                        M[:3, 1] = np.cross(a_hat, g)     # col1 = col2 × col0
                        M[:3, 3] = p
                        mats.append(M)
    return np.stack(mats, axis=0)


def _make_coin_grasp_gate(rim_z, arm, coin_radius=COIN_RADIUS, tag="gate"):
    """Build a grasp_pose_filter for coin contact, tray-rim clearance and wrist posture.

    The filter consumes candidate poses and object pose, and returns a boolean
    mask. Evaluate final closed-fingertip clearance against the tray rim.
    """
    arm_body_sign = 1.0 if arm == "left" else -1.0   # World-Y sign of this arm's side.

    def gate(grasp_poses, obj_pose, grasp_widths):
        poses = np.asarray(grasp_poses, dtype=float)
        widths = np.asarray(grasp_widths, dtype=float)
        if len(poses) == 0:
            return poses, widths
        centre = np.asarray(obj_pose, dtype=float)[:3, 3]

        g = poses[:, :3, 0]          # Grip axis=+/-disc normal. The third axis's world-Z component determines wrist roll.
                                     # Commanded point (TCP).
        c1 = poses[:, :3, 1]         #
        a = poses[:, :3, 2]          #
        p = poses[:, :3, 3]          #

        elev = np.arcsin(np.clip(-a[:, 2], -1.0, 1.0))
        ok_elev = (elev >= GRASP_ELEV_MIN - 1e-9) & (elev <= GRASP_ELEV_MAX + 1e-9)

        gz = np.abs(g[:, 2])
        ok_flat = gz <= GRASP_GRIP_AXIS_MAX_TILT

        # Check closed fingertips, whose final position is 0.25 mm ahead of TCP.
        # Open fingertips are farther behind and above the commanded point.
        tip_closed = p + CLOSED_TIP_OFFSET * a
        ok_tip = tip_closed[:, 2] >= rim_z + FINGERTIP_RIM_MARGIN

        # Pad contact length: intersect the line along a with the disc, then intersect with the CLOSED pad span.
        # w=TCP-disc center lies in the disc plane; t^2 +2*t*(w dot a)+|w|^2-R^2<=0.
        w = p - centre[None, :]
        wa = np.einsum("ij,ij->i", w, a)
        disc = wa ** 2 - np.einsum("ij,ij->i", w, w) + coin_radius ** 2
        has_chord = disc > 0.0
        root = np.sqrt(np.maximum(disc, 0.0))
        t_lo, t_hi = -wa - root, -wa + root          # The coin occupies [t_lo, t_hi] along a.
        overlap = (np.minimum(t_hi, PAD_SPAN_CLOSED[1])
                   - np.maximum(t_lo, PAD_SPAN_CLOSED[0]))
        ok_pad = has_chord & (overlap >= MIN_PAD_CONTACT)

        pre_z = p[:, 2] - PRE_GRASP_DISTANCE * a[:, 2]
        ok_pre = pre_z >= rim_z + PRE_GRASP_RIM_MARGIN

        # Body side: approach horizontal component toward +Y puts the gripper body on -Y (right).
        ok_side = (-np.sign(a[:, 1])) == arm_body_sign
        ok_roll = (c1[:, 2] < 0.0) if arm == "left" else (c1[:, 2] > 0.0)

        keep = ok_elev & ok_flat & ok_tip & ok_pad & ok_pre & ok_side & ok_roll
        print(f"[coin][{tag}] Candidates {len(poses)} → {int(keep.sum())}  "
              f"(elevation {int((~ok_elev).sum())}, "
              f"coin insufficiently vertical {int((ok_elev & ~ok_flat).sum())}, "
              f"closed fingertips below rim {int((ok_elev & ok_flat & ~ok_tip).sum())}, "
              f"pads cannot contact {int((ok_elev & ok_flat & ok_tip & ~ok_pad).sum())}, "
              f"pre-grasp too low {int((ok_elev & ok_flat & ok_tip & ok_pad & ~ok_pre).sum())}, "
              f"gripper crosses centerline {int((ok_elev & ok_flat & ok_tip & ok_pad & ok_pre & ~ok_side).sum())}, "
              f"wrist flipped {int((ok_elev & ok_flat & ok_tip & ok_pad & ok_pre & ok_side & ~ok_roll).sum())})")
        # Log the actual constrained output quantities, not only surviving candidate counts.
        if keep.any():
            print(f"[coin][{tag}] Measured output: elevation {np.degrees(elev[keep]).min():.1f}~"
                  f"{np.degrees(elev[keep]).max():.1f}°  "
                  f"pad contact {overlap[keep].min()*1000:.1f}~{overlap[keep].max()*1000:.1f}mm  "
                  f"closed fingertips above rim {(tip_closed[keep][:, 2] - rim_z).min()*1000:.1f}~"
                  f"{(tip_closed[keep][:, 2] - rim_z).max()*1000:.1f}mm  "
                  f"|grip-axis z| {gz[keep].max():.3f}  "
                  f"col1_z closest to 0 is "
                  f"{(c1[keep, 2].max() if arm == 'left' else c1[keep, 2].min()):+.2f}"
                  f"({arm} arm, requires {'<0' if arm == 'left' else '>0'})")
        return poses[keep], widths[keep]

    return gate


def _find_by_keyword(scene_info, kw):
    """Find the first object in scene_info['layout'] containing keyword; return(obj_id, usd)."""
    for oid, info in scene_info["layout"].items():
        kws = [str(k).lower() for k in (info.get("keywords") or [])]
        if kw.lower() in kws:
            return oid, info.get("usd", "")
    return None, None


def _robot_frame_y(robot, position):
    """Target y in robot coordinates, left positive, right negative.

    robot_init_pose.quaternion=[1,0,0,0]; randomization changes position only, so robot y
    equals world y minus base y. Robot yaw would require a changed calculation; explicitly
    check it below rather than silently approximate.
    """
    p = np.asarray(position, dtype=float)
    base = np.asarray(getattr(robot, "init_position", (0.0, 0.0, 0.0)), dtype=float)
    quat = np.asarray(getattr(robot, "init_rotation", (1.0, 0.0, 0.0, 0.0)), dtype=float)
    if abs(float(quat[0])) < 0.999:
        print(f"[coin] ⚠ Robot has yaw quat={np.round(quat, 4)}, "
              f"so the world-y≈robot-y approximation no longer holds; arm selection may reverse")
    return float(p[1]) - float(base[1])


def _arm_by_side(robot, position):
    """Return (arm, robot_frame_y) for the point's side of the robot.

    Use the sign of robot-frame Y for deterministic side selection.
    """
    y = _robot_frame_y(robot, position)
    return ("left" if y >= 0.0 else "right"), y


def _slot_world_axis(robot, mb_id):
    """Derive slot axes in world coordinates from randomized money-box yaw.

    Slot long axis follows asset-local X. Return(slot_top_center_world,
    slot_long_axis_world, slot_normal_axis_world): opening center; unit54.5mm long-axis
    direction; upward+Z slot-plane normal, all in world coordinates.
    """
    pos, quat_wxyz = robot.get_object_pose(mb_id)
    if pos is None:
        return None, None, None
    R = quat2mat_wxyz(np.asarray(quat_wxyz, dtype=float))
    size_w = robot.get_object_size_in_world_frame(mb_id)
    if size_w is None:
        half_h = MB_HALF_H
    else:
        half_h = float(size_w[2]) / 2.0

    slot_top_z = float(pos[2]) + half_h
    slot_center = np.array([float(pos[0]), float(pos[1]), slot_top_z])
    # Asset-local X is slot long axis; rotation column 0 gives that axis in world coordinates.
    slot_long = R[:, 0].copy()
    slot_long[2] = 0.0
    norm = np.linalg.norm(slot_long)
    if norm > 1e-6:
        slot_long /= norm
    else:
        slot_long = np.array([1.0, 0.0, 0.0])
    slot_normal = np.array([0.0, 0.0, 1.0])
    return slot_center, slot_long, slot_normal


def _coin_drop_trajectory(slot_center, slot_long, slot_normal,
                          coin_pos, coin_quat_wxyz,
                          hover_height=HOVER_HEIGHT,
                          drop_depth=INSERT_DROP,
                          n_candidates=5):
    """Build coin-entity insertion trajectories with shape (M, N, 7).

    Waypoints use [x, y, z, qw, qx, qy, qz] in world coordinates. Derive the
    gripper trajectory from the live coin-to-EE relation and slot axes.
    """
    coin_R = quat2mat_wxyz(np.asarray(coin_quat_wxyz, dtype=float))
    n_now = _coin_disc_normal_h(coin_quat_wxyz)

    short_axis = np.cross(slot_normal, slot_long)
    short_axis[2] = 0.0
    short_axis /= np.linalg.norm(short_axis)

    def _yaw_between(a, b):
        """World+Z angle rotating horizontal unit vector a to b."""
        return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))

    # Both +/-short_axis are physically equivalent; choose the smaller rotation.
    cands = [_yaw_between(n_now, short_axis), _yaw_between(n_now, -short_axis)]
    base_yaw = min(cands, key=abs)

    def _Rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _mat_to_wxyz(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                             (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        if R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                             0.25 * s, (R[1, 2] + R[2, 1]) / s])
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    p0 = np.asarray(coin_pos, dtype=float)
    q0 = np.asarray(coin_quat_wxyz, dtype=float)
    slot_xy = np.array([slot_center[0], slot_center[1]])
    slot_top_z = float(slot_center[2])

    trajectories = []
    # Add modest yaw tolerance and varied hover/insertion heights to provide multiple complete trajectories.
    for dyaw in np.deg2rad([0.0, 6.0, -6.0, 12.0, -12.0])[:n_candidates]:
        R_goal = _Rz(base_yaw + dyaw) @ coin_R
        q_goal = _mat_to_wxyz(R_goal)
        for hov, dz in ((hover_height, drop_depth),
                        (hover_height + 0.02, drop_depth + 0.004)):
            p1 = np.array([slot_xy[0], slot_xy[1], slot_top_z + hov])
            p2 = np.array([slot_xy[0], slot_xy[1], slot_top_z + dz])
            trajectories.append(np.array([
                [p0[0], p0[1], p0[2], q0[0], q0[1], q0[2], q0[3]],
                [p1[0], p1[1], p1[2], q_goal[0], q_goal[1], q_goal[2], q_goal[3]],
                [p2[0], p2[1], p2[2], q_goal[0], q_goal[1], q_goal[2], q_goal[3]],
            ]))

    return np.stack(trajectories, axis=0)   # (M, 3, 7)


def _coin_disc_normal_h(coin_quat_wxyz):
    """Return the unit horizontal projection of the disc normal.

    The Z-up entity-frame disc normal is col1.
    """
    R = quat2mat_wxyz(np.asarray(coin_quat_wxyz, dtype=float))
    n = R[:, 1].copy()
    n[2] = 0.0
    if np.linalg.norm(n) < 1e-6:
        n = np.array([1.0, 0.0, 0.0])
    return n / np.linalg.norm(n)


def _coin_slot_beta(coin_quat_wxyz, slot_long):
    """Return disc-normal/slot-short-axis angle in degrees and zeroing Z-yaw in radians.

    Use the disc normal to evaluate projected coin thickness across the slot.
    """
    n = _coin_disc_normal_h(coin_quat_wxyz)
    short = np.cross(np.array([0.0, 0.0, 1.0]), np.asarray(slot_long, dtype=float))
    short[2] = 0.0
    short /= max(np.linalg.norm(short), 1e-9)
    beta = np.degrees(np.arccos(np.clip(abs(float(n @ short)), -1.0, 1.0)))
    # Zeroing yaw: choose the nearer of +/-short; either disc-normal sign is equivalent.
    def _yaw(a, b):
        return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    dyaw = min((_yaw(n, short), _yaw(n, -short)), key=abs)
    return beta, dyaw


def _coin_thickness_projection(beta_deg):
    """Coin projected width along slot short axis at beta(m);9.6mm slot width is the hard maximum."""
    b = np.radians(float(beta_deg))
    return COIN_THICKNESS * np.cos(b) + (2.0 * COIN_RADIUS) * np.sin(b)


def _coin_hover_poses(slot_center, slot_long, slot_normal,
                      coin_pos, coin_quat_wxyz,
                      hover_height=HOVER_HEIGHT):
    """Build hover targets directly above the slot; return(N,4,4) pose list and base_yaw.

    For move_actor_to_target_pose target_pose(world4*4 matrices), rotate current coin pose
    minimally around world Z to align its normal with slot short axis. The upright coin fits
    diameter 37.5mm along54.5mm length and thickness 2mm across9.6mm width.
    Include small yaw perturbation rungs for additional candidates.
    """
    coin_R = quat2mat_wxyz(np.asarray(coin_quat_wxyz, dtype=float))
    n_now = _coin_disc_normal_h(coin_quat_wxyz)

    short_axis = np.cross(slot_normal, slot_long)
    short_axis[2] = 0.0
    short_axis /= np.linalg.norm(short_axis)

    def _yaw_between(a, b):
        return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))

    cands = [_yaw_between(n_now, short_axis), _yaw_between(n_now, -short_axis)]
    base_yaw = min(cands, key=abs)

    def _Rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    slot_top_z = float(slot_center[2])
    target_xyz = np.array([slot_center[0], slot_center[1], slot_top_z + hover_height])

    # Generate distinct candidate orientations while keeping yaw perturbations small
    # to limit wrist rotation during transport.
    hover_mats = []
    for dz in (0.0, 0.02):                       # Two height rungs.
        for dyaw in np.deg2rad([0.0, 4.0, -4.0]):
            R_yaw = _Rz(base_yaw + dyaw) @ coin_R
            n_axis = R_yaw[:, 1]                 # Disc normal after rotation (entity col1).
            for spin in np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 135.0, 180.0]):
                c, si = np.cos(spin), np.sin(spin)
                K = np.array([[0.0, -n_axis[2], n_axis[1]],
                              [n_axis[2], 0.0, -n_axis[0]],
                              [-n_axis[1], n_axis[0], 0.0]])
                R_spin = np.eye(3) + si * K + (1.0 - c) * (K @ K)   # Rodrigues rotation.
                M = np.eye(4)
                M[:3, :3] = R_spin @ R_yaw
                M[:3, 3] = target_xyz + np.array([0.0, 0.0, dz])
                hover_mats.append(M)

    return hover_mats, base_yaw


def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    robot.client.set_task_task_basic_info({
        "task_name": "pick the coin from beside the coin box and insert it through the slot on top of the money box",
        "english_task_name": "pick the coin from beside the coin box and insert it through the slot on top of the money box",
        "init_scene_text": (
            "A coin box sits toward the back of the table with a coin standing "
            "almost upright in one of its cells, its lower edge dropped into a "
            "moulded slot in the cell floor and its upper half standing clear "
            "of the box rim. A money box (piggy bank) sits beside it. The robot "
            "must pick the coin and drop it through the slot on top of the "
            "money box."
        ),
    })

    # 1. Locate objects.
    coin_id, _ = _find_by_keyword(scene_info, "coin_target")
    cb_id, _ = _find_by_keyword(scene_info, "coin_box")
    mb_id, _ = _find_by_keyword(scene_info, "money_box")

    if coin_id is None:
        print("[coin] Coin not found (keyword: coin_target)")
        return False
    if cb_id is None:
        print("[coin] Coin tray not found (keyword: coin_box)")
        return False
    if mb_id is None:
        print("[coin] Money box not found (keyword: money_box)")
        return False
    print(f"[coin] coin={coin_id}  coin_box={cb_id}  money_box={mb_id}")

    coin_pos, coin_quat = robot.get_object_pose(coin_id)
    if coin_pos is None:
        print("[coin] Cannot read coin pose")
        return False
    coin_pos = np.asarray(coin_pos, dtype=float)

    # 2. Derive world slot coordinates.
    slot_center, slot_long, slot_normal = _slot_world_axis(robot, mb_id)
    if slot_center is None:
        print("[coin] Cannot read money-box pose")
        return False
    print(f"[coin] Slot center={np.round(slot_center, 3)}  long axis={np.round(slot_long, 3)}")

    # Choose the arm on the coin side so pickup and transport use the same arm.
    coin_y_rf = _robot_frame_y(robot, coin_pos)
    slot_y_rf = _robot_frame_y(robot, slot_center)
    arm = "left" if coin_y_rf >= 0.0 else "right"
    side_sign = 1.0 if arm == "left" else -1.0
    cb_pose_now, _ = robot.get_object_pose(cb_id)
    cb_y_rf = _robot_frame_y(robot, cb_pose_now) if cb_pose_now is not None else float("nan")
    print(f"[coin] Using {arm} arm (selected by coin side: coin y={coin_y_rf:+.3f} "
          f"slot y={slot_y_rf:+.3f} tray y={cb_y_rf:+.3f})")
    # Warn when slot placement crosses the centerline within the configured tolerance;
    # reject larger crossings before attempting the move.
    if slot_y_rf * side_sign < -0.040:
        print(f"[coin] ⚠⚠ Slot is across from the {arm} arm by {abs(slot_y_rf)*1000:.0f}mm (y={slot_y_rf:+.3f})"
              f"— beyond 40mm. Insertion crosses the centerline at z≈0.96; all 3/3 measured trials at 110~134mm failed.")
        print(f"[coin] Check LLM_RESULT.py PAIR_CENTRE_Y / PAIR_CLEARANCE, "
              f"and whether the scene hash matches the current LLM_RESULT; do not change arm selection here.")
        return False
    if slot_y_rf * side_sign < 0.0:
        print(f"[coin] Note: slot is across from the {arm} arm by {abs(slot_y_rf)*1000:.0f}mm (≤40mm, an expected cost of wider "
              f"spacing). No measurements cover this range — if vertical descent to hover above the slot fails, "
              f"first restore LLM_RESULT.py PAIR_CLEARANCE to 0.030.")
    # Recheck reachability to detect scene/code version mismatch; layout generation already filters against the measured envelope.
    base_xy = np.asarray(getattr(robot, "init_position", (0.0, 0.0, 0.0)), dtype=float)[:2]
    pick_d = float(np.hypot(coin_pos[0] - base_xy[0], coin_pos[1] - base_xy[1]))
    insert_d = float(np.hypot(slot_center[0] - base_xy[0], slot_center[1] - base_xy[1]))
    if not (0.555 <= pick_d <= 0.790):
        print(f"[coin] ⚠ Pickup horizontal distance {pick_d*1000:.0f}mm is outside the measured envelope 555~790 — "
              f"IK may reject all grasp candidates. Check LLM_RESULT.py REACH_PICK.")
    if not (0.575 <= insert_d <= 0.755):
        print(f"[coin] ⚠ Insertion horizontal distance {insert_d*1000:.0f}mm is outside the measured envelope 575~755 — "
              f"both too near and too far cause the same failure when moving above the slot."
              f"Check LLM_RESULT.py REACH_INSERT.")

    # 4. Grasp the coin. It stands upright between tray ribs, exposing14.92mm above the rim.
    # Synthesize runtime candidates (_synthesize_coin_grasps), covering both faces and approach variations,
    # then filter to45-60-degree table elevation with both tips above the rim. Bypass the eight annotated
    # top-down poses.
    # Remove both money box and coin tray from obstacles during grasp:
    # * Money box otherwise blocks the oblique pre-grasp/descent path, failing every rung with move pose failed.
    # * With the trimmed tray's15 colliders present, curobo detours around cell walls; an oblique fingertip
    #   entry hits a wall and rotates the coin about 90 degrees, failing step1 completion
    #   (angle_diff approximately 92 degrees, pos_diff approximately 20mm). Removal permits the specified approach.
    # curobo obstacles affect planning; unmovable_objects independently compare motion afterward.
    robot.remove_objs_from_obstacle([mb_id, cb_id])
    unmovable = []
    for oid in scene_info["layout"]:
        if oid not in (coin_id, cb_id, mb_id):
            unmovable.append(oid)

    # Tray-rim height is the entity-origin height plus CB_RIM_ABOVE_ORIGIN.
    # Use that measured offset rather than half the asset height.
    cb_pos_now, cb_quat_now = robot.get_object_pose(cb_id)
    if cb_pos_now is None:
        print("[coin] Cannot read tray pose to determine rim height")
        return False
    rim_z = float(cb_pos_now[2]) + CB_RIM_ABOVE_ORIGIN

    # Verify the disc-normal column before synthesis. A wrong column places the candidate ring outside the
    # coin while looking like a layout failure; report frame errors before empty-pool errors.
    if not _check_disc_normal(coin_quat, cb_quat_now):
        return False
    # Check radius before synthesis; mismatch silently corrupts both pad-contact and fingertip gates.
    if not _check_coin_radius(coin_pos[2], rim_z)[1]:
        return False

    coin_grasps = _synthesize_coin_grasps(coin_pos, coin_quat)
    # Rehearse the same gate locally to confirm candidates exist before running the stage.
    preview = _make_coin_grasp_gate(rim_z, arm, tag="preview")
    obj_pose_now = np.eye(4)
    obj_pose_now[:3, :3] = quat2mat_wxyz(np.asarray(coin_quat, dtype=float))
    obj_pose_now[:3, 3] = coin_pos
    kept, _ = preview(coin_grasps, obj_pose_now, np.full(len(coin_grasps), 0.08))
    n_now, u_now, _ = _coin_disc_frame(coin_quat)
    tilt_deg = np.degrees(np.arcsin(min(1.0, abs(float(n_now[2])))))
    expo_mm = (float(coin_pos[2]) + COIN_RADIUS * np.cos(np.radians(tilt_deg)) - rim_z) * 1000.0
    print(f"[coin] Settled coin tilt {tilt_deg:.1f}°  rim z={rim_z:.4f}  "
          f"disc center z={coin_pos[2]:.4f}  exposure above rim {expo_mm:.2f}mm")
    # Report the settled coin pose in the selected central tray cell.
    # Use the measured pose for subsequent grasp selection.
    if tilt_deg > 27.5:
        print(f"[coin] ⚠ Settled tilt {tilt_deg:.1f}° exceeds gap 3's jamming angle 27.42° — the inference above "
              f"was wrong: the V-shaped slopes did not support the coin between the ribs. Restore LLM_RESULT.py "
              f"COIN_SLOT_LOCAL_Y to 0.023 (end gap, capped at 9.09°), change cropping-script indices 22/23 "
              f"back to 25, then regenerate the scene.")
    elif tilt_deg > 10.0:
        print(f"[coin] ⚠ Settled tilt {tilt_deg:.1f}° exceeds the end-gap partition-side limit 9.09° — "
              f"exposure dropped from 14.92mm to {expo_mm:.2f}mm. It has not reached the jamming angle; the end gap helps only if "
              f"the coin falls toward the partition. Inspect several trials to see whether the fall direction is fixed.")
    if len(kept) == 0:
        # Report object and support poses before classifying a grasp-planning failure.
        print("[coin] Diagnosis: hard filters removed all candidates. Use the per-filter counts above; "
              "do not assume a scene issue:")
        print("[coin]   · Normal verification passed ⇒ coordinate frame is correct; if that check also failed, fix it first")
        print(f"[coin]   · Coin insufficiently vertical ⇒ check settled tilt (this run {tilt_deg:.1f}°, "
              f"threshold {np.degrees(np.arcsin(GRASP_GRIP_AXIS_MAX_TILT)):.1f}°)")
        print("[coin]   · Fingertips below rim / pads cannot contact ⇒ tune COIN_CONTACT_HEIGHTS or "
              "COIN_GRASP_DEPTHS; do not relax the gate")
        print("[coin]   · All grippers cross centerline ⇒ arm and coin are on opposite sides; see the Using X arm line above")
        return False

    grasp_cfg = GraspConfig(
        pick_up_distance=PICK_UP_DISTANCE,
        # Top-down approaches have commanded col1_z approximately 0, satisfying neither arm's wrist-flip
        # inequality and eliminating all candidates; that motivated disabling it. Oblique approaches now have
        # well-conditioned col1_z (measured 0.65-0.70), but _make_coin_grasp_gate already implements and logs
        # the same inequality alongside other gates. Keep the framework check disabled to avoid duplicate logic.
        # Synthesis already emits both grip-axis signs, so flip only duplicates candidates and doubles IK cost.
        # Its180-degree rotation around col2 recreates the wrist-flipped poses our gate rejects.
        # Flip occurs at grasp.py:151 before our gate, so they would be rejected again rather than leak through,
        # but enabling it adds needless work.
        disable_upside_down=False,
        flip_grasp=False,
        grasp_pose=coin_grasps,
        grasp_pose_frame="world",
        grasp_pose_filter=_make_coin_grasp_gate(rim_z, arm, tag="stage"),
        # Pre-grasp retreat was0.12 for top-down grasps. With oblique approach,0.06 is over three coin radii,
        # sufficient clearance without a large detour. Pass via extra_params: GraspConfig pre_grasp_offset
        # is ineffective because grasp.py:50's value is overwritten by the same-named local at:548.
        extra_params={"pre_grasp_distance": PRE_GRASP_DISTANCE},
        unmovable_objects=unmovable,
        action_description={
            "zh": "从硬币盒的格子里斜向抓起竖立的硬币",
            "en": "pick up the coin standing in the coin box cell",
        },
    )
    coin_z_before = float(coin_pos[2])
    # Also record the tray position for relative-rise verification below.
    cb_before = np.asarray(cb_pos_now, dtype=float).copy()
    picked = robot.pick_actor(coin_id, arm=arm, grasp_config=grasp_cfg)
    if not picked:
        print("[coin] Grasp -> framework reported failure")
        return False

    # Refresh coin pose after grasp; position may have shifted slightly.
    coin_pos, coin_quat = robot.get_object_pose(coin_id)
    if coin_pos is None:
        print("[coin] Cannot read coin pose after grasp")
        robot.detach_obj()
        return False
    coin_pos = np.asarray(coin_pos, dtype=float)
    coin_quat = np.asarray(coin_quat, dtype=float)

    # Check actual coin lift and rotation after pick_actor, using the tray as the
    # relative height reference so lifting the tray cannot count as lifting the coin.
    cb_pos_after, _ = robot.get_object_pose(cb_id)
    cb_pos_after = (np.asarray(cb_pos_after, dtype=float)
                    if cb_pos_after is not None else cb_before)
    cb_moved = float(np.linalg.norm(cb_pos_after - cb_before))
    cb_lifted = float(cb_pos_after[2]) - float(cb_before[2])
    rise_abs = float(coin_pos[2]) - coin_z_before
    rise_rel = rise_abs - cb_lifted
    held = rise_rel >= 0.6 * PICK_UP_DISTANCE
    # Evaluate coin rise relative to the tray; report tray motion separately.
    if not held:
        print(f"[coin] Grasp -> **failure**: coin rose only {rise_rel * 1000:+.1f}mm relative to tray "
              f"(commanded lift {PICK_UP_DISTANCE * 1000:.0f}mm, criterion ≥"
              f"{0.6 * PICK_UP_DISTANCE * 1000:.0f}mm; absolute rise {rise_abs * 1000:+.1f}mm, "
              f"tray rise {cb_lifted * 1000:+.1f}mm, total displacement {cb_moved * 1000:.1f}mm)")
        # Separate the two diagnostic causes because their remedies differ.
        if cb_lifted > 0.005:
            print(f"[coin] Diagnosis: **the entire tray was gripped and lifted** (rise {cb_lifted*1000:.1f}mm)."
                  f"Closed fingertips almost certainly fell below the rim — open-finger clearance is 89.7mm and tips are "
                  f"19~25mm outside the cell partitions, so descending grips the partition exterior. Check the stage line above: "
                  f"closed fingertips above rim must be positive; if positive yet this still happens, check COIN_GRASP_DEPTHS.")
        else:
            print(f"[coin] Diagnosis: gripper did not hold the coin (tray was not lifted). First check grasp.py "
                  f"Step 1 (closure) angle_diff — >20° means closure first rotated the coin, "
                  f"so off-center initial contact pushed it over. This is a grasp-point/centering issue, not an insertion issue.")
        robot.detach_obj()
        return False
    print(f"[coin] Grasp -> success (coin rose {rise_rel * 1000:+.1f}mm relative to tray)")
    if cb_moved > 0.005:
        # Warn without failing when extraction succeeds but the tray is nudged: the coin is held and insertion
        # can proceed. Record this image-quality issue because it may indicate fingertips scraping the rim;
        # further parameter drift could turn a scrape into a divider-wall grasp.
        print(f"[coin] ⚠ Tray was nudged {cb_moved * 1000:.1f}mm "
              f"(rise {cb_lifted * 1000:+.1f}mm, remaining motion lateral) — grasp is unaffected, but this is "
              f"an early sign of fingertips brushing the rim. Increase FINGERTIP_RIM_MARGIN to tighten clearance, "
              f"at the cost of fewer candidates.")

    # Move directly from the post-pick pose to HOVER_HEIGHT above the slot.
    # Check the achieved hover pose before starting descent.
    print(f"[coin] Held-coin pose pos={np.round(coin_pos, 4)} quat={np.round(coin_quat, 4)}")
    _slot_arm, _slot_y = _arm_by_side(robot, slot_center)
    print(f"[coin] Pickup used {arm} arm; slot is on {_slot_arm} side "
          f"(coin_y={coin_pos[1]:+.3f} slot_y={slot_center[1]:+.3f})")
    if _slot_arm != arm:
        print(f"[coin] ⚠ Pickup arm and slot are on opposite sides — moving above the slot is likely unreachable")

    # Remove the money box again before hovering over it. Attach reloads obstacles, so grasp-phase removal
    # cannot be assumed to remain effective.
    robot.remove_objs_from_obstacle([mb_id])

    hover_mats, base_yaw = _coin_hover_poses(
        slot_center=slot_center,
        slot_long=slot_long,
        slot_normal=slot_normal,
        coin_pos=coin_pos,
        coin_quat_wxyz=coin_quat,
        hover_height=HOVER_HEIGHT,
    )
    print(f"[coin] Target above slot z={hover_mats[0][2, 3]:.4f}  "
          f"aligning with slot short axis requires Z rotation {np.degrees(base_yaw):+.1f}°  candidates {len(hover_mats)} poses")
    # Coin normal and slot short axis are both approximately along X; use their
    # small relative yaw rather than the complementary angle.
    if abs(np.degrees(base_yaw)) > 30.0:
        print(f"[coin] ⚠⚠ Required rotation {np.degrees(base_yaw):+.1f}° greatly exceeds the expected few degrees."
              f"Every candidate inherits this yaw and most will likely be rejected."
              f"First suspect the wrong coin-normal column (use _coin_disc_normal_h, entity col1), "
              f"then the mb_yaw randomization range.")

    at_hover = robot.move_actor_to_target_pose(
        actor=coin_id,
        target_pose=np.stack(hover_mats),
        arm=arm,
        move_config=MoveConfig(
            unmovable_objects=[cb_id],
            action_description={
                "zh": "把硬币一步移到存钱罐投币孔上方",
                "en": "move the coin above the money box slot in one step",
            },
        ),
    )
    print(f"[coin] Move above slot -> {'success' if at_hover else 'failure'}")
    if not at_hover:
        # After combining transport into one move, crossing versus descent failures cannot be distinguished:
        # position, side, reachability, and tray avoidance now share one verdict.
        print("[coin] Diagnosis: the single-step move failed. Before merging, lateral crossing and descent failures "
              "could be distinguished; the merged move cannot distinguish them.")
        print("[coin]   · Check the aligning with slot short axis requires Z rotation line: it should be only a few degrees")
        print("[coin]   · Is the money box still an obstacle? attach restores obstacles, so remove it again")
        print("[coin]   · Is the tray blocking the path? unmovable_objects=[cb_id] only requests avoidance; "
              "it does not guarantee the planner can find a path around it")
        robot.back_to_origin(arm=arm)
        return False

    # Derive the second leg from the achieved pose instead of recreating candidates.
    coin_pos, coin_quat = robot.get_object_pose(coin_id)
    if coin_pos is None:
        robot.back_to_origin(arm=arm)
        return False
    coin_pos = np.asarray(coin_pos, dtype=float)
    coin_quat = np.asarray(coin_quat, dtype=float)
    beta_hover, dyaw_fix = _coin_slot_beta(coin_quat, slot_long)
    tproj = _coin_thickness_projection(beta_hover)
    print(f"[coin] Actual hover pos={np.round(coin_pos, 4)}  "
          f"β={beta_hover:.1f}° thickness projection={tproj*1000:.1f}mm (slot width 9.6mm)  "
          f"zeroing requires Z rotation {np.degrees(dyaw_fix):+.1f}°")
    if tproj > 0.0096:
        print(f"[coin] ⚠ At β={beta_hover:.1f}°, coin spans {tproj*1000:.1f}mm on the slot short axis > slot width 9.6mm — "
              f"direct insertion hits the domed lid and contact torque twists the coin in the gripper (worst measured 19°→90°), "
              f"leaving it flat on the lid. The first trajectory candidate below zeros β.")

    # Descend through N_DESCENT_STEPS linear targets to INSERT_DROP above the slot.
    # Derive endpoint orientation from the live coin and slot axes.
    def _Rz_q(t, q):
        c, s = np.cos(t), np.sin(t)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return mat2quat_wxyz(Rz @ quat2mat_wxyz(np.asarray(q, dtype=float)))

    N_DESCENT_STEPS = 3   # Insert intermediate waypoints from hover to entry rather than jumping directly to the endpoint.

    trajectories = []
    for frac in (1.0, 0.5, 0.0):                       # Correction strengths: full, half, none.
        for dz in (INSERT_DROP, INSERT_DROP + 0.004, INSERT_DROP + 0.008):
            p_end = np.array([slot_center[0], slot_center[1], float(slot_center[2]) + dz])
            waypoints = [[coin_pos[0], coin_pos[1], coin_pos[2],
                          coin_quat[0], coin_quat[1], coin_quat[2], coin_quat[3]]]
            for k in range(1, N_DESCENT_STEPS + 1):
                t = k / N_DESCENT_STEPS
                p_k = (1.0 - t) * coin_pos + t * p_end
                q_k = coin_quat if frac == 0.0 else _Rz_q(t * frac * dyaw_fix, coin_quat)
                waypoints.append([p_k[0], p_k[1], p_k[2],
                                  q_k[0], q_k[1], q_k[2], q_k[3]])
            trajectories.append(np.array(waypoints))
    trajectories = np.stack(trajectories, axis=0)   # (9, N_DESCENT_STEPS +1, 7)

    traj_cfg = FollowTrajectoryConfig(
        # noise_range=[dx, dy, dz, droll, dpitch, dyaw], explicitly ordered in robot_api.py:451.
        # Index4 is PITCH, not yaw;0.14rad has always been pitch tolerance. It is not active generalization noise:
        # follow_trajectory.py:150 skips sampling when original_solvable[waypoint_index] is true. Only failed
        # original waypoint IK triggers perturbation, selecting the smallest normalized one.
        # Thus it is a tolerance budget rather than per-run jitter. An8-degree budget cannot explain measured
        # beta growth19 ->90 degrees, which came from dome-contact torque.
        noise_range=[0.003, 0.003, 0.002, 0.0, 0.14, 0.0],
        noise_frame="local",
        sample_count=16,
        action_description={
            "zh": "将硬币沿轨迹投入存钱罐",
            "en": "follow trajectory to drop the coin through the money box slot",
        },
    )

    inserted = robot.follow_trajectory(
        actor=coin_id,
        waypoints=trajectories,
        arm=arm,
        trajectory_config=traj_cfg,
    )
    print(f"[coin] follow_trajectory insertion -> {'success' if inserted else 'failure'}")
    if not inserted:
        # On insertion failure, return while holding the coin; opening would drop it onto the table.
        print("[coin] Diagnosis: hover is reachable but insertion is not — an insertion-segment geometry issue")
        robot.back_to_origin(arm=arm)
        return False

    # 7. Open and let the coin fall into the money box.
    robot.open_gripper(arm=arm)
    robot.wait(0.5)
    robot.detach_obj()
    robot.wait(1.0)

    # 8. Retreat along negative gripper approach to avoid sweeping against the money box.
    ee = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
    retreat = ee.copy()
    retreat[:3, 3] = ee[:3, 3] - 0.12 * ee[:3, 2]  # col2 = approach
    robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=retreat[np.newaxis, ...],
        arm=arm,
        move_config=MoveConfig(
            unmovable_objects=[mb_id],
            action_description={
                "zh": "退离存钱罐上方",
                "en": "retreat the hand away from the money box",
            },
        ),
    )
    robot.wait(0.3)

    # 9. Reset. back_to_origin first rotates the wrist; the hand has already retreated clear of the box.
    robot.back_to_origin(arm=arm)

    # 10. Final verification: is the coin inside the money box?
    robot.wait(1.5)
    coin_pos_final, _ = robot.get_object_pose(coin_id)
    mb_pos_final, _ = robot.get_object_pose(mb_id)
    if coin_pos_final is None or mb_pos_final is None:
        print("[coin] Final check: cannot read poses; result unknown")
        return False

    coin_pos_final = np.asarray(coin_pos_final, dtype=float)
    mb_pos_final = np.asarray(mb_pos_final, dtype=float)
    dz = float(coin_pos_final[2] - mb_pos_final[2])
    radial = float(np.linalg.norm(coin_pos_final[:2] - mb_pos_final[:2]))

    # Evaluate slot entry from the coin plane, lower edge and slot clearance
    # using the achieved pose, followed by the post-release check.
    inside = bool(dz < MB_INSIDE_MAX_DZ and radial < MB_INSIDE_MAX_RADIAL)
    print(f"[coin] Final check: Δz={dz*1000:+.1f}mm (requires <{MB_INSIDE_MAX_DZ*1000:+.0f}) "
          f"radial distance={radial*1000:.1f}mm (requires <{MB_INSIDE_MAX_RADIAL*1000:.0f}) "
          f"-> {'insertion succeeded' if inside else 'insertion failed'}")
    if not inside and dz > 0.060:
        print(f"[coin] Diagnosis: coin stopped on the domed lid (Δz={dz*1000:+.1f}mm, dome +89.8mm), "
              f"without passing through the slot. Check insertion: does INSERT_DROP put the lower coin edge into the slot, "
              f"and is the coin still vertical at release? A flat coin cannot pass through a 9.6mm-wide slot.")
    return inside
