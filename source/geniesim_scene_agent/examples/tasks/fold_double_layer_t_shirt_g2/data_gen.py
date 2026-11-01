# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""cloth_fold_dual_g2_tshirt: G2 dual-arm FEM double-layer T-shirt folding.

Usage: dual_act(left=DualPick(actor, GraspConfig(grasp_pose=T_l)), right=DualPick(...))
uses server/client grasp-stage state machine for pre-grasp + grasp + close. pick_up_distance=0
disables lift; cycle performs synchronized lift via DualCuroboMove.

cloth_001 is registered in ASSETS_INDEX ($SIM_ASSETS/objects/edited/cloth_001/ has item.py,
Aligned.usda, object_parameters.json, and embedded tshirt_001/Aligned_fem.usda), so
get_object("cloth_175fb7fa") works. Vertex indices match
cloth_fold_fem_tshirt001_gripper_standalone.py exactly (19060 vertices).

Flow cycle1 -> transport -> cycle2 -> cycle3:
- grasp: dual_act or single-arm pick_actor with grasp_pose; pre-grasp + grasp + close.
- attach: FEM pins for cloth nodes within radius.
- fold/lift: DualCuroboMove +z lift_dz.
- fold/move: DualCuroboMove to place XY.
- fold/lower: DualCuroboMove -z descend_dz.
- release: fem_pin_release + open_gripper.
- retract: DualCuroboMove backward along EE local z.
- transport between cycle1/cycle2: +z 3.5 cm -> X -(10 cm + scene X offset) -> -z to table.

Grasp orientations: left z parallel to world -y, rotated +45 degrees about world x;
right z parallel to +y, rotated -45 degrees about world x.
"""
import numpy as np

from client.llm_gen.robot_api import (
    RobotAPI,
    DualPick,
    DualMoveToPose,
    GraspConfig,
    MoveConfig,
)


# ---------------------------------------------------------------------------
# Configuration switches.
# ---------------------------------------------------------------------------

# Whether to draw debug cubes in IsaacSim GUI: grasp poses red/orange, cloth nodes blue.
# Disable for recording/production collection; enable when checking grasp points.
DEBUG = True


def _draw_cube(robot: RobotAPI, *args, **kwargs):
    """DEBUG-controlled wrapper for robot.draw_debug_cube."""
    if not DEBUG:
        return
    try:
        robot.draw_debug_cube(*args, **kwargs)
    except Exception as exc:
        print(f"[cf] draw_debug_cube failed: {exc}")


# ---------------------------------------------------------------------------
# Geometry helpers: grasp pose construction and world-to-base conversion.
# ---------------------------------------------------------------------------

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_axis(axis, deg):
    rad = np.deg2rad(deg) * 0.5
    s = np.sin(rad)
    return np.array([np.cos(rad), axis[0]*s, axis[1]*s, axis[2]*s])


def _quat_from_xz(x_img, z_img):
    x_img = np.asarray(x_img, dtype=np.float64)
    z_img = np.asarray(z_img, dtype=np.float64)
    z_img = z_img / np.linalg.norm(z_img)
    x_img = x_img - np.dot(x_img, z_img) * z_img
    x_img = x_img / np.linalg.norm(x_img)
    y_img = np.cross(z_img, x_img)
    R = np.column_stack([x_img, y_img, z_img])
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _quat_to_mat(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),  1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),  2*(y*z + x*w),  1-2*(x*x+y*y)],
    ])


def _grasp_pose(target_xyz, side, world_x_deg=45.0):
    """Construct a world-frame 4x4 grasp pose with gripper z toward cloth surface."""
    z_dir = np.array([0.0, -1.0, 0.0]) if side == "left" else np.array([0.0, 1.0, 0.0])
    x_dir = np.array([-1.0, 0.0, 0.0])
    q_base = _quat_from_xz(x_dir, z_dir)
    q_rot = _quat_axis([1.0, 0.0, 0.0], world_x_deg)
    q = _quat_mul(q_rot, q_base)
    T = np.eye(4)
    T[:3, :3] = _quat_to_mat(q)
    T[:3, 3] = np.asarray(target_xyz, dtype=np.float64)
    return T, q


def _shift_toward_origin_xy(p_xyz, dist=0.003):
    """Shift xyz toward world origin (0,0,_) in XY by dist m (default 3 mm), preserving z.
    Fine-tunes grab/place points 3 mm toward robot before approach to avoid landing beyond cloth edge."""
    p = np.asarray(p_xyz, dtype=np.float64).copy()
    r = float(np.linalg.norm(p[:2]))
    if r < 1e-9:
        return p
    p[:2] = p[:2] * (1.0 - dist / r)
    return p


def _grasp_pose_candidates(target_xyz, side, world_x_deg=45.0,
                            shift_toward_origin=0.003,
                            angle_jitter_deg=5.0):
    """Generate 11 grasp candidates with deterministic XY samples and random angle variation.

      0: original target_xyz index point, always retained.
      1: center = target_xyz shifted shift_toward_origin m toward (0,0,_).
      2-5: center + inner ring r=1.5 mm, angles 0/90/180/270 degrees.
      6-10: center + outer ring r=3 mm, angles 36/108/180/252/324 degrees,
            36-degree angular offset avoiding shared radial directions.

    Each adds random world_x_deg +/-angle_jitter_deg rotation (default +/-5 degrees).
    Coverage is a <=3 mm disk plus original point; concentric equal-angle samples let server IK
    try multiple reachable poses within a small neighborhood.
    """
    target = np.asarray(target_xyz, dtype=np.float64).copy()
    center = _shift_toward_origin_xy(target_xyz, shift_toward_origin)
    samples = [target, center.copy()]
    # Inner ring r=1.5 mm, four evenly spaced points.
    for k in range(4):
        theta = 2 * np.pi * k / 4.0
        samples.append(center + np.array(
            [0.0015 * np.cos(theta), 0.0015 * np.sin(theta), 0.0],
            dtype=np.float64,
        ))
    # Outer ring r=3 mm, five evenly spaced points, rotated 36 degrees to avoid inner-ring radial overlap.
    for k in range(5):
        theta = 2 * np.pi * k / 5.0 + np.pi / 5.0
        samples.append(center + np.array(
            [0.003 * np.cos(theta), 0.003 * np.sin(theta), 0.0],
            dtype=np.float64,
        ))
    rng = np.random.default_rng()
    Ts = []
    for s in samples:
        d_ang = rng.uniform(-angle_jitter_deg, angle_jitter_deg)
        T, _ = _grasp_pose(s, side, world_x_deg=world_x_deg + d_ang)
        Ts.append(T)
    return np.stack(Ts, axis=0)


def _world_to_base(pos, base_xyz):
    return np.asarray(pos) - np.asarray(base_xyz)


# ---------------------------------------------------------------------------
# Direct DualCuroboMove dispatch for lift/move/lower/retract, bypassing stage state machine.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dual/single-arm moves via robot.dual_act + DualMoveToPose use the same stage
# pipeline as grasp, aligning recording/playback with ROS ticks rather than bypassing stages through command_any.
# ---------------------------------------------------------------------------

def _pose_4x4(pos, quat_wxyz):
    """(pos, quat_wxyz) -> 4x4 world matrix."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_mat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _make_move_cfg(path_constraint, offset_in_goal_frame,
                   from_current_pose, goal_offset, slowdown_factor,
                   action_description=None):
    return MoveConfig(
        path_constraint=list(path_constraint) if path_constraint else [],
        offset_and_constraint_in_goal_frame=bool(offset_in_goal_frame),
        action_description=action_description or {"zh": "双臂同步移动", "en": "dual sync move"},
        extra_params={
            "from_current_pose": bool(from_current_pose),
            "goal_offset": list(goal_offset) if goal_offset is not None else [0, 0, 0, 1, 0, 0, 0],
            "slowdown_factor": float(slowdown_factor),
        },
    )


def _dual_move(robot: RobotAPI,
               left_pos, left_quat,
               right_pos, right_quat,
               base_xyz,
               path_constraint=None,
               offset_in_goal_frame=True,
               from_current_pose=False,
               goal_offset=None,
               slowdown_factor=1.0,
               action_description=None,
               label=""):
    """Synchronized dual-arm move via robot.dual_act + DualMoveToPose + MoveConfig stage state machine
    to server DualCuroboMove. ROS recording ticks automatically capture frame_state/cloth vertices,
    sharing single-arm grasp pipeline. With from_current_pose=True, server-current EE + goal_offset
    defines planning goal; client target_pose is placeholder only. base_xyz stays for signature compatibility
    but server handles base conversion. action_description={zh:..., en:...} passes through dual_act;
    recorder set_frame_state stores this text per frame for replay semantic labels.
    """
    _ = base_xyz   # Preserve old signature; server computes value using robot.init_position.
    # _build_move_to_pose_stage expects target_pose[0][:3, 3] from (N,4,4); wrap a 4x4 as (1,4,4).
    left_T = _pose_4x4(left_pos, left_quat)[np.newaxis, ...]
    right_T = _pose_4x4(right_pos, right_quat)[np.newaxis, ...]
    cfg = _make_move_cfg(path_constraint, offset_in_goal_frame,
                          from_current_pose, goal_offset, slowdown_factor)
    left_spec = DualMoveToPose(actor="gripper", target_pose=left_T, move_config=cfg)
    right_spec = DualMoveToPose(actor="gripper", target_pose=right_T, move_config=cfg)
    desc = action_description or {"zh": f"双臂 {label}", "en": f"dual {label}"}
    print(f"[cf/{label}] dual_act L={left_T[0,:3,3].tolist()} R={right_T[0,:3,3].tolist()} "
          f"fcp={from_current_pose} go={cfg.extra_params['goal_offset']} sd={slowdown_factor} "
          f"desc={desc.get('zh','')}")
    success, _ = robot.dual_act(
        left=left_spec, right=right_spec, action_description=desc,
    )
    if not success:
        print(f"[cf/{label}] dual_act FAIL")
    return bool(success)


def _single_move(robot: RobotAPI, arm, pos, quat, base_xyz,
                 path_constraint=None, offset_in_goal_frame=True,
                 from_current_pose=False, goal_offset=None,
                 slowdown_factor=1.0, action_description=None, label=""):
    """Single-arm move via robot.move_actor_to_target_pose(actor='gripper', ...), sharing dual_act stage
    pipeline with automatic ROS frame_state/cloth capture. base_xyz remains unused for compatibility.
    target_pose wraps to (1,4,4) for _build_move_to_pose_stage. action_description enters MoveConfig
    for recorder labels. cycle3's single-arm description already describes lowering the hem and needs
    no automatic left-arm prefix; extra_params strip_arm_prefix=True makes get_move_stage skip it.
    """
    _ = base_xyz
    T = _pose_4x4(pos, quat)[np.newaxis, ...]   # (1, 4, 4)
    cfg = _make_move_cfg(path_constraint, offset_in_goal_frame,
                          from_current_pose, goal_offset, slowdown_factor,
                          action_description=action_description or {"zh": label, "en": label})
    cfg.extra_params["strip_arm_prefix"] = True
    print(f"[cf/{label}] move_actor {arm} target={T[0,:3,3].tolist()} "
          f"fcp={from_current_pose} go={cfg.extra_params['goal_offset']} sd={slowdown_factor} "
          f"desc={cfg.action_description.get('zh','')}")
    success = robot.move_actor_to_target_pose(
        actor="gripper", target_pose=T, arm=arm, move_config=cfg,
    )
    if not success:
        print(f"[cf/{label}] move_actor ({arm}) FAIL")
    return bool(success)


# ---------------------------------------------------------------------------
# Pick (DualPick / pick_actor): pre-grasp + grasp + close; pick_up_distance=0 disables lift.
# ---------------------------------------------------------------------------

def _make_grasp_cfg(T_candidates, lift_dz_in_pick, side, action_description=None):
    """Build GraspConfig bypassing USD preannotations.

    T_candidates accepts (4,4) or (N,4,4), normalized internally to (N,4,4). Multiple candidates
    let server IK select reachable small variations instead of failing one rigid pose.
    Caller provides action_description per cycle/transport segment, defaulting to cloth grasp.
    """
    T_arr = np.asarray(T_candidates, dtype=np.float64)
    if T_arr.ndim == 2:
        T_arr = T_arr[np.newaxis, ...]
    return GraspConfig(
        grasp_pose=T_arr,
        grasp_pose_frame="world",
        pre_grasp_offset=0.05,
        pick_up_distance=float(lift_dz_in_pick),  # 0 disables lift inside pick.
        pick_up_direction="z",
        disable_upside_down=False,
        flip_grasp=False,
        action_description=action_description or {
            "zh": "抓取布料",
            "en": "grasp cloth",
        },
        unmovable_objects=[],
        # Slow the entire grasp stage (pre_grasp + grasp + close) by 1.5x using stage extra_params,
        # flowing into client _dispatch_dual_step_move / _step_dispatch_move_single plan
        # dictionary and then dual_moveto_curobo / client.moveto slowdown_factor field.
        extra_params={"slowdown_factor": 1.5},
    )


def _dual_pick_cloth(robot, cloth_actor, T_l, T_r, lift_dz_in_pick=0.0,
                     action_description=None):
    """Synchronized DualPick: pre-grasp + grasp + close; pick_up_distance=0 disables lift.

    T_l/T_r accept single (4,4) or multiple (N,4,4) candidates; server IK selects reachable ones.
    """
    T_l_arr = np.asarray(T_l)
    T_r_arr = np.asarray(T_r)
    nl = T_l_arr.shape[0] if T_l_arr.ndim == 3 else 1
    nr = T_r_arr.shape[0] if T_r_arr.ndim == 3 else 1
    L_pos0 = T_l_arr[0, :3, 3] if T_l_arr.ndim == 3 else T_l_arr[:3, 3]
    R_pos0 = T_r_arr[0, :3, 3] if T_r_arr.ndim == 3 else T_r_arr[:3, 3]
    print(f"[cf/dual_pick] L(n={nl})={L_pos0} R(n={nr})={R_pos0}")
    left_spec = DualPick(
        actor=cloth_actor,
        grasp_config=_make_grasp_cfg(T_l, lift_dz_in_pick, "left",
                                      action_description=action_description),
    )
    right_spec = DualPick(
        actor=cloth_actor,
        grasp_config=_make_grasp_cfg(T_r, lift_dz_in_pick, "right",
                                      action_description=action_description),
    )
    success, _ = robot.dual_act(left=left_spec, right=right_spec,
                                 action_description=action_description)
    if not success:
        print(f"[cf/dual_pick] FAIL")
    return success


def _single_pick_cloth(robot, cloth_actor, arm, T, lift_dz_in_pick=0.0,
                       action_description=None):
    """Single-arm pick_actor. T may be (4,4) or (N,4,4)."""
    T_arr = np.asarray(T)
    n = T_arr.shape[0] if T_arr.ndim == 3 else 1
    pos0 = T_arr[0, :3, 3] if T_arr.ndim == 3 else T_arr[:3, 3]
    print(f"[cf/single_pick] {arm}(n={n})={pos0}")
    cfg = _make_grasp_cfg(T, lift_dz_in_pick, arm, action_description=action_description)
    ok = robot.pick_actor(actor=cloth_actor, arm=arm, grasp_config=cfg)
    if not ok:
        print(f"[cf/single_pick] {arm} FAIL")
    return ok


# ---------------------------------------------------------------------------
# Cycle flow.
# ---------------------------------------------------------------------------

# Fold completion criteria inherited from cloth_fold_fem_tshirt001_gripper_standalone.py:
# world distances between three vertex pairs plus flat-spread gate (sufficient footprint and thin enough thickness).
# Pair distances only prove fold points meet, which a crumpled ball also satisfies; footprint/thickness distinguish tidy folding.
# DIST_PAIRS ("upper/lower", 17310, 1349) is critical (~44 cm unfolded, ~12 cm folded).
_DIST_PAIRS = [("上左右", 2573, 1652), ("下左右", 17766, 18532), ("上下", 17310, 1349)]
_FOLD_DONE_M = {"上左右": 0.14, "下左右": 0.10, "上下": 0.20}
_FOOT_MIN_CM2 = 350.0   # Minimum XY bounding-box footprint in cm squared; smaller means crumpled clump.
_THICK_MAX_M = 0.12     # Maximum thickness z95-z5 in m; larger means piled up.


def _check_fold(robot, cloth_prim, tag: str) -> bool:
    """Read all cloth FEM-node world positions and assess folding. Log only; does not block main flow."""
    try:
        P = robot.get_cloth_nodal_positions(cloth_prim)
    except Exception as exc:
        print(f"[fold/{tag}] get_cloth_nodal_positions failed: {exc}")
        return False
    if P is None or len(P) == 0:
        print(f"[fold/{tag}] no cloth nodes")
        return False
    n = P.shape[0]
    ok = True
    parts = []
    for nm, i, j in _DIST_PAIRS:
        if i >= n or j >= n:
            parts.append(f"{nm}=N/A(idx>{n})")
            ok = False
            continue
        d = float(np.linalg.norm(P[i] - P[j]))
        thr = _FOLD_DONE_M[nm]
        g = d <= thr
        ok = ok and g
        parts.append(f"{nm}={d*100:.1f}{'✓' if g else '✗'}≤{thr*100:.0f}")
    z = P[:, 2]
    z_hi = float(np.quantile(z, 0.95))
    z_lo = float(np.quantile(z, 0.05))
    thick = z_hi - z_lo
    sx = float(P[:, 0].max() - P[:, 0].min())
    sy = float(P[:, 1].max() - P[:, 1].min())
    foot = sx * sy * 1e4  # cm²
    foot_ok = foot >= _FOOT_MIN_CM2
    thick_ok = thick <= _THICK_MAX_M
    ok = ok and foot_ok and thick_ok
    print(
        f"[fold/{tag}] {'★fold complete★' if ok else 'incomplete'}  " + "  ".join(parts)
        + f"  | thickness={thick*100:.1f}{'✓' if thick_ok else '✗'}≤{_THICK_MAX_M*100:.0f}"
        + f"(z range {z_lo*100:.1f}~{z_hi*100:.1f})"
        + f"  footprint={sx*100:.0f}x{sy*100:.0f}={foot:.0f}"
        + f"{'✓' if foot_ok else '✗'}≥{_FOOT_MIN_CM2:.0f}cm² (cm)"
    )
    return ok


# ---------------------------------------------------------------------------
# Automatic left/right assignment by grasp/target y: y>0 left, y<0 right.
# ---------------------------------------------------------------------------

def _node_y(robot, cloth_prim, idx: int) -> float:
    """Read world y of cloth node idx."""
    pos = robot.get_cloth_nodal_positions(cloth_prim, indices=[int(idx)])
    return float(pos[0][1])


def _assign_pair_by_y(robot, cloth_prim, idx_a: int, idx_b: int):
    """Order (idx_a, idx_b) as (for_left, for_right) by world y.
    Positive y goes left, negative right; if both have the same sign, use descending y for left/right."""
    ya = _node_y(robot, cloth_prim, idx_a)
    yb = _node_y(robot, cloth_prim, idx_b)
    if ya >= yb:
        return int(idx_a), int(idx_b), ya, yb
    return int(idx_b), int(idx_a), yb, ya


def _pick_arm_by_y(robot, cloth_prim, idx: int) -> str:
    """Choose arm from one point: y>0 left, y<=0 right."""
    y = _node_y(robot, cloth_prim, idx)
    return "left" if y > 0.0 else "right"


# Action descriptions for each cycle/transport stage: grasp / lift / move / lower / release,
# five substages. Recorder writes corresponding text into frame_state.json on every set_frame_state;
# replay uses time_stamp to retrieve current semantic action label.
_PHASE_DESC = {
    "cycle1": {
        "grasp":   {"zh": "双臂夹取衣物下方两角",        "en": "dual-arm grasp lower corners"},
        "lift":    {"zh": "双臂提起衣物下半部分",        "en": "dual-arm lift lower half"},
        "move":    {"zh": "把衣物下半部分对折到中部",    "en": "dual-arm fold lower half over the middle"},
        "lower":   {"zh": "把衣物边缘放回桌面",          "en": "dual-arm lower folded edge"},
        "release": {"zh": "双臂松开并后撤",              "en": "dual-arm release and retract"},
    },
    "cycle2": {
        "grasp":   {"zh": "双臂夹取衣物上方两角",        "en": "dual-arm grasp upper corners"},
        "lift":    {"zh": "双臂提起衣物上半部分",        "en": "dual-arm lift upper half"},
        "move":    {"zh": "把衣物上半部分对折到中部",    "en": "dual-arm fold upper half over the middle"},
        "lower":   {"zh": "把衣物边缘放回桌面",          "en": "dual-arm lower folded edge"},
        "release": {"zh": "双臂松开并后撤",              "en": "dual-arm release and retract"},
    },
    "cycle3": {
        "grasp":   {"zh": "拿起衣服下摆",                "en": "grasp the hem"},
        "lift":    {"zh": "把下摆提起",                  "en": "lift the hem up"},
        "move":    {"zh": "把下摆对折到衣领",            "en": "fold the hem to the collar"},
        "lower":   {"zh": "把下摆放下",                  "en": "place the hem down"},
        "release": {"zh": "松开并后撤",                  "en": "release and retract"},
    },
    "transport": {
        "grasp":   {"zh": "双臂夹取衣物中段",            "en": "dual-arm grasp middle of the cloth"},
        "lift":    {"zh": "双臂把衣物整体抬起",          "en": "dual-arm lift the cloth as a whole"},
        "move":    {"zh": "把衣服拿近",                  "en": "dual-arm pull the cloth closer"},
        "lower":   {"zh": "双臂把衣物放回桌面",          "en": "dual-arm place the cloth back"},
        "release": {"zh": "双臂松开并后撤",              "en": "dual-arm release and retract"},
    },
}


# ---------------------------------------------------------------------------
# Fold parameters previously lived in scene_info.json cloth_175fb7fa.cycles / transport;
# now embedded in data_gen.py. Vertex indices, lift_dz, pin_radius, etc. belong to task orchestration,
# independently of scene layout.
# ---------------------------------------------------------------------------
_CYCLES = [
    {
        "name": "cycle1",
        "grab_left": 2573, "grab_right": 17766,
        "place_left": 1349, "place_right": 17310,
        "lift_dz": 0.15, "descend_dz": 0.13,
        "place_offset_left": [0.05, 0, 0],
        "place_offset_right": [0, 0, 0],
        "pin_radius": 0.03,
    },
    {
        "name": "cycle2",
        "grab_left": 1652, "grab_right": 18532,
        "place_left": 1349, "place_right": 17310,
        "lift_dz": 0.15, "descend_dz": 0.12,
        "place_offset_left": [-0.05, 0, 0],
        "place_offset_right": [0, 0, 0],
        "pin_radius": 0.03,
    },
    {
        "name": "cycle3",
        "grab_left": 17310, "grab_right": 17310,
        "place_left": 1349, "place_right": 1349,
        "lift_dz": 0.20, "descend_dz": 0.13,
        "place_offset_left": [0, 0.0, 0],
        "place_offset_right": [0, 0, 0],
        "pin_radius": 0.05,
        "single_arm": True,
    },
]

_TRANSPORT = {
    "grab_left": 4298, "grab_right": 18266,
    "lift_dz": 0.035,
}

_BASE_TRANSPORT_PULL_M = 0.1
_MAX_TABLE_X_OFFSET_M = 0.20


def _transport_move_dx_from_layout(layout):
    """Return the negative-X pull needed for this generalized scene.

    The canonical workspace starts at x=0 and pulls 0.1 m toward the robot.
    When the shared table/proxy/cloth pose is shifted +x, pull that additional
    distance as well. The scene generator bounds the offset to [0, 0.20] m;
    clipping here keeps the controller safe if hand-edited metadata is passed.
    """
    tables = [info for info in layout.values() if info.get("usd") == "table_000"]
    if len(tables) != 1:
        raise ValueError(f"expected exactly one table_000 in layout, got {len(tables)}")
    table_x = float(tables[0]["xyz"][0])
    x_offset = float(np.clip(table_x, 0.0, _MAX_TABLE_X_OFFSET_M))
    return -(_BASE_TRANSPORT_PULL_M + x_offset)


def _desc(phase_name: str, step: str):
    """phase_name in {cycle1, cycle2, cycle3, transport}; step in {grasp, lift, move, lower, release}.
    Unconfigured descriptions fall back to {zh: step, en: step}."""
    return _PHASE_DESC.get(phase_name, {}).get(step, {"zh": step, "en": step})


def _do_grasp_and_pin(robot, cloth_actor, cloth_prim, base_xyz,
                       grab_idx_l, grab_idx_r,
                       pin_radius, pin_group_l="L", pin_group_r="R",
                       single_arm=False, single_arm_side="left",
                       grasp_desc=None):
    """Read cloth vertices -> construct grasp poses -> pick_actor/DualPick pre-grasp + grasp + close
    -> FEM pin attach. Return (grasp_l_T, grasp_l_q, grasp_r_T, grasp_r_q) or None.
    grasp_desc passes through to GraspConfig.action_description for recording labels. In single-arm mode,
    single_arm_side chooses the hand grasping grab_idx_l; returned l slot belongs to that actual arm.
    Existing l naming minimizes downstream changes; _run_cycle single branch forwards single_arm_side
    to _single_move and _do_release_and_retract.
    """
    indices = [grab_idx_l, grab_idx_r] if not single_arm else [grab_idx_l]
    nodes = robot.get_cloth_nodal_positions(cloth_prim, indices=indices)
    grasp_l_xyz = nodes[0]
    if not single_arm:
        grasp_r_xyz = nodes[1]
        print(f"[cf] grasp_l={grasp_l_xyz}  grasp_r={grasp_r_xyz}")
    else:
        grasp_r_xyz = None
        print(f"[cf] grasp_single({single_arm_side})={grasp_l_xyz}")

    if single_arm:
        side = single_arm_side
        world_x_deg = +45.0 if side == "left" else -45.0
        T_s, q_s = _grasp_pose(grasp_l_xyz, side=side, world_x_deg=world_x_deg)
        _draw_cube(robot, name=f"grasp_{side}_{pin_group_l}",
                   pose_4x4=T_s, size=0.02, color=[1.0, 0.0, 0.0])
        _draw_cube(robot, name=f"node_{side}_{pin_group_l}",
                   position=grasp_l_xyz.tolist(),
                   size=0.01, color=[0.0, 0.3, 1.0])
        T_s_cands = _grasp_pose_candidates(
            grasp_l_xyz, side=side, world_x_deg=world_x_deg,
        )
        if not _single_pick_cloth(robot, cloth_actor, side, T_s_cands,
                                   lift_dz_in_pick=0.0, action_description=grasp_desc):
            return None
        robot.wait(0.5)
        n_s = robot.set_fem_pin(arm=side, cloth_prim=cloth_prim,
                                 radius=pin_radius, group_id=pin_group_l)
        print(f"[cf] pin attached {side}={n_s}")
        if n_s == 0:
            print("[cf] pinned 0 nodes — abort")
            return None
        return T_s[:3, 3], q_s, None, None

    T_l, q_l = _grasp_pose(grasp_l_xyz, side="left",  world_x_deg=+45.0)
    # Debug: small red cube at reference grasp_l pose and blue cubes at FEM nodes;
    # global DEBUG controls all drawing; disabled mode is a no-op.
    _draw_cube(robot, name=f"grasp_l_{pin_group_l}",
               pose_4x4=T_l, size=0.02, color=[1.0, 0.0, 0.0])
    _draw_cube(robot, name=f"node_l_{pin_group_l}",
               position=grasp_l_xyz.tolist(),
               size=0.01, color=[0.0, 0.3, 1.0])
    # Small variation: reference plus four candidates (XY +/-1 cm, world-X rotation +/-5 degrees); server IK filter
    # selects reachable poses. FEM vertex positions vary slightly each run, so multiple candidates avoid fragile single-pose IK failure.
    T_l_cands = _grasp_pose_candidates(
        grasp_l_xyz, side="left", world_x_deg=+45.0,
    )

    T_r, q_r = _grasp_pose(grasp_r_xyz, side="right", world_x_deg=-45.0)
    _draw_cube(robot, name=f"grasp_r_{pin_group_r}",
               pose_4x4=T_r, size=0.02, color=[1.0, 0.5, 0.0])
    _draw_cube(robot, name=f"node_r_{pin_group_r}",
               position=grasp_r_xyz.tolist(),
               size=0.01, color=[0.0, 0.3, 1.0])
    T_r_cands = _grasp_pose_candidates(
        grasp_r_xyz, side="right", world_x_deg=-45.0,
    )
    if not _dual_pick_cloth(robot, cloth_actor, T_l_cands, T_r_cands,
                             lift_dz_in_pick=0.0, action_description=grasp_desc):
        return None
    robot.wait(0.5)

    # FEM pin attach after closure, with gripper at cloth surface.
    n_l = robot.set_fem_pin(arm="left", cloth_prim=cloth_prim,
                             radius=pin_radius, group_id=pin_group_l)
    n_r = robot.set_fem_pin(arm="right", cloth_prim=cloth_prim,
                             radius=pin_radius, group_id=pin_group_r)
    print(f"[cf] pin attached L={n_l} R={n_r}")
    if n_l == 0 or n_r == 0:
        print("[cf] pinned 0 nodes — abort")
        return None
    return T_l[:3, 3], q_l, T_r[:3, 3], q_r


def _do_release_and_retract(robot, grasp_l_pos, grasp_l_q,
                             grasp_r_pos, grasp_r_q, base_xyz,
                             retract_dist=0.10, single_arm=False,
                             single_arm_side="left",
                             release_desc=None):
    print("[cf] === release(pin + open) ===")
    robot.release_fem_pin(group_id="all")
    if single_arm:
        robot.open_gripper(arm=single_arm_side)
    else:
        robot.open_gripper(arm="left")
        robot.open_gripper(arm="right")
    robot.wait(0.5)

    # Retreat along EE local z, mapped into world by current grasp_l_q.
    w, x, y, z = grasp_l_q
    z_axis = np.array([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)])
    ret_l_pos = grasp_l_pos - z_axis * retract_dist
    print(f"[cf] === retract -z {retract_dist}m ===")
    if not single_arm:
        w2, x2, y2, z2 = grasp_r_q
        z_axis_r = np.array([2*(x2*z2 + y2*w2), 2*(y2*z2 - x2*w2), 1 - 2*(x2*x2 + y2*y2)])
        ret_r_pos = grasp_r_pos - z_axis_r * retract_dist
        return _dual_move(robot, ret_l_pos, grasp_l_q, ret_r_pos, grasp_r_q,
                          base_xyz, action_description=release_desc, label="retract")
    return _single_move(robot, single_arm_side, ret_l_pos, grasp_l_q, base_xyz,
                        action_description=release_desc,
                        label=f"retract({single_arm_side[0].upper()})")


def _run_cycle(robot, cloth_actor, cloth_prim, base_xyz, cycle):
    name = cycle["name"]
    single = bool(cycle.get("single_arm", False))
    is_last_cycle = bool(cycle.get("is_last_cycle", single))   # cycle3 is the last by default.
    print(f"[cf] === cycle {name} === (single_arm={single}, last_cycle={is_last_cycle})")

    # -- Automatic left/right arm assignment --
    # grasp: assign by grasp-point y (y>0 left, y<0 right); one point determines single_side for single-arm mode.
    # move/lift/lower inherit grasp left/right assignment without reassessment.
    # place: except last cycle, assign by target y too, ensuring left arm goes to positive-y point;
    # place_offset_left/right **swap with** place_left/right, because offsets name
    # actual hands, not scene_info field names.
    if single:
        grab_idx_l_use = int(cycle["grab_left"])
        grab_idx_r_use = grab_idx_l_use   # Ignore r slot in single-arm mode.
        single_side = _pick_arm_by_y(robot, cloth_prim, grab_idx_l_use)
        place_l_use = int(cycle["place_left"])
        place_r_use = place_l_use
        place_offset_l = np.array(cycle.get("place_offset_left",  [0, 0, 0]))
        place_offset_r = np.array(cycle.get("place_offset_right", [0, 0, 0]))
        print(f"[cf] {name}: single_side={single_side} grab={grab_idx_l_use} place={place_l_use}")
    else:
        grab_idx_l_use, grab_idx_r_use, gy_l, gy_r = _assign_pair_by_y(
            robot, cloth_prim, cycle["grab_left"], cycle["grab_right"]
        )
        single_side = "left"   # Unused placeholder.
        print(f"[cf] {name}: grab assign by y → L={grab_idx_l_use}(y={gy_l:.3f}) "
              f"R={grab_idx_r_use}(y={gy_r:.3f})")
        if is_last_cycle:
            # Last-cycle target points retain configured assignment.
            place_l_use = int(cycle["place_left"])
            place_r_use = int(cycle["place_right"])
            place_offset_l = np.array(cycle.get("place_offset_left",  [0, 0, 0]))
            place_offset_r = np.array(cycle.get("place_offset_right", [0, 0, 0]))
            print(f"[cf] {name}: keep place allocation (last_cycle) L={place_l_use} R={place_r_use}")
        else:
            # Swap place and offset together by y, keeping offsets associated with actual hands.
            a, b = cycle["place_left"], cycle["place_right"]
            oa = np.array(cycle.get("place_offset_left",  [0, 0, 0]))
            ob = np.array(cycle.get("place_offset_right", [0, 0, 0]))
            ya = _node_y(robot, cloth_prim, a)
            yb = _node_y(robot, cloth_prim, b)
            if ya >= yb:
                place_l_use, place_r_use = int(a), int(b)
                place_offset_l, place_offset_r = oa, ob
                py_l, py_r = ya, yb
            else:
                place_l_use, place_r_use = int(b), int(a)
                place_offset_l, place_offset_r = ob, oa
                py_l, py_r = yb, ya
            print(f"[cf] {name}: place assign by y → L={place_l_use}(y={py_l:.3f}) "
                  f"R={place_r_use}(y={py_r:.3f})")

    pinned = _do_grasp_and_pin(
        robot, cloth_actor, cloth_prim, base_xyz,
        grab_idx_l=grab_idx_l_use,
        grab_idx_r=grab_idx_r_use,
        pin_radius=float(cycle["pin_radius"]),
        pin_group_l=f"{name}_L", pin_group_r=f"{name}_R",
        single_arm=single, single_arm_side=single_side,
        grasp_desc=_desc(name, "grasp"),
    )
    if pinned is None:
        return False
    grasp_l_pos, grasp_l_q, grasp_r_pos, grasp_r_q = pinned

    # fold/lift +z: lock orientation and XY, leave world z free.
    # Use **from_current_pose=True and goal_offset=[0,0,dz] together**: server computes planning goal from
    # its own current EE plus offset, matching path_constraint start, also server current.
    # Old lift_l_pos = grasp_l_pos + dz used theoretical grasp position, causing plan goal
    # to differ from server-current path_constraint start and a "Partial position" error.
    # between start and goal is not equal" + "update pose cost metric failed".
    # Client still supplies reference target = client-read current + dz, unused by server,
    # but useful for log comparisons.
    cur_ee_l = robot.get_ee_pose(arm="left" if not single else single_side)
    lift_dz = float(cycle["lift_dz"])
    lift_l_pos = cur_ee_l[:3, 3].copy(); lift_l_pos[2] += lift_dz
    print(f"[cf] === fold/lift +z {lift_dz} === (cur_ee_l_pos={cur_ee_l[:3,3].tolist()})")
    _lift_pc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]   # rx,ry,rz, x,y,z; z free.
    _lift_go = [0.0, 0.0, lift_dz, 1.0, 0.0, 0.0, 0.0]
    _lift_desc = _desc(name, "lift")
    if not single:
        cur_ee_r = robot.get_ee_pose(arm="right")
        lift_r_pos = cur_ee_r[:3, 3].copy(); lift_r_pos[2] += lift_dz
        if not _dual_move(robot, lift_l_pos, grasp_l_q, lift_r_pos, grasp_r_q,
                          base_xyz, path_constraint=_lift_pc,
                          offset_in_goal_frame=False,
                          from_current_pose=True, goal_offset=_lift_go,
                          slowdown_factor=5.0,
                          action_description=_lift_desc,
                          label=f"{name}/lift"):
            return False
    else:
        lift_r_pos = None
        if not _single_move(robot, single_side, lift_l_pos, grasp_l_q, base_xyz,
                            path_constraint=_lift_pc, offset_in_goal_frame=False,
                            from_current_pose=True, goal_offset=_lift_go,
                            slowdown_factor=5.0,
                            action_description=_lift_desc,
                            label=f"{name}/lift({single_side[0].upper()})"):
            return False
    robot.wait(0.5)

    # fold/move to place-vertex XY with unchanged z: lock orientation and z, leave XY free.
    # Absolute XY cannot be represented by from_current_pose offsets, so keep
    # client-derived target with from_current_pose=False. XY comes from place vertex; use server
    # current EE z to prevent previous-step z drift.
    # place_l_use / place_r_use already reassigned by y at entry except single-arm/last cycle;
    # pair with grab_idx_l_use / grab_idx_r_use so each hand's grasp maps to its same-side place.
    place_indices = [place_l_use, place_r_use] if not single else [place_l_use]
    place_pos_world = robot.get_cloth_nodal_positions(cloth_prim, indices=place_indices)
    # Shift place point 3 mm toward (0,0), without sampling; server fold/move uses a single target.
    place_pos_world = np.stack(
        [_shift_toward_origin_xy(p, dist=0.003) for p in place_pos_world],
        axis=0,
    )
    # place_offset_l/r already swapped with place indices at entry; use directly.
    cur_ee_l = robot.get_ee_pose(arm="left" if not single else single_side)
    move_l_pos = cur_ee_l[:3, 3].copy()
    move_l_pos[0] = place_pos_world[0][0] + place_offset_l[0]
    if not single:
        move_l_pos[1] = place_pos_world[0][1] + place_offset_l[1]
        cur_ee_r = robot.get_ee_pose(arm="right")
        move_r_pos = cur_ee_r[:3, 3].copy()
        move_r_pos[0] = place_pos_world[1][0] + place_offset_r[0]
        move_r_pos[1] = place_pos_world[1][1] + place_offset_r[1]
    else:
        move_l_pos[1] = place_pos_world[0][1] + place_offset_l[1]
        move_r_pos = None
    print(f"[cf] === fold/move === (cur_ee_l_pos={cur_ee_l[:3,3].tolist()})")
    _move_pc = [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]   # XY free, z locked.
    _move_desc = _desc(name, "move")
    if not single:
        if not _dual_move(robot, move_l_pos, grasp_l_q, move_r_pos, grasp_r_q,
                          base_xyz, path_constraint=_move_pc,
                          offset_in_goal_frame=False,
                          slowdown_factor=2.0,
                          action_description=_move_desc, label=f"{name}/move"):
            return False
    else:
        if not _single_move(robot, single_side, move_l_pos, grasp_l_q, base_xyz,
                            path_constraint=_move_pc, offset_in_goal_frame=False,
                            slowdown_factor=2.0,
                            action_description=_move_desc,
                            label=f"{name}/move({single_side[0].upper()})"):
            return False
    robot.wait(0.5)

    # fold/lower -z: same constraints and from_current_pose mode as lift.
    cur_ee_l = robot.get_ee_pose(arm="left" if not single else single_side)
    descend_dz = float(cycle["descend_dz"])
    lower_l_pos = cur_ee_l[:3, 3].copy(); lower_l_pos[2] -= descend_dz
    print(f"[cf] === fold/lower -z {descend_dz} === (cur_ee_l_pos={cur_ee_l[:3,3].tolist()})")
    _lower_go = [0.0, 0.0, -descend_dz, 1.0, 0.0, 0.0, 0.0]
    _lower_desc = _desc(name, "lower")
    if not single:
        cur_ee_r = robot.get_ee_pose(arm="right")
        lower_r_pos = cur_ee_r[:3, 3].copy(); lower_r_pos[2] -= descend_dz
        if not _dual_move(robot, lower_l_pos, grasp_l_q, lower_r_pos, grasp_r_q,
                          base_xyz, path_constraint=_lift_pc,
                          offset_in_goal_frame=False,
                          from_current_pose=True, goal_offset=_lower_go,
                          slowdown_factor=2.0,
                          action_description=_lower_desc,
                          label=f"{name}/lower"):
            return False
    else:
        if not _single_move(robot, single_side, lower_l_pos, grasp_l_q, base_xyz,
                            path_constraint=_lift_pc, offset_in_goal_frame=False,
                            from_current_pose=True, goal_offset=_lower_go,
                            slowdown_factor=2.0,
                            action_description=_lower_desc,
                            label=f"{name}/lower({single_side[0].upper()})"):
            return False
    robot.wait(0.5)

    if not _do_release_and_retract(
        robot, lower_l_pos, grasp_l_q,
        None if single else lower_r_pos, grasp_r_q,
        base_xyz, retract_dist=0.10, single_arm=single,
        single_arm_side=single_side,
        release_desc=_desc(name, "release"),
    ):
        return False
    print(f"[cf] === cycle {name} done ===")
    return True


def _run_transport(robot, cloth_actor, cloth_prim, base_xyz, transport):
    # Synchronized dual-arm transport assigns by grasp y: y<0 left, y>0 right.
    # Transport has no place point; swap only grab.
    grab_l_use, grab_r_use, gy_l, gy_r = _assign_pair_by_y(
        robot, cloth_prim, transport["grab_left"], transport["grab_right"]
    )
    print(f"[cf] === transport === grab assign by y → "
          f"L={grab_l_use}(y={gy_l:.3f})  R={grab_r_use}(y={gy_r:.3f})")
    pinned = _do_grasp_and_pin(
        robot, cloth_actor, cloth_prim, base_xyz,
        grab_idx_l=grab_l_use,
        grab_idx_r=grab_r_use,
        pin_radius=0.03,
        pin_group_l="T_L", pin_group_r="T_R",
        single_arm=False,
        grasp_desc=_desc("transport", "grasp"),
    )
    if pinned is None:
        return False
    grasp_l_pos, grasp_l_q, grasp_r_pos, grasp_r_q = pinned

    lift_dz = float(transport["lift_dz"])
    move_dx = float(transport["move_dx"])

    # Every segment computes target from server-current EE with from_current_pose=True and goal_offset,
    # keeping server plan goal strictly consistent with server-current path_constraint start.
    cur_ee_l = robot.get_ee_pose(arm="left")
    cur_ee_r = robot.get_ee_pose(arm="right")
    lift_l_pos = cur_ee_l[:3, 3].copy(); lift_l_pos[2] += lift_dz
    lift_r_pos = cur_ee_r[:3, 3].copy(); lift_r_pos[2] += lift_dz
    print(f"[cf] === T/lift +z {lift_dz} ===")
    _T_lift_pc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]   # Lock orientation and XY, leave z free.
    _T_lift_go = [0.0, 0.0, lift_dz, 1.0, 0.0, 0.0, 0.0]
    if not _dual_move(robot, lift_l_pos, grasp_l_q, lift_r_pos, grasp_r_q,
                      base_xyz, path_constraint=_T_lift_pc,
                      offset_in_goal_frame=False,
                      from_current_pose=True, goal_offset=_T_lift_go,
                      slowdown_factor=5.0,
                      action_description=_desc("transport", "lift"),
                      label="T/lift"):
        return False
    robot.wait(0.3)

    cur_ee_l = robot.get_ee_pose(arm="left")
    cur_ee_r = robot.get_ee_pose(arm="right")
    move_l_pos = cur_ee_l[:3, 3].copy(); move_l_pos[0] += move_dx
    move_r_pos = cur_ee_r[:3, 3].copy(); move_r_pos[0] += move_dx
    print(f"[cf] === T/move dx={move_dx} ===")
    _T_move_pc = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1]   # Lock orientation and y/z, leave only x free.
    _T_move_go = [move_dx, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if not _dual_move(robot, move_l_pos, grasp_l_q, move_r_pos, grasp_r_q,
                      base_xyz, path_constraint=_T_move_pc,
                      offset_in_goal_frame=False,
                      from_current_pose=True, goal_offset=_T_move_go,
                      slowdown_factor=4.0,
                      action_description=_desc("transport", "move"),
                      label="T/move"):
        return False
    robot.wait(0.3)

    cur_ee_l = robot.get_ee_pose(arm="left")
    cur_ee_r = robot.get_ee_pose(arm="right")
    lower_l_pos = cur_ee_l[:3, 3].copy(); lower_l_pos[2] -= lift_dz
    lower_r_pos = cur_ee_r[:3, 3].copy(); lower_r_pos[2] -= lift_dz
    print(f"[cf] === T/lower -z {lift_dz} ===")
    _T_lower_go = [0.0, 0.0, -lift_dz, 1.0, 0.0, 0.0, 0.0]
    if not _dual_move(robot, lower_l_pos, grasp_l_q, lower_r_pos, grasp_r_q,
                      base_xyz, path_constraint=_T_lift_pc,
                      offset_in_goal_frame=False,
                      from_current_pose=True, goal_offset=_T_lower_go,
                      slowdown_factor=2.0,
                      action_description=_desc("transport", "lower"),
                      label="T/lower"):
        return False
    robot.wait(0.3)

    if not _do_release_and_retract(
        robot, lower_l_pos, grasp_l_q,
        lower_r_pos, grasp_r_q,
        base_xyz, retract_dist=0.08, single_arm=False,
        release_desc=_desc("transport", "release"),
    ):
        return False
    print("[cf] === transport done ===")
    return True


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def execute_task(robot: RobotAPI) -> bool:
    description_json = {
        "task_name": "dual-arm double-layer T-shirt folding via FEM pin servo",
        "english_task_name": "dual-arm double-layer T-shirt folding via FEM pin servo",
        "init_scene_text": "Double-layer T-shirt on table; both arms grasp two corners and fold.",
    }
    robot.client.set_task_task_basic_info(description_json)

    layout = robot.scene_info["layout"]
    cloth_id = next(
        k for k, v in layout.items()
        if "cloth" in v.get("keywords", []) or "deformable" in v.get("keywords", [])
    )
    cycles = _CYCLES
    transport = dict(_TRANSPORT)
    transport["move_dx"] = _transport_move_dx_from_layout(layout)
    print(
        f"[cf] table x={-transport['move_dx'] - _BASE_TRANSPORT_PULL_M:.3f}m; "
        f"transport pull={-transport['move_dx']:.3f}m"
    )

    base_xyz = np.array(robot.config.robot_config["robot_init_pose"]["position"])
    # Actual cloth-body prim path is determined by server init_cloth_views finding the unique SurfaceDeformableBody;
    # set_fem_pin and get_cloth_nodal_positions require the same path. Obtain it once by server scan,
    # falling back to layout if get_cloth_nodal_positions(prim=None) is unsupported.
    # Simplification: scene.usda now payloads cloth_000/Aligned.usda into cloth_175fb7fa, giving grafted
    # body mesh path /World/Objects/{cloth_id}/body.
    cloth_prim = f"/World/Objects/{cloth_id}/entity/body"
    print(f"[cf] cloth_id={cloth_id}, cloth_prim={cloth_prim}, cycles={len(cycles)}, base={base_xyz}")

    # As in cloth_fold_dual_g2_attach.py, curobo plans only robot self-collision, not scene collisions.
    # Remove layout objects with table/deformable/cloth keywords from curobo obstacle
    # list; otherwise near-table cloth grasp is rejected by obstacle IK because gripper nearly touches table.
    _obstacle_exclude_prims = []
    for obj_id, info in layout.items():
        kw = set(info.get("keywords", []) or [])
        if kw & {"table", "workspace_table", "deformable", "cloth", "softbody", "fem"}:
            _obstacle_exclude_prims.append(robot.get_object_prim_path(obj_id))
    if _obstacle_exclude_prims:
        try:
            robot.remove_objs_from_obstacle(_obstacle_exclude_prims)
            print(f"[cf] removed from curobo obstacle: {_obstacle_exclude_prims}")
        except Exception as _exc:
            print(f"[cf] remove_objs_from_obstacle failed (continuing): {_exc}")

    # FEM cooking and physics settling.
    robot.wait(0.5)

    # Disable gripper colliders for the entire task: during constrained lift/fold/move,
    # gripper mesh contact with cloth/table can mark planning collision, reporting success while EE remains stuck.
    # FEM pin servo handles node following, so gripper physics collision is unnecessary. Do not restore at task end.
    try:
        robot.disable_gripper_colliders(arm="both")
        print("[cf] gripper colliders disabled (task-wide)")
    except Exception as _exc:
        print(f"[cf] disable_gripper_colliders failed (continuing): {_exc}")

    # cycle1
    if not _run_cycle(robot, cloth_id, cloth_prim, base_xyz, cycles[0]):
        print("[cf] cycle1 FAIL")
        return False
    _check_fold(robot, cloth_prim, "cycle1")

    # Transport between cycle1 and cycle2.
    if transport is not None:
        if not _run_transport(robot, cloth_id, cloth_prim, base_xyz, transport):
            print("[cf] transport FAIL")
            return False
        _check_fold(robot, cloth_prim, "transport")

    # cycle2
    if len(cycles) > 1:
        if not _run_cycle(robot, cloth_id, cloth_prim, base_xyz, cycles[1]):
            print("[cf] cycle2 FAIL")
            return False
        _check_fold(robot, cloth_prim, "cycle2")

    # cycle3, single arm.
    if len(cycles) > 2:
        if not _run_cycle(robot, cloth_id, cloth_prim, base_xyz, cycles[2]):
            print("[cf] cycle3 FAIL")
            return False
        _check_fold(robot, cloth_prim, "cycle3")

    print("[cf] === ALL CYCLES DONE ===")
    robot.wait(3.0)
    final_ok = _check_fold(robot, cloth_prim, "final")
    print(f"[cf] final fold check: {'PASS' if final_ok else 'FAIL'}")
    return True
