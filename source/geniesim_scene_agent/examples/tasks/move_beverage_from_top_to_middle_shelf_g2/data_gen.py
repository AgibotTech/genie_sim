# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import json


# layers.json index of the plank the beverage is placed DOWN ONTO (the MIDDLE
# layer, which is the DESTINATION). For benchmark_supermarket_shelf_006 (5
# planks, 0=lowest .. 4=highest) the beverage starts on the OPEN TOP plank
# (index 4) and is moved DOWN onto the MIDDLE plank (index 3). This mirrors
# the proven move_shelf_fruit_onto_plate_g2 direction (top -> middle).
TARGET_LAYER_INDEX = 3

# layers.json index of the plank directly ABOVE the destination (the top plank).
# Used to cap the release z so the gripper never rises into it while descending
# into the middle compartment.
CEILING_LAYER_INDEX = 4

# Safety margin (meters) above the destination plank before releasing. Small:
# we want the beverage to land flush, not drop from a height (a tall, narrow
# can topples more easily when dropped). 5 mm keeps a tiny cushion.
PLACE_MARGIN = 0.005

# ---- Upright thresholds -------------------------------------------------------
# Cosine of the beverage's tilt away from its reference-upright orientation.
# 1.0 = perfectly upright, 0.0 = tipped 90 deg.
PLACE_UPRIGHT_COS = 0.90        # ~25.8 deg, release-instant checker (lenient)
FINAL_UPRIGHT_COS = 0.96        # ~16.3 deg, post-settle authoritative gate (strict)

# Seconds to wait after back_to_origin before the final upright check, giving
# the scene time to settle after the arm withdraws.
SETTLE_SECONDS = 2.0

# Small post-grasp vertical lift (meters). The TOP layer is OPEN (no plank
# above it), so there is no ceiling constraint during the pick. 0.04 m is
# enough to lift the beverage clear of the top plank for a clean pull-out.
POST_GRASP_LIFT = 0.04

# World x between robot and shelf front, used as the pull-out waypoint.
WAYPOINT_X = 0.0

# Keep pickup wrist configurations reachable at both shelf heights.
# Prefer near-horizontal approaches at or below the beverage center,
# and retain the framework's per-arm upright-gripper sign constraint.
PICK_PRE_GRASP_DISTANCE = 0.06
# Strict/relaxed thresholds on approach-axis Z (world) and grasp-point Z relative
# to bottle center. First the strict pool; if empty, the relaxed fallback.
PICK_HORIZONTAL_APPROACH_Z_STRICT = 0.30
PICK_HORIZONTAL_APPROACH_Z_RELAXED = 0.50
PICK_MID_LOW_MARGIN_STRICT = 0.02   # grasp z <= center_z + 2 cm
PICK_MID_LOW_MARGIN_RELAXED = 0.05  # grasp z <= center_z + 5 cm

# Front-bias on the DESTINATION plank: the middle layer sits below the top
# plank, so releases deep under the shelf force the arm to fully extend (EE far
# from base) -> large wrist pitch -> cuRobo trajopt converges slowly / IK fails.
# Keep the drop hard against the shelf front edge (near the robot) so the arm
# stays inside its comfortable reach envelope.
DEST_FRONT_DEPTH_FRAC = 0.15
DEST_FRONT_INSET = 0.02


def _find_shelf_layers_json(shelf_usd: str) -> str:
    path = f"{ASSETS_PATH}/interaction/{shelf_usd}/layers.json"
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"layers.json not found for {shelf_usd} at {path}. "
        f"Run generate-shelf-layers first."
    )


def _world_transform_from_shelf(robot: RobotAPI, shelf_id: str):
    shelf_pos, shelf_quat_wxyz = robot.get_object_pose(shelf_id)
    R = quat2mat_wxyz(shelf_quat_wxyz)
    return np.asarray(shelf_pos, dtype=float), np.asarray(R, dtype=float)


def _layer_top_z_world(layer, shelf_origin_world, R_world_from_local):
    center_world = shelf_origin_world + R_world_from_local @ np.asarray(layer["center_local"])
    return float(center_world[2])


def _layer_world_bounds(layer, shelf_origin_world, R_world_from_local):
    xs, ys = [], []
    for corner_local in layer["corners_local"]:
        cw = shelf_origin_world + R_world_from_local @ np.asarray(corner_local, dtype=float)
        xs.append(float(cw[0]))
        ys.append(float(cw[1]))
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "cx": 0.5 * (x_min + x_max),
        "cy": 0.5 * (y_min + y_max),
    }


def _align_R(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-8:
        return np.eye(3) if c >= 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _filter_horizontal_midlow_grasps(grasp_poses, obj_pose, grasp_widths):
    """Keep HORIZONTAL side-approach grasps on the middle-lower body:
      - approach axis (gripper +Z in world) nearly horizontal (|z| small)
        -> gripper enters from the SIDE, not top or bottom
      - grasp point z at or slightly below bottle center z (mid-lower body)
        -> bottle sticks up out of the gripper for stable pull-out / place
    We deliberately do NOT constrain jaw axis Z: upstream already forces the
    jaw sign (grasp.py L180-188 upright_mask: left arm jaw_z<0, right arm
    jaw_z>0) to avoid an upside-down wrist. On an upright cylindrical bottle
    the natural side grasps have jaw axis vertical (jaw_z ~ ±1) -- forcing
    |jaw_z|<0.3 kills every candidate. What we actually need is a horizontal
    APPROACH; the wrist ends up level regardless of jaw orientation."""
    grasp_poses = np.asarray(grasp_poses, dtype=float)
    grasp_widths = np.asarray(grasp_widths)
    if grasp_poses.size == 0:
        return grasp_poses, grasp_widths

    approach_z = grasp_poses[:, 2, 2]  # gripper +Z axis, world Z component
    grasp_z = grasp_poses[:, 2, 3]     # world Z of grasp point
    obj_z = float(np.asarray(obj_pose, dtype=float)[2, 3])

    strict = (
        (np.abs(approach_z) < PICK_HORIZONTAL_APPROACH_Z_STRICT)
        & (grasp_z <= obj_z + PICK_MID_LOW_MARGIN_STRICT)
    )
    if np.any(strict):
        return grasp_poses[strict], grasp_widths[strict]

    relaxed = (
        (np.abs(approach_z) < PICK_HORIZONTAL_APPROACH_Z_RELAXED)
        & (grasp_z <= obj_z + PICK_MID_LOW_MARGIN_RELAXED)
    )
    if np.any(relaxed):
        return grasp_poses[relaxed], grasp_widths[relaxed]

    # Final fallback: only enforce mid-lower height; better a slightly tilted
    # grasp than no grasp (pick_actor's own IK filter will trim unreachable).
    fallback = grasp_z <= obj_z + PICK_MID_LOW_MARGIN_RELAXED
    if np.any(fallback):
        return grasp_poses[fallback], grasp_widths[fallback]

    return grasp_poses[:0], grasp_widths[:0]


def _upright_cos_vs_ref(quat_wxyz, ref_R) -> float:
    """Cosine of tilt away from the reference-upright orientation."""
    if quat_wxyz is None:
        return -1.0
    R_now = quat2mat_wxyz(quat_wxyz)
    ref_R = np.asarray(ref_R, dtype=float)
    return float((R_now @ ref_R.T)[2, 2])


def _build_target_poses_on_middle_plank(
    dest_center_xy: np.ndarray,
    upright_half_height: float,
    dest_plank_top_z: float,
    ceiling_top_z: float,
    initial_pose_world: np.ndarray,
    n_xy: int = 3,
    n_yaw: int = 12,
) -> np.ndarray:
    """Generate upright candidate poses for setting the beverage DOWN onto the
    middle plank.

    - Translation: tight +-5 mm grid around the front-biased destination centre
    - Rotation: beverage local +Y aligned to world +Z, swept 0-360 deg about Z
    - z: resting release height (PLACE_MARGIN above plank) up to a small band
      capped well below the ceiling plank above
    """
    cx, cy = float(dest_center_xy[0]), float(dest_center_xy[1])

    z_floor = dest_plank_top_z + PLACE_MARGIN + float(upright_half_height)
    # Keep the hand under the top plank ceiling. 0.20 m headroom as in the fruit task.
    z_ceiling_cap = ceiling_top_z - 0.20
    z_high = min(z_floor + 0.02, z_ceiling_cap)
    if z_high < z_floor:
        z_high = z_floor
    z_grid = np.linspace(z_floor, z_high, 2)

    initial_pose_world = np.asarray(initial_pose_world, dtype=float).copy()
    if initial_pose_world.shape != (4, 4):
        initial_pose_world = np.eye(4)

    cur_R = np.asarray(initial_pose_world[:3, :3], dtype=float)
    R_up = _align_R(cur_R[:, 1], [0.0, 0.0, 1.0]) @ cur_R

    base_pose = np.eye(4)
    base_pose[:3, :3] = R_up
    yaw_deg_grid = np.linspace(0.0, 360.0, n_yaw, endpoint=False)
    rotated_samples = []
    for yaw_deg in yaw_deg_grid:
        rotated = rotate_along_axis(
            base_pose, float(yaw_deg), rot_axis="z", use_local=False,
        )
        rotated_samples.append(np.asarray(rotated, dtype=float))

    offs = np.linspace(-0.005, 0.005, n_xy)
    poses = []
    for dx in offs:
        for dy in offs:
            for z in z_grid:
                for rotated in rotated_samples:
                    pose = rotated.copy()
                    pose[:3, 3] = [cx + float(dx), cy + float(dy), float(z)]
                    poses.append(pose)

    if len(poses) < 3:
        for rotated in rotated_samples:
            pose = rotated.copy()
            pose[:3, 3] = [cx, cy, float(z_floor)]
            poses.append(pose)

    return np.array(poses)


def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    # ------------------------------------------------------------------
    # 1. Locate the shelf and the single target beverage (top layer).
    # ------------------------------------------------------------------
    shelf_id = None
    shelf_usd = None
    beverage_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        if shelf_id is None and (
            "target_shelf" in keywords or "shelf" in keywords
            or "benchmark_supermarket_shelf_" in usd
        ):
            shelf_id = obj_id
            shelf_usd = usd
            continue

        if beverage_id is None and (
            "target_beverage" in keywords or "beverage" in keywords
            or "drink" in keywords
            or "beverage_bottle" in usd
        ):
            beverage_id = obj_id
            continue

    assert shelf_id is not None, "shelf not found in scene_info"
    assert shelf_usd, "shelf_usd empty, cannot resolve layers.json"
    assert beverage_id is not None, "target beverage not found in scene_info"

    # ------------------------------------------------------------------
    # 2. Set task description BEFORE the first robot action.
    # ------------------------------------------------------------------
    description_json = {
        "task_name": 'Move a beverage from the top shelf to the middle shelf',
        "english_task_name": (
            'Move a beverage from the top shelf to the middle shelf'
        ),
        "init_scene_text": (
            "The robot is in front of a multi-layer supermarket shelf. The open "
            "top layer holds one beverage container (the target); the middle layer "
            "is empty (the destination); the other layers are empty."
        ),
    }
    robot.client.set_task_task_basic_info(description_json)

    # ------------------------------------------------------------------
    # 3. Load layers.json, resolve shelf transform and destination geometry.
    # ------------------------------------------------------------------
    layers_path = _find_shelf_layers_json(shelf_usd)
    with open(layers_path) as f:
        layers_data = json.load(f)
    target_layer = layers_data["layers"][TARGET_LAYER_INDEX]
    ceiling_layer = layers_data["layers"][CEILING_LAYER_INDEX]

    shelf_origin_world, R_world_from_local = _world_transform_from_shelf(robot, shelf_id)
    target_plank_top_z = _layer_top_z_world(target_layer, shelf_origin_world, R_world_from_local)
    ceiling_top_z = _layer_top_z_world(ceiling_layer, shelf_origin_world, R_world_from_local)

    dest_bounds = _layer_world_bounds(target_layer, shelf_origin_world, R_world_from_local)

    shelf_pos0_raw, shelf_quat0 = robot.get_object_pose(shelf_id)
    shelf_pos0 = np.asarray(shelf_pos0_raw, dtype=float).copy()
    shelf_quat0 = np.asarray(shelf_quat0, dtype=float).copy()

    # ------------------------------------------------------------------
    # 4. Move the beverage from the top layer down onto the middle plank.
    # ------------------------------------------------------------------
    protect_ids = [shelf_id]

    beverage_size = np.asarray(robot.get_object_size_in_world_frame(beverage_id), dtype=float)
    beverage_size_local = robot.get_object_size_in_local_frame(beverage_id)
    if beverage_size_local is None:
        beverage_size_local = beverage_size
    beverage_size_local = np.asarray(beverage_size_local, dtype=float)
    bev_half_x = float(beverage_size[0]) / 2.0
    bev_diag = max(float(beverage_size[0]), float(beverage_size[1]))
    bev_upright_half_height = float(beverage_size_local[1]) / 2.0
    bev_upright_height = float(beverage_size_local[1])

    # Front-biased destination spot on the middle plank.
    x_front = dest_bounds["x_min"] + bev_half_x + DEST_FRONT_INSET
    x_band_back = dest_bounds["x_min"] + DEST_FRONT_DEPTH_FRAC * (dest_bounds["x_max"] - dest_bounds["x_min"])
    dest_x = max(x_front, min(x_band_back, dest_bounds["x_max"] - bev_half_x - 0.005))
    dest_y = dest_bounds["cy"]
    dest_center_xy = np.array([float(dest_x), float(dest_y)])

    bev_pos_w, bev_quat_w = robot.get_object_pose(beverage_id)
    bev_init_pose = np.eye(4)
    bev_init_pose[:3, :3] = quat2mat_wxyz(bev_quat_w)
    bev_init_pose[:3, 3] = np.asarray(bev_pos_w, dtype=float)
    bev_ref_R = bev_init_pose[:3, :3].copy()

    target_poses = _build_target_poses_on_middle_plank(
        dest_center_xy,
        bev_upright_half_height,
        target_plank_top_z,
        ceiling_top_z,
        initial_pose_world=bev_init_pose,
    )

    arm = robot.judge_arm_to_use(bev_pos_w)

    # Checker: beverage lands on the middle plank within 3 cm, at the right z,
    # and is still upright at the instant of release.
    def _make_checker(bid=beverage_id, cx=float(dest_center_xy[0]), cy=float(dest_center_xy[1]),
                      plank_z=target_plank_top_z, bev_h=bev_upright_height, ref_R=bev_ref_R):
        def _checker():
            pos, rot = robot.get_object_pose(bid)
            pos = np.asarray(pos, dtype=float)
            if rot is None:
                return False
            in_xy = (abs(pos[0] - cx) <= 0.03) and (abs(pos[1] - cy) <= 0.03)
            in_z = (pos[2] >= plank_z - 0.02) and (pos[2] <= plank_z + bev_h + 0.04)
            upright = _upright_cos_vs_ref(rot, ref_R) >= PLACE_UPRIGHT_COS
            return bool(in_xy and in_z and upright)
        return _checker

    checker = _make_checker()

    grasp_config = GraspConfig(
        unmovable_objects=list(protect_ids),
        action_description={
            "zh": "拿起货架顶层的饮料",
            "en": "picks up the beverage from the open top layer of the shelf",
        },
        # Top layer is OPEN: no ceiling above. Prefer HORIZONTAL grasps on the
        # bottle's middle-lower body (see _filter_horizontal_midlow_grasps).
        # A level wrist stays Lula-IK-solvable across the vertical range from
        # the top plank down to the middle plank; a top-down grasp pitches
        # the wrist by 60+ deg and Lula cannot unroll it to a level upright
        # place. is_from_up_side=False lets side-approach candidates through.
        is_from_up_side=False,
        pick_up_distance=POST_GRASP_LIFT,
        pre_grasp_offset=PICK_PRE_GRASP_DISTANCE,
        grasp_pose_filter=_filter_horizontal_midlow_grasps,
    )

    # ---- Step 1 (pick): grasp the beverage on the OPEN top layer, then lift it
    #      POST_GRASP_LIFT to clear the plank. Step 2 pulls it horizontally out
    #      of the shelf at constant height. ----
    ok_pick = robot.pick_actor(actor=beverage_id, arm=arm, grasp_config=grasp_config)
    if not ok_pick:
        return False

    # ---- Step 2 (move): pull the beverage straight OUT toward the robot at
    #      CONSTANT height. Mirrors move_shelf_fruit_onto_plate_g2: horizontal pull-out
    #      to WAYPOINT_X in open space, z fixed at the lifted height. ----
    held_pos, held_quat = robot.get_object_pose(beverage_id)
    held_pose = np.eye(4)
    held_pose[:3, :3] = quat2mat_wxyz(held_quat)
    held_pose[:3, 3] = np.asarray(held_pos, dtype=float)

    held_z = float(held_pos[2])
    held_y = float(held_pos[1])

    pullout_poses = []
    for dy in np.linspace(-0.03, 0.03, 3):
        for yaw_deg in np.linspace(0.0, 360.0, 8, endpoint=False):
            rotated = rotate_along_axis(
                held_pose, float(yaw_deg), rot_axis="z", use_local=False,
            )
            pose = np.asarray(rotated, dtype=float).copy()
            pose[:3, 3] = [WAYPOINT_X, held_y + dy, held_z]
            pullout_poses.append(pose)
    pullout_poses = np.array(pullout_poses)

    pull_out_config = MoveConfig(
        unmovable_objects=list(protect_ids),
        action_description={
            "zh": "把饮料水平拉出货架",
            "en": "pulls the beverage horizontally out of the shelf at constant height",
        },
    )
    ok_pull = robot.move_actor_to_target_pose(
        actor=beverage_id,
        target_pose=pullout_poses,
        arm=arm,
        move_config=pull_out_config,
    )
    if not ok_pull:
        return False

    # ---- Step 3 (place): set the beverage DOWN onto the middle layer in ONE
    #      motion from the pull-out waypoint. The place stage's obstacle-aware
    #      planner descends from the pull-out height (at/above the top plank z)
    #      into the middle compartment -- the same cuRobo motion that works for
    #      the apple in move_shelf_fruit_onto_plate_g2. ----
    # ---- Step 3 (place, TWO-PHASE MOVES): the built-in place_actor_to_target_pose
    #      uses ONLY target_pose[0][:3, 3] as the fix_pose position and lets the
    #      planner freely pick the object orientation via R_align. On our bottle
    #      pool (multiple primitives, some named 'default') the planner routinely
    #      settles on a target that TILTS the bottle by 30+ deg -- the release-
    #      instant upright checker then fails and the whole stage returns fail.
    #
    #      Instead follow the proven `straighten_and_place_beverage_g2` recipe:
    #      call move_actor_to_target_pose directly with a batch of UPRIGHT bottle
    #      target poses (translated to the destination, swept over world-Z yaw).
    #      move_actor_to_target_pose honours the entire target_pose array and
    #      IK-filters the batch, so cuRobo settles on a candidate that keeps the
    #      bottle upright by construction. A descending ladder (dz = 0..0.08 m
    #      above the plank) lets the planner pick the LOWEST z that IK-passes,
    #      then we open the gripper to release. If every rung fails, we return
    #      fail rather than dropping the bottle from mid-air.
    def _Rz(deg):
        a = float(np.deg2rad(deg))
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # Rotation aligning the bottle's CURRENT local +Y to world +Z (upright).
    cur_pos_now, cur_quat_now = robot.get_object_pose(beverage_id)
    cur_R_now = quat2mat_wxyz(cur_quat_now)
    R_up_bottle = _align_R(cur_R_now[:, 1], [0.0, 0.0, 1.0]) @ cur_R_now

    dest_x_f = float(dest_center_xy[0])
    dest_y_f = float(dest_center_xy[1])
    z_floor = target_plank_top_z + PLACE_MARGIN + bev_upright_half_height

    def _upright_batch(cz):
        out = []
        for deg in range(0, 360, 15):
            P = np.eye(4)
            P[:3, :3] = _Rz(deg) @ R_up_bottle
            P[:3, 3] = [dest_x_f, dest_y_f, float(cz)]
            out.append(P)
        return np.array(out)

    # Descend ladder: prefer the LOWEST reachable release height. Starting a bit
    # above z_floor gives the planner slack under the top plank while still
    # setting the bottle down flush.
    placed = False
    for dz in (0.0, 0.02, 0.04, 0.06, 0.08):
        place_cfg = MoveConfig(
            unmovable_objects=list(protect_ids),
            action_description={
                "zh": "把饮料放到货架中层",
                "en": "places the beverage onto the middle layer of the shelf",
            },
        )
        if robot.move_actor_to_target_pose(
            beverage_id, _upright_batch(z_floor + dz), arm=arm, move_config=place_cfg,
        ):
            placed = True
            break
    if not placed:
        return False

    # Release: gripper opens where the arm ended (bottle is upright at z_floor+dz).
    robot.open_gripper(arm=arm)

    # ------------------------------------------------------------------
    # Step 4: retract arm, settle, then authoritative upright + position gate.
    #
    # The release-instant checker is lenient (PLACE_UPRIGHT_COS ~25.8 deg).
    # A narrow beverage can topple AFTER release as the arm withdraws. Retract
    # first, wait SETTLE_SECONDS, then re-verify the beverage is still upright
    # and still on the middle plank. Mirrors the shelf_beverage_middle_to_top
    # final gate recipe.
    # ------------------------------------------------------------------
    robot.back_to_origin(arm=arm)
    robot.wait(SETTLE_SECONDS)

    final_pos, final_quat = robot.get_object_pose(beverage_id)
    if _upright_cos_vs_ref(final_quat, bev_ref_R) < FINAL_UPRIGHT_COS:
        return False

    final_pos = np.asarray(final_pos, dtype=float)
    on_middle_plank = (
        final_pos[2] >= target_plank_top_z - 0.02
        and final_pos[2] <= target_plank_top_z + bev_upright_height + 0.06
        and abs(final_pos[0] - float(dest_center_xy[0])) <= 0.15
        and abs(final_pos[1] - float(dest_center_xy[1])) <= 0.15
    )
    if not on_middle_plank:
        return False

    # ------------------------------------------------------------------
    # Shelf integrity gate: a shelf knocked during back_to_origin would not be
    # caught by the framework's per-stage unmovable check. Sample the shelf
    # over a short settling window; fail if peak displacement / tilt exceed
    # the framework's own thresholds (1 cm / 5 deg).
    # ------------------------------------------------------------------
    def _quat_angle_deg(q_now, q_ref):
        q_now = np.asarray(q_now, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        dot = abs(float(np.dot(q_now / (np.linalg.norm(q_now) + 1e-9),
                               q_ref / (np.linalg.norm(q_ref) + 1e-9))))
        dot = min(1.0, max(-1.0, dot))
        return float(np.degrees(2.0 * np.arccos(dot)))

    peak_shift = 0.0
    peak_tilt = 0.0
    for _ in range(12):
        robot.wait(0.1)
        pos_now, quat_now = robot.get_object_pose(shelf_id)
        shift = float(np.linalg.norm(np.asarray(pos_now, dtype=float) - shelf_pos0))
        tilt = _quat_angle_deg(quat_now, shelf_quat0)
        peak_shift = max(peak_shift, shift)
        peak_tilt = max(peak_tilt, tilt)

    if peak_shift > 0.01 or peak_tilt > 5.0:
        print(
            f"[shelf-gate] shelf disturbed during task "
            f"(peak shift {peak_shift * 100:.1f} cm, peak tilt {peak_tilt:.1f} deg) -> task FAIL"
        )
        return False

    return True
