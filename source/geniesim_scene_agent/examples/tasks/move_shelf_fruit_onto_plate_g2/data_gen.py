# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import json


# Layer the plate (the PLACE target surface) sits on (layers.json index). For
# benchmark_supermarket_shelf_006 (5 planks, 0=lowest .. 4=highest) the plate
# sits on the MIDDLE plank (index 3) and the apple starts on the OPEN TOP plank
# (index 4): the apple is moved DOWN from the top layer onto the plate on the
# middle layer. Picking from the OPEN top layer needs no
# reaching under a plank (so the grasp does not graze the shelf), and placing
# DOWN onto the middle plate is a reachable motion (the gripper stays at/below
# the shoulder), so the arm never has to reach up/over the shelf to a pose at the
# edge of its range -- which is what toppled the shelf when the target was on the
# TOP layer. A plate has NO walls, so the descent onto it is cleaner than
# dropping into a box: the gripper never has to dip below a box rim under the top
# plank.
TARGET_LAYER_INDEX = 3

# layers.json index of the plank directly ABOVE the plate (the ceiling of the
# middle compartment). Used to cap the approach/place heights so the gripper
# never drives up into it.
CEILING_LAYER_INDEX = 4

# Clearance (meters) above the plate TOP at which the apple is RELEASED. A plate
# is a walled dish with a raised RIM (unlike a flat coaster), so releasing flush
# (~0.01) drove the gripper/fingers DOWN into the rim and PUSHED the plate away.
# Releasing ~0.03 above the rim keeps the gripper clear of it; the centred apple
# then drops the last cm into the plate well. Mirrors the proven
# place_fruit_into_bowl recipe (release at bowl_top + 0.03).
PLACE_MARGIN = 0.03

# Small post-grasp vertical lift (meters, world +z). After the grasp closes on
# the top-layer apple we lift it just enough to clear the plank surface before
# pulling it out toward the robot, so the gripper does not graze/drag along the
# shelf during the extraction. Kept SMALL on purpose: the top layer is already
# at the EDGE of the G2's reach, and the framework's DEFAULT 0.12 m lift drove
# the gripper to z~1.52 -> kinematically out of reach (the pick-up "Simple" IK
# gate rejected every candidate -> unreachable). 0.04 m is enough to lift the apple
# off the plank for a clean pull-out while staying inside the arm's reach.
POST_GRASP_LIFT = 0.04


def _find_shelf_layers_json(shelf_usd: str) -> str:
    """Resolve the on-disk path of the shelf asset's layers.json
    (<ASSETS_PATH>/interaction/<asset_id>/layers.json)."""
    path = f"{ASSETS_PATH}/interaction/{shelf_usd}/layers.json"
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"layers.json not found for {shelf_usd} at {path}. "
        f"Run generate-shelf-layers first."
    )


def _world_transform_from_shelf(robot: RobotAPI, shelf_id: str):
    """Return (origin_world[3], R_world_from_local[3x3]) for the shelf instance."""
    shelf_pos, shelf_quat_wxyz = robot.get_object_pose(shelf_id)
    R = quat2mat_wxyz(shelf_quat_wxyz)
    return np.asarray(shelf_pos, dtype=float), np.asarray(R, dtype=float)


def _layer_top_z_world(layer, shelf_origin_world, R_world_from_local):
    """World-z of the plank's top face."""
    center_world = shelf_origin_world + R_world_from_local @ np.asarray(layer["center_local"])
    return float(center_world[2])


def _build_target_poses_on_plate(
    plate_center_xy: np.ndarray,
    plate_size_world: np.ndarray,
    plate_top_z: float,
    ceiling_top_z: float,
    obj_size_world: np.ndarray,
    initial_pose_world: np.ndarray,
    n_xy: int = 3,
    n_z: int = 4,
    n_yaw: int = 12,
) -> np.ndarray:
    """Generate candidate poses that set the apple DOWN onto the CENTRE of the
    plate, using the proven coaster-placement recipe (place_fruit_on_coaster /
    gather_fruits_onto_center_coasters), which lands the object ACCURATELY on a
    small round target instead of drifting toward the rim / off the plate:
      * target xy PINNED at the plate centre, swept only over a TIGHT +-0.5 cm
        grid -- a WIDE grid (or a front-of-plate bias) lets the collision-aware
        IK satisfy reach by drifting the release off-centre and dropping the
        apple near/over the round plate's rim, i.e. beside the plate;
      * a small release z a CLEARANCE (PLACE_MARGIN, ~3 cm) ABOVE the plate top,
        NOT flush: a plate has a raised RIM (unlike a flat coaster), so releasing
        flush drove the gripper/fingers into the rim and PUSHED the plate away.
        Releasing a few cm above the rim keeps the gripper clear; the centred
        apple then drops the last cm into the plate well (mirrors the proven
        place_fruit_into_bowl: release at bowl_top + 0.03). A small upward band
        (capped under the plank above) gives the collision-aware IK freedom;
      * the apple's grasped world rotation swept about WORLD +Z (0..360 deg),
        which gives the IK many reachable wrist yaws ALL aimed at the plate
        centre WITHOUT moving the centred translation.

    n_z is accepted only for signature compatibility (unused); ceiling_top_z caps
    the release band so the hand stays under the plank above.

    Returns: np.ndarray (N, 4, 4).
    """
    plate_cx, plate_cy = float(plate_center_xy[0]), float(plate_center_xy[1])
    apple_half_z = float(obj_size_world[2]) / 2.0

    # Release the apple a CLEARANCE (PLACE_MARGIN) ABOVE the plate top so the
    # gripper/fingers clear the plate's raised RIM and do NOT push the plate.
    # The centred apple then drops the last cm into the plate well. A small
    # upward band (capped under the plank above) gives the IK vertical freedom.
    z_floor = plate_top_z + PLACE_MARGIN + apple_half_z
    z_ceiling_cap = ceiling_top_z - 0.20             # keep the hand under the plank
    z_high = min(z_floor + 0.02, z_ceiling_cap)
    if z_high < z_floor:
        z_high = z_floor
    z_grid = np.linspace(z_floor, z_high, 2)

    initial_pose_world = np.asarray(initial_pose_world, dtype=float).copy()
    if initial_pose_world.shape != (4, 4):
        initial_pose_world = np.eye(4)
    base_rot = initial_pose_world[:3, :3].copy()

    # TIGHT centred xy grid (+-1 cm, as in the proven place_fruit_into_bowl):
    # every candidate stays over the plate centre so the IK cannot satisfy reach
    # by sliding the apple toward the rim (which dropped it off the plate).
    offs = np.linspace(-0.01, 0.01, n_xy)
    yaw_deg_grid = np.linspace(0.0, 360.0, n_yaw, endpoint=False)

    poses = []
    for dx in offs:
        for dy in offs:
            for z in z_grid:
                for yaw_deg in yaw_deg_grid:
                    pose = np.eye(4)
                    pose[:3, :3] = base_rot
                    pose[:3, 3] = [plate_cx + float(dx), plate_cy + float(dy), float(z)]
                    # rotate_along_axis(use_local=False) spins ONLY the
                    # orientation about world +Z and keeps the translation fixed.
                    pose = rotate_along_axis(
                        pose, float(yaw_deg), rot_axis="z", use_local=False,
                    )
                    poses.append(np.asarray(pose, dtype=float))

    return np.array(poses)


def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    # ------------------------------------------------------------------
    # 1. Locate the shelf, the apple(s) to MOVE (top layer), and the
    #    plate on the middle layer (the target surface).
    # ------------------------------------------------------------------
    shelf_id = None
    shelf_usd = None
    apple_ids = []          # top-layer apples to move
    plate_id = None         # plate on the middle layer

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        # Plate first (it must NOT be picked up as a "fruit").
        if plate_id is None and (
            "plate" in keywords or "target_plate" in keywords
            or "benchmark_plate_" in usd
        ):
            plate_id = obj_id
            continue

        if shelf_id is None and (
            "target_shelf" in keywords or "shelf" in keywords
            or "benchmark_supermarket_shelf_" in usd
        ):
            shelf_id = obj_id
            shelf_usd = usd
            continue

        is_apple_by_kw = any(kw == "apple" for kw in keywords)
        is_apple_by_usd = ("benchmark_apple_" in usd) or ("benchmark_green_apple_" in usd)
        if is_apple_by_kw or is_apple_by_usd:
            apple_ids.append(obj_id)

    assert shelf_id is not None, "shelf not found in scene_info"
    assert shelf_usd, "shelf_usd empty, cannot resolve layers.json"
    assert plate_id is not None, "plate not found in scene_info"
    assert len(apple_ids) >= 1, "no top-layer apples found in scene_info"

    # ------------------------------------------------------------------
    # 2. Set task description BEFORE the first robot action.
    # ------------------------------------------------------------------
    description_json = {
        "task_name": 'Move a fruit from the top shelf onto a plate on the middle shelf',
        "english_task_name": (
            'Move a fruit from the top shelf onto a plate on the middle shelf'
        ),
        "init_scene_text": (
            "The robot is in front of a multi-layer shelf. The top layer holds "
            "an apple and a couple of oranges; the middle layer holds an empty "
            "round plate; the other layers are empty."
        ),
    }
    robot.client.set_task_task_basic_info(description_json)

    # ------------------------------------------------------------------
    # 3. Load layers.json, resolve shelf transform and the plate geometry.
    # ------------------------------------------------------------------
    layers_path = _find_shelf_layers_json(shelf_usd)
    with open(layers_path) as f:
        layers_data = json.load(f)
    target_layer = layers_data["layers"][TARGET_LAYER_INDEX]

    shelf_origin_world, R_world_from_local = _world_transform_from_shelf(robot, shelf_id)
    plate_plank_top_z = _layer_top_z_world(target_layer, shelf_origin_world, R_world_from_local)

    # Record the shelf's initial world POSE (position + orientation) for the
    # end-of-task shelf-integrity gate. Read it via get_object_pose(shelf_id) --
    # the SAME source the gate samples with -- so the baseline and the samples
    # share one coordinate convention (shelf_origin_world above comes from the
    # transform resolver and is not necessarily the same reference point).
    shelf_pos0_raw, shelf_quat0 = robot.get_object_pose(shelf_id)
    shelf_pos0 = np.asarray(shelf_pos0_raw, dtype=float).copy()
    shelf_quat0 = np.asarray(shelf_quat0, dtype=float).copy()

    # Ceiling: top face of the plank directly above the plate (the top-layer
    # plank). Cap the approach/place heights so the gripper never rises into it.
    ceiling_layer = layers_data["layers"][CEILING_LAYER_INDEX]
    ceiling_top_z = _layer_top_z_world(ceiling_layer, shelf_origin_world, R_world_from_local)

    # Plate geometry in world. get_object_pose -> the plate origin (~footprint
    # centre in xy); size -> world AABB. The plate rests flush on the MIDDLE
    # plank, so its top surface = middle_plank_top + plate_height (robust to the
    # origin convention).
    plate_pos, plate_quat0 = robot.get_object_pose(plate_id)
    plate_size = np.asarray(robot.get_object_size_in_world_frame(plate_id), dtype=float)
    plate_center_xy = np.array([float(plate_pos[0]), float(plate_pos[1])])
    plate_height = float(plate_size[2])
    plate_top_z = plate_plank_top_z + plate_height
    plate_half_x = float(plate_size[0]) / 2.0
    plate_half_y = float(plate_size[1]) / 2.0

    # ------------------------------------------------------------------
    # 4. Move each top-layer apple down onto the plate on the middle layer
    #    (left->right order).
    # ------------------------------------------------------------------
    def _apple_x(aid):
        pos, _ = robot.get_object_pose(aid)
        return float(pos[0])

    apple_ids_sorted = sorted(apple_ids, key=_apple_x)
    protect_ids = [shelf_id, plate_id]

    # Baselines for the end-of-task integrity gate (the FINAL gate after the loop).
    # Recorded BEFORE any manipulation so a plate knocked over LATER (during a place
    # settle, or during the reset that no stage-level check covers) is caught against
    # a clean baseline. `placed_apple_ids` collects the apples actually released so
    # the gate can re-verify each one is still ON the plate.
    plate_pos0 = np.asarray(plate_pos, dtype=float).copy()
    plate_quat0 = np.asarray(plate_quat0, dtype=float).copy()
    plate_radius = float(min(plate_half_x, plate_half_y))
    placed_apple_ids = []

    for apple_id in apple_ids_sorted:
        apple_size = np.asarray(robot.get_object_size_in_world_frame(apple_id), dtype=float)
        apple_diag = max(float(apple_size[0]), float(apple_size[1]))

        apple_pos_w, apple_quat_w = robot.get_object_pose(apple_id)
        apple_init_pose = np.eye(4)
        apple_init_pose[:3, :3] = quat2mat_wxyz(apple_quat_w)
        apple_init_pose[:3, 3] = np.asarray(apple_pos_w, dtype=float)

        target_poses = _build_target_poses_on_plate(
            plate_center_xy,
            plate_size,
            plate_top_z,
            ceiling_top_z,
            apple_size,
            initial_pose_world=apple_init_pose,
        )

        arm = robot.judge_arm_to_use(apple_pos_w)

        # Checker: the apple ends up over the ROUND plate disc (radial xy
        # containment -- NOT a rectangular box test, which would accept the
        # plate's bounding-box corners where there is no plate) and resting on it.
        def _make_checker(aid=apple_id, cx=float(plate_center_xy[0]), cy=float(plate_center_xy[1]),
                          plate_r=min(plate_half_x, plate_half_y),
                          plank_z=plate_plank_top_z, top_z=plate_top_z, adiag=apple_diag):
            def _checker():
                pos, _rot = robot.get_object_pose(aid)
                pos = np.asarray(pos, dtype=float)
                xy = float(np.hypot(pos[0] - cx, pos[1] - cy))
                in_xy = xy <= plate_r
                # z lower bound from the PLANK (a fruit settled in the plate well
                # sits below the rim top), upper bound a little above the rim.
                in_z = (pos[2] >= plank_z - 0.02) and (pos[2] <= top_z + adiag + 0.08)
                return bool(in_xy and in_z)
            return _checker

        checker = _make_checker()

        grasp_config = GraspConfig(
            unmovable_objects=list(protect_ids),
            action_description={
                "zh": "拿起货架顶层的苹果",
                "en": "picks up the apple from the top layer of the shelf",
            },
            # SMALL post-grasp lift. The TOP plank (apple z~1.40) is at the EDGE
            # of the G2's reach: the grasp itself IS reachable (the front bias
            # keeps it so), but the framework's DEFAULT +0.12 straight-up lift
            # drove the gripper to z~1.52 -> kinematically out of reach (the
            # pick-up IK gate, grasp.py solve_ik type="Simple" = PURE reach,
            # rejected EVERY candidate -> "Pick-up IK filter (single 0.12): 0
            # poses passed" -> unreachable). pick_up_distance=0.0 fixed the reach but
            # then the apple was pulled straight out along the plank, grazing the
            # shelf. A SMALL lift (POST_GRASP_LIFT, 0.04 m) splits the difference:
            # it raises the apple just clear of the plank surface for a clean
            # pull-out, while z~1.44 stays inside the arm's reach.
            pick_up_distance=POST_GRASP_LIFT,
        )

        # ---- Step 1 (pick): grasp the apple on the OPEN top layer, then lift it a
        #      small amount (POST_GRASP_LIFT) to clear the plank. Step 2 then pulls
        #      it HORIZONTALLY out of the shelf (constant z) so the apple leaves the
        #      shelf without grazing the top plank. The lift is kept small because
        #      the top layer is at the arm's reach edge; the default 0.12 m lift was
        #      out of reach. ----
        ok_pick = robot.pick_actor(actor=apple_id, arm=arm, grasp_config=grasp_config)
        if not ok_pick:
            return False

        # ---- Step 2 (move): pull the apple straight OUT toward the robot at
        #      CONSTANT height. The grasp only lifts the apple POST_GRASP_LIFT (a
        #      few cm) off the top plank -- just clear of the surface, not high --
        #      so the extraction MUST be horizontal: move out in world -x while
        #      holding z fixed at the lifted height, carrying the gripper clear of
        #      the shelf's front edge without grazing/dragging along the top plank.
        #      Step 3 then places onto the plate in a SINGLE motion:
        #      pick -> pull-out -> place, with no separate descend leg. ----
        held_pos, held_quat = robot.get_object_pose(apple_id)
        held_pose = np.eye(4)
        held_pose[:3, :3] = quat2mat_wxyz(held_quat)
        held_pose[:3, 3] = np.asarray(held_pos, dtype=float)

        WAYPOINT_X = 0.0  # world x between robot and shelf, clear of the planks
        held_z = float(held_pos[2])   # keep the apple at its lifted height (z fixed)
        held_y = float(held_pos[1])

        pullout_poses = []
        for dy in np.linspace(-0.03, 0.03, 3):
            for yaw_deg in np.linspace(0.0, 360.0, 8, endpoint=False):
                rotated = rotate_along_axis(
                    held_pose, float(yaw_deg), rot_axis="z", use_local=False,
                )
                pose = np.asarray(rotated, dtype=float).copy()
                # Pure horizontal pull-out: x -> WAYPOINT_X, z held constant.
                pose[:3, 3] = [WAYPOINT_X, held_y + dy, held_z]
                pullout_poses.append(pose)
        pullout_poses = np.array(pullout_poses)

        pull_out_config = MoveConfig(
            unmovable_objects=list(protect_ids),
            action_description={
                "zh": "把苹果水平拉出货架",
                "en": "pulls the apple horizontally out of the shelf at constant height",
            },
        )
        ok_pull = robot.move_actor_to_target_pose(
            actor=apple_id,
            target_pose=pullout_poses,
            arm=arm,
            move_config=pull_out_config,
        )
        if not ok_pull:
            return False

        # ---- Step 3 (place): set the apple ONTO the plate on the middle layer in a
        #      SINGLE AvoidObs motion straight from the pull-out waypoint. No
        #      separate descend leg: the place stage's own obstacle-aware planner
        #      routes the apple down onto
        #      the plate, and the tight centred target grid (built above) lands it
        #      accurately on the plate. ----
        place_config = PlaceConfig(
            unmovable_objects=list(protect_ids),
            action_description={
                "zh": "把苹果放到中层的盘子上",
                "en": "places the apple onto the plate on the middle layer",
            },
            extra_checkers=[checker],
            # Do NOT set use_near_point=True. On G2 the place.py near-point
            # fallback is buggy (find_near_point_grasp_pose refreshes the joint
            # data but NOT ik_link_poses -> the G2 human-like sorter crashes on an
            # empty link_poses) AND -- worse for this task -- it lets the release
            # drift off the plate centre toward/over the rim, i.e. off the plate.
            # An unreachable place should fail GRACEFULLY. Accuracy comes from the
            # tight centred target grid above, not a post-hoc offset search.
        )
        ok = robot.place_actor_to_target_pose(
            actor=apple_id,
            arm=arm,
            target_pose=target_poses,
            place_config=place_config,
        )
        if not ok:
            return False
        placed_apple_ids.append(apple_id)

        # After the place the hand sits INSIDE the middle compartment (above the
        # plate). Reset the arm to its start pose with the FRAMEWORK's own
        # back_to_origin, which plans ONE CONTINUOUS cuRobo whole-arm obstacle-
        # avoiding trajectory (move_pose type="AvoidObs", block=True) and falls back
        # to a Simple move only if avoidance fails.
        #
        # WHY not a hand-built FollowTrajectory here: the mid-shelf compartment is
        # only ~0.315 m tall (plate plank -> plank above, from layers.json). A
        # FollowTrajectory (server Command 72) only solves batch IK per END-EFFECTOR
        # waypoint -- it does NOT collision-check the forearm/wrist links. In a
        # compartment this shallow the forearm rises into the plank above no matter
        # how the EE waypoints are arranged, so every FollowTrajectory reset drove
        # the forearm into the top plank (server: reason=timeout, max_joint_err
        # ~0.29 rad) and knocked the shelf. Only cuRobo's whole-arm AvoidObs planner
        # (what place itself used to descend INTO the compartment) routes the whole
        # arm back out without grazing. back_to_origin is that planner, in one call.
        robot.back_to_origin(arm)

    # ------------------------------------------------------------------
    # Fallback gate: the framework's unmovable-object check runs ONLY inside each
    # stage (pick/move/place). The reset is not a stage, so a shelf knocked DURING
    # the reset is never checked by the framework -> the episode passes as SUCCESS.
    # Add an explicit shelf-integrity gate here.
    #
    # A grazed shelf WOBBLES and springs back: by the time the arm is home its net
    # displacement can be < 1 cm again, so a single end-of-task position check
    # MISSES it (this is exactly how a reset-collision run was wrongly passed). So
    # instead of one snapshot, sample the shelf over a short settling window right
    # after the reset and take the PEAK deviation -- both a position peak and an
    # orientation (tilt) peak, since a shove tips the shelf transiently even when it
    # rocks back to the same footprint. Either peak over threshold => the shelf was
    # struck => FAIL. Thresholds match the framework's unmovable check (1 cm / 5 deg).
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
    # Plate integrity, sampled over the SAME settling window: a plate knocked or
    # toppled AFTER the checker's release-instant snapshot (or during the reset)
    # drops off the middle plank -- and the apple goes with it -- yet the release-
    # only checker already passed. Track the worst z-drop / horizontal shift / tilt.
    plate_peak_zdrop = 0.0
    plate_peak_shift = 0.0
    plate_peak_tilt = 0.0
    for _ in range(12):                       # ~12 frames of settling
        robot.wait(0.1)
        pos_now, quat_now = robot.get_object_pose(shelf_id)
        shift = float(np.linalg.norm(np.asarray(pos_now, dtype=float) - shelf_pos0))
        tilt = _quat_angle_deg(quat_now, shelf_quat0)
        peak_shift = max(peak_shift, shift)
        peak_tilt = max(peak_tilt, tilt)

        ppos, pquat = robot.get_object_pose(plate_id)
        ppos = np.asarray(ppos, dtype=float)
        plate_peak_zdrop = max(plate_peak_zdrop, float(plate_pos0[2] - ppos[2]))
        plate_peak_shift = max(plate_peak_shift,
                               float(np.linalg.norm(ppos[:2] - plate_pos0[:2])))
        plate_peak_tilt = max(plate_peak_tilt, _quat_angle_deg(pquat, plate_quat0))

    if peak_shift > 0.01 or peak_tilt > 5.0:
        print(
            f"[shelf-gate] shelf disturbed during task "
            f"(peak shift {peak_shift * 100:.1f} cm, peak tilt {peak_tilt:.1f} deg) -> task FAIL"
        )
        return False

    # Plate must still rest on the middle plank: not dropped (z-fall), not shoved
    # off, not tipped over. A plate knocked to the floor falls ~1 m and rolls ~180
    # deg, so these thresholds are generous yet catch every real topple/drop.
    if plate_peak_zdrop > 0.05 or plate_peak_shift > 0.05 or plate_peak_tilt > 30.0:
        print(
            f"[plate-gate] plate displaced from the middle plank "
            f"(z-drop {plate_peak_zdrop * 100:.1f} cm, shift {plate_peak_shift * 100:.1f} cm, "
            f"tilt {plate_peak_tilt:.1f} deg) -> task FAIL"
        )
        return False

    # Every placed apple must still be ON the plate after settling. The release-
    # instant checker used the plate's ORIGINAL centre; re-verify against the plate's
    # CURRENT centre so an apple that rolled off -- or fell to the floor with the
    # plate -- is caught even if the plate itself passed the gate above.
    plate_now, _ = robot.get_object_pose(plate_id)
    plate_now = np.asarray(plate_now, dtype=float)
    for aid in placed_apple_ids:
        apos, _ = robot.get_object_pose(aid)
        apos = np.asarray(apos, dtype=float)
        radial = float(np.hypot(apos[0] - plate_now[0], apos[1] - plate_now[1]))
        # xy: inside the plate disc (+ a small apple-radius tolerance).
        # z : not fallen below the plate plank (a floored apple sits ~1 m lower).
        if radial > plate_radius + 0.04 or apos[2] < plate_plank_top_z - 0.05:
            print(
                f"[apple-gate] apple {aid} left the plate after settling "
                f"(radial {radial * 100:.1f} cm vs plate r {plate_radius * 100:.1f} cm, "
                f"apple z {apos[2]:.3f} vs plank {plate_plank_top_z:.3f}) -> task FAIL"
            )
            return False

    return True
