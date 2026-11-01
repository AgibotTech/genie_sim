# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
find_hidden_block_in_three_drawers_set_2_g2 — open drawers one by one until the hidden
building block is revealed.

Demonstrates the *custom grasp pose + articulation part grasp* API extension
(see data-collection/docs/custom_grasp_pose_and_part_grasp.md):

  - `pick_actor(..., part="part_01", grasp_config=GraspConfig(grasp_pose=...))`
    grasps a specific articulation sub-part using caller-supplied candidates.
  - The candidate poses are world-frame robot_gripper poses — the API does NOT
    apply the legacy `grasp_gripper -> robot_gripper` rotation to them, so we
    encode the final desired EE orientation directly.
"""
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# USD prefixes used to identify scene objects from `scene_info["layout"]`.
BLOCK_USD_PREFIXES = (
    "benchmark_building_blocks_073",
    "benchmark_building_blocks_085",
    "benchmark_building_blocks_089",
)
BOX_USD_PREFIX = "benchmark_storage_box_019"
TABLE_USD_PREFIX = "table_000"

# Tunables.
#   PULL_DISTANCE                  — how far the gripper retreats along its
#                                    own local +Z (i.e. the approach axis,
#                                    pointing into the handle) when pulling.
#   DRAWER_OPENED_MIN_DISPLACEMENT — `extra_checkers` threshold; pull is
#                                    accepted once the drawer body (part_00)
#                                    has moved this far along pull_dir.
#   BLOCK_FOUND_DISPLACEMENT       — after a pull, the block is considered
#                                    "found in this drawer" if its xy
#                                    displacement projected onto pull_dir
#                                    exceeds this threshold.
PULL_DISTANCE = 0.10
DRAWER_OPENED_MIN_DISPLACEMENT = 0.05
BLOCK_FOUND_DISPLACEMENT = 0.04


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_storage_box_019": {
            "has_interactions": False,
            "object_type": "drawer_box",
        },
        "benchmark_building_blocks_073": {
            "has_interactions": True,
            "object_type": "building_block",
        },
        "benchmark_building_blocks_085": {
            "has_interactions": True,
            "object_type": "building_block",
        },
        "benchmark_building_blocks_089": {
            "has_interactions": True,
            "object_type": "building_block",
        },
        "table_000": {
            "has_interactions": False,
            "object_type": "table",
        },
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    # Bilingual task description shown in the data-collection UI / metadata.
    description_json = {
        "task_name": "pull open the three drawers on the table one by one until the hidden building block is found",
        "english_task_name": "pull open the three drawers on the table one by one until the hidden building block is found",
        "init_scene_text": "The robot is in front of a table with three side-by-side drawer boxes; a building block is hidden inside one of them.",
    }
    robot.client.set_task_task_basic_info(description_json)

    # ------------------------------------------------------------------
    # 1. Locate objects in scene_info.
    # ------------------------------------------------------------------
    table_id = None
    block_id = None
    box_entries = []

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if TABLE_USD_PREFIX in usd:
            table_id = obj_id
        elif BOX_USD_PREFIX in usd:
            xyz = obj_info.get("xyz", [0.0, 0.0, 0.0])
            box_entries.append((obj_id, float(xyz[1])))
        elif any(p in usd for p in BLOCK_USD_PREFIXES):
            block_id = obj_id

    if table_id is None or block_id is None or len(box_entries) != 3:
        print(
            f"Scene parsing failed: table={table_id}, block={block_id}, "
            f"boxes={[b[0] for b in box_entries]}"
        )
        return False

    # Inspect from left (+Y) to right (-Y).
    box_entries.sort(key=lambda kv: -kv[1])
    box_ids = [b[0] for b in box_entries]
    print(f"Table: {table_id}")
    print(f"Drawers in inspection order (left → right): {box_ids}")
    print(f"Hidden block id: {block_id}")

    initial_block_pos, _ = robot.get_object_pose(block_id)
    initial_block_pos = np.array(initial_block_pos, dtype=float)
    print(f"Initial block xyz: {initial_block_pos}")

    # ------------------------------------------------------------------
    # 2. Helper: build a candidate set of grasp poses for a handle.
    # ------------------------------------------------------------------
    # The drawer asset (`benchmark_storage_box_019`) ships no USD-annotated
    # `passive.grasp` for `part_01`, so we must supply our own candidates via
    # GraspConfig.grasp_pose (custom-grasp-pose API).
    #
    # Strategy: define a small set of robot_gripper orientations valid for ONE
    # canonical drawer placement (handle at world (0,0,0) and the storage_box
    # asset yawed 180° around world Z), then transport those orientations to
    # each runtime drawer using its actual handle world quat.
    #
    # Two world-frame reference quats verified by hand for the canonical pose
    # — they pre-position the gripper to grab the handle from above with the
    # robot_gripper Z axis approximately aligned with -X_world (i.e. into the
    # handle). Stored as wxyz.
    _REF_WORLD_QUATS_WXYZ = [
        np.array([0.707, 0.0, 0.707, 0.0]),
        np.array([0.0, 0.707, 0.0, 0.707]),
    ]
    # Canonical world quat of `part_01` at the canonical drawer pose. Probed
    # offline (see scripts/visualize_grasp_poses.py for the probe code) — the
    # asset's internal `entity.xformOp:orient = (0.707, 0.707, 0, 0)` (90°
    # around X) makes this NOT the identity, so the conversion below is
    # necessary even at canonical pose.
    _CANONICAL_HANDLE_QUAT_WXYZ = np.array(
        [-0.49999773, 0.49999773, 0.50000227, 0.50000227]
    )
    # Pre-compute reference quats expressed in the HANDLE LOCAL frame:
    #   q_local = q_canonical_handle⁻¹ ⊗ q_world_ref
    # At runtime we just rotate by the actual handle's world quat to get the
    # final world-frame grasp orientation, regardless of how the drawer is
    # placed/yawed in the scene.
    _q_handle_inv = quat_inverse(_CANONICAL_HANDLE_QUAT_WXYZ)
    _LOCAL_REF_QUATS = [
        quat_multiplication_wxyz(_q_handle_inv, q) for q in _REF_WORLD_QUATS_WXYZ
    ]
    # The reference rotations point gripper local +Z AWAY from the handle.
    # Flip local Z so the approach axis points INTO the handle. Equivalent to
    # a 180° rotation around local Y: X→-X, Y→+Y, Z→-Z (right-handed). The
    # flip was verified visually with scripts/visualize_grasp_poses.py.
    _FLIP_Z = np.diag([-1.0, 1.0, -1.0])

    def _rot_axis(axis, deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "z":
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        raise ValueError(axis)

    def build_handle_grasp_poses(handle_pos, handle_quat_wxyz):
        """Generate (N,4,4) candidate grasp poses for a drawer handle.

        Position sampling: in the handle's LOCAL frame, x=0 (on the handle
        face), y∈[0.01,0.02] m (small standoff towards the gripper), and
        z∈[-0.02,0.02] m (small spread along the handle bar).

        Orientation sampling: each precomputed handle-local reference rotation
        (with local-Z flip applied so +Z points into the handle), perturbed
        by ±30° around handle-local Z (roll) and ±30° around handle-local X
        (pitch). The roll/pitch tolerance gives downstream IK enough
        candidates to find a feasible solution.

        Returned poses are world-frame robot_gripper poses, ready to be passed
        directly into `GraspConfig.grasp_pose` (with default
        `grasp_pose_frame="world"`).
        """
        handle_pos = np.asarray(handle_pos, dtype=float)
        R_handle = quat2mat_wxyz(np.asarray(handle_quat_wxyz, dtype=float))

        ys = np.linspace(0.01, 0.02, 3)
        zs = np.linspace(-0.02, 0.02, 5)
        rolls_z =  np.linspace(-30, 30, 5)
        pitches_x =  np.linspace(-30, 30, 5)

        poses = []
        for q_local in _LOCAL_REF_QUATS:
            R_local_base = quat2mat_wxyz(q_local) @ _FLIP_Z
            for ly in ys:
                for lz in zs:
                    local_pos = np.array([0.0, float(ly), float(lz)])
                    # local → world: rotate by handle frame, translate to handle.
                    world_pos = handle_pos + R_handle @ local_pos
                    for rz in rolls_z:
                        for px in pitches_x:
                            R_local = R_local_base @ _rot_axis("z", rz) @ _rot_axis("x", px)
                            world_R = R_handle @ R_local
                            T = np.eye(4)
                            T[:3, :3] = world_R
                            T[:3, 3] = world_pos
                            poses.append(T)
        return np.array(poses)

    # ------------------------------------------------------------------
    # 3. Main loop: inspect each drawer.
    # ------------------------------------------------------------------
    found_box_id = None

    # Drawers are sorted left → right (descending world-Y), so index maps to
    # human-readable position labels for action descriptions.
    POSITION_LABELS = [
        {"zh": "左边", "en": "left"},
        {"zh": "中间", "en": "middle"},
        {"zh": "右边", "en": "right"},
    ]

    for idx, box_id in enumerate(box_ids):
        other_boxes = [b for b in box_ids if b != box_id]
        pos_label = POSITION_LABELS[idx]
        print("\n" + "=" * 60)
        print(f"[{idx+1}/{len(box_ids)}] Trying drawer {box_id} ({pos_label['en']})")
        print("=" * 60)

        # 3a. Locate handle (part_01) and drawer body (part_00). The pull
        #     direction follows the handle→body offset projected onto xy
        #     (drawer slides horizontally), inverted: i.e. the gripper
        #     retreats AWAY from the body.
        handle_pos, handle_quat = robot.get_object_pose(
            box_id, articulation_part="part_01"
        )
        body_pos, _ = robot.get_object_pose(box_id)
        handle_pos = np.array(handle_pos, dtype=float)
        body_pos = np.array(body_pos, dtype=float)

        delta = handle_pos - body_pos
        delta[2] = 0.0
        if np.linalg.norm(delta) < 1e-6:
            print(
                f"❌ Cannot infer pull direction (handle ≈ body) for {box_id}; "
                "aborting"
            )
            return False
        pull_dir = delta / np.linalg.norm(delta)
        print(f"Handle world xyz: {handle_pos}")
        print(f"Pull direction (world): {pull_dir}")

        # Pick which arm to use based on which side of the robot the handle is.
        grasp_arm = robot.judge_arm_to_use(handle_pos)
        print(f"Using arm: {grasp_arm}")

        # 3b. Record initial part_00 position so we can verify drawer opened.
        part00_init_pos, _ = robot.get_object_pose(
            box_id, articulation_part="part_00"
        )
        part00_init_pos = np.array(part00_init_pos, dtype=float)

        # 3c. Move gripper to handle (auto close fingers on arrival).
        # Use the new pick_actor(part="part_01", grasp_config=GraspConfig(
        # grasp_pose=...)) API: caller-supplied world-frame candidates that
        # bypass the missing USD grasp annotations on the handle part.
        grasp_poses = build_handle_grasp_poses(handle_pos, handle_quat)
        grasp_config = GraspConfig(
            action_description={
                "zh": f"抓握{pos_label['zh']}抽屉的把手",
                "en": f"grasps the handle of the {pos_label['en']} drawer",
            },
            pick_up_distance=0.0,  # do NOT lift after grasp — just hold the handle.
            grasp_pose=grasp_poses,
            unmovable_objects=[table_id] + other_boxes,
        )
        if not robot.pick_actor(
            box_id,  grasp_arm, grasp_config, part="part_01"
        ):
            print(f"❌ Could not reach the handle of {box_id}; aborting")
            return False

        # 3d. Pull the drawer along its prismatic axis.
        # The `extra_checkers` callable is queried each step by the planner;
        # it returns True once part_00 has displaced more than
        # DRAWER_OPENED_MIN_DISPLACEMENT m along pull_dir, terminating the
        # motion early once the drawer is visibly open.
        def make_drawer_opened_checker(box_id_local, init_xy, dir_xy):
            def _checker():
                cur, _ = robot.get_object_pose(
                    box_id_local, articulation_part="part_00"
                )
                disp = np.array(cur[:2]) - init_xy
                projected = float(np.dot(disp, dir_xy))
                print(
                    f"  drawer part_00 displacement along pull: {projected:.4f}m "
                    f"(need > {DRAWER_OPENED_MIN_DISPLACEMENT:.3f})"
                )
                return projected > DRAWER_OPENED_MIN_DISPLACEMENT

            return _checker


        # Build pull target from the gripper's ACTUAL current pose (after the
        # approach + close finished), translating it along its own local Z by
        # -PULL_DISTANCE so the gripper retreats along its approach axis and
        # drags the drawer with it. Using current_ee_pose (not the planned
        # grasp pose) handles the small residual offset from IK / collision
        # avoidance that may have shifted the gripper off the nominal target.
        current_ee_pose = robot.get_ee_pose(arm=grasp_arm)
        pull_target = current_ee_pose.copy()
        local_z_axis = pull_target[:3, 2]
        pull_target[:3, 3] -= local_z_axis * PULL_DISTANCE
        pull_targets = np.array([pull_target])

        pull_cfg = MoveConfig(
            action_description={
                "zh": f"拉开{pos_label['zh']}抽屉",
                "en": f"pulls open the {pos_label['en']} drawer",
            },
            extra_params={
                # Plan a constrained motion from current pose. The `goal_offset`
                # value here is unused by the planner because we pass an
                # explicit target_pose, but `from_current_pose` plus the
                # path_constraint mask below keeps motion to a single linear
                # axis matching the prismatic joint.
                "from_current_pose": True,
                "goal_offset": [-0.1, 0, 0, 1, 0, 0, 0],  # Move -0.1m in X direction
                "path_constraint": [0.1, 0.1, 0.1, 0.0, 0.1, 0.1],  # Only allow motion along X axis
                "offset_and_constraint_in_goal_frame": False,
            },
            unmovable_objects=[table_id] + other_boxes,
            extra_checkers=[
                make_drawer_opened_checker(
                    box_id, part00_init_pos[:2].copy(), pull_dir[:2].copy()
                )
            ],
        )
        # Free the drawer body and handle from obstacle list before pulling —
        # otherwise the planner refuses to drag the gripper through them.
        robot.remove_objs_from_obstacle(
            [
                robot.get_object_prim_path(box_id, "part_00"),
                robot.get_object_prim_path(box_id, "part_01"),
            ]
        )
        if not robot.move_actor_to_target_pose(
            "gripper", pull_targets, grasp_arm, pull_cfg
        ):
            print(f"❌ Failed to pull drawer {box_id} open; aborting")
            return False

        # 3e. Wait for physics to settle, then check whether the block came
        #     out together with this drawer. If it did, the block's xy will
        #     have moved together with part_00 along pull_dir.
        robot.wait(1.0)

        cur_block_pos, _ = robot.get_object_pose(block_id)
        block_displacement_xy = (
            np.array(cur_block_pos[:2], dtype=float) - initial_block_pos[:2]
        )
        block_along_pull = float(np.dot(block_displacement_xy, pull_dir[:2]))
        print(
            f"Block displacement along pull_dir: {block_along_pull:.4f}m "
            f"(need > {BLOCK_FOUND_DISPLACEMENT:.3f})"
        )

        if block_along_pull > BLOCK_FOUND_DISPLACEMENT:
            found_box_id = box_id
            print(f"✅ Block was hiding in {box_id}; leaving the drawer open")
            break

        # 3f. Wrong drawer — close it back, release gripper, return arm.
        # Push the gripper back along its own local +Z by a bit more than
        # PULL_DISTANCE so the drawer fully snaps shut.
        close_targets = grasp_poses.copy()
        for i in range(len(close_targets)):
            close_targets[i, :3, 3] += local_z_axis * 0.005  # small over-push
        close_cfg = MoveConfig(
            action_description={
                "zh": f"合上{pos_label['zh']}抽屉",
                "en": f"closes the {pos_label['en']} drawer",
            },
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0.101, 0, 0, 1, 0, 0, 0],  # Move +0.101m in X direction (close)
            "path_constraint": [0.1, 0.1, 0.1, 0.0, 0.1, 0.1],  # Only allow motion along X axis
            "offset_and_constraint_in_goal_frame": False,
        },
            unmovable_objects=[table_id] + other_boxes,
        )
        if not robot.move_actor_to_target_pose(
            "gripper", close_targets, grasp_arm, close_cfg
        ):
            print(f"❌ Failed to close drawer {box_id}; aborting")
            return False

        robot.open_gripper(arm=grasp_arm)
        robot.back_to_origin(arm=grasp_arm)
        robot.wait(0.3)

    if found_box_id is None:
        print("❌ Pulled every drawer but the block never moved with one of them")
        return False

    print("\n" + "=" * 60)
    print(f"TASK SUCCESS — block located in drawer {found_box_id}")
    print("=" * 60)
    return success
