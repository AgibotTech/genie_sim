# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_button_000": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_012": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_013": {"has_interactions": True, "object_type": "button"},
        "benchmark_building_blocks_073": {"has_interactions": False, "object_type": "building_block"},
        "benchmark_building_blocks_085": {"has_interactions": False, "object_type": "building_block"},
        "benchmark_building_blocks_089": {"has_interactions": False, "object_type": "building_block"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": 'Observe a block sequence and press buttons in the same order',
        "english_task_name": 'Observe a block sequence and press buttons in the same order',
        "init_scene_text": "The robot is in front of a table. Three same-size buttons with different colors are arranged left/middle/right at the table center. A single building block is initially placed in front of one of the three buttons. The block will be shown in front of each button one at a time; the robot must remember the order and then press the buttons in the same order.",
    }
    robot.client.set_task_task_basic_info(description_json)

    # ------------------------------------------------------------
    # 1. Dynamic lookup of the three buttons (by position keyword) and the block.
    # ------------------------------------------------------------
    POSITIONS = ["left", "middle", "right"]
    button_id_by_pos = {}     # {"left": <id>, "middle": <id>, "right": <id>}
    block_id = None
    initial_block_position = None  # one of "left"/"middle"/"right"

    for oid, info in scene_info["layout"].items():
        kws = info.get("keywords", [])
        usd = info.get("usd", "")

        # Hint block: tagged in scene generator with "hint_block".
        if "hint_block" in kws:
            block_id = oid
            for kw in kws:
                if kw.startswith("in_front_of_") and kw.endswith("_button"):
                    initial_block_position = kw[len("in_front_of_"):-len("_button")]
                    break
            continue

        # Buttons: each tagged with its position via "<pos>_button".
        if "button" in kws or "benchmark_button" in usd:
            for pos in POSITIONS:
                if f"{pos}_button" in kws:
                    button_id_by_pos[pos] = oid
                    break

    if block_id is None or initial_block_position is None:
        print(f"Block lookup failed: block_id={block_id}, init_pos={initial_block_position}")
        return False
    if set(button_id_by_pos.keys()) != set(POSITIONS):
        print(f"Button lookup incomplete: {button_id_by_pos}")
        return False

    print(f"Initial block position: in front of {initial_block_position} button")
    print(f"Button mapping: {button_id_by_pos}")
    print(f"Block id: {block_id}")

    # ------------------------------------------------------------
    # 2. Build visit order: start where the block already is, then the
    #    other two positions in random order.
    # ------------------------------------------------------------
    remaining = [p for p in POSITIONS if p != initial_block_position]
    random.shuffle(remaining)
    visit_order = [initial_block_position] + remaining
    print(f"Visit order: {visit_order}")

    # ------------------------------------------------------------
    # 3. Resolve table top z (used to keep block sitting on the table).
    # ------------------------------------------------------------
    table_id = None
    for oid, info in scene_info["layout"].items():
        if info.get("usd", "") == "table_000" or "table" in info.get("keywords", []):
            table_id = oid
            break
    if table_id is None:
        print("Table not found")
        return False
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + 0.5 * table_size[2]

    block_size = robot.get_object_size_in_world_frame(block_id)
    block_half_h = 0.5 * block_size[2]
    block_z = table_top_z + block_half_h + 0.005  # tiny clearance, then settles

    block_prim_path = robot.get_object_prim_path(block_id)

    # X offset of the block w.r.t. its button. The block now sits on the +X
    # side of the button (further from the robot, away from the chest).
    # Button x range -0.05..-0.03, block x range 0.18..0.22 ⇒ offset 0.21..0.27.
    # 0.24 m is a comfortable single value that keeps the gap well above the
    # 0.02 m minimum used at scene generation.
    BLOCK_X_OFFSET_IN_FRONT = 0.24

    all_button_ids = list(button_id_by_pos.values())

    # Per-position dwell time for the block teleport events. delay is applied
    # BEFORE the property is re-written, so a positive value keeps the block
    # visible at its CURRENT position for that many seconds before jumping
    # to the next observation pose.
    BLOCK_DWELL_SEC = 2.5

    # ------------------------------------------------------------
    # 4. Sequentially show the block in front of each button in visit order.
    #    The teleport itself is not a robot action — it is a stimulus the
    #    robot OBSERVES. We frame every step (including the initial one)
    #    as "observe the block at position X" so the policy learns to track
    #    the order via vision before deciding which button to press.
    #
    #    set_object_parameter_action records a frame-action with the given
    #    description; for the initial position we re-apply the current pose
    #    so the observation step is recorded consistently with the others.
    # ------------------------------------------------------------
    POS_ZH = {"left": "左侧", "middle": "中间", "right": "右侧"}

    for visit_idx, pos in enumerate(visit_order):
        btn_id = button_id_by_pos[pos]
        btn_pos, _ = robot.get_object_pose(btn_id)
        target_xyz = [
            float(btn_pos[0] + BLOCK_X_OFFSET_IN_FRONT),
            float(btn_pos[1]),
            float(block_z),
        ]
        if visit_idx == 0:
            cur_block_pos, _ = robot.get_object_pose(block_id)
            target_xyz = [float(cur_block_pos[0]),
                          float(cur_block_pos[1]),
                          float(cur_block_pos[2])]
        ok = robot.set_object_parameter_action(
            block_prim_path,
            "xformOp:translate",
            target_xyz,
            delay=BLOCK_DWELL_SEC,
            description={
                "zh": f"观察积木出现在哪个按钮前",
                "en": f"observes which button the building block is in front of",
            },
        )
        if not ok:
            print(f"Failed to show block at {pos}")
            return False

    # ------------------------------------------------------------
    # 5. Hide the block (move it far away) so it does not interfere with
    #    pressing. This is also an observation event from the robot's
    #    perspective — "the cue disappears".
    # ------------------------------------------------------------
    far_xyz = [20.0, 20.0, float(block_z)]
    ok = robot.set_object_parameter_action(
        block_prim_path,
        "xformOp:translate",
        far_xyz,
        delay=BLOCK_DWELL_SEC,
        description={
            "zh": "观察积木被移走，准备按按钮",
            "en": "observes the building block being removed, ready to press buttons",
        },
    )
    if not ok:
        print("Failed to hide block")
        return False

    # ------------------------------------------------------------
    # After all block-observation events finish, close both grippers and
    # capture each arm's initial ee pose. We do this AFTER observation so
    # the gripper-close events are not interleaved with the cue events,
    # and so the captured pose reflects the actual idle pose right before
    # the press phase begins.
    # ------------------------------------------------------------
    robot.close_gripper(arm="left")
    robot.close_gripper(arm="right")
    initial_ee_pose = {
        "left": robot.get_ee_pose(arm="left"),
        "right": robot.get_ee_pose(arm="right"),
    }

    # ------------------------------------------------------------
    # 6. Press the three buttons in the same order as the visit order.
    #    After each press, return the used arm to its captured initial ee
    #    pose using move_actor_to_target_pose. We:
    #      - keep the gripper closed (do NOT open),
    #      - first lift straight up so the fingertip leaves the button,
    #        then move to the initial ee pose,
    #      - mark every button as unmovable_objects so the planner avoids
    #        touching any of them during the return motion.
    # ------------------------------------------------------------
    for press_idx, pos in enumerate(visit_order):
        used_arm = _press_button(robot, button_id_by_pos[pos], pos)
        if used_arm is None:
            return False

        # Lift the gripper straight up first to clear the just-pressed button.
        cur_ee = robot.get_ee_pose(arm=used_arm)
        lift_pose = cur_ee.copy()
        lift_pose[2, 3] += 0.10
        arm_zh = "左臂" if used_arm == "left" else "右臂"
        lift_cfg = MoveConfig(
            unmovable_objects=all_button_ids,
            action_description={
                "zh": f"抬起",
                "en": f"lifts up",
            },
        )
        if not robot.move_actor_to_target_pose(
            "gripper", np.array([lift_pose]), arm=used_arm, move_config=lift_cfg
        ):
            print(f"Lift gripper after pressing {pos} failed")
            return False

        # Move the used arm back to its captured initial ee pose.
        home_cfg = MoveConfig(
            unmovable_objects=all_button_ids,
            action_description={
                "zh": f"归位",
                "en": f"returns to initial pose",
            },
        )
        if not robot.move_actor_to_target_pose(
            "gripper",
            np.array([initial_ee_pose[used_arm]]),
            arm=used_arm,
            move_config=home_cfg,
        ):
            print(f"Return {used_arm} arm to initial pose failed")
            return False

    print("\n=== Task summary ===")
    print("SUCCESS: observed block sequence and pressed buttons in the same order")
    return success


def _press_button(robot: RobotAPI, button_id: str, position_tag: str):
    """Press a single button using the standard press_button_g2 pattern.

    Returns the arm string ("left"/"right") that was used on success, or None on failure.
    """
    button_pos, _ = robot.get_object_pose(button_id)
    button_origin_pos = button_pos.copy()
    part_pos_before, _ = robot.get_object_pose(button_id, articulation_part="part_00")
    button_size = robot.get_object_size_in_world_frame(button_id)
    if button_pos is None or button_size is None:
        print(f"Cannot get button pose/size for {button_id}")
        return None

    arm = robot.judge_arm_to_use(button_pos)

    button_top_z = button_pos[2] + 0.5 * button_size[2]
    target_height = button_top_z + 0.05

    target_poses = []
    down_z_rotation = euler2mat([np.pi, 0, 0])
    for dx in np.linspace(-0.001, 0.001, 3):
        for dy in np.linspace(-0.001, 0.001, 3):
            position = np.array([button_pos[0] + dx, button_pos[1] + dy, target_height])
            target_pose = np.eye(4)
            target_pose[:3, 3] = position
            target_pose[:3, :3] = down_z_rotation
            for angle in np.linspace(180, 270, 12, endpoint=False):
                rotated_target_pose = rotate_along_axis(target_pose, angle, rot_axis="z", use_local=False)
                for angle_x in np.random.uniform(-10, 10, 10):
                    rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_x, rot_axis="x", use_local=False)
                    for angle_y in np.random.uniform(-10, 10, 10):
                        rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_y, rot_axis="y", use_local=False)
                target_poses.append(rotated_target_pose)
    target_poses = np.array(target_poses)

    robot.close_gripper(arm=arm)

    approach_cfg = MoveConfig(
        action_description={
            "zh": f"将夹爪移动到{ {'left':'左侧','middle':'中间','right':'右侧'}[position_tag] }按钮上方",
            "en": f"moves the gripper above the {position_tag} button",
        },
    )
    move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=arm,
        move_config=approach_cfg,
    )
    if not move_success:
        print(f"Move gripper above {position_tag} button failed")
        return None

    def check_button_pressed():
        cur_pos, _ = robot.get_object_pose(button_id)
        if cur_pos is None:
            return False
        if np.linalg.norm(cur_pos - button_origin_pos) > 0.01:
            print(f"{position_tag} button base moved")
            return False
        part_pos_after, _ = robot.get_object_pose(button_id, articulation_part="part_00")
        if part_pos_after[2] > part_pos_before[2] - 0.001:
            print(f"{position_tag} button not pressed: after={part_pos_after[2]} before={part_pos_before[2]}")
            return False
        return True

    press_cfg = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.05, 1, 0, 0, 0],
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_button_pressed],
        action_description={
            "zh": f"按压{ {'left':'左侧','middle':'中间','right':'右侧'}[position_tag] }按钮",
            "en": f"presses the {position_tag} button",
        },
        unmovable_objects=[button_id],
    )
    robot.client.remove_objs_from_obstacle([robot.get_object_prim_path(button_id)])
    press_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=arm,
        move_config=press_cfg,
    )
    if not press_success:
        print(f"Press {position_tag} button failed")
        return None

    return arm
