# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    'Pick up the juice bottle from the side and shake it.\n    The bottle is benchmark_beverage_bottle_008 or genie_beverage_bottle_009.'

    # Initialize objects_list from the provided scene generator.
    # Two juice-bottle asset IDs are possible; check interaction data for both.
    objects_list = {
        "benchmark_beverage_bottle_008": {
            "has_interactions": True,  # Assume interaction data exists.
            "object_type": "beverage_bottle",
            "description": "果汁瓶"
        },
        "genie_beverage_bottle_009": {
            "has_interactions": True,  # Assume interaction data exists.
            "object_type": "beverage_bottle",
            "description": "果汁瓶"
        },
        "table_000": {
            "has_interactions": False,  # Tables usually have no interaction data.
            "object_type": "table"
        }
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    # Task description: pick up the juice bottle on the table.
    description_json = {
        "task_name": "pick up the juice bottle on the table and shake it",
        "english_task_name": "pick up the juice bottle on the table and shake it",
        "init_scene_text": "The robot is in front of the table, there is a juice bottle placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table (table_000).
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords or "workspace" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find the juice bottle.
    juice_bottle_id = None
    juice_bottle_usd = None
    juice_bottle_info = None

    # Find the juice bottle by either of its two possible usd asset IDs.
    bottle_usds = ["benchmark_beverage_bottle_008", "genie_beverage_bottle_009"]
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether the object is a juice bottle.
        is_bottle = False
        for bottle_usd in bottle_usds:
            if bottle_usd in usd:
                is_bottle = True
                juice_bottle_usd = bottle_usd
                break

        if is_bottle or any("juice_bottle" in kw.lower() for kw in keywords) or any("beverage" in kw.lower() for kw in keywords):
            juice_bottle_id = obj_id
            juice_bottle_info = obj_info
            print(f"Found juice bottle: {obj_id}, usd: {usd}, keywords: {keywords}")
            break

    if not juice_bottle_id:
        print("Juice bottle not found")
        return False

    print(f"Using juice bottle: {juice_bottle_id}")

    # 3. Read the juice bottle's position and size.
    juice_bottle_position, juice_bottle_rotation = robot.get_object_pose(juice_bottle_id)
    if juice_bottle_position is None:
        print("Cannot get juice bottle position")
        return False

    # Read bottle dimensions for grasp-position calculation.
    juice_bottle_size = robot.get_object_size_in_world_frame(juice_bottle_id)
    if juice_bottle_size is None:
        print("Cannot get juice bottle size")
        return False

    print(f"Juice bottle position: {juice_bottle_position}")
    print(f"Juice bottle size: {juice_bottle_size}")

    # 4. Choose the working arm.
    arm = robot.judge_arm_to_use(juice_bottle_position)
    print(f"Using arm: {arm}")

    # 5. Inspect interaction data for side-grasp points.
    # The bottle is a container and may have no interaction data or only specific grasp points.
    # If interaction.json was not supplied, inspect the runtime interaction data.
    bottle_interaction = interactions.get(juice_bottle_usd, {})
    passive_grasp = bottle_interaction.get("passive", {}).get("grasp", {})

    print(f"Bottle has passive.grasp primitives: {bool(passive_grasp)}")
    if passive_grasp:
        print(f"Available grasp primitives: {list(passive_grasp.keys())}")

    # 6. Prepare the grasp configuration.
    # Use default grasp points for the requested side grasp; the system selects the best candidate.
    # A specialized side-grasp configuration can also be tried.

    # Immovable object: keep the table fixed.
    unmovable_objects = [table_id]

    # Set the required action_description in the grasp configuration.
    grasp_action_desc = {
        "zh": "抓起果汁瓶",
        "en": "picks up the juice bottle"
    }

    # Create the grasp configuration.
    # Configure grasp parameters for the requested side grasp.
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",
        set_grasp_pose_center=True,
        # Use middle-region grasp points, avoiding the top and bottom.
        grasp_lower_percentile=40,
        grasp_upper_percentile=60,
        # Immovable objects.
        unmovable_objects=unmovable_objects,
        # Required action_description.
        action_description=grasp_action_desc
    )

    # 7. Execute the grasp.
    print("Attempting to pick up juice bottle from the side...")
    current_success = robot.pick_actor(
        actor=juice_bottle_id,
        arm=arm,
        grasp_config=grasp_config
    )

    success = success and current_success
    if not current_success:
        print("Failed to pick up juice bottle")
        return False

     # 6. Move to shake Pose (Hand A)
    gripper_pose = robot.get_ee_pose(arm)
    # Position: x(-0.1 ~ -0.05), y(0.0). Z set to a reasonable chest height (e.g., 0.3m)
    shake_x = np.random.uniform(-0.1, 0.0)
    shake_y = 0.0
    shake_z = gripper_pose[2, 3] # Chest height approximation

    gripper_direction_quaternion = np.array([0,0,0.70711, -0.70711]) if arm == "left" else np.array([0,0,0.70711, 0.70711])
    base_rot_mat = quat2mat_wxyz(gripper_direction_quaternion)
    # Generate candidates with small Z rotation (±5 deg)
    shake_poses = []
    z_jitter = np.linspace(-0.05, 0.15, 5)
    y_jitter = np.linspace(-0.02, 0.02, 5) # Small y jitter
    angle_jitter = np.linspace(-15, 15, 5)

    for z_off in z_jitter:
        for y_off in y_jitter:
            for angle in angle_jitter:
                # Apply rotation around World Z

                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                pose[:3, 3] = [shake_x, shake_y + y_off, shake_z + z_off]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                shake_poses.append(pose)

    shake_poses = np.array(shake_poses)

    def bottle_in_hand_checker():
        bottle_pos, bottle_rot = robot.get_object_pose(juice_bottle_id)
        gripper_pos = robot.get_ee_pose(arm)
        gripper_position = gripper_pos[:3, 3]
        distance = np.linalg.norm(bottle_pos - gripper_position)
        if distance > 0.08:
            print("Bottle is not in hand: distance = ", distance)
            return False
        return True
    move_config = MoveConfig(
        extra_checkers=[bottle_in_hand_checker],
        action_description={"zh": "移动到胸前位置", "en": "moves to chest position"}
    )
    success = robot.move_actor_to_target_pose("gripper", shake_poses, arm=arm, move_config=move_config)
    if not success:
        print("Failed to move to shake pose")
        return False

    # shake
    angle_ranges = []
    for angle in np.linspace(15, 30, 5):
        angle = angle * np.pi / 180
        angle_ranges.append([-angle, angle])
    import random
    random.shuffle(angle_ranges)
    horizontal = False
    shake_config = ShakeConfig(
        action_description={"zh": "摇晃果汁瓶", "en": "shakes the juice bottle"},
        extra_checkers=[bottle_in_hand_checker]
    )
    success = robot.shake(actor=juice_bottle_id, arm=arm, horizontal=horizontal, angle_ranges=angle_ranges, times=4, shake_config=shake_config)

    print("\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: Juice bottle picked up and shaken")
    else:
        print("❌ FAILED: Could not pick up juice bottle or shake it")

    return success
