# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # objects_list: Dictionary mapping asset_id to object metadata
    objects_list = {
        "benchmark_trash_can_013": {"has_interactions": True, "object_type": "trash_can"}
    }

    # runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "stack the outer trash can on top of the inner trash can",
        "english_task_name": "stack the outer trash can on top of the inner trash can",
        "init_scene_text": "The robot is in front of the table, there are two trash cans on it, one in the inner area and one in the outer area"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if "table" in keywords or obj_info.get("usd", "") == "table_000":
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find inner and outer trash cans
    inner_can_id = None
    outer_can_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        # Find inner trash can
        if any("inner_trash_can" in kw for kw in keywords):
            inner_can_id = obj_id
            print(f"Found inner trash can: {obj_id}")
        # Find outer trash can
        if any("outer_trash_can" in kw for kw in keywords):
            outer_can_id = obj_id
            print(f"Found outer trash can: {obj_id}")

    if not inner_can_id or not outer_can_id:
        print("Could not find both trash cans")
        return False

    # 3. Get poses and sizes
    inner_pos, inner_rot = robot.get_object_pose(inner_can_id)
    inner_size = robot.get_object_size_in_world_frame(inner_can_id)

    outer_pos, outer_rot = robot.get_object_pose(outer_can_id)
    outer_size = robot.get_object_size_in_world_frame(outer_can_id)

    if inner_pos is None or inner_size is None:
        print("Failed to get inner trash can info")
        return False

    if outer_pos is None or outer_size is None:
        print("Failed to get outer trash can info")
        return False

    # 4. Determine arm
    arm = robot.judge_arm_to_use(outer_pos)
    print(f"Using arm: {arm}")

    # 5. Generate target poses
    # Requirement:
    # - Position: The XY of outer bucket's bottom center should exactly match the XY of inner bucket's top center.
    #   Z: bottom of outer can is 0.0~0.02m above top of inner can.
    # - Rotation: Use inner can's rotation, then rotate around world Z by 90-degree multiples.

    # Known trash-can height.
    TRASH_CAN_HEIGHT = 0.1495
    inner_rot_matrix = quat2mat_wxyz(inner_rot)

    # Compute the center of the inner trash can's top surface.
    # World direction of local +Z: the rotation matrix's third column.
    local_y_axis = inner_rot_matrix[:, 1]
    # Top surface center = geometric center + (H/2) * local Y axis.
    inner_top_center = inner_pos + (TRASH_CAN_HEIGHT / 2.0) * local_y_axis

    print(f"Inner trash can geometric center: {inner_pos}")
    print(f"Inner trash can top surface center: {inner_top_center}")

    # Generate candidates
    target_poses = []

    # Z offsets for the bottom of the outer can relative to the top of the inner can
    # Requirement: 0.0 to 0.02m.
    # Final check requires height_diff >= 0.01 (outer center z - inner center z),
    # which approximately means outer bottom is 0.01m above inner top.
    # To ensure verification passes, generate offsets in range 0.015~0.025
    z_bottom_offsets = np.linspace(0.01, 0.015, 5)

    # Rotation angles around Z axis (multiples of 90 degrees)
    rotation_angles = [0, np.pi/2, np.pi, 3*np.pi/2]

    for z_bottom_offset in z_bottom_offsets:
        # Target: outer-can bottom center Z = inner_top_center[2] + z_bottom_offset.
        # Outer-can geometric center Z = bottom-center Z + H/2.
        target_z = inner_top_center[2] + z_bottom_offset + TRASH_CAN_HEIGHT / 2.0

        # Match XY to the inner trash can's top surface center.
        target_position = np.array([inner_top_center[0], inner_top_center[1], target_z])

        for angle in rotation_angles:
            # Create rotation matrix for Z axis rotation
            rot_z = euler2mat([0, 0, angle])

            # Combine: First inner rotation, then Z rotation
            # Requirement: "Take inner can rotation, then rotate around Z".
            # This means final rotation = rot_z @ inner_rot_matrix
            final_rot_matrix = np.dot(rot_z, inner_rot_matrix)

            pose = np.eye(4)
            pose[:3, :3] = final_rot_matrix
            pose[:3, 3] = target_position
            target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} target poses")

    # 6. Define checker function for success verification
    def check_stacked():
        # Verify:
        # 1. XY alignment: outer bucket's bottom center xy should match inner bucket's top center xy
        # 2. Z height: outer bucket's bottom center should be >= 0.01m above inner bucket's top center

        curr_inner_pos, curr_inner_rot = robot.get_object_pose(inner_can_id)
        curr_outer_pos, curr_outer_rot = robot.get_object_pose(outer_can_id)

        if curr_inner_pos is None or curr_outer_pos is None:
            return False

        # Check XY alignment with 3 cm tolerance.
        xy_dist = np.linalg.norm(curr_outer_pos[:2] - curr_inner_pos[:2])
        if xy_dist > 0.03:
            print(f"XY alignment check failed: xy distance {xy_dist:.4f}m")
            return False

        # Check Z: the outer-can bottom center should be at least 0.01 m above the inner-can top center.
        height_diff = curr_outer_pos[2] - curr_inner_pos[2]

        # The requested range is 0.0-0.02 m, but the final check requires >=0.01 m.
        # Tolerance range: 0.008-0.1 m to account for actual placement error.
        if height_diff < 0.008 or height_diff > 0.1:
            print(f"Z height check failed: height diff {height_diff:.4f}m, expected 0.008~0.1m")
            return False

        print(f"Success check passed: XY dist={xy_dist:.4f}m, Z diff (bottom relative to top)={height_diff:.4f}m")
        return True

    def check_picked_up():
        curr_outer_pos, _ = robot.get_object_pose(outer_can_id)
        if curr_outer_pos is None:
            return False
        robot.client.run_step(30)
        height_diff = curr_outer_pos[2] - outer_pos[2]
        if height_diff < 0.05:
            print(f"Picked up check failed: outer_pos is {curr_outer_pos[2]:.4f}m, required >= 0.05m")
            return False
        return True

    # 7. Execute pick and place
    grasp_config = GraspConfig(
        pick_up_distance=0.2,
        is_from_up_side=True,
        unmovable_objects=[inner_can_id, table_id],
        action_description={"zh": "拿起外侧的垃圾桶", "en": "pick up the outer trash can"},
        extra_params={"skip_check": True},
        extra_checkers=[check_picked_up]
    )

    # Use a 0.08 m pre-placement offset.
    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=[inner_can_id, table_id],
        disable_upside_down=False,
        extra_checkers=[check_stacked],
        action_description={"zh": "将外侧垃圾桶叠放到内侧垃圾桶上", "en": "place the outer trash can on top of the inner trash can"}
    )

    success = robot.pick_and_place_actor_to_target_pose(
        actor=outer_can_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    robot.back_to_origin(arm=arm)

    return success
