# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on provided interaction data
    objects_list = {
        "benchmark_beverage_bottle_001": {"has_interactions": True, "object_type": "beverage"},
        "benchmark_beverage_bottle_002": {"has_interactions": True, "object_type": "beverage"},
        "benchmark_beverage_bottle_003": {"has_interactions": True, "object_type": "beverage"},
        "benchmark_beverage_bottle_029": {"has_interactions": True, "object_type": "beverage"}
    }
    # Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. Find all beverages and the table
    beverages = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_beverage_bottle" in usd:
            # Determine beverage Chinese name from keywords
            bev_name_cn = "饮料"
            bev_name_en = "beverage"
            if "coca_cola" in keywords:
                bev_name_cn = "可乐"
                bev_name_en = "coca cola"
            elif "sprite" in keywords:
                bev_name_cn = "雪碧"
                bev_name_en = "sprite"
            elif "fanta" in keywords:
                bev_name_cn = "芬达"
                bev_name_en = "fanta"
            elif "wang_lao_ji" in keywords:
                bev_name_cn = "王老吉"
                bev_name_en = "wang lao ji"

            pos, rot = robot.get_object_pose(obj_id)
            beverages.append({
                "id": obj_id,
                "name_cn": bev_name_cn,
                "name_en": bev_name_en,
                "pos": pos.copy(),
                "rot": rot
            })

    if len(beverages) < 2:
        print("Not enough beverages found")
        return False

    # 2. Randomly select two distinct beverages A and B
    # Using random.sample to ensure they are different
    indices = list(range(len(beverages)))
    idx_B, idx_A = random.sample(indices, 2)

    bev_B = beverages[idx_B]
    bev_A = beverages[idx_A]

    print(f"Task: Place {bev_A['name_en']} at {bev_B['name_en']}'s position")

    # MANDATORY: Set task description based on random selection
    description_json = {
        "task_name": f"place {bev_A['name_en']} at {bev_B['name_en']}'s position",
        "english_task_name": f"place {bev_A['name_en']} at {bev_B['name_en']}'s position",
        "init_scene_text": "The robot is in front of the table, there are 4 beverage cans placed in a 2x2 grid"
    }
    robot.client.set_task_task_basic_info(description_json)

    # Store initial positions for verification
    initial_positions = {bev['id']: bev['pos'].copy() for bev in beverages}
    B_initial_z = bev_B['pos'][2]

    # Helper function to check if other beverages (excluding a specific one) have unchanged xy positions
    def create_others_xy_unchanged_checker(exclude_id):
        def checker():
            for bev in beverages:
                if bev['id'] == exclude_id:
                    continue

                curr_pos, curr_rot = robot.get_object_pose(bev['id'])
                if curr_pos is None:
                    return False

                initial_pos = initial_positions[bev['id']]

                # Check if xy positions are unchanged (tolerance 1cm)
                xy_distance = np.linalg.norm(curr_pos[:2] - initial_pos[:2])
                if xy_distance > 0.01:
                    print(f"ERROR: {bev['name_cn']} xy position changed by {xy_distance:.3f}m")
                    return False

            return True
        return checker

    # Helper function to check if B's z value is higher than initial by at least 0.05m
    def create_B_z_elevated_checker(B_id, initial_z):
        def checker():
            curr_pos, curr_rot = robot.get_object_pose(B_id)
            if curr_pos is None:
                return False

            z_elevation = curr_pos[2] - initial_z
            if z_elevation < 0.05:
                print(f"ERROR: B z elevation {z_elevation:.3f}m < 0.05m requirement")
                return False

            print(f"SUCCESS: B z elevation {z_elevation:.3f}m >= 0.05m")
            return True
        return checker

    # Get list of all beverage IDs except the one being manipulated
    def get_other_beverage_ids(exclude_id):
        return [bev['id'] for bev in beverages if bev['id'] != exclude_id]

    # 3. Pick up B - ensure other beverages don't move
    pos_B = bev_B['pos']
    arm_B = robot.judge_arm_to_use(pos_B)

    # Checker: Ensure other beverages don't move during B's pickup
    check_B_pickup_stable = create_others_xy_unchanged_checker(bev_B['id'])

    grasp_config_B = GraspConfig(
        pick_up_distance=0.2,
        grasp_upper_percentile=90,
        grasp_lower_percentile=60,
        unmovable_objects=get_other_beverage_ids(bev_B['id']),
        extra_checkers=[check_B_pickup_stable],
        action_description={"zh": f"拿起{bev_B['name_cn']}", "en": f"picks up the {bev_B['name_en']}"}
    )

    if not robot.pick_actor(bev_B['id'], arm=arm_B, grasp_config=grasp_config_B):
        print(f"Failed to pick {bev_B['name_cn']}")
        return False

    # 4. Reset arm B (holding B) as requested
    robot.back_to_origin(arm=arm_B)

    # Check B's z elevation and other beverages' xy stability after reset
    check_B_reset_ok = create_B_z_elevated_checker(bev_B['id'], B_initial_z)
    check_others_stable_after_B_reset = create_others_xy_unchanged_checker(bev_B['id'])

    if not check_B_reset_ok() or not check_others_stable_after_B_reset():
        print("ERROR: After resetting arm B, conditions not met")
        print("  - B's z should be >= 0.05m above initial")
        print("  - Other beverages' xy should remain unchanged")
        return False

    print("SUCCESS: After resetting arm B, B is elevated and other beverages are stable")

    # 5. Pick up A using the other arm, as explicitly required.
    # Determine the other arm
    arm_A = "left" if arm_B == "right" else "right"

    # Important: When picking up A, we should NOT include B in unmovable_objects
    # because B is being held and will move with the arm
    # We only need to protect the other two beverages
    unmovable_for_A = [bev['id'] for bev in beverages if bev['id'] not in [bev_A['id'], bev_B['id']]]

    grasp_config_A = GraspConfig(
        unmovable_objects=unmovable_for_A,
        grasp_upper_percentile=90,
        grasp_lower_percentile=60,
        action_description={"zh": f"拿起{bev_A['name_cn']}", "en": f"picks up the {bev_A['name_en']}"}
    )

    if not robot.pick_actor(bev_A['id'], arm=arm_A, grasp_config=grasp_config_A):
        print(f"Failed to pick {bev_A['name_cn']} with {arm_A} arm")
        return False

    # 6. Place A at B's original position
    # Requirement: Use A's initial rotation rotated arbitrarily around world Z axis
    base_rot_matrix = quat2mat_wxyz(bev_A['rot'])
    target_poses = []

    # Generate multiple candidate rotations around World Z
    z_angles = np.linspace(0, 2*np.pi, 36, endpoint=False)

    for angle in z_angles:
        # Rotate around World Z
        rot_z = euler2mat([0, 0, angle])
        final_rot = np.dot(rot_z, base_rot_matrix)

        pose = np.eye(4)
        pose[:3, :3] = final_rot
        pose[:3, 3] = bev_B['pos'] # Target is B's initial position
        target_poses.append(pose)

    target_poses = np.array(target_poses)

    # Define checker:
    # 1. Verify that for all beverages EXCEPT B, the angle between their local Y axis and World Z axis is < 10 degrees
    # 2. Verify that B's z value is still >= 0.05m above initial
    def verify_final_conditions():
        world_z_axis = np.array([0, 0, 1])

        # Check condition 1: All beverages except B have upright orientation (y-axis within 10° of world z)
        for bev in beverages:
            # Skip B (it's still being held)
            if bev['id'] == bev_B['id']:
                continue

            curr_pos, curr_rot = robot.get_object_pose(bev['id'])
            if curr_pos is None:
                return False

            # Get local Y axis from rotation matrix
            rot_mat = quat2mat_wxyz(curr_rot)
            local_y_axis = rot_mat[:, 1]

            # Calculate angle
            dot_product = np.dot(local_y_axis, world_z_axis)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.rad2deg(angle_rad)

            if angle_deg > 10.0:
                print(f"Verification failed: {bev['name_cn']} local Y is tilted {angle_deg:.2f} degrees (target < 10)")
                return False

        # Check condition 2: B's z value is still >= 0.05m above initial
        B_curr_pos, B_curr_rot = robot.get_object_pose(bev_B['id'])
        if B_curr_pos is None:
            return False

        B_z_elevation = B_curr_pos[2] - B_initial_z
        if B_z_elevation < 0.05:
            print(f"Verification failed: B z elevation {B_z_elevation:.3f}m < 0.05m requirement")
            return False

        print(f"SUCCESS: Final verification passed")
        print(f"  - B z elevation: {B_z_elevation:.3f}m >= 0.05m")

        return True

    place_config_A = PlaceConfig(
        unmovable_objects=unmovable_for_A,  # Only protect the other two beverages
        extra_checkers=[verify_final_conditions],
        action_description={"zh": f"将{bev_A['name_cn']}放置到{bev_B['name_cn']}的位置", "en": f"places the {bev_A['name_en']} at the {bev_B['name_en']}'s position"}
    )

    if not robot.place_actor_to_target_pose(bev_A['id'], arm=arm_A, target_pose=target_poses, place_config=place_config_A):
        print(f"Failed to place {bev_A['name_cn']}")
        return False

    # 7. Reset arm A
    robot.back_to_origin(arm=arm_A)

    return success
