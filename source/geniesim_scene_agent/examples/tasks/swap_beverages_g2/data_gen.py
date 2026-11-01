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

    # Store initial positions for verification
    A_initial_pos = bev_A['pos'].copy()
    B_initial_pos = bev_B['pos'].copy()
    B_initial_z = bev_B['pos'][2]

    # Store initial positions for all beverages (before any manipulation)
    initial_positions = {bev['id']: bev['pos'].copy() for bev in beverages}

    print(f"Task: Swap {bev_A['name_en']} and {bev_B['name_en']}")

    # Set the task description to match the beverage-swap operation.
    description_json = {
        "task_name": f"swap {bev_A['name_en']} and {bev_B['name_en']}",
        "english_task_name": f"swap {bev_A['name_en']} and {bev_B['name_en']}",
        "init_scene_text": "The robot is in front of the table, there are 4 beverage cans placed in a 2x2 grid"
    }
    robot.client.set_task_task_basic_info(description_json)

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

    # 4. Reset arm B while holding B.
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

    # 5. Pick up A using the other arm (strictly following "the other hand").
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
    base_rot_matrix_A = quat2mat_wxyz(bev_A['rot'])
    target_poses_A = []

    # Generate multiple candidate rotations around World Z
    z_angles = np.linspace(0, 2*np.pi, 36, endpoint=False)

    for angle in z_angles:
        # Rotate around World Z
        rot_z = euler2mat([0, 0, angle])
        final_rot = np.dot(rot_z, base_rot_matrix_A)

        pose = np.eye(4)
        pose[:3, :3] = final_rot
        pose[:3, 3] = B_initial_pos  # Target is B's initial position
        target_poses_A.append(pose)

    target_poses_A = np.array(target_poses_A)

    # Define checker for placing A:
    # Verify that for all beverages EXCEPT B, the angle between their local Y axis and World Z axis is < 10 degrees
    def verify_A_placement_conditions():
        world_z_axis = np.array([0, 0, 1])

        # Check condition: All beverages except B have upright orientation (y-axis within 10° of world z)
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

        print(f"SUCCESS: A placement verification passed")
        return True

    place_config_A = PlaceConfig(
        unmovable_objects=unmovable_for_A,  # Only protect the other two beverages
        extra_checkers=[verify_A_placement_conditions],
        action_description={"zh": f"将{bev_A['name_cn']}放置到{bev_B['name_cn']}的位置", "en": f"place the {bev_A['name_en']} at the {bev_B['name_en']}'s position"}
    )

    if not robot.place_actor_to_target_pose(bev_A['id'], arm=arm_A, target_pose=target_poses_A, place_config=place_config_A):
        print(f"Failed to place {bev_A['name_cn']}")
        return False

    # 7. NEW: Place B at A's original position
    print(f"\n=== Step 7: Place B ({bev_B['name_cn']}) at A's original position ===")

    # Requirement: Use B's initial rotation rotated arbitrarily around world Z axis
    base_rot_matrix_B = quat2mat_wxyz(bev_B['rot'])
    target_poses_B = []

    # Generate multiple candidate rotations around World Z
    z_angles_B = np.linspace(0, 2*np.pi, 36, endpoint=False)

    for angle in z_angles_B:
        # Rotate around World Z
        rot_z = euler2mat([0, 0, angle])
        final_rot = np.dot(rot_z, base_rot_matrix_B)

        pose = np.eye(4)
        pose[:3, :3] = final_rot
        pose[:3, 3] = A_initial_pos  # Target is A's initial position
        target_poses_B.append(pose)

    target_poses_B = np.array(target_poses_B)

    # Define checker for placing B:
    # Verify that for all beverages, the angle between their local Y axis and World Z axis is < 10 degrees
    def verify_B_placement_conditions():
        world_z_axis = np.array([0, 0, 1])

        # Check condition: All beverages have upright orientation (y-axis within 10° of world z)
        for bev in beverages:
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

        print(f"SUCCESS: B placement verification passed")
        return True

    place_config_B = PlaceConfig(
        unmovable_objects=unmovable_for_A,  # Only protect the other two beverages (not A, which we just placed)
        extra_checkers=[verify_B_placement_conditions],
        action_description={"zh": f"将{bev_B['name_cn']}放置到{bev_A['name_cn']}的位置", "en": f"place the {bev_B['name_en']} at the {bev_A['name_en']}'s position"}
    )

    if not robot.place_actor_to_target_pose(bev_B['id'], arm=arm_B, target_pose=target_poses_B, place_config=place_config_B):
        print(f"Failed to place {bev_B['name_cn']}")
        return False

    # 8. Final verification: Check that A is at B's original position and B is at A's original position
    print("\n=== Final Verification ===")

    def verify_final_swap():
        # Check A is at B's original position
        A_curr_pos, A_curr_rot = robot.get_object_pose(bev_A['id'])
        if A_curr_pos is None:
            return False

        distance_A = np.linalg.norm(A_curr_pos[:2] - B_initial_pos[:2])
        if distance_A > 0.05:
            print(f"Final verification failed: A is {distance_A:.3f}m from B's original position (target < 0.05m)")
            return False
        print(f"SUCCESS: A is at B's original position (distance: {distance_A:.3f}m)")

        # Check B is at A's original position
        B_curr_pos, B_curr_rot = robot.get_object_pose(bev_B['id'])
        if B_curr_pos is None:
            return False

        distance_B = np.linalg.norm(B_curr_pos[:2] - A_initial_pos[:2])
        if distance_B > 0.05:
            print(f"Final verification failed: B is {distance_B:.3f}m from A's original position (target < 0.05m)")
            return False
        print(f"SUCCESS: B is at A's original position (distance: {distance_B:.3f}m)")

        # Check all other beverages are upright
        world_z_axis = np.array([0, 0, 1])
        for bev in beverages:
            curr_pos, curr_rot = robot.get_object_pose(bev['id'])
            if curr_pos is None:
                return False

            rot_mat = quat2mat_wxyz(curr_rot)
            local_y_axis = rot_mat[:, 1]

            dot_product = np.dot(local_y_axis, world_z_axis)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.rad2deg(angle_rad)

            if angle_deg > 10.0:
                print(f"Final verification failed: {bev['name_cn']} local Y is tilted {angle_deg:.2f} degrees (target < 10)")
                return False

        print("SUCCESS: All beverages are upright")

        return True

    if not verify_final_swap():
        print("Final verification failed")
        success = False
    else:
        print("✅ SUCCESS: Successfully swapped positions of A and B")

    # 9. Reset both arms
    robot.back_to_origin(arm=arm_A)
    robot.back_to_origin(arm=arm_B)

    return success
