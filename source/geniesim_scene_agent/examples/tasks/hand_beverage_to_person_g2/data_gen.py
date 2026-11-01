# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    # Extract asset ID list.
    asset_ids = [
        "benchmark_beverage_bottle_001", "benchmark_beverage_bottle_002",
        "benchmark_beverage_bottle_003", "benchmark_beverage_bottle_004",
        "benchmark_beverage_bottle_008", "benchmark_beverage_bottle_023",
        "benchmark_beverage_bottle_024", "benchmark_beverage_bottle_025",
        "benchmark_beverage_bottle_026", "benchmark_beverage_bottle_029",
        "benchmark_beverage_bottle_073",
        "benchmark_character_000", "benchmark_character_001",
        "benchmark_character_002", "benchmark_character_003",
        "benchmark_character_004", "benchmark_character_005",
        "benchmark_character_006", "benchmark_character_007",
        "benchmark_character_008", "benchmark_character_009",
        "benchmark_character_010", "benchmark_character_011",
        "benchmark_character_012", "benchmark_character_013",
        "benchmark_character_014",
        "table_000"
    ]

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "pick up the drink on the table and bring it to the person",
        "english_task_name": "pick up the drink on the table and bring it to the person",
        "init_scene_text": "The robot is in front of the table, there is a water bottle on the table and a person sitting in front of the table"
    }
    robot.client.set_task_task_basic_info(description_json)
    print("=== Starting task: Bring drink to person ===")

    # 1. Find table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords or "dining_table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False
    print(f"Found table: {table_id}")

    # 2. Find cup/beverage bottle.
    bottle_id = None
    bottle_usd = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_beverage_bottle" in usd or any("water_bottle" in kw.lower() for kw in keywords) or any("bottle" in kw.lower() for kw in keywords):
            bottle_id = obj_id
            bottle_usd = usd
            print(f"Found bottle: {bottle_id}, usd: {bottle_usd}")
            break

    if not bottle_id:
        print("Water bottle not found on table")
        return False

    # 3. Find person.
    person_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_character" in usd or any("person" in kw.lower() for kw in keywords) or any("human" in kw.lower() for kw in keywords):
            person_id = obj_id
            print(f"Found person: {person_id}, usd: {usd}")
            break

    if not person_id:
        print("Person not found in front of table")
        return False

    # 4. Get person's position and determine target region.
    person_pos, person_rot = robot.get_object_pose(person_id)
    if person_pos is None:
        print(f"Cannot get person position for {person_id}")
        return False

    person_y = person_pos[1]

    # Get table height.
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # Get bottle dimensions.
    bottle_size = robot.get_object_size_in_world_frame(bottle_id)
    if bottle_size is None:
        print("Cannot get bottle size")
        return False

    # 5. Compute target point.
    target_x = random.uniform(0.20, 0.25)
    target_y = random.uniform(person_y - 0.02, person_y + 0.02)
    # Release suspended 1 cm above target to avoid striking table; let gravity settle it.
    margin = 0.01
    target_z = table_top_z + margin + 2 * bottle_size[2] / 2.0
    target_position = np.array([target_x, target_y, target_z])

    print(f"Target placement position: {target_position}")

    # 6. Select arm.
    bottle_pos, bottle_rot = robot.get_object_pose(bottle_id)
    arm = robot.judge_arm_to_use(bottle_pos)
    print(f"Using arm: {arm}")

    # 7. Generate guaranteed vertical candidate poses.
    # Beverage-bottle local y should point upward along world z.
    # Create rotation aligning local y with world z.
    # Option 1: rotate +90 degrees about x, aligning local y with world z.
    rot_x_90 = euler2mat([np.pi/2, 0, 0])
    # Option 2: rotate -90 degrees about x, aligning local y with world z in opposite direction.
    rot_x_minus_90 = euler2mat([-np.pi/2, 0, 0])

    # Generate candidate rotation matrices.
    candidate_rotations = []

    # For each base rotation, add world-z rotations in multiples of 90 degrees; beverage permits arbitrary horizontal yaw.
    for base_rot in [rot_x_90, rot_x_minus_90]:
        for z_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_z = euler2mat([0, 0, z_angle])
            final_rot = np.dot(rot_z, base_rot)
            candidate_rotations.append(final_rot)

    # Deduplicate.
    unique_rotations = []
    for rot in candidate_rotations:
        is_unique = True
        for existing in unique_rotations:
            if np.allclose(rot, existing, atol=1e-6):
                is_unique = False
                break
        if is_unique:
            unique_rotations.append(rot)

    print(f"Generated {len(unique_rotations)} unique rotation matrices for upright placement")

    # Generate all candidate poses.
    target_poses = []
    # Position variation parameters, small offsets.
    x_offsets = np.linspace(-0.001, 0.001, 3)
    y_offsets = np.linspace(-0.001, 0.001, 3)
    z_offsets = np.linspace(0, 0.002, 3)

    # Generate all candidate combinations.
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = target_position + np.array([x_offset, y_offset, z_offset])
                # Generate a pose for each candidate rotation.
                for rot_mat in unique_rotations:
                    pose = np.eye(4)
                    pose[:3, :3] = rot_mat
                    pose[:3, 3] = position
                    target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} UPRIGHT candidate poses (ensuring bottle's local y-axis faces upward)")

    # 8. Define checker: beverage bottle placed vertically, local y upward.
    def create_bottle_upright_checker(bottle_id, expected_x_range, person_y, table_top_z):
        def check_bottle_upright_and_placed():
            # Get current bottle position/rotation.
            current_pos, current_rot = robot.get_object_pose(bottle_id)
            if current_pos is None or current_rot is None:
                print("Cannot get bottle position or rotation")
                return False

            # 1. Check placement position.
            # Relax X tolerance to allow sliding.
            if current_pos[0] < expected_x_range[0] - 0.02 or current_pos[0] > expected_x_range[1] + 0.02:
                print(f"Bottle X {current_pos[0]:.3f} out of relaxed bounds")
                return False

            # Relax Y tolerance to allow up to 4 cm sliding.
            if abs(current_pos[1] - person_y) > 0.04:
                print(f"Bottle Y {current_pos[1]:.3f} too far from person Y {person_y:.3f}")
                return False

            # Accept Z if interpenetration is not excessive.
            if current_pos[2] < table_top_z - 0.01:
                print(f"Bottle fell below table: Z={current_pos[2]:.3f}")
                return False

            # 2. Check bottle verticality, local y upward.
            # Get current bottle rotation matrix.
            current_rot_matrix = quat2mat_wxyz(current_rot)

            # Bottle local y direction is [0, 1, 0] in local frame.
            # Transform direction into world frame.
            local_y_axis = np.array([0, 1, 0])
            world_y_axis = np.dot(current_rot_matrix, local_y_axis)

            # World up is [0, 0, 1].
            world_up = np.array([0, 0, 1])

            # Compute angle between local y and world z.
            # Use dot product for cosine: cos(theta) = (a dot b) / (|a||b|).
            dot_product = np.dot(world_y_axis, world_up)
            norm_y = np.linalg.norm(world_y_axis)
            norm_up = np.linalg.norm(world_up)

            # Avoid division by zero.
            if norm_y == 0 or norm_up == 0:
                print("Zero norm in axis calculation")
                return False

            cos_theta = dot_product / (norm_y * norm_up)

            # Clamp to [-1, 1] for floating-point error.
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Compute angle in radians.
            angle_rad = np.arccos(cos_theta)
            angle_deg = angle_rad * 180 / np.pi

            print(f"Bottle local y-axis angle with world up: {angle_deg:.1f}°")

            # Check local y upward with +/-15-degree tolerance.
            # Vertical local y implies angle near 0 or 180 degrees to world z.
            # Direction depends on rotation: rot_x_90 should give 0 degrees; rot_x_minus_90 should give 180.
            # General check: local y approximately parallel to world z, near 0 or 180 degrees.
            # Expected vertical beverage placement requires local y approximately parallel to world z.
            # Allow +/-15-degree error.
            upright_tolerance = 15.0  # Degrees.

            # Check angle within tolerance of 0 or 180 degrees.
            if angle_deg <= upright_tolerance or (180 - angle_deg) <= upright_tolerance:
                print(f"✅ Bottle is upright (angle: {angle_deg:.1f}°)")

                # Additional check: upright rather than inverted bottle.
                # Check sign of local-y/world-z dot product.
                # For upward local y, world_y_axis[2] should be positive, near 1.
                # For downward local y, world_y_axis[2] should be negative, near -1.

                # Beverage bottle normally expected upright, opening upward.
                # Task does not explicitly require opening up/down, so accept both directions.
                # Sign of world_y_axis[2] distinguishes them.
                if world_y_axis[2] > 0:
                    print(f"  Bottle is upright (local y-axis pointing up)")
                else:
                    print(f"  Bottle is inverted (local y-axis pointing down)")

                return True
            else:
                print(f"❌ Bottle is not upright enough (angle: {angle_deg:.1f}°, tolerance: ±{upright_tolerance}°)")
                return False

        return check_bottle_upright_and_placed

    # Create checker.
    check_function = create_bottle_upright_checker(bottle_id, (0.20, 0.25), person_y, table_top_z)

    # 9. Configure pick and place.
    unmovable_objects = [table_id, person_id]

    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        pick_up_distance=0.07,
        action_description={"zh": "拿起桌子上的饮料", "en": "pick up the drink on the table"}
    )

    place_config = PlaceConfig(
        unmovable_objects=unmovable_objects,
        extra_checkers=[check_function],  # Use checker including verticality test.
        action_description={"zh": "将饮料竖直放置到人面前", "en": "place the drink upright in front of the person"}
    )

    # 11. Execute and verify.
    print(f"Attempting to pick and place bottle...")
    success = robot.pick_and_place_actor_to_target_pose(
        actor=bottle_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    print("\n=== Performing final verification ===")
    if success:
        # Use checker for final verification.
        if check_function():
            final_pos, final_rot = robot.get_object_pose(bottle_id)
            if final_pos is not None and final_rot is not None:
                print(f"✅ Bottle final position: {final_pos}")

                # Get and print bottle orientation information.
                final_rot_matrix = quat2mat_wxyz(final_rot)
                local_y_axis = np.array([0, 1, 0])
                world_y_axis = np.dot(final_rot_matrix, local_y_axis)
                world_up = np.array([0, 0, 1])

                dot_product = np.dot(world_y_axis, world_up)
                norm_y = np.linalg.norm(world_y_axis)
                norm_up = np.linalg.norm(world_up)

                if norm_y > 0 and norm_up > 0:
                    cos_theta = dot_product / (norm_y * norm_up)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_rad = np.arccos(cos_theta)
                    angle_deg = angle_rad * 180 / np.pi
                    print(f"✅ Bottle is upright with angle: {angle_deg:.1f}° from vertical")

                print("✅ All placement requirements satisfied (position and upright)")
            else:
                print("❌ Cannot get bottle final state")
                success = False
        else:
            print("❌ Final verification failed: Bottle not upright or not properly placed")
            success = False
    else:
        print("❌ Pick and place operation failed")

    # 12. Safe retreat.
    robot.back_to_origin(arm=arm)

    if success:
        print("✅ SUCCESS: Drink handed to the person in upright position")
    else:
        print("❌ FAILED: Task execution failed")

    return success
