# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from asset information in scene generation.
    objects_list = {
        "benchmark_building_blocks_074": {
            "has_interactions": True,  # Assume blocks have interaction data.
            "object_type": "building_block",
            "size": "large"  # 0.05m cube.
        },
        "table_000": {
            "has_interactions": False,  # The table may have no interaction data.
            "object_type": "table"
        }
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. Locate table_000.
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

    # 2. Locate all benchmark_building_blocks_074 cubes.
    cubes = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether this is benchmark_building_blocks_074.
        if "benchmark_building_blocks_074" in usd:
            size = robot.get_object_size_in_world_frame(obj_id)
            if size is not None:
                volume = size[0] * size[1] * size[2]
                pos, rot = robot.get_object_pose(obj_id)
                cubes.append({
                    "id": obj_id,
                    "size": size,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found cube: {obj_id}, size: {size}, volume: {volume:.6f}")

    if len(cubes) < 3:
        print(f"Found only {len(cubes)} cubes, need at least 3")
        return False

    print(f"Total cubes found: {len(cubes)}")

    # 3. Get table information for stack placement.
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    # Tabletop height = table_pos[2] + table_size[2]/2.0.
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": "stack all the building blocks on the middle of the table",
        "english_task_name": "stack all the building blocks on the middle of the table",
        "init_scene_text": "The robot is in front of the table, there are several cube building blocks of the same size scattered on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 4. Calculate stack placement parameters.
    # Target is table center (0, 0) in table coordinates.
    # table_pos may be the table center; use (0, 0) as the offset to ensure center placement.
    target_x = 0.0  # Table center x.
    target_y = 0.0  # Table center y.

    # Get the first block's size; all blocks are identical.
    if len(cubes) > 0:
        cube_size = cubes[0]["size"]
        print(f"Cube size: {cube_size}")
    else:
        print("No cubes found")
        return False

    # 5. Generate candidate orientations satisfying task requirements.
    def generate_valid_rotations():
        """Generate candidate rotations with y upward and horizontal yaw in multiples of 90 degrees."""
        candidate_rotations = []

        # Base rotation ensures local y points up, along world z.
        # Rotate the cube's local frame so its local y points upward (world z).
        # The task requires its upward y axis to remain upward after placement.

        # Option 1: rotate 90 degrees about x, aligning local y with world z.
        rot_x_90 = euler2mat([np.pi/2, 0, 0])

        # Option 2: rotate -90 degrees about x, aligning the axes with possibly opposite signs.
        rot_x_minus_90 = euler2mat([-np.pi/2, 0, 0])

        # For each base rotation, add world-z rotations in multiples of 90 degrees.
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

        print(f"Generated {len(unique_rotations)} unique rotation matrices")
        return unique_rotations

    candidate_rotations = generate_valid_rotations()

    # 6. Stack blocks individually.
    placed_cubes = []
    last_arm = ""

    # Stack from bottom to top.
    for i in range(len(cubes)):
        cube_id = cubes[i]["id"]
        cube_size = cubes[i]["size"]

        print(f"\n=== Processing cube {i+1}/{len(cubes)}: {cube_id} ===")

        # Get current block position.
        cube_position, cube_rotation = robot.get_object_pose(cube_id)
        if cube_position is None:
            print(f"Warning: Cannot get position for cube {cube_id}")
            continue

        # Choose the arm.
        cube_arm = robot.judge_arm_to_use(cube_position)
        print(f"Using arm: {cube_arm}")

        # Return to origin when switching arms.
        if last_arm != cube_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = cube_arm

        # Calculate stacking position.
        # Place the first block at table center and subsequent blocks atop the preceding block.
        if i == 0:
            # First block: table center.
            # CRITICAL: safety margin avoids gripper collisions.
            margin = 0.01  # 1cm safety margin.
            target_z = table_top_z + margin + cube_size[2] / 2.0
        else:
            # Later blocks: atop the previous block.
            prev_cube_id = placed_cubes[-1]["id"]
            prev_cube_pos, prev_cube_rot = robot.get_object_pose(prev_cube_id)
            prev_cube_size = robot.get_object_size_in_world_frame(prev_cube_id)

            if prev_cube_pos is None or prev_cube_size is None:
                print(f"Cannot get previous cube {prev_cube_id} info")
                success = False
                break

            margin = 0.005  # 1cm safety margin.
            target_z = prev_cube_pos[2] + prev_cube_size[2] / 2.0 + margin + cube_size[2] / 2.0

        base_position = np.array([target_x, target_y, target_z])

        # Create grasp configuration.
        unmovable_objects = [table_id]
        # Protect blocks already stacked.
        unmovable_objects.extend([c["id"] for c in placed_cubes])

        # Describe the block by stack order.
        if i == 0:
            block_desc_zh = "第一个的积木"
            block_desc_en = "first block"
        elif i == 1:
            block_desc_zh = "第二个的积木"
            block_desc_en = "second block"
        elif i == 2:
            block_desc_zh = "第三个的积木"
            block_desc_en = "third block"
        elif i == len(cubes) - 1:
            block_desc_zh = "最后一个的积木"
            block_desc_en = "last block"
        else:
            block_desc_zh = f"第{i+1}个积木"
            block_desc_en = f"the {i+1}th block"

        grasp_config = GraspConfig(
            grasp_offset=-0.005,
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{block_desc_zh}", "en": f"picks up the {block_desc_en}"}  # REQUIRED
        )

        # Generate many candidate poses satisfying the task.
        target_poses = []

        # Small position variations improve the chance of success.
        x_offsets = np.linspace(-0.0001, 0.0001, 3)  # Small x offset.
        y_offsets = np.linspace(-0.0001, 0.0001, 3)  # Small y offset.
        z_offsets = np.linspace(0, 0.002, 3)  # Small z offset.

        # Generate all candidate-pose combinations.
        for z_offset in z_offsets:
            position = base_position + np.array([0,0, z_offset])
            for rot_mat in candidate_rotations:
                pose = np.eye(4)
                pose[:3, :3] = rot_mat
                pose[:3, 3] = position
                target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for cube {cube_id}")

        # Define a checker for successful stacking.
        def create_check_function(cube_id, expected_position, expected_height, is_bottom=False):
            def check_cube_stacked():
                current_pos, current_rot = robot.get_object_pose(cube_id)
                if current_pos is None or current_rot is None:
                    return False

                # Position check: 2cm XY tolerance and appropriate z height range.
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                xy_tolerance = 0.02  # 2cm

                # Use different z checks for bottom and upper blocks.
                if is_bottom:
                    # Bottom block should be above the table.
                    z_min = table_top_z + 0.005  # Slightly above the tabletop.
                    z_max = table_top_z + cube_size[2] + 0.02  # Allow some error.
                else:
                    # Upper blocks should be near expected height.
                    z_min = expected_height - 0.02  # Allow 1cm error.
                    z_max = expected_height + 0.02  # Allow 1cm error.

                if xy_distance > xy_tolerance:
                    print(f"Cube {cube_id} xy position error: {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    return False

                if current_pos[2] < z_min or current_pos[2] > z_max:
                    print(f"Cube {cube_id} z position {current_pos[2]:.3f}m out of range [{z_min:.3f}, {z_max:.3f}]")
                    return False

                return True

            return check_cube_stacked

        # Create the checker.
        is_bottom = (i == 0)
        check_function = create_check_function(cube_id, base_position, target_z, is_bottom)

        # Describe placement by stack position.
        if i == 0:
            place_desc_zh = "将第一个积木放置到桌子正中心"
            place_desc_en = "places the first block at the center of the table"
        elif i == 1:
            place_desc_zh = "将第二个积木放置到第一个积木上方"
            place_desc_en = "places the second block on top of the first block"
        elif i == 2:
            place_desc_zh = "将第三个积木放置到塔的顶部"
            place_desc_en = "places the third block on top of the tower"
        else:
            place_desc_zh = f"将第{i+1}个积木放置到塔的顶部"
            place_desc_en = f"places the {i+1}th block on top of the tower"

        place_config = PlaceConfig(
            pre_insert_offset=0.02,
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en},  # REQUIRED
            extra_params={"constraint_insert": True}
        )

        # Execute grasp and placement with pose-based API; blocks have no annotated passive.place point.
        print(f"Attempting to pick and place cube {cube_id} with {len(target_poses)} candidate poses")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=cube_id,
            target_pose=target_poses,
            arm=cube_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"Failed to place cube {cube_id}")
            success = False
            # Regrasp/replacement could be optional, but instructions require aborting on chained-task failure.
            break
        else:
            print(f"Successfully placed cube {cube_id}")

        # Add to placed list.
        placed_cubes.append({"id": cube_id, "position": base_position})

    # 7. Final verification.
    print("\n=== Performing final verification ===")

    if len(placed_cubes) > 0:
        # Check tower stability.
        tower_height = 0
        for i, cube_info in enumerate(placed_cubes):
            cube_id = cube_info["id"]
            current_pos, current_rot = robot.get_object_pose(cube_id)

            if current_pos is None:
                print(f"Cube {cube_id} not found after placement")
                success = False
                continue

            # Check proximity to table center.
            center_distance = np.linalg.norm(current_pos[:2])
            if center_distance > 0.03:  # Allow 3cm offset.
                print(f"  Warning: Cube {cube_id} is {center_distance:.3f}m from center")

            # Calculate tower height.
            if i == 0:
                # Bottom height relative to tabletop.
                tower_height = current_pos[2] - table_top_z
            else:
                # Check each upper block is directly above its supporting block.
                prev_cube_id = placed_cubes[i-1]["id"]
                prev_cube_pos, _ = robot.get_object_pose(prev_cube_id)
                if prev_cube_pos is not None:
                    height_diff = current_pos[2] - prev_cube_pos[2]
                    expected_diff = cube_size[2]  # Block height.
                    if abs(height_diff - expected_diff) > 0.015:  # Allow 1.5cm error.
                        print(f"  Warning: Height difference between cubes {i} and {i-1}: {height_diff:.3f}m, expected {expected_diff:.3f}m")

        print(f"Tower height: {tower_height:.3f}m")
        print(f"Number of cubes in tower: {len(placed_cubes)}")

        # Check tower verticality.
        if len(placed_cubes) >= 2:
            bottom_pos, _ = robot.get_object_pose(placed_cubes[0]["id"])
            top_pos, _ = robot.get_object_pose(placed_cubes[-1]["id"])

            if bottom_pos is not None and top_pos is not None:
                # Calculate tower lean.
                horizontal_offset = np.linalg.norm((top_pos - bottom_pos)[:2])
                vertical_height = abs(top_pos[2] - bottom_pos[2])

                if vertical_height > 0:
                    tilt_ratio = horizontal_offset / vertical_height
                    if tilt_ratio > 0.1:  # Lean exceeds 10%.
                        print(f"  Warning: Tower is tilted, horizontal offset: {horizontal_offset:.3f}m, tilt ratio: {tilt_ratio:.3f}")
                    else:
                        print(f"  Tower is stable, horizontal offset: {horizontal_offset:.3f}m")

    # Return the arm to its initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success and len(placed_cubes) == len(cubes):
        print("✅ SUCCESS: All cubes stacked into a tower at the center of the table")
    else:
        print(f"⚠️ PARTIAL SUCCESS: Stacked {len(placed_cubes)} out of {len(cubes)} cubes")

    return success and (len(placed_cubes) == len(cubes))
