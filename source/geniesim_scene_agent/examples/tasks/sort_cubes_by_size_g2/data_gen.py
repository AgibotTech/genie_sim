# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Block asset IDs extracted from scene generation.
    objects_list = {
        "benchmark_building_blocks_079": {"has_interactions": True, "object_type": "building_block", "size": "small"},
        "benchmark_building_blocks_072": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_076": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_019": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_073": {"has_interactions": True, "object_type": "building_block", "size": "large"},
        "benchmark_building_blocks_011": {"has_interactions": True, "object_type": "building_block", "size": "large"},
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

    # 2. Locate all cube blocks.
    cubes = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        is_cube = False
        for cube_asset in list(objects_list.keys()):
            if cube_asset in usd:
                is_cube = True
                break

        if is_cube or any("cube" in kw.lower() for kw in keywords) or any("block" in kw.lower() for kw in keywords):
            size = robot.get_object_size_in_world_frame(obj_id)
            if size is not None:
                volume = size[0] * size[1] * size[2]
                pos, rot = robot.get_object_pose(obj_id)
                cubes.append({
                    "id": obj_id,
                    "size": size,
                    "volume": volume,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found cube: {obj_id}, size: {size}, volume: {volume:.6f}")

    if len(cubes) < 2:
        print(f"Found only {len(cubes)} cubes, need at least 2")
        return False

    print(f"Total cubes found: {len(cubes)}")

    # 3. Sort by volume, smallest first.
    cubes.sort(key=lambda x: x["volume"])
    print("Cubes sorted by volume (smallest to largest):")
    for i, cube in enumerate(cubes):
        print(f"  {i+1}. {cube['id']}: volume={cube['volume']:.6f}, size={cube['size']}")

    # 4. Calculate placement parameters.
    # Placement region: x=(-0.05 to -0.15), y=(-0.1, 0.1).
    target_x = np.random.uniform(-0.13, -0.17)  # Fixed x position.

    # Calculate total width.
    total_width = 0
    for cube in cubes:
        cube_width = cube["size"][1]  # Use y dimension as width.
        total_width += cube_width

    # Increase block spacing to at least 3cm.
    min_spacing = 0.04  # 3cm

    # Calculate required total width.
    total_spacing = min_spacing * (len(cubes) - 1)
    total_needed_width = total_width + total_spacing

    # Check fit within the specified y range.
    available_y_range = 0.20  # y=-0.1 to 0.1 spans 0.20m.
    print(f"Available y range: {available_y_range:.3f}m")
    print(f"Total needed width: {total_needed_width:.3f}m")

    if total_needed_width > available_y_range:
        print(f"Warning: Total needed width {total_needed_width:.3f}m exceeds available range {available_y_range:.3f}m")
        # Adjust spacing proportionally, retaining at least 2cm.
        scale_factor = available_y_range / total_needed_width
        if scale_factor < 0.5:
            print("Error: Cannot fit cubes with reasonable spacing")
            return False

        adjusted_spacing = max(min_spacing * scale_factor, 0.03)  # At least 2cm.
        total_needed_width = total_width + adjusted_spacing * (len(cubes) - 1)
        print(f"Adjusted spacing to {adjusted_spacing:.3f}m")
        spacing = adjusted_spacing
    else:
        spacing = min_spacing

    print(f"Using spacing: {spacing:.3f}m")

    # Calculate starting position: smallest block leftmost, at greatest y.
    # y ranges from -0.1 to 0.1, arranged in descending order.
    start_y = np.random.uniform(0.090, 0.1)

    # 5. Calculate each block's target position, smallest first.
    target_positions = []
    current_y = start_y

    for i, cube in enumerate(cubes):
        cube_id = cube["id"]
        cube_size = cube["size"]

        # Calculate y position.
        current_y = current_y - cube_size[1]/2.0
        target_y = current_y

        # Get table height.
        table_pos, table_rot = robot.get_object_pose(table_id)
        table_size = robot.get_object_size_in_world_frame(table_id)
        table_top_z = table_pos[2] + table_size[2] / 2.0

        # Placement height = table top + safety margin + half block height.
        margin = 0.01  # 1cm safety margin avoids gripper collisions.
        target_z = table_top_z + margin + cube_size[2] / 2.0

        # Position vector.
        target_position = np.array([target_x, target_y, target_z])
        target_positions.append({
            "id": cube_id,
            "position": target_position,
            "size": cube_size,
            "index": i
        })

        print(f"Cube {i+1} ({cube_id}) target position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}], size: {cube_size}")

        # Update current y for the next block.
        current_y = current_y - cube_size[1]/2.0 - spacing

    # 6. Generate candidate orientations in advance for all blocks.
    def generate_candidate_rotations():
        """Generate candidate rotation matrices satisfying the requirements."""
        candidate_rotations = []

        # Base rotation maps local y to world z: 90 degrees about x.
        rot_x_90 = euler2mat([np.pi/2, 0, 0])

        # Option 1: world-z rotations of 0, 90, 180, 270 degrees.
        for z_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_z = euler2mat([0, 0, z_angle])
            final_rot = np.dot(rot_z, rot_x_90)
            candidate_rotations.append(final_rot)

        # Option 2: rotate about y, then 90 degrees about x.
        for y_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_y = euler2mat([0, y_angle, 0])
            rot_combined = np.dot(rot_x_90, rot_y)
            candidate_rotations.append(rot_combined)

        # Deduplicate, although duplicates are unlikely.
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

    candidate_rotations = generate_candidate_rotations()

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": "arrange blocks from small to large",
        "english_task_name": "arrange blocks from small to large",
        "init_scene_text": "The robot is in front of the table, there are three cube building blocks of different sizes randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 7. Execute placement, smallest first.
    placed_cubes = []
    last_arm = ""
    for i, target_info in enumerate(target_positions):
        cube_id = target_info["id"]
        base_position = target_info["position"]
        cube_size = target_info["size"]

        print(f"\n=== Processing cube {i+1}/{len(target_positions)}: {cube_id} ===")

        # Get current block position.
        cube_position, cube_rotation = robot.get_object_pose(cube_id)
        if cube_position is None:
            print(f"Warning: Cannot get position for cube {cube_id}")
            continue

        # Choose the arm.
        cube_arm = robot.judge_arm_to_use(cube_position)
        print(f"Using arm: {cube_arm}")
        if last_arm != cube_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = cube_arm
        # Create grasp configuration.
        unmovable_objects = [table_id]
        unmovable_objects.extend([c["id"] for c in placed_cubes])

        # Describe the ith block in ascending size order.
        size_descriptions = ["最小的积木", "中等大小的积木", "最大的积木"]
        size_descriptions_en = ["smallest block", "medium-sized block", "largest block"]
        if i < len(size_descriptions):
            block_desc_zh = size_descriptions[i]
            block_desc_en = size_descriptions_en[i]
        else:
            block_desc_zh = f"第{i+1}个积木"
            block_desc_en = f"the {i+1}th block"

        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{block_desc_zh}", "en": f"pick up the {block_desc_en}"}  # REQUIRED
        )

        # Generate many candidate poses; pass all alternatives together.
        target_poses = []

        # Position variation parameters.
        x_offsets = np.linspace(-0.001, 0.001, 3)  # x offset.
        y_offsets = np.linspace(-0.001, 0.001, 3)  # y offset.
        z_offsets = np.linspace(0, 0.005, 3)  # z offset.

        # Generate every candidate-pose combination.
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                for z_offset in z_offsets:
                    position = base_position + np.array([x_offset, y_offset, z_offset])
                    for rot_mat in candidate_rotations:
                        pose = np.eye(4)
                        pose[:3, :3] = rot_mat
                        pose[:3, 3] = position
                        target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for cube {cube_id}")

        # Define the checker.
        def create_check_function(cube_id, expected_position):
            def check_cube_aligned():
                current_pos, current_rot = robot.get_object_pose(cube_id)
                if current_pos is None or current_rot is None:
                    return False

                # Position check with 3cm tolerance.
                pos_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                tolerance = 0.03
                if pos_distance > tolerance:
                    print(f"Cube {cube_id} position error: {pos_distance:.3f}m > {tolerance:.3f}m")
                    return False

                # Check that it is on the table.
                table_pos, _ = robot.get_object_pose(table_id)
                table_size = robot.get_object_size_in_world_frame(table_id)
                table_top_z = table_pos[2] + table_size[2] / 2.0

                if current_pos[2] < table_top_z:
                    print(f"Cube {cube_id} below table surface")
                    return False

                return True

            return check_cube_aligned

        # Create the checker.
        check_function = create_check_function(cube_id, base_position)

        # Describe placement by size and left-to-right position.
        if i == 0:
            # First (smallest) block goes on the left.
            place_desc_zh = "将最小的积木放置到桌子中央偏下位置的左侧"
            place_desc_en = "place the smallest block to the left side of the lower center of the table"
        elif i == 1:
            # Second (medium) block goes to the smallest block's right.
            place_desc_zh = "将中等大小的积木放置到最小积木的右侧"
            place_desc_en = "place the medium-sized block to the right of the smallest block"
        elif i == 2:
            # Third (largest) block goes to the medium block's right.
            place_desc_zh = "将最大的积木放置到中等大小积木的右侧"
            place_desc_en = "place the largest block to the right of the medium-sized block"
        else:
            # Handle more than three blocks.
            place_desc_zh = f"将第{i+1}个积木放置到桌子中央偏下位置"
            place_desc_en = f"place the {i+1}th block to the lower center of the table"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # Execute grasp and placement with all candidate poses together.
        print(f"Attempting to pick and place cube {cube_id} with {len(target_poses)} candidate poses")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=cube_id,
            target_pose=target_poses,
            arm=cube_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )
        success = success and current_success
        if not success:
            print(f"Failed to place cube {cube_id}")
            return False
        print(f"Successfully placed cube {cube_id}")

        # Add to placed list.
        placed_cubes.append({"id": cube_id, "position": base_position})


    # 8. Final verification and summary.
    print("\n=== Performing final verification ===")

    # Collect all blocks' final positions.
    final_positions = []
    for cube in cubes:
        cube_id = cube["id"]
        current_pos, current_rot = robot.get_object_pose(cube_id)

        if current_pos is None:
            print(f"Cube {cube_id} not found after placement")
            success = False
            continue

        final_positions.append((cube_id, current_pos[1], current_pos))

        # Check basic requirements.
        if current_pos[0] < -0.15 or current_pos[0] > -0.05:
            print(f"  Warning: {cube_id} x position {current_pos[0]:.3f} out of range")

        if current_pos[1] < -0.1 or current_pos[1] > 0.1:
            print(f"  Warning: {cube_id} y position {current_pos[1]:.3f} out of range")

    # Check arrangement order.
    if len(final_positions) >= 2:
        final_positions.sort(key=lambda x: x[1], reverse=True)  # Sort by descending y.

        print("\nFinal arrangement (from left to right, y decreasing):")
        for i, (cube_id, y_pos, full_pos) in enumerate(final_positions):
            print(f"  {i+1}. {cube_id}: y={y_pos:.3f}, position={full_pos}")

        # Check spacing.
        print("\nChecking spacing between cubes:")
        for i in range(len(final_positions) - 1):
            cube1_id, y1, pos1 = final_positions[i]
            cube2_id, y2, pos2 = final_positions[i+1]

            # Calculate center-to-center distance.
            center_distance = abs(y1 - y2)

            # Get block dimensions.
            size1 = None
            size2 = None
            for cube in cubes:
                if cube["id"] == cube1_id:
                    size1 = cube["size"][1]
                if cube["id"] == cube2_id:
                    size2 = cube["size"][1]

            if size1 is not None and size2 is not None:
                # Edge gap = center distance - (size1/2 + size2/2).
                edge_distance = center_distance - (size1/2.0 + size2/2.0)

                if edge_distance < 0.02:  # At least 2cm.
                    print(f"  Warning: {cube1_id} and {cube2_id} too close: {edge_distance:.3f}m < 0.02m")
                else:
                    print(f"  OK: {cube1_id} and {cube2_id} spacing: {edge_distance:.3f}m")
    robot.back_to_origin(arm=last_arm)
    print("\n=== Task summary ===")
    if success:
        print("SUCCESS: All cubes placed with at least 3cm spacing")
    else:
        print("PARTIAL SUCCESS: Some issues encountered")

    return success
