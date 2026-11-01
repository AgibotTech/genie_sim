# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from scene generation and interaction data.
    objects_list = {
        "benchmark_cup_105": {
            "has_interactions": True,
            "object_type": "paper_cup",
            "interaction_data": {
                "active": {
                    "place": ["侧放", "正放"]
                },
                "passive": {
                    "place": ["正放"],
                    "grasp": ["default"]
                }
            }
        },
        "table_000": {
            "has_interactions": False,
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
        if usd == "table_000" or "table" in keywords or "dining_table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Locate benchmark_cup_105 paper cups and record initial states.
    cups = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether this is benchmark_cup_105.
        if "benchmark_cup_105" in usd or any("paper_cup" in kw.lower() for kw in keywords):
            pos, rot = robot.get_object_pose(obj_id)
            if pos is not None:
                # Calculate distance to table center.
                table_pos, _ = robot.get_object_pose(table_id)
                if table_pos is None:
                    distance_to_center = np.linalg.norm(pos[:2])
                else:
                    distance_to_center = np.linalg.norm(pos[:2] - table_pos[:2])

                # Get initial rotation matrix and upward axis.
                initial_rot_matrix = quat2mat_wxyz(rot)

                # Determine the local axis closest to world z.
                world_z = np.array([0, 0, 1])
                col_0 = initial_rot_matrix[:, 0]
                col_1 = initial_rot_matrix[:, 1]
                col_2 = initial_rot_matrix[:, 2]

                dot_0 = np.abs(np.dot(col_0, world_z))
                dot_1 = np.abs(np.dot(col_1, world_z))
                dot_2 = np.abs(np.dot(col_2, world_z))

                # Choose the axis closest to world z as upward.
                dots = [dot_0, dot_1, dot_2]
                up_axis_idx = np.argmax(dots)
                initial_up_axis = initial_rot_matrix[:, up_axis_idx]

                # Get cup size.
                cup_size = robot.get_object_size_in_world_frame(obj_id)
                if cup_size is None:
                    print(f"Warning: Cannot get size for cup {obj_id}")
                    cup_size = np.array([0.08, 0.08, 0.12])  # Default size.

                cups.append({
                    "id": obj_id,
                    "position": pos,
                    "rotation": rot,
                    "size": cup_size,
                    "initial_rot_matrix": initial_rot_matrix,
                    "initial_up_axis": initial_up_axis,
                    "up_axis_idx": up_axis_idx,
                    "distance": distance_to_center,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found cup: {obj_id}, distance={distance_to_center:.3f}m, size={cup_size}")

    if len(cups) < 3:
        print(f"Found only {len(cups)} cups, need at least 3")
        return False

    print(f"\nTotal cups found: {len(cups)}")

    # 3. Sort nearest to farthest.
    cups.sort(key=lambda x: x["distance"])
    print("Cups sorted by distance from center (nearest to farthest):")
    for i, cup in enumerate(cups):
        print(f"  {i+1}. {cup['id']}: distance={cup['distance']:.3f}m, size={cup['size']}")

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": 'Stack paper cups',
        "english_task_name": 'Stack paper cups',
        "init_scene_text": "The robot is in front of the table, there are paper cups on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 4. Get table information for stack placement.
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    # Tabletop height.
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 5. Stack nearest to farthest.
    placed_cups = []
    last_arm = ""

    # Target stack position: table center (0, 0).
    base_x = 0.0
    base_y = 0.0

    # Record the first cup's final position for later alignment.
    first_cup_final_pos = None
    first_cup_final_size = None
    last_cup_final_pos = None


    for i, cup_info in enumerate(cups):
        cup_id = cup_info["id"]
        cup_pos = cup_info["position"]
        cup_size = cup_info["size"]
        cup_distance = cup_info["distance"]
        initial_rot_matrix = cup_info["initial_rot_matrix"]
        initial_up_axis = cup_info["initial_up_axis"]
        up_axis_idx = cup_info["up_axis_idx"]

        print(f"\n=== Processing cup {i+1}/{len(cups)}: {cup_id} ===")
        print(f"  Distance: {cup_distance:.3f}m")
        print(f"  Size: {cup_size}")
        print(f"  Initial up axis: {initial_up_axis} (idx: {up_axis_idx})")

        # Choose the arm.
        cube_arm = robot.judge_arm_to_use(cup_pos)
        print(f"Using arm: {cube_arm}")

        # Return to origin when switching arms.
        if last_arm != cube_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = cube_arm

        # Calculate placement for nested stacking.
        if i == 0:
            # First cup: near table center, allowing small offsets.
            margin = 0.01  # 1cm safety margin.
            base_z = table_top_z + margin + cup_size[2] / 2.0

            # Use a small random offset for the first cup.
            x_offset = np.random.uniform(-0.02, 0.02)
            y_offset = np.random.uniform(-0.02, 0.02)
            base_position = np.array([base_x + x_offset, base_y + y_offset, base_z])

            print(f"  First cup offset: x={x_offset:.3f}m, y={y_offset:.3f}m")
        else:
            # Later cups: nest into the preceding cup.
            # Calculate insertion depth.
            margin = 0.01
            insertion_depth = cup_size[2] * 0.3
            base_z = last_cup_final_pos[2] - insertion_depth + cup_size[2] + margin
            base_position = last_cup_final_pos.copy()
            base_position[2] = base_z

        print(f"  Target position: [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]")

        # Create grasp configuration.
        unmovable_objects = [table_id]
        unmovable_objects.extend([c["id"] for c in placed_cups])

        # Describe by distance and stack order.
        if i == 0:
            cup_desc_zh = "离中心最近的纸杯"
            cup_desc_en = "cup closest to the center"
        elif i == 1:
            cup_desc_zh = "第二近的纸杯"
            cup_desc_en = "second closest cup"
        else:
            cup_desc_zh = f"第{i+1}个纸杯"
            cup_desc_en = f"{i+1}th cup"

        grasp_config = GraspConfig(
            passive_primitive="default",  # Use the paper cup's default grasp point [1].
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{cup_desc_zh}", "en": f"pick up the {cup_desc_en}"}
        )

        # Generate candidate orientations by rotating the initial rotation about world z.
        target_poses = []

        # Position variation parameters; nested stacking needs precise alignment.
        if i == 0:
            # First cup: small position variation.
            x_offsets = np.linspace(-0.003, 0.003, 3)
            y_offsets = np.linspace(-0.003, 0.003, 3)
            position_offsets = []
            for dx in x_offsets:
                for dy in y_offsets:
                    position_offsets.append(np.array([dx, dy, 0]))
        else:
            # Later cups: insertion needs very precise alignment.
            position_offsets = [
                np.array([0, 0, 0]),
            ]

        # Fine z adjustment is important for insertion.
        if i == 0:
            z_offsets = np.linspace(0, 0.002, 3)
        else:
            # Use a smaller z adjustment range when inserting.
            z_offsets = np.linspace(-0.001, 0.001, 3)

        # Generate several world-z rotation angles.
        rotation_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

        # Generate all candidate-pose combinations.
        for pos_offset in position_offsets:
            for z_offset in z_offsets:
                position = base_position + pos_offset + np.array([0, 0, z_offset])

                for z_angle in rotation_angles:
                    # Rotate about world z.
                    rot_z = euler2mat([0, 0, z_angle])

                    # Compose world-z rotation with initial orientation.
                    final_rot = np.dot(rot_z, initial_rot_matrix)

                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"  Generated {len(target_poses)} candidate poses")

        # Define a checker specialized for nested stacking.
        def create_check_function(cup_id, expected_position, cup_size, initial_up_axis,
                                  up_axis_idx, placed_cup_ids, is_first=False):
            def check_cup_stacked():
                # Get current cup state.
                current_pos, current_rot = robot.get_object_pose(cup_id)
                if current_pos is None or current_rot is None:
                    print(f"    Error: Cannot get cup state")
                    return False

                # Get current rotation matrix.
                current_rot_matrix = quat2mat_wxyz(current_rot)
                current_up_axis = current_rot_matrix[:, up_axis_idx]

                # 1. Check upward axis remains upward using its angle to world z.
                world_z = np.array([0, 0, 1])
                current_up_dot = np.abs(np.dot(current_up_axis, world_z))

                # Check deviation from initial upward axis.
                up_axis_dot = np.dot(current_up_axis, initial_up_axis)
                up_axis_dot = np.clip(up_axis_dot, -1.0, 1.0)
                up_axis_angle = np.arccos(abs(up_axis_dot))

                # Check upward orientation: angle to z less than 30 degrees.
                if current_up_dot < np.cos(np.deg2rad(30)):
                    print(f"    Error: Cup not upright enough, dot={current_up_dot:.3f}")
                    return False

                # Check axis change: less than 45 degrees from initial.
                if up_axis_angle > np.deg2rad(45):
                    print(f"    Error: Up axis changed too much: {np.rad2deg(up_axis_angle):.1f}°")
                    return False

                # 2. Check XY position.
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                xy_tolerance = 0.03 if is_first else 0.01  # Insertion requires greater precision.

                if xy_distance > xy_tolerance:
                    print(f"    Error: XY position off by {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    return False

                # 3. For later cups, verify actual nesting.
                if not is_first and len(placed_cup_ids) > 0:
                    # Get the preceding receiving cup.
                    prev_cup_id = placed_cup_ids[-1]
                    prev_cup_pos, prev_cup_rot = robot.get_object_pose(prev_cup_id)
                    prev_cup_size = robot.get_object_size_in_world_frame(prev_cup_id)

                    if prev_cup_pos is not None and prev_cup_size is not None:
                        # Check center alignment required for insertion.
                        center_distance_xy = np.linalg.norm(current_pos[:2] - prev_cup_pos[:2])
                        if center_distance_xy > 0.015:  # 1.5cm tolerance.
                            print(f"    Error: Not centered for insertion")
                            print(f"      Center offset: {center_distance_xy:.3f}m")
                            return False
                        # Height check.
                        height_diff = current_pos[2] - prev_cup_pos[2]
                        if height_diff < 0.001:
                            print(f"    Error: Not inserted properly - current cup should be below the previous cup")
                            print(f"      Current cup top: {current_pos[2]:.3f}m")
                            print(f"      Previous cup top: {prev_cup_pos[2]:.3f}m")
                            print(f"      Difference: {height_diff:.3f}m")
                            return False

                print(f"    Check passed:")
                print(f"      XY error: {xy_distance:.3f}m")
                # print(f"      Z error: {abs(current_pos[2] - expected_z):.3f}m")
                print(f"      Up axis deviation: {np.rad2deg(up_axis_angle):.1f}°")
                print(f"      Uprightness (dot): {current_up_dot:.3f}")
                return True

            return check_cup_stacked

        # Create the checker.
        is_first = (i == 0)
        placed_cup_ids = [c["id"] for c in placed_cups]
        check_function = create_check_function(cup_id, base_position, cup_size,
                                              initial_up_axis, up_axis_idx,
                                              placed_cup_ids, is_first)

        # Describe placement.
        if i == 0:
            place_desc_zh = "将离中心最近的纸杯放置到桌子中心附近"
            place_desc_en = "place the cup closest to the center near the table center"
        elif i == 1:
            place_desc_zh = "将第二近的纸杯插入第一个纸杯中"
            place_desc_en = "insert the second closest cup into the first cup"
        else:
            place_desc_zh = f"将第{i+1}个纸杯插入纸杯塔中"
            place_desc_en = f"insert the {i+1}th cup into the cup tower"

        # Use pre_insert_offset for the insertion motion.
        place_config = PlaceConfig(
            active_primitive="正放",  # Use upright placement point [1].
            pre_insert_offset=0.1,    # 8cm pre-placement offset for insertion.
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}
        )

        # Execute grasp and placement.
        print(f"  Executing pick and place...")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=cup_id,
            target_pose=target_poses,
            arm=cube_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"  ❌ Failed to place cup {cup_id}")
            success = False
            break
        else:
            print(f"  ✅ Successfully placed cup {cup_id}")
            cup_final_pos, _ = robot.get_object_pose(cup_id)
            last_cup_final_pos = cup_final_pos

            # Record first cup's final position and size.
            if i == 0:
                first_cup_final_pos, _ = robot.get_object_pose(cup_id)
                first_cup_final_size = robot.get_object_size_in_world_frame(cup_id)
                print(f"  First cup final position: {first_cup_final_pos}")

        # Add to placed list.
        placed_cups.append({
            "id": cup_id,
            "position": base_position,
            "size": cup_size,
            "initial_up_axis": initial_up_axis,
            "up_axis_idx": up_axis_idx
        })

    # 6. Final detailed verification.
    print("\n" + "="*60)
    print("DETAILED FINAL VERIFICATION - INSERTION STACKING")
    print("="*60)

    all_checks_passed = True

    if len(placed_cups) > 0:
        print(f"\nChecking {len(placed_cups)} inserted cups:")

        for i, cup_info in enumerate(placed_cups):
            cup_id = cup_info["id"]
            cup_size = cup_info["size"]
            initial_up_axis = cup_info["initial_up_axis"]
            up_axis_idx = cup_info["up_axis_idx"]

            current_pos, current_rot = robot.get_object_pose(cup_id)
            if current_pos is None:
                print(f"Cup {i+1} ({cup_id}): ❌ Not found")
                all_checks_passed = False
                continue

            # Get current rotation matrix.
            current_rot_matrix = quat2mat_wxyz(current_rot)
            current_up_axis = current_rot_matrix[:, up_axis_idx]

            # Check upward axis.
            world_z = np.array([0, 0, 1])
            current_up_dot = np.abs(np.dot(current_up_axis, world_z))

            # Check deviation from initial upward axis.
            up_axis_dot = np.dot(current_up_axis, initial_up_axis)
            up_axis_dot = np.clip(up_axis_dot, -1.0, 1.0)
            up_axis_angle = np.arccos(abs(up_axis_dot))

            # Check position.
            center_distance = np.linalg.norm(current_pos[:2])

            # Check insertion state.
            insertion_ok = True
            insertion_info = ""

            if i > 0:
                # Check nesting into the preceding cup.
                prev_cup_id = placed_cups[i-1]["id"]
                prev_cup_pos, _ = robot.get_object_pose(prev_cup_id)
                prev_cup_size = robot.get_object_size_in_world_frame(prev_cup_id)

                if prev_cup_pos is not None and prev_cup_size is not None:
                    # Calculate height relationship.
                    current_cup_top = current_pos[2] + cup_size[2] / 2.0
                    prev_cup_top = prev_cup_pos[2] + prev_cup_size[2] / 2.0

                    # For nesting, current cup top should be near or below preceding cup top.
                    top_diff = current_cup_top - prev_cup_top

                    # Check center alignment.
                    center_offset = np.linalg.norm(current_pos[:2] - prev_cup_pos[:2])

                    insertion_ok = (top_diff <= 0.05) and (center_offset <= 0.02)
                    insertion_info = f"Top diff: {top_diff:.3f}m, Center offset: {center_offset:.3f}m"

            # Combined assessment.
            upright_ok = current_up_dot > np.cos(np.deg2rad(30))
            axis_ok = up_axis_angle < np.deg2rad(45)
            position_ok = center_distance < 0.03

            cup_status = "✅" if (upright_ok and axis_ok and position_ok and (insertion_ok if i > 0 else True)) else "⚠️"

            print(f"Cup {i+1} ({cup_id}): {cup_status}")
            print(f"  Position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
            print(f"  Height above table: {current_pos[2] - table_top_z:.3f}m")
            print(f"  Center distance: {center_distance:.3f}m {'OK' if position_ok else 'TOO FAR'}")
            print(f"  Uprightness (dot): {current_up_dot:.3f} {'OK' if upright_ok else 'NOT UPRIGHT'}")
            print(f"  Up axis deviation: {np.rad2deg(up_axis_angle):.1f}° {'OK' if axis_ok else 'TOO LARGE'}")

            if i > 0:
                print(f"  Insertion: {'OK' if insertion_ok else 'NOT PROPERLY INSERTED'}")
                if insertion_info:
                    print(f"    {insertion_info}")

            if not (upright_ok and axis_ok and position_ok and (insertion_ok if i > 0 else True)):
                all_checks_passed = False

    # Return the arm to its initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n" + "="*60)
    print("TASK SUMMARY - INSERTION STACKING")
    print("="*60)

    if success and len(placed_cups) == len(cups) and all_checks_passed:
        print("✅ SUCCESS: All cups properly inserted and stacked!")
        print(f"   Stacking order (by distance from center):")
        for i, cup in enumerate(cups):
            print(f"     {i+1}. {cup['id']}: {cup['distance']:.3f}m")
        print(f"   Total cups inserted: {len(placed_cups)}")
    elif success and len(placed_cups) == len(cups) and not all_checks_passed:
        print("⚠️ PARTIAL: All cups placed but some insertion checks failed")
        print(f"   This may indicate shallow insertion or alignment issues")
    else:
        print(f"❌ PARTIAL: Inserted {len(placed_cups)}/{len(cups)} cups")

    return success and (len(placed_cups) == len(cups)) and all_checks_passed
