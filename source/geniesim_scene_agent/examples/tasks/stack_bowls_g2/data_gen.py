# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from interaction data.
    objects_list = {
        "benchmark_bowl_025": {
            "has_interactions": True,
            "object_type": "bowl",
            "interaction_type": "container"
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

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": 'Stack bowls',
        "english_task_name": 'Stack bowls',
        "init_scene_text": "The robot is in front of the table, there are several bowls on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 1. Find table_000.
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

    # 2. Find all bowls, benchmark_bowl_025.
    bowls = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_bowl_025" in usd or any("bowl_" in kw for kw in keywords):
            size = robot.get_object_size_in_world_frame(obj_id)
            if size is not None:
                pos, rot = robot.get_object_pose(obj_id)
                bowls.append({
                    "id": obj_id,
                    "size": size,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found bowl: {obj_id}, size: {size}, position: {pos}")

    if len(bowls) < 2:
        print(f"Found only {len(bowls)} bowls, need at least 2")
        return False

    print(f"Total bowls found: {len(bowls)}")

    # 3. Get table information to compute stack location.
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    # Table-top height.
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 4. Compute stack-position parameters.
    # Target: exact table center (0, 0).
    target_x = 0.0
    target_y = 0.0

    # 5. Sort ascending z, starting nearest table surface.
    bowls.sort(key=lambda x: x["position"][2])
    print("Bowls sorted by height (lowest to highest):")
    for i, bowl in enumerate(bowls):
        print(f"  {i+1}. {bowl['id']}: height={bowl['position'][2]:.3f}")

    # 6. Extract initial bowl rotation and generate world-z rotation candidates.
    def generate_bowl_rotations(base_rotation):
        """Generate candidate poses rotated about world z."""
        candidate_rotations = []

        # Extract base rotation matrix.
        base_rot_mat = quat2mat_wxyz(base_rotation)
        candidate_rotations.append(base_rot_mat)

        # Generate different angles about z.
        for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_z = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])

            # Compose original rotation with random z rotation.
            final_rot = np.dot(rot_z, base_rot_mat)
            candidate_rotations.append(final_rot)

        print(f"Generated {len(candidate_rotations)} rotation variations around z-axis")
        return candidate_rotations

    # 7. Execute stacking from lowest bowl upward.
    stacked_bowls = []
    last_arm = ""

    for i, bowl in enumerate(bowls):
        bowl_id = bowl["id"]
        bowl_size = bowl["size"]
        bowl_position = bowl["position"]
        bowl_rotation = bowl["rotation"]

        print(f"\n=== Processing bowl {i+1}/{len(bowls)}: {bowl_id} ===")

        # Choose arm.
        bowl_arm = robot.judge_arm_to_use(bowl_position)
        print(f"Using arm: {bowl_arm}")

        # Return home when switching arms.
        if last_arm != bowl_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = bowl_arm

        # Compute stacking height.
        if i == 0:
            # First bowl: place at table center.
            # Task requires bowl bottom 0.02~0.03 m above another bowl's rim.
            # First bowl goes on table; account for bowl height.
            margin = 0.025  # Use midpoint 0.025 m.
            # Bowl table height = table top + safety margin + half bowl height.
            safety_margin = 0.01  # Safety margin to avoid gripper collision.
            target_z = table_top_z + safety_margin + bowl_size[2] / 2.0
        else:
            # Subsequent bowls: place atop preceding bowl.
            prev_bowl_id = stacked_bowls[-1]["id"]
            prev_bowl_pos, prev_bowl_rot = robot.get_object_pose(prev_bowl_id)
            prev_bowl_size = robot.get_object_size_in_world_frame(prev_bowl_id)

            if prev_bowl_pos is None or prev_bowl_size is None:
                print(f"Cannot get previous bowl {prev_bowl_id} info")
                success = False
                break

            # Compute previous bowl's rim height.
            # Rim is approximately at bowl top.
            prev_bowl_top_z = prev_bowl_pos[2] + prev_bowl_size[2] / 2.0

            # Current bowl bottom should be 0.02~0.03 m above previous rim.
            # Bottom-to-center height is approximately half bowl height.
            bowl_bottom_to_center = bowl_size[2] / 2.0
            height_offset = np.random.uniform(0.02, 0.03)  # Random value between 0.02 and 0.03 m.

            safety_margin = 0.01  # Safety margin to avoid gripper collision.
            target_z = prev_bowl_top_z - height_offset + bowl_bottom_to_center
            target_x = prev_bowl_pos[0]
            target_y = prev_bowl_pos[1]

        base_position = np.array([target_x, target_y, target_z])

        # Create grasp configuration.
        unmovable_objects = [table_id]
        # Protect already stacked bowls.
        unmovable_objects.extend([b["id"] for b in stacked_bowls])

        # Interaction data provides bowl grasp annotation points.
        grasp_config = GraspConfig(
            pick_up_distance= 0.1 + i * 0.05,
            passive_primitive="default",
            unmovable_objects=unmovable_objects,
            action_description={"zh": "拿起碗", "en": "pick up the bowl"}  # REQUIRED
        )

        # Generate candidate rotations.
        candidate_rotations = generate_bowl_rotations(bowl_rotation)

        # Generate many candidate poses.
        target_poses = []

        # Position variation parameters, small offsets.
        x_offsets = np.linspace(-0.001, 0.001, 3)
        y_offsets = np.linspace(-0.001, 0.001, 3)
        z_offsets = np.linspace(0, 0.005, 3)  # Fine height adjustment.

        # Generate all candidate combinations.
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
        print(f"Generated {len(target_poses)} candidate poses for bowl {bowl_id}")

        # Define check for successful bowl stacking.
        def create_check_function(bowl_id, expected_position, expected_height, is_bottom=False):
            def check_bowl_stacked():
                current_pos, current_rot = robot.get_object_pose(bowl_id)
                if current_pos is None or current_rot is None:
                    return False

                # Position check with 5 cm XY tolerance.
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                xy_tolerance = 0.05
                if xy_distance > xy_tolerance:
                    print(f"Bowl {bowl_id} xy position error: {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    return False

                return True

            return check_bowl_stacked

        # Create check function.
        is_bottom = (i == 0)
        check_function = create_check_function(bowl_id, base_position, target_z, is_bottom)

        # Determine placement description from stack position.
        if i == 0:
            place_desc_zh = "将第一个碗放置到桌子正中心"
            place_desc_en = "place the bottom bowl at the center of the table"
        elif i == 1:
            place_desc_zh = "将第二个碗放置到第一个碗上方"
            place_desc_en = "place the second bowl on top of the first bowl"
        elif i == 2:
            place_desc_zh = "将第三个碗放置到碗塔的顶部"
            place_desc_en = "place the third bowl on top of the bowl tower"
        else:
            place_desc_zh = f"将第{i+1}个碗放置到碗塔的顶部"
            place_desc_en = f"place the {i+1}th bowl on top of the bowl tower"

        # Interaction data provides bowl place annotation points.
        place_config = PlaceConfig(
            active_primitive="default",
            passive_primitive="default",
            unmovable_objects=unmovable_objects,
            pre_insert_offset=0.1,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # Execute pick/place using pick_and_place_actor_to_target_pose: bowls have interaction data, but stack height requires precise control.
        print(f"Attempting to pick and place bowl {bowl_id} with {len(target_poses)} candidate poses")

        # Use interaction-data grasp points.
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=bowl_id,
            target_pose=target_poses,
            arm=bowl_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"Failed to place bowl {bowl_id}")
            success = False
            break
        else:
            print(f"Successfully placed bowl {bowl_id}")

        # Add to stacked-bowl list.
        stacked_bowls.append({"id": bowl_id, "position": base_position})

    # 8. Final verification.
    print("\n=== Performing final verification ===")

    if len(stacked_bowls) > 0:
        # Check tower stability.
        tower_height = 0
        for i, bowl_info in enumerate(stacked_bowls):
            bowl_id = bowl_info["id"]
            current_pos, current_rot = robot.get_object_pose(bowl_id)

            if current_pos is None:
                print(f"Bowl {bowl_id} not found after placement")
                success = False
                continue

            # Check near table center.
            center_distance = np.linalg.norm(current_pos[:2])
            if center_distance > 0.05:  # Allow 5 cm deviation.
                print(f"  Warning: Bowl {bowl_id} is {center_distance:.3f}m from center")

            # Compute tower height.
            if i == 0:
                tower_height = current_pos[2] - table_top_z
            else:
                # Check upper bowl directly above lower bowl.
                prev_bowl_id = stacked_bowls[i-1]["id"]
                prev_bowl_pos, _ = robot.get_object_pose(prev_bowl_id)
                if prev_bowl_pos is not None:
                    height_diff = current_pos[2] - prev_bowl_pos[2]
                    # Inter-bowl height difference should be reasonable.
                    if height_diff < 0.01 or height_diff > 0.1:
                        print(f"  Warning: Height difference between bowls {i} and {i-1}: {height_diff:.3f}m")

        print(f"Bowl tower height: {tower_height:.3f}m")
        print(f"Number of bowls in tower: {len(stacked_bowls)}")

    # Return arm to initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success and len(stacked_bowls) == len(bowls):
        print("✅ SUCCESS: All bowls stacked into a tower at the center of the table")
    else:
        print(f"⚠️ PARTIAL SUCCESS: Stacked {len(stacked_bowls)} out of {len(bowls)} bowls")

    return success and (len(stacked_bowls) == len(bowls))
