# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## Initialize objects_list.
    objects_list = {
        "benchmark_beverage_bottle_001": {
            "has_interactions": True,
            "object_type": "beverage",
            "beverage_type": "coca_cola_can"
        },
        "benchmark_beverage_bottle_002": {
            "has_interactions": True,
            "object_type": "beverage",
            "beverage_type": "sprite_can"
        },
        "benchmark_beverage_bottle_003": {
            "has_interactions": True,
            "object_type": "beverage",
            "beverage_type": "fanta_can"
        },
        "benchmark_beverage_bottle_029": {
            "has_interactions": True,
            "object_type": "beverage",
            "beverage_type": "wang_lao_ji_can"
        },
        "table_000": {
            "has_interactions": False,
            "object_type": "table"
        }
    }

    ## Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": 'Sort beverages by type',
        "english_task_name": 'Sort beverages by type',
        "init_scene_text": "The robot is in front of the table, there are 3-5 canned beverages of two types randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if usd == "table_000":
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    ## 2. Find all beverages.
    beverages = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        ## Check whether object is a beverage.
        is_beverage = False
        beverage_type = None
        for bev_usd, bev_info in objects_list.items():
            if bev_usd in usd and bev_info.get("object_type") == "beverage":
                is_beverage = True
                beverage_type = bev_info.get("beverage_type")
                break

        if is_beverage or any("can" in kw.lower() or "beverage" in kw.lower() for kw in keywords):
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)

            ## Determine beverage type.
            if beverage_type is None:
                if "coca_cola" in usd.lower() or any("coke" in kw.lower() for kw in keywords):
                    beverage_type = "coca_cola_can"
                elif "sprite" in usd.lower() or any("sprite" in kw.lower() for kw in keywords):
                    beverage_type = "sprite_can"
                elif "fanta" in usd.lower() or any("fanta" in kw.lower() for kw in keywords):
                    beverage_type = "fanta_can"
                elif "wang_lao_ji" in usd.lower() or any("wang" in kw.lower() for kw in keywords):
                    beverage_type = "wang_lao_ji_can"
                else:
                    beverage_type = "unknown"

            beverages.append({
                "id": obj_id,
                "usd": usd,
                "position": pos,
                "rotation": rot,
                "size": size,
                "type": beverage_type,
                "keywords": keywords
            })
            print(f"Found beverage: {obj_id}, type: {beverage_type}, position: {pos}")

    if len(beverages) < 2:
        print(f"Found only {len(beverages)} beverages, need at least 2")
        return False

    print(f"Total beverages found: {len(beverages)}")

    ## 3. Classify beverages by type.
    type_groups = {}
    for bev in beverages:
        bev_type = bev["type"]
        if bev_type not in type_groups:
            type_groups[bev_type] = []
        type_groups[bev_type].append(bev)

    print(f"Beverage types: {list(type_groups.keys())}")
    for bev_type, bev_list in type_groups.items():
        print(f"  {bev_type}: {len(bev_list)} items")

    if len(type_groups) < 2:
        print("Warning: Less than 2 beverage types found")

    ## 4. Plan placement for both beverage types.
    ## Region 1: x(-0.2 ~ -0.1), y(-0.15, -0.03).
    region1 = {
        "x_range": (-0.2, -0.1),
        "y_range": (-0.15, -0.03),
        "center": np.array([-0.15, -0.09])
    }

    ## Region 2: x(-0.2 ~ -0.1), y(0.03, 0.15).
    region2 = {
        "x_range": (-0.2, -0.1),
        "y_range": (0.03, 0.15),
        "center": np.array([-0.15, 0.09])
    }

    ## Get table height.
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    def plan_positions(num_items, region):
        """
        Plan beverage placement:
        - One: region center.
        - Two: one row, identical X.
        - Three: triangle.
        - Four: two rows, quadrilateral.
        """
        positions = []
        min_spacing = 0.1

        if num_items == 1:
            ## Place near region center, X toward front (smaller X is farther forward), Y centered.
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y = region["center"][1]
            positions.append(np.array([x, y]))

        elif num_items == 2:
            ## Arrange one row with identical X, separated Y.
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]
            y_offset = min_spacing / 2.0
            positions.append(np.array([x, y_center - y_offset]))
            positions.append(np.array([x, y_center + y_offset]))

        elif num_items == 3:
            ## Arrange a triangle.
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            ## Two lower positions.
            positions.append(np.array([x, y_center - min_spacing / 2.0]))
            positions.append(np.array([x, y_center + min_spacing / 2.0]))
            ## One upper position.
            positions.append(np.array([x + min_spacing * 0.866, y_center]))

        elif num_items == 4:
            ## Arrange two rows forming quadrilateral.
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            ## First/front row.
            positions.append(np.array([x, y_center - min_spacing / 2.0]))
            positions.append(np.array([x, y_center + min_spacing / 2.0]))
            ## Second/rear row.
            positions.append(np.array([x + min_spacing, y_center - min_spacing / 2.0]))
            positions.append(np.array([x + min_spacing, y_center + min_spacing / 2.0]))

        else:
            ## For larger counts, use grid arrangement.
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            for i in range(num_items):
                row = i // 2
                col = i % 2
                x_pos = x + row * min_spacing
                y_pos = y_center + (col - 0.5) * min_spacing
                positions.append(np.array([x_pos, y_pos]))

        return positions

    ## Plan target positions for each beverage type.
    type_positions = {}
    type_regions = {}

    type_index = 0
    for bev_type, bev_list in type_groups.items():
        if type_index == 0:
            region = region1
        else:
            region = region2

        num_items = len(bev_list)
        positions = plan_positions(num_items, region)
        type_positions[bev_type] = positions
        type_regions[bev_type] = region

        print(f"\nType {bev_type} ({num_items} items) will be placed in region:")
        print(f"  x_range: {region['x_range']}, y_range: {region['y_range']}")
        for i, pos in enumerate(positions):
            print(f"    Position {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}]")

        type_index += 1

    ## 5. Sort beverages by grasp order, smaller X first.
    all_beverages_sorted = sorted(beverages, key=lambda b: b["position"][0])

    print("\nBeverages sorted by x (front to back):")
    for i, bev in enumerate(all_beverages_sorted):
        print(f"  {i+1}. {bev['id']} (type: {bev['type']}), x={bev['position'][0]:.3f}")

    ## 6. Execute grasp and placement.
    last_arm = ""
    processed_beverages = set()

    for bev_index, bev in enumerate(all_beverages_sorted):
        bev_id = bev["id"]
        bev_type = bev["type"]
        bev_pos = bev["position"]
        bev_rot = bev["rotation"]
        bev_size = bev["size"]

        if bev_id in processed_beverages:
            continue

        print(f"\n=== Processing beverage {bev_index+1}/{len(all_beverages_sorted)}: {bev_id} (type: {bev_type}) ===")

        ## Choose arm.
        arm = robot.judge_arm_to_use(bev_pos)
        print(f"Using arm: {arm}")

        ## Return home when switching arms.
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        ## Get initial rotation angle.
        initial_rot_matrix = quat2mat_wxyz(bev_rot)

        ## Plan target position.
        if bev_type not in type_positions:
            print(f"Warning: Unknown beverage type {bev_type}, skipping")
            continue

        target_positions = type_positions[bev_type]

        ## Find current beverage index in list.
        type_beverages = [b for b in beverages if b["type"] == bev_type and b["id"] not in processed_beverages]
        if len(type_beverages) == 0:
            continue

        target_xy = target_positions.pop(0)

        ## Compute target z 0.03~0.05 above initial z.
        height_offset = np.random.uniform(0.03, 0.05)
        target_z = table_top_z + height_offset + bev_size[2] / 2.0

        ## Generate 12 candidate rotation angles sampled over 0~2*pi.
        rotation_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        ## Generate candidate poses.
        target_poses = []

        ## Position variation with small offsets.
        x_offsets = np.linspace(-0.001, 0.001, 3)
        y_offsets = np.linspace(-0.001, 0.001, 3)
        z_offsets = np.linspace(0, 0.005, 3)

        for x_offset in x_offsets:
            for y_offset in y_offsets:
                for z_offset in z_offsets:
                    position = np.array([
                        target_xy[0] + x_offset,
                        target_xy[1] + y_offset,
                        target_z + z_offset
                    ])

                    for rot_angle in rotation_angles:
                        ## Get initial rotation, then rotate by arbitrary angle about z.
                        rot_z = euler2mat([0, 0, rot_angle])
                        final_rot = np.dot(rot_z, initial_rot_matrix)

                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = position
                        target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for beverage {bev_id}")

        ## Prepare grasp configuration.
        unmovable_objects = [table_id]
        unmovable_objects.extend(processed_beverages)

        ## Determine description by beverage type.
        bev_desc_zh = ""
        bev_desc_en = ""
        if bev_type == "coca_cola_can":
            bev_desc_zh = "可乐罐"
            bev_desc_en = "Coca Cola can"
        elif bev_type == "sprite_can":
            bev_desc_zh = "雪碧罐"
            bev_desc_en = "Sprite can"
        elif bev_type == "fanta_can":
            bev_desc_zh = "芬达罐"
            bev_desc_en = "Fanta can"
        elif bev_type == "wang_lao_ji_can":
            bev_desc_zh = "王老吉罐"
            bev_desc_en = "Wang Lao Ji can"
        else:
            bev_desc_zh = "饮料罐"
            bev_desc_en = "beverage can"

        grasp_config = GraspConfig(
            grasp_lower_percentile=75,
            grasp_upper_percentile=95,
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{bev_desc_zh}", "en": f"picks up the {bev_desc_en}"}
        )

        ## Define check: beverage successfully placed in correct region.
        def create_beverage_checker(bev_id, target_xy, region):
            def checker():
                current_pos, current_rot = robot.get_object_pose(bev_id)
                if current_pos is None:
                    return False

                ## Check XY near target region.
                x_range = region["x_range"]
                y_range = region["y_range"]

                x_in_range = x_range[0] - 0.05 <= current_pos[0] <= x_range[1] + 0.05
                y_in_range = y_range[0] - 0.05 <= current_pos[1] <= y_range[1] + 0.05

                if not x_in_range or not y_in_range:
                    print(f"  Beverage {bev_id} not in target region")
                    print(f"    Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
                    print(f"    Target region: x[{x_range[0]:.3f}, {x_range[1]:.3f}], y[{y_range[0]:.3f}, {y_range[1]:.3f}]")
                    return False

                ## Check correct z height.
                if current_pos[2] < table_top_z + 0.02:
                    print(f"  Beverage {bev_id} too low: z={current_pos[2]:.3f}")
                    return False

                return True

            return checker

        target_region = type_regions.get(bev_type, region1)
        region_desc_zh = "左侧" if target_region == region2 else "右侧"
        region_desc_en = "left side" if target_region == region2 else "right side"

        place_desc_zh = f"将{bev_desc_zh}放置到{region_desc_zh}"
        place_desc_en = f"places the {bev_desc_en} to the {region_desc_en}"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_beverage_checker(bev_id, target_xy, target_region)],
            action_description={"zh": place_desc_zh, "en": place_desc_en}
        )

        ## Execute pick and place.
        print(f"Attempting to pick and place beverage {bev_id} with {len(target_poses)} candidate poses")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=bev_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"Failed to place beverage {bev_id}")
            success = False
        else:
            print(f"Successfully placed beverage {bev_id}")
            processed_beverages.add(bev_id)

    ## 7. Final verification.
    print("\n=== Performing final verification ===")

    ## Verify all beverages correctly classified and placed.
    type_region_mapping = {}
    for bev_type, region in type_regions.items():
        type_region_mapping[bev_type] = region

    for bev in beverages:
        bev_id = bev["id"]
        bev_type = bev["type"]
        current_pos, current_rot = robot.get_object_pose(bev_id)

        if current_pos is None:
            print(f"  Beverage {bev_id} not found")
            success = False
            continue

        expected_region = type_region_mapping.get(bev_type)
        if expected_region is None:
            continue

        x_range = expected_region["x_range"]
        y_range = expected_region["y_range"]

        x_in_range = x_range[0] - 0.05 <= current_pos[0] <= x_range[1] + 0.05
        y_in_range = y_range[0] - 0.05 <= current_pos[1] <= y_range[1] + 0.05

        if x_in_range and y_in_range:
            print(f"  ✓ Beverage {bev_id} (type: {bev_type}) correctly placed")
        else:
            print(f"  ✗ Beverage {bev_id} (type: {bev_type}) NOT in correct region")
            print(f"    Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
            print(f"    Expected region: x[{x_range[0]:.3f}, {x_range[1]:.3f}], y[{y_range[0]:.3f}, {y_range[1]:.3f}]")
            success = False
        rot_mat = quat2mat_wxyz(current_rot)
        local_y_axis = rot_mat[:, 1]
        world_z_axis = np.array([0, 0, 1])
        dot_product = np.dot(local_y_axis, world_z_axis)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.rad2deg(angle_rad)

        if angle_deg > 10.0:
            print(f"Verification failed: {bev_id} local Y is tilted {angle_deg:.2f} degrees (target < 10)")
            success = False

    ## Return arm to initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: All beverages sorted into two piles by type")
    else:
        print("⚠️ PARTIAL SUCCESS: Some beverages not correctly placed")

    return success
