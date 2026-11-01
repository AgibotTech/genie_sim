# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from the supplied interaction data.
    objects_list = {
        "benchmark_storage_box_001": {"has_interactions": True, "object_type": "storage_box"},
        "benchmark_storage_box_004": {"has_interactions": True, "object_type": "storage_box"},
        "benchmark_storage_box_005": {"has_interactions": True, "object_type": "storage_box"},
        "benchmark_storage_box_003": {"has_interactions": True, "object_type": "storage_box"},
        "benchmark_building_blocks_057": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_005": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_car_018": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_plane_029": {"has_interactions": True, "object_type": "toy"},
        "benchmark_building_blocks_029": {"has_interactions": True, "object_type": "toy"},
        "benchmark_building_blocks_028": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_008": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_021": {"has_interactions": True, "object_type": "toy"},
        "benchmark_rubik_cube_001": {"has_interactions": True, "object_type": "toy"},
        "benchmark_garage_kit_064": {"has_interactions": True, "object_type": "toy"},
        "benchmark_chess_001": {"has_interactions": True, "object_type": "toy"},
        "benchmark_toy_018": {"has_interactions": True, "object_type": "toy"}
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "put all toys into the storage box",
        "english_task_name": "put all toys into the storage box",
        "init_scene_text": "The robot is in front of the table, there is a storage box and several toys on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if any("table" in kw.lower() for kw in keywords) or obj_info.get("usd", "") == "table_000":
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find the storage box.
    storage_box_id = None
    storage_box_usd = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Identify the storage box by its usd value.
        is_storage_box = False
        for box_usd in ["benchmark_storage_box_001", "benchmark_storage_box_004",
                        "benchmark_storage_box_005", "benchmark_storage_box_003"]:
            if box_usd in usd:
                is_storage_box = True
                storage_box_usd = box_usd
                break

        if is_storage_box or any("storage_box" in kw.lower() for kw in keywords):
            storage_box_id = obj_id
            break

    if not storage_box_id:
        print("Storage box not found")
        return False

    print(f"Found storage box: {storage_box_id} (usd: {storage_box_usd})")

    # 3. Find all toys, excluding the storage box itself.
    all_toys = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Exclude the storage box and table.
        if obj_id == storage_box_id or obj_id == table_id:
            continue

        # Identify toys by their usd values.
        is_toy = False
        for toy_usd in list(objects_list.keys()):
            if toy_usd in ["benchmark_storage_box_001", "benchmark_storage_box_004",
                          "benchmark_storage_box_005", "benchmark_storage_box_003"]:
                continue  # Skip the storage box.
            if toy_usd in usd:
                is_toy = True
                break

        # Keywords can also identify toys; the scene generator may assign "toy".
        if not is_toy and any("toy" in kw.lower() for kw in keywords):
            is_toy = True

        if is_toy:
            toy_pos, toy_rot = robot.get_object_pose(obj_id)
            all_toys.append({
                "id": obj_id,
                "usd": usd,
                "position": toy_pos,
                "rotation": toy_rot,
                "keywords": keywords
            })
            print(f"Found toy: {obj_id}, usd: {usd}, keywords: {keywords}")

    if len(all_toys) == 0:
        print("No toys found")
        return True  # No toys remain to process, so the task succeeds.

    print(f"Total toys found: {len(all_toys)}")

    # 4. Use keywords to identify toys already inside the box.
    # The generator labels toys inside the box with "toy_in_box_{i}".
    # Toys on the table have "toy_on_desk_{i}", "small", or "toy" keywords.
    toys_in_box = []
    toys_on_desk = []

    for toy in all_toys:
        toy_id = toy["id"]
        keywords = toy["keywords"]

        # Check whether the toy is inside the box.
        is_in_box = False
        for kw in keywords:
            if "toy_in_box" in kw:
                is_in_box = True
                break

        if is_in_box:
            toys_in_box.append(toy)
            print(f"Toy {toy_id} is already in box (keyword: {[kw for kw in keywords if 'inside_storage' in kw]})")
        else:
            toys_on_desk.append(toy)
            print(f"Toy {toy_id} is on desk (needs to be placed in box)")

    # 5. Return success immediately if there are no toys to process.
    if len(toys_on_desk) == 0:
        print("All toys are already in the storage box")
        return True

    print(f"Toys already in box: {len(toys_in_box)}")
    print(f"Toys to process (on desk): {len(toys_on_desk)}")

    # 6. Read the storage box position and dimensions.
    storage_box_pos, storage_box_rot = robot.get_object_pose(storage_box_id)
    storage_box_size = robot.get_object_size_in_world_frame(storage_box_id)

    if storage_box_pos is None or storage_box_size is None:
        print("Cannot get storage box position/size")
        return False

    # Compute the storage box's top height.
    storage_box_top_z = storage_box_pos[2] + storage_box_size[2] / 2.0
    print(f"Storage box top z: {storage_box_top_z:.3f}")

    # 7. Process each toy that must be put into the box.
    last_arm = ""

    for i, toy in enumerate(toys_on_desk):
        toy_id = toy["id"]
        toy_usd = toy["usd"]
        toy_pos = toy["position"]

        print(f"\n=== Processing toy {i+1}/{len(toys_on_desk)}: {toy_id} ===")

        # Choose the working arm.
        arm = robot.judge_arm_to_use(toy_pos)
        print(f"Using arm: {arm}")

        # Return the previous arm to its origin before switching arms.
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        # Protect the table, storage box, and toys already placed inside it.
        unmovable_objects = [table_id, storage_box_id]

        # Choose an action description based on the toy type.
        toy_desc_zh = "玩具"
        toy_desc_en = "toy"

        # Extract toy type information from usd or keywords.
        if "building_blocks" in toy_usd:
            toy_desc_zh = "积木玩具"
            toy_desc_en = "building block toy"
        elif "toy_car" in toy_usd:
            toy_desc_zh = "玩具车"
            toy_desc_en = "toy car"
        elif "toy_plane" in toy_usd:
            toy_desc_zh = "玩具飞机"
            toy_desc_en = "toy plane"
        elif "rubik_cube" in toy_usd:
            toy_desc_zh = "魔方"
            toy_desc_en = "Rubik's cube"
        elif "garage_kit" in toy_usd:
            toy_desc_zh = "小火龙摆件"
            toy_desc_en = "charmander figure"
        elif "chess" in toy_usd:
            toy_desc_zh = "国际象棋"
            toy_desc_en = "chess piece"

        # Grasp configuration.
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects.copy(),
            action_description={"zh": f"拿起{toy_desc_zh}", "en": f"picks up the {toy_desc_en}"}  # REQUIRED
        )

        # Execute the grasp.
        print(f"Attempting to pick up toy {toy_id}")
        pick_success = robot.pick_actor(toy_id, arm=arm, grasp_config=grasp_config)

        if not pick_success:
            print(f"Failed to pick up toy {toy_id}")
            success = False
            continue

        # Read the toy dimensions.
        toy_size = robot.get_object_size_in_world_frame(toy_id)
        if toy_size is None:
            print(f"Cannot get toy size for {toy_id}")
            success = False
            # Try returning the toy to its original position.
            robot.back_to_origin(arm=arm)
            continue

        # Compute the placement position.
        # The stated 0.03 cm clearance appears to mean 0.03 m = 3 cm between the toy bottom and box top.
        # Note: 0.03 cm is 0.0003 m, which is too small; the intended clearance is 0.03 m = 3 cm.
        # The context indicates a 3 cm safety clearance.
        min_clearance = 0.03  # 3cm clearance

        # Toy bottom-to-center distance = toy_size[2] / 2.0.
        # Placement Z = storage_box_top_z + min_clearance + toy_size[2] / 2.0.
        placement_z = storage_box_top_z + min_clearance + toy_size[2] / 2.0

        # Generate placement candidates with small X and Y offsets.
        target_poses = []

        # Place near the center while allowing small offsets.
        x_offsets = np.linspace(-0.02, 0.02, 5)  # +/-2 cm X offset.
        y_offsets = np.linspace(-0.02, 0.02, 5)  # +/-2 cm Y offset.

        # Use the storage box rotation matrix as the base orientation.
        base_rot_matrix = quat2mat_wxyz(storage_box_rot)

        # The task allows arbitrary toy rotations about all three axes.
        # Generate multiple random orientations.
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                target_position = storage_box_pos.copy()
                target_position[0] += x_offset
                target_position[1] += y_offset
                target_position[2] = placement_z

                rxs = np.linspace(0, 2*np.pi, 6)
                rys = np.linspace(0, 2*np.pi, 6)
                rzs = np.linspace(0, 2*np.pi, 12)

                for rx in rxs:
                    for ry in rys:
                        for rz in rzs:
                            rot_x = euler2mat([rx, 0, 0])
                            rot_y = euler2mat([0, ry, 0])
                            rot_z = euler2mat([0, 0, rz])
                            final_rot = np.dot(rot_z, np.dot(rot_y, rot_x))
                            final_rot = np.dot(base_rot_matrix, final_rot)
                            pose = np.eye(4)
                            pose[:3, :3] = final_rot
                            pose[:3, 3] = target_position
                            target_poses.append(pose)

        if len(target_poses) == 0:
            print(f"No valid target poses generated for toy {toy_id}")
            success = False
            robot.back_to_origin(arm=arm)
            continue

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for placement")

        # Define a checker for successful placement inside the box.
        def create_toy_in_box_checker(toy_id, box_id):
            def checker():
                toy_pos, toy_rot = robot.get_object_pose(toy_id)
                toy_size = robot.get_object_size_in_local_frame(toy_id)
                box_pos, box_rot = robot.get_object_pose(box_id)
                box_size = robot.get_object_size_in_local_frame(box_id)

                if toy_pos is None or box_pos is None:
                    return False

                # Check the toy's XY projection against the box, ignoring Z.
                box_contains_toy, _ = check_box_containment(
                    box_pos, box_rot, box_size,
                    toy_pos, toy_rot, toy_size,
                    ignore_axes=["z"],
                    overlap_threshold=0.4
                )

                return box_contains_toy

            return checker

        # Placement configuration.
        place_desc_zh = f"将{toy_desc_zh}放置到收纳盒"
        place_desc_en = f"places the {toy_desc_en} in the storage box"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects.copy(),
            extra_checkers=[create_toy_in_box_checker(toy_id, storage_box_id)],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # Execute pose-based placement.
        print(f"Attempting to place toy {toy_id} above storage box")
        place_success = robot.place_actor_to_target_pose(
            actor=toy_id,
            arm=arm,
            target_pose=target_poses,
            place_config=place_config
        )

        if not place_success:
            print(f"Failed to place toy {toy_id} above storage box")
            success = False
        else:
            print(f"Successfully placed toy {toy_id} above storage box")

    # 8. Final validation.
    print("\n=== Performing final verification ===")

    # Recheck the physical positions of all toys.
    final_toys_in_box = []
    final_toys_not_in_box = []

    for toy in toys_on_desk:
        toy_id = toy["id"]
        toy_pos, toy_rot = robot.get_object_pose(toy_id)
        toy_size = robot.get_object_size_in_local_frame(toy_id)
        box_pos, box_rot = robot.get_object_pose(storage_box_id)
        box_size = robot.get_object_size_in_local_frame(storage_box_id)

        if toy_pos is None or box_pos is None:
            final_toys_not_in_box.append(toy_id)
            print(f"  ❓ Cannot verify toy {toy_id} (missing position data)")
            continue

        # Use a physical-position check.
        box_contains_toy, _ = check_box_containment(
            box_pos, box_rot, box_size,
            toy_pos, toy_rot, toy_size,
            ignore_axes=["z"],
            overlap_threshold=0.4
        )

        if box_contains_toy:
            final_toys_in_box.append(toy_id)
            print(f"  ✅ Toy {toy_id} is in storage box (physical check)")
        else:
            final_toys_not_in_box.append(toy_id)
            print(f"  ❌ Toy {toy_id} is NOT in storage box (physical check)")

    print(f"\nFinal count: {len(final_toys_in_box)} toys in box, {len(final_toys_not_in_box)} toys not in box")

    # Return the arm to its initial position.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if len(final_toys_not_in_box) == 0:
        print("✅ SUCCESS: All toys placed into storage box")
        return True
    else:
        print(f"❌ FAILED: {len(final_toys_not_in_box)} toys are not in the storage box")
        return False
