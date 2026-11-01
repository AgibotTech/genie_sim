# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from provided interaction data.
    objects_list = {
        "benchmark_apple_001": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_green_apple_000": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_orange_004": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_fruit_021": {"has_interactions": True, "object_type": "fruit", "product": True},
        "benchmark_beverage_bottle_073": {"has_interactions": True, "object_type": "beverage", "product": True},
        "benchmark_handbag_004": {"has_interactions": False, "object_type": "handbag", "container": True}
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "place the product into the bag",
        "english_task_name": "place the product into the bag",
        "init_scene_text": "The robot is in front of the table, there is a handbag and a fruit product on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find bag/handbag.
    bag_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_handbag_004" in usd or any("handbag" in kw.lower() for kw in keywords):
            bag_id = obj_id
            print(f"Found bag: {obj_id}")
            break

    if not bag_id:
        print("Bag not found")
        return False

    # 3. Find products, fruit.
    product_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check known fruit product.
        if any(product_usd in usd for product_usd in ["benchmark_apple_001", "benchmark_green_apple_000", "benchmark_orange_004", "benchmark_fruit_021", "benchmark_beverage_bottle_073"]):
            product_id = obj_id
            print(f"Found product: {obj_id}")
            break

    if not product_id:
        print("Product not found")
        return False

    # 4. Choose arm.
    product_pos, product_rot = robot.get_object_pose(product_id)
    arm = robot.judge_arm_to_use(product_pos)
    print(f"Using arm: {arm}")

    # 5. Check interaction data.
    bag_interaction = interactions.get("benchmark_handbag_004", {})

    # Safer interaction-data access.
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == product_id:
            product_usd = obj_info.get("usd", "")
            product_interaction = interactions.get(product_usd, {})
            break

    bag_has_place = "passive" in bag_interaction and "place" in bag_interaction.get("passive", {})
    product_has_grasp = "passive" in product_interaction and "grasp" in product_interaction.get("passive", {})

    print(f"Bag has passive.place: {bag_has_place}")
    print(f"Product has passive.grasp: {product_has_grasp}")

    # 6. Prepare immovable-object list.
    unmovable_objects = [table_id, bag_id]

    # 7. Configure grasp parameters.
    product_desc_zh = ""
    product_desc_en = ""

    # Determine product-type description.
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == product_id:
            usd = obj_info.get("usd", "")
            if "apple" in usd.lower():
                product_desc_zh = "苹果"
                product_desc_en = "apple"
            elif "green" in usd.lower():
                product_desc_zh = "青苹果"
                product_desc_en = "green apple"
            elif "orange" in usd.lower():
                product_desc_zh = "橙子"
                product_desc_en = "orange"
            elif "fruit_021" in usd.lower():
                product_desc_zh = "香蕉"
                product_desc_en = "fruit"
            elif "beverage" in usd.lower():
                product_desc_zh = "红牛"
                product_desc_en = "red bull"
            break

    grasp_config = GraspConfig(
        pick_up_distance=0.1,
        unmovable_objects=unmovable_objects,
        action_description={"zh": f"拿起{product_desc_zh}", "en": f"picks up the {product_desc_en}"}
    )

    # 8. Define check: product successfully placed in bag.
    def create_product_in_bag_checker(product_id, bag_id):
        def checker():
            product_pos, product_rot = robot.get_object_pose(product_id)
            product_size = robot.get_object_size_in_local_frame(product_id)
            bag_pos, bag_rot = robot.get_object_pose(bag_id)
            bag_size = robot.get_object_size_in_local_frame(bag_id)

            if product_pos is None or bag_pos is None:
                return False

            # Check containment ignoring z, using XY projection only and allowing different heights.
            from common.base_utils.transform_utils import check_box_containment
            bag_contains_product, _ = check_box_containment(
                bag_pos, bag_rot, bag_size,
                product_pos, product_rot, product_size,
                overlap_threshold=0.7
            )

            return bag_contains_product

        return checker

    # 9. Execute placement.
    place_desc_zh = f"将{product_desc_zh}装进袋子里"
    place_desc_en = f"places the {product_desc_en} into the bag"
    print(f"Using place_actor_to_target_pose for bag {bag_id}")

    bag_pos, bag_rot = robot.get_object_pose(bag_id)
    bag_size = robot.get_object_size_in_world_frame(bag_id)
    product_size = robot.get_object_size_in_world_frame(product_id)

    # Generate multiple candidate positions inside bag.
    target_poses = []
    margin = 0.01  # Safety margin prevents gripper collision.

    # Generate multiple interior positions accounting for bag opening dimensions.
    x_offsets = np.linspace(-0.05, 0.05,  10)
    y_offsets = np.linspace(-0.05, 0.0, 5)
    z_offsets = np.linspace(0.008, 0.012, 3)

    rot_matrix = quat2mat_wxyz(bag_rot)
    product_rot_matrix = quat2mat_wxyz(product_rot)
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                target_position = bag_pos.copy()
                target_position[0] += x_offset
                target_position[1] += y_offset

                # Placement height = bag bottom + safety margin + product half-height.
                bag_top = bag_pos[2] + bag_size[2] / 2.0
                target_position[2] = bag_top  + z_offset + product_size[2] / 2.0

                # Generate multiple random rotations for each position.
                for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    pose = np.eye(4)
                    pose[:3, :3] = product_rot_matrix
                    pose[:3, 3] = target_position
                    pose = rotate_along_axis(pose, angle, "z", use_local=False)
                    target_poses.append(pose)

    if len(target_poses) == 0:
        print(f"No valid target poses generated for product {product_id}")
        success = False
        return success

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses for product {product_id}")

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects,
        extra_checkers=[create_product_in_bag_checker(product_id, bag_id)],
        action_description={"zh": place_desc_zh, "en": place_desc_en}
    )
    pick_success = robot.pick_actor(product_id, arm=arm, grasp_config=grasp_config)
    if not pick_success:
        print(f"Failed to pick product {product_id}")
        success = False
        return success
    else:
        print(f"Successfully picked product {product_id}")

    place_success = robot.place_actor_to_target_pose(product_id, arm=arm, target_pose=target_poses, place_config=place_config)
    if not place_success:
        print(f"Failed to place product {product_id} into bag {bag_id}")
        success = False
        return success
    else:
        print(f"Successfully placed product {product_id} into bag {bag_id}")

    # Return to initial position.
    robot.back_to_origin(arm=arm)

    # 10. Final verification.
    print("\n=== Performing final verification ===")

    final_product_pos, _ = robot.get_object_pose(product_id)
    if final_product_pos is not None:
        in_bag = False
        bag_pos, bag_rot = robot.get_object_pose(bag_id)
        bag_size = robot.get_object_size_in_local_frame(bag_id)
        product_size = robot.get_object_size_in_local_frame(product_id)

        bag_contains_product, _ = check_box_containment(
            bag_pos, bag_rot, bag_size,
            final_product_pos, _, product_size,
            ignore_axes=["z"],
            overlap_threshold=0.7
        )

        if bag_contains_product:
            print(f"✅ Verification PASSED: Product is inside the bag")
        else:
            print(f"⚠️ Verification FAILED: Product may not be fully inside the bag")
            success = False
    else:
        print(f"⚠️ Cannot verify product position after placement")
        success = False

    if success:
        print("\n🎉 SUCCESS: Product successfully placed into the bag")
    else:
        print("\n❌ FAILED: Task execution encountered issues")

    return success
