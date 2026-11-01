# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def execute_task(robot: RobotAPI):
    ## Initialize objects_list with every asset that may appear in scene.
    objects_list = {
        "benchmark_beverage_bottle_001": {"zh": "可口可乐", "en": "coke cola"},
        "benchmark_beverage_bottle_003": {"zh": "芬达", "en": "fanta"},
        "benchmark_beverage_bottle_004": {
            "zh": "乌龙茶饮料",
            "en": "oolong tea beverage",
        },
        "genie_beverage_bottle_009": {"zh": "葡萄汁", "en": "grape juice"},
        "benchmark_toy_car_018": {"zh": "玩具车", "en": "toy car"},
        "benchmark_building_blocks_072": {"zh": "积木", "en": "building blocks"},
        "benchmark_cup_043": {"zh": "马克杯", "en": "mug"},
        "benchmark_cup_105": {"zh": "纸杯", "en": "paper cup"},
        "benchmark_toiletries_009": {"zh": "洗面奶", "en": "face cleanser"},
        "benchmark_stationery_009": {"zh": "黄色订书机", "en": "yellow stapler"},
        "benchmark_mouse_000": {"zh": "鼠标", "en": "mouse"},
        "benchmark_bottle_017": {"zh": "保温杯", "en": "water bottle"},
        "benchmark_apple_002": {"zh": "苹果", "en": "apple"},
        "benchmark_orange_004": {"zh": "橙子", "en": "orange"},
        "benchmark_lemon_030": {"zh": "柠檬", "en": "lemon"},
        "benchmark_peach_021": {"zh": "桃子", "en": "peach"},
        "benchmark_green_apple_001": {"zh": "青苹果", "en": "green apple"},
        "benchmark_bowl_sponge_000": {"zh": "洗碗海绵", "en": "bowl sponge"},
        "benchmark_clock_044": {"zh": "闹钟", "en": "clock"},
        "benchmark_tape_measure_007": {"zh": "卷尺", "en": "tape measure"},
        "benchmark_toy_plane_029": {"zh": "玩具飞机", "en": "toy plane"},
        "benchmark_rubik_cube_001": {"zh": "魔方", "en": "rubik cube"},
        "benchmark_medicine_bottle_000": {"zh": "药瓶", "en": "medicine bottle"},
        "benchmark_garage_kit_064": {"zh": "小火龙摆件", "en": "charmander figure"},
    }

    ## Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## 1. Find table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if "table_000" in obj_info.get("usd", "") or any(
            "table" in kw.lower() for kw in keywords
        ):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    ## 2. Find all objects and identify TARGET objects.
    all_objects = []
    target_objects = []

    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == table_id:
            continue

        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        ## Check whether object is TARGET.
        is_target = "target" in keywords

        pos, rot = robot.get_object_pose(obj_id)
        all_objects.append(
            {
                "id": obj_id,
                "usd": usd,
                "position": pos,
                "keywords": keywords,
                "is_target": is_target,
            }
        )

        if is_target:
            target_objects.append(
                {"id": obj_id, "usd": usd, "position": pos, "keywords": keywords}
            )

    if len(target_objects) < 2:
        print(f"Found {len(target_objects)} targets, need at least 2")
        return False

    print(f"Found {len(all_objects)} objects, including {len(target_objects)} targets")

    ## 3. Choose one TARGET randomly to grasp, using the other for comparison.
    import random

    target_idx_to_pick = random.choice([0, 1])
    target_pick = target_objects[target_idx_to_pick]
    target_other = target_objects[1 - target_idx_to_pick]

    target_pick_id = target_pick["id"]
    target_other_id = target_other["id"]
    pos_pick = target_pick["position"]
    pos_other = target_other["position"]

    print(f"Target to pick: {target_pick_id}")
    print(f"Other target: {target_other_id}")

    ## 4. Generate unambiguous position descriptions.
    ## Threshold: distance difference must exceed this value for an unambiguous description.
    DISTANCE_THRESHOLD = 0.08  # 8cm

    candidates = []

    ## Helper: get object's Chinese name.
    def get_obj_name(obj_id, keywords, usd):
        # First try matching USD.
        if usd in objects_list:
            return objects_list[usd]["zh"], objects_list[usd]["en"]
        # Fallback: search keywords.
        for kw in keywords:
            if kw not in ["target", "object"]:
                return kw, kw
        return "物体", "object"

    ## Get table information.
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    table_center_x = table_pos[0]
    table_center_y = table_pos[1]
    table_left_y = table_pos[1] + table_size[1] / 2.0  # World Y is left
    table_right_y = table_pos[1] - table_size[1] / 2.0

    ## -- Table-relative descriptions --

    ## 4.1 Left/right position by Y coordinate.
    # Distance from table's left edge.
    dist_pick_left = abs(pos_pick[1] - table_left_y)
    dist_other_left = abs(pos_other[1] - table_left_y)

    if (
        abs(dist_pick_left - dist_other_left) > DISTANCE_THRESHOLD
        and dist_pick_left < 0.1
    ):
        if dist_pick_left < dist_other_left:
            candidates.append(
                {
                    "zh": f"靠近桌子左侧边缘的",
                    "en": "closer to the left edge of the table",
                }
            )

    dist_pick_right = abs(pos_pick[1] - table_right_y)
    dist_other_right = abs(pos_other[1] - table_right_y)
    if (
        abs(dist_pick_right - dist_other_right) > DISTANCE_THRESHOLD
        and dist_pick_right < 0.1
    ):
        if dist_pick_right < dist_other_right:
            candidates.append(
                {
                    "zh": f"靠近桌子右侧边缘的",
                    "en": "closer to the right edge of the table",
                }
            )

    # Absolute left/right position relative to table center.
    dy = pos_pick[1] - pos_other[1]
    if abs(dy) > DISTANCE_THRESHOLD:
        if dy > 0:  # pick is left of other, with larger Y.
            candidates.append(
                {
                    "zh": "位于桌子偏左侧的",
                    "en": "located on the left side of the table",
                }
            )
        else:
            candidates.append(
                {
                    "zh": "位于桌子偏右侧的",
                    "en": "located on the right side of the table",
                }
            )

    ## 4.2 Front/back and robot distance by X coordinate.
    # Robot assumed in front of table, negative X; larger X is farther from robot.
    if abs(pos_pick[0] - pos_other[0]) > DISTANCE_THRESHOLD:
        if pos_pick[0] > pos_other[0]:
            candidates.append({"zh": "更远离机器人的", "en": "farther from the robot"})
        else:
            candidates.append({"zh": "更靠近机器人的", "en": "closer to the robot"})

    # Distance from table center.
    dist_pick_center = np.linalg.norm(pos_pick[:2] - table_pos[:2])
    dist_other_center = np.linalg.norm(pos_other[:2] - table_pos[:2])

    if (
        abs(dist_pick_center - dist_other_center) > DISTANCE_THRESHOLD
        and dist_pick_center < 0.15
    ):
        if dist_pick_center < dist_other_center:
            candidates.append(
                {"zh": "靠近桌子中心的", "en": "closer to the center of the table"}
            )
        else:
            candidates.append(
                {"zh": "远离桌子中心的", "en": "farther from the center of the table"}
            )

    ## -- Descriptions relative to other objects --
    ## Additional requirement: reference object must be unique.

    # Count each object type in scene.
    usd_counts = {}
    for obj in all_objects:
        usd = obj["usd"]
        usd_counts[usd] = usd_counts.get(usd, 0) + 1

    # Reference candidates must be unique and different from target to grasp.
    potential_references = []
    for obj in all_objects:
        if obj["id"] == target_pick_id:
            continue
        # Skip other target as reference; theoretically possible, but prefer other objects for semantic clarity.
        if obj["is_target"]:
            continue

        # Check whether object type is unique.
        if usd_counts[obj["usd"]] == 1:
            potential_references.append(obj)
        else:
            print(
                f"Skipping {obj['usd']} as reference because it is not unique (count: {usd_counts[obj['usd']]})"
            )

    for ref_obj in potential_references:
        ref_id = ref_obj["id"]
        ref_pos = ref_obj["position"]
        ref_name_zh, ref_name_en = get_obj_name(
            ref_id, ref_obj["keywords"], ref_obj["usd"]
        )

        # Compute Euclidean distance.
        dist_pick_ref = np.linalg.norm(pos_pick - ref_pos)
        dist_other_ref = np.linalg.norm(pos_other - ref_pos)

        # Distance description.
        if (
            abs(dist_pick_ref - dist_other_ref) > DISTANCE_THRESHOLD
        ):
            if dist_pick_ref < dist_other_ref:
                    candidates.append(
                        {
                            "zh": f"更靠近{ref_name_zh}的",
                            "en": f"closer to the {ref_name_en}",
                        }
                    )
                    if dist_other_ref > 0.2 and dist_pick_ref < 0.15:
                        candidates.append(
                            {
                                "zh": f"在{ref_name_zh}旁边的",
                                "en": f"next to the {ref_name_en}",
                            }
                        )
            else:
                candidates.append(
                    {
                        "zh": f"更远离{ref_name_zh}的",
                        "en": f"farther from the {ref_name_en}",
                    }
                )
                if dist_other_ref < 0.15 and dist_pick_ref > 0.2:
                    candidates.append(
                        {
                            "zh": f"远离{ref_name_zh}的",
                            "en": f"far from the {ref_name_en}",
                        }
                    )

        # Relative left/right description.
        dy_ref = pos_pick[1] - ref_pos[1]
        # Add only if left/right relation clearly distinguishes the two targets.
        dy_other_ref = pos_other[1] - ref_pos[1]
        dx_ref = pos_pick[0] - ref_pos[0]
        dx_other_ref = pos_other[0] - ref_pos[0]
        is_left_ref = 0.1 > dy_ref > 0 and abs(dx_ref) < 0.05
        is_right_ref = -0.1 < dy_ref < 0 and abs(dx_ref) < 0.05
        is_front_ref = -0.1 < dx_ref < 0 and abs(dy_ref) < 0.05
        is_back_ref = 0.1 > dx_ref > 0 and abs(dy_ref) < 0.05

        is_left_other = dy_other_ref > 0 and abs(dx_other_ref) < 0.05
        is_right_other = dy_other_ref < 0 and abs(dx_other_ref) < 0.05
        is_front_other = dx_other_ref < 0 and abs(dy_other_ref) < 0.05
        is_back_other = dx_other_ref > 0 and abs(dy_other_ref) < 0.05

        if is_left_ref and not is_left_other:
            candidates.append(
                {
                    "zh": f"在{ref_name_zh}左侧的",
                    "en": f"to the left of the {ref_name_en}",
                }
            )
        if is_right_ref and not is_right_other:
            candidates.append(
                {
                    "zh": f"在{ref_name_zh}右侧的",
                    "en": f"to the right of the {ref_name_en}",
                }
            )
        if is_front_ref and not is_front_other:
            candidates.append(
                {"zh": f"在{ref_name_zh}前方的", "en": f"in front of the {ref_name_en}"}
            )
        if is_back_ref and not is_back_other:
            candidates.append(
                {"zh": f"在{ref_name_zh}后方的", "en": f"behind the {ref_name_en}"}
            )

    ## 5. Randomly select a description.
    if not candidates:
        print(
            "No unique spatial description found significant enough difference, return False."
        )
        return False
    else:
        selected_desc = random.choice(candidates)
        print(f"Generated candidates: {candidates}")
        print(f"Number of candidates: {len(candidates)}")
        print(
            f"Selected description: ZH: {selected_desc['zh']}, EN: {selected_desc['en']}"
        )

    ## 6. Get target object's name.
    target_name_zh, target_name_en = get_obj_name(
        target_pick_id, target_pick["keywords"], target_pick["usd"]
    )

    ## Build complete action description.
    final_desc_zh = f"拿起{selected_desc['zh']}{target_name_zh}"
    final_desc_en = f"pick up the {selected_desc['en']}{target_name_en}"

    action_description = {"zh": final_desc_zh, "en": final_desc_en}
    print(f"Action Description: {action_description}")

    task_description = {
        "task_name": f"pick up the {selected_desc['en']}{target_name_en}",
        "english_task_name": f"pick up the {selected_desc['en']}{target_name_en}",
        "init_scene_text": f"The robot is in front of the table, the table has some objects on it"
    }
    robot.client.set_task_task_basic_info(task_description)
    ## 7. Execute grasp.
    arm = robot.judge_arm_to_use(pos_pick)

    grasp_config = GraspConfig(action_description=action_description)

    print(f"Attempting to pick {target_pick_id} using {arm} arm")

    pick_success = robot.pick_actor(actor=target_pick_id, arm=arm, grasp_config=grasp_config)
    final_success = False
    if pick_success:
        print("Pick action executed by API (reported success). Verifying physical lift...")

        # Get object position after grasp.
        current_pos, _ = robot.get_object_pose(target_pick_id)
        initial_z = pos_pick[2]
        if current_pos is not None:
            current_z = current_pos[2]
            lift_height = current_z - initial_z

            print(f"Current Z position: {current_z:.4f}m")
            print(f"Lift height: {lift_height:.4f}m")

            # Check lift exceeds 3 cm (0.03 m).
            if lift_height > 0.03:
                print(f"✅ Verification SUCCESS: Object lifted by {lift_height:.4f}m (> 0.03m)")
                final_success = True
            else:
                print(f"❌ Verification FAILED: Object only lifted by {lift_height:.4f}m (<= 0.03m)")
        else:
            print("❌ Verification FAILED: Could not retrieve object position after pick.")
    else:
        print(f"❌ API reported failed pick for {target_pick_id}")

    return final_success
