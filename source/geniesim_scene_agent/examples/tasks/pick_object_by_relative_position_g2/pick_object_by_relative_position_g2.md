# Pick an object by its relative position

**Tags:** `pick_only` `dynamic_task_description` `spatial_relation` `tags_target` `disambiguate_by_position` `table_edges` `left_right_by_y` `closer_to_robot` `reference_object` `unique_reference` `DISTANCE_THRESHOLD` `action_description` `set_task_task_basic_info_before_action` `GraspConfig` `pick_actor` `objects_list_zh_en` `get_object_pose` `table_size`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find two targets (keywords contain "target" or tags_target). Get table pose/size for edges (left/right by y, or center). Get both targets' positions.
2. **Disambiguate:** Build spatial description that uniquely identifies one target: e.g. "closer to the left edge" (compare y or distance to table edge), "to the left of the apple" (compare y), "closer to the robot" (compare x). Use DISTANCE_THRESHOLD if needed.
3. **Task description:** Set task_task_basic_info before action: e.g. "拿起靠近桌子左侧边缘的苹果" / "pick up the object closer to the left edge of the table". Use objects_list_zh_en for names.
4. **Execution:** pick_actor(selected_target_id, arm, GraspConfig with action_description). No place step.

---

# INPUT

**scene_description:** A table with multiple objects. Exactly two of them are marked as "target" in their keywords (or tags). The task is to pick one of the two targets; the choice is arbitrary (e.g. random). The robot must generate a spatial description that uniquely identifies the chosen target relative to the other (e.g. "closer to the left edge of the table", "to the left of the apple", "closer to the robot").

**task description:** Pick up the object that matches the generated spatial description. The task description and action description must be set at runtime after computing the disambiguating description (e.g. "pick up the object closer to the left edge of the table" / "拿起靠近桌子左侧边缘的苹果"). Use table geometry (center, left/right edges) and optionally other unique reference objects to build descriptions that differentiate the two targets.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: pick_object_by_relative_position_g2
description: A table with 3-6 randomly selected objects placed within a specified range.
One type of object appears twice and is marked as 'target'. All objects are randomly rotated,
non-colliding (min distance 0.15m), and labeled with both Chinese and English names.
"""


def get_desktop_workspace(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the desktop in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def sample_non_colliding_positions(
    num_objects: int,
    workspace_center: P,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_dist: float = 0.15,
) -> List[P]:
    """
    Sample non-colliding positions for objects on the table.
    """
    positions = []
    max_attempts = 1000

    for i in range(num_objects):
        for _ in range(max_attempts):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            z = workspace_center[2]
            candidate = np.array([x, y, z])

            # Check distance to all previously placed objects
            collision = False
            for pos in positions:
                if np.linalg.norm(candidate[:2] - pos[:2]) < min_dist:
                    collision = True
                    break

            if not collision:
                positions.append(candidate)
                break
        else:
            # If we can't find a non-colliding position, just place it (should rarely happen with given constraints)
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            positions.append(np.array([x, y, workspace_center[2]]))

    return positions


@register()
def create_tabletop_object(oid: str, zh_name: str, en_name: str, is_target: bool = False) -> Shape:
    """
    Create a single object with proper labeling.
    """
    keywords = [en_name.replace(" ", "_"), "object", en_name, zh_name]
    if is_target:
        keywords.append("target")

    return library_call("usd", oid=oid, keywords=keywords)


@register()
def place_objects_on_table(desk_shape: Shape) -> Shape:
    # Available objects with their Chinese and English names
    available_objects = {
        "benchmark_beverage_bottle_001": {"zh": "可口可乐", "en": "coke cola"},
        "benchmark_beverage_bottle_003": {"zh": "芬达", "en": "fanta"},
        "benchmark_beverage_bottle_004": {"zh": "乌龙茶饮料", "en": "oolong tea beverage"},
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

    # Determine total number of objects (3-6)
    total_objects = random.randint(3, 6)

    # Select which object will appear twice (target object)
    target_oid = random.choice(list(available_objects.keys()))
    target_info = available_objects[target_oid]

    # Create list of object IDs to place
    object_ids = [target_oid, target_oid]  # Two instances of target object

    # Fill remaining slots with unique objects (excluding the target)
    remaining_slots = total_objects - 2
    other_objects = [oid for oid in available_objects.keys() if oid != target_oid]
    for _ in range(remaining_slots):
        object_ids.append(random.choice(other_objects))

    # Get desktop workspace
    workspace_center, workspace_size = get_desktop_workspace(desk_shape)

    # Define placement range (x: -0.2 to 0.2, y: -0.25 to 0.25 relative to workspace center)
    x_min = workspace_center[0] - 0.2
    x_max = workspace_center[0] + 0.2
    y_min = workspace_center[1] - 0.25
    y_max = workspace_center[1] + 0.25

    # Sample non-colliding positions
    positions = sample_non_colliding_positions(len(object_ids), workspace_center, (x_min, x_max), (y_min, y_max))

    # Create and place all objects
    placed_objects = []
    for i, oid in enumerate(object_ids):
        is_target = oid == target_oid
        obj_shape = library_call(
            "create_tabletop_object",
            oid=oid,
            zh_name=available_objects[oid]["zh"],
            en_name=available_objects[oid]["en"],
            is_target=is_target,
        )

        # Get object size for proper placement
        obj_info = get_object_info(obj_shape)
        obj_height = obj_info["size"][2]

        # Position at sampled location (adjusting for object height if needed)
        placement_pos = positions[i].copy()
        placement_pos[2] = workspace_center[2] - 0.1  # Place on table surface

        # Apply translation
        obj_shape = transform_shape(obj_shape, translation_matrix(placement_pos))

        # Apply random rotation around Z-axis
        obj_center = compute_shape_center(obj_shape)
        rotation_angle = random.uniform(0, 2 * math.pi)
        obj_shape = transform_shape(obj_shape, rotation_matrix(rotation_angle, direction=(0, 0, 1), point=obj_center))

        placed_objects.append(obj_shape)

    return concat_shapes(desk_shape, *placed_objects)


@register()
def pick_object_by_relative_position_g2() -> Shape:
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_objects_on_table(desk_shape)


@register()
def root_scene() -> Shape:
    return pick_object_by_relative_position_g2()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def execute_task(robot: RobotAPI):
    ## 初始化objects_list，包含场景中可能出现的所有资产
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

    ## 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## 1. 查找桌子
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

    ## 2. 查找所有物体，并识别TARGET物体
    all_objects = []
    target_objects = []

    for obj_id, obj_info in scene_info["layout"].items():
        if obj_id == table_id:
            continue

        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        ## 判断是否是TARGET
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

    ## 3. 选择要抓取的TARGET (随机选择一个作为目标，另一个作为对比)
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

    ## 4. 生成无歧义的位置描述
    ## 定义阈值：距离差超过这个值才认为描述是“无歧义”的
    DISTANCE_THRESHOLD = 0.08  # 8cm

    candidates = []

    ## 辅助函数：获取物体的中文名称
    def get_obj_name(obj_id, keywords, usd):
        # 首先尝试从usd匹配
        if usd in objects_list:
            return objects_list[usd]["zh"], objects_list[usd]["en"]
        # Fallback: 从keywords中查找
        for kw in keywords:
            if kw not in ["target", "object"]:
                return kw, kw
        return "物体", "object"

    ## 获取桌子信息
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    table_center_x = table_pos[0]
    table_center_y = table_pos[1]
    table_left_y = table_pos[1] + table_size[1] / 2.0  # World Y is left
    table_right_y = table_pos[1] - table_size[1] / 2.0

    ## --- 基于桌子的描述 ---

    ## 4.1 左右位置 (基于Y坐标)
    # 距离桌子左侧边缘的距离
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

    # 绝对左右位置 (相对于桌子中心)
    dy = pos_pick[1] - pos_other[1]
    if abs(dy) > DISTANCE_THRESHOLD:
        if dy > 0:  # pick在other的左边 (Y更大)
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

    ## 4.2 前后/距离机器人位置 (基于X坐标)
    # 距离机器人远近 (假设机器人在桌子前方，即X负方向。X越大越远离机器人)
    if abs(pos_pick[0] - pos_other[0]) > DISTANCE_THRESHOLD:
        if pos_pick[0] > pos_other[0]:
            candidates.append({"zh": "更远离机器人的", "en": "farther from the robot"})
        else:
            candidates.append({"zh": "更靠近机器人的", "en": "closer to the robot"})

    # 距离桌子中心的距离
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

    ## --- 基于其他物体的描述 ---
    ## 新增要求：确保参考物体是唯一的

    # 统计场景中每种物体的数量
    usd_counts = {}
    for obj in all_objects:
        usd = obj["usd"]
        usd_counts[usd] = usd_counts.get(usd, 0) + 1

    # 筛选可以作为参考的物体（必须是唯一的，且不是要抓取的target）
    potential_references = []
    for obj in all_objects:
        if obj["id"] == target_pick_id:
            continue
        # 跳过另一个target作为参考（虽然理论上也可以，但为了语义清晰，优先使用其他物体）
        if obj["is_target"]:
            continue

        # 检查该物体类型是否唯一
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

        # 计算欧几里得距离
        dist_pick_ref = np.linalg.norm(pos_pick - ref_pos)
        dist_other_ref = np.linalg.norm(pos_other - ref_pos)

        # 距离描述
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

        # 相对位置描述 (左右)
        dy_ref = pos_pick[1] - ref_pos[1]
        # 只有当该左右关系对于两个target有明显区别时才添加
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

    ## 5. 随机选择一个描述
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

    ## 6. 获取目标物体的名称
    target_name_zh, target_name_en = get_obj_name(
        target_pick_id, target_pick["keywords"], target_pick["usd"]
    )

    ## 构建完整的动作描述
    final_desc_zh = f"拿起{selected_desc['zh']}{target_name_zh}"
    final_desc_en = f"pick up the {selected_desc['en']}{target_name_en}"

    action_description = {"zh": final_desc_zh, "en": final_desc_en}
    print(f"Action Description: {action_description}")

    task_description = {
        "task_name": f"拿起{selected_desc['zh']}{target_name_zh}",
        "english_task_name": f"pick up the {selected_desc['en']}{target_name_en}",
        "init_scene_text": f"The robot is in front of the table, the table has some objects on it"
    }
    robot.client.set_task_task_basic_info(task_description)
    ## 7. 执行抓取
    arm = robot.judge_arm_to_use(pos_pick)

    grasp_config = GraspConfig(action_description=action_description)

    print(f"Attempting to pick {target_pick_id} using {arm} arm")

    pick_success = robot.pick_actor(actor=target_pick_id, arm=arm, grasp_config=grasp_config)
    final_success = False
    if pick_success:
        print("Pick action executed by API (reported success). Verifying physical lift...")

        # 获取抓取后的物体位置
        current_pos, _ = robot.get_object_pose(target_pick_id)
        initial_z = pos_pick[2]
        if current_pos is not None:
            current_z = current_pos[2]
            lift_height = current_z - initial_z

            print(f"Current Z position: {current_z:.4f}m")
            print(f"Lift height: {lift_height:.4f}m")

            # 检查提升高度是否超过3cm (0.03m)
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
```
