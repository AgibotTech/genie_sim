# Place an object at a relative position

**Tags:** `place` `relative_position` `on_desk` `OBJECT_NAMES` `two_objects_selection` `reference_object` `target_pose_from_reference` `dynamic_task_description` `keywords_on_desk` `place_actor_to_target_pose` `GraspConfig` `PlaceConfig` `pick_and_place`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd or keywords. Collect desktop_objects (keyword "on_desk" or position near table); need at least 2. Select object_to_move and reference_object (e.g. random). Get names from OBJECT_NAMES[usd].
2. **Task description:** Set dynamic task_description: e.g. f"将{obj_zh}放置到{ref_zh}的{position_desc}" / "place {obj_en} to the {position_desc} of {ref_en}".
3. **Target pose:** Compute target position = reference_pos + offset (e.g. [0, offset_y, table_z + size/2 + margin]). Build multiple candidate poses (e.g. rotations around z).
4. **Execution:** pick_and_place_actor_to_target_pose(object_to_move_id, target_poses, arm, grasp_config, place_config) with GraspConfig/PlaceConfig.

---

# INPUT

**scene_description:** A table with multiple objects (from OBJECT_NAMES: beverages, toys, cups, fruits, etc.). Objects on the desk have keyword "on_desk". Two objects are selected (e.g. one to move, one as reference). The robot must place the first object at a position relative to the second (e.g. to the left of, next to, or at a specified offset).

**task description:** Place object A relative to object B (e.g. "place the apple to the left of the cup"). Task description and target pose are determined at runtime: select which object to move and which is reference; compute target pose from reference object's pose + offset (e.g. along Y for left/right).

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_random_objects
description: A white desk (table_000) with 3 to 6 unique randomly selected objects placed within a bounded region (x: -0.2 to 0.2, y: -0.25 to 0.25), ensuring no duplicates and minimum distance of 0.15m between objects.
"""


def get_desktop_placement_area(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the desktop for object placement.

    Args:
        desk_shape (Shape): The shape of the desk.

    Returns:
        Tuple[P, P]: (center_position, size) of the desktop's top surface in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Compute desktop top surface center in world frame
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def sample_position_within_bounds(center: P, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> P:
    """
    Sample a position within specified x and y bounds relative to a center point.
    """
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    return np.array([center[0] + x, center[1] + y, center[2]])


def check_minimum_distance(new_obj: Shape, existing_objs: List[Shape], min_dist: float = 0.15) -> bool:
    """
    Check if the new object is at least min_dist away from all existing objects (center-to-center distance).
    """
    new_center = compute_shape_center(new_obj)
    for obj in existing_objs:
        obj_center = compute_shape_center(obj)
        distance = np.linalg.norm(new_center[:2] - obj_center[:2])  # Only consider x,y distance
        if distance < min_dist:
            return False
    return True


# Object ID to semantic name mapping
OBJECT_SEMANTIC_NAMES = {
    "benchmark_beverage_bottle_001": "coke cola",
    "benchmark_beverage_bottle_003": "fanta",
    "benchmark_beverage_bottle_004": "oolong tea beverage",
    "genie_beverage_bottle_009": "grape juice",
    "benchmark_toy_car_018": "toy car",
    "benchmark_building_blocks_072": "building blocks",
    "benchmark_cup_043": "mug",
    "benchmark_cup_105": "paper cup",
    "benchmark_toiletries_009": "face cleanser",
    "benchmark_pen_006": "marker pen",
    "benchmark_stationery_009": "yellow stapler",
    "benchmark_mouse_000": "mouse",
    "benchmark_bottle_017": "water bottle",
    "benchmark_apple_002": "apple",
    "benchmark_orange_004": "orange",
    "benchmark_lemon_030": "lemon",
    "benchmark_peach_021": "peach",
    "benchmark_green_apple_001": "green apple",
    "benchmark_bowl_sponge_000": "bowl sponge",
    "benchmark_clock_044": "clock",
    "benchmark_tape_measure_007": "tape measure",
    "benchmark_toy_plane_029": "toy plane",
    "benchmark_rubik_cube_001": "rubik cube",
    "benchmark_medicine_bottle_000": "medicine bottle",
    "benchmark_garage_kit_064": "charmander figure",
}


@register()
def place_random_objects_on_desk(desk_shape: Shape) -> Shape:
    # Available object IDs (validated to exist in assets library)
    available_objects = [
        "benchmark_beverage_bottle_001",
        "benchmark_beverage_bottle_003",
        "benchmark_beverage_bottle_004",
        "genie_beverage_bottle_009",
        "benchmark_toy_car_018",
        "benchmark_building_blocks_072",
        "benchmark_cup_043",
        "benchmark_cup_105",
        "benchmark_toiletries_009",
        "benchmark_stationery_009",
        "benchmark_mouse_000",
        "benchmark_bottle_017",
        "benchmark_apple_002",
        "benchmark_orange_004",
        "benchmark_lemon_030",
        "benchmark_peach_021",
        "benchmark_green_apple_001",
        "benchmark_bowl_sponge_000",
        "benchmark_clock_044",
        "benchmark_tape_measure_007",
        "benchmark_toy_plane_029",
        "benchmark_rubik_cube_001",
        "benchmark_medicine_bottle_000",
        "benchmark_garage_kit_064",
    ]

    # Determine number of objects to place (3 to 6)
    num_objects = random.randint(3, 6)

    # Ensure we don't try to place more objects than available
    num_objects = min(num_objects, len(available_objects))

    # Randomly select unique objects
    selected_objects = random.sample(available_objects, num_objects)

    # Get desktop placement area
    desktop_center, desktop_size = get_desktop_placement_area(desk_shape)

    # Define placement bounds relative to desktop center
    x_bounds = (-0.2, 0.15)
    y_bounds = (-0.25, 0.25)

    placed_objects = []
    attempts_per_object = 0
    max_attempts_per_object = 20

    for obj_id in selected_objects:
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts_per_object:
            # Load the object with semantic name in keywords
            semantic_name = OBJECT_SEMANTIC_NAMES.get(obj_id, obj_id)

            # Determine position tag based on x coordinate
            base_x = np.random.uniform(x_bounds[0], x_bounds[1])
            base_y = np.random.uniform(y_bounds[0], y_bounds[1])
            position_tag = "left" if base_x < 0 else "right"

            obj_shape = library_call("usd", oid=obj_id, keywords=[semantic_name, obj_id, "on_desk", position_tag])

            # Sample a position within bounds
            base_pos = np.array([desktop_center[0] + base_x, desktop_center[1] + base_y, desktop_center[2]])

            # Get object size to adjust placement height
            obj_min = compute_shape_min(obj_shape)
            obj_height = compute_shape_sizes(obj_shape)[2]
            placement_z = desktop_center[2] + obj_height / 2.0 - obj_min[2]  # Place on surface

            # Apply translation
            obj_transformed = transform_shape(obj_shape, translation_matrix([base_pos[0], base_pos[1], placement_z]))

            # Add small random rotation around z-axis for natural look
            obj_center = compute_shape_center(obj_transformed)
            obj_transformed = transform_shape(
                obj_transformed, rotation_matrix(np.random.uniform(-0.3, 0.3), direction=(0, 0, 1), point=obj_center)
            )

            # Check minimum distance constraint
            if check_minimum_distance(obj_transformed, placed_objects, min_dist=0.15):
                placed_objects.append(obj_transformed)
                placed = True

            attempts += 1

        # If we couldn't place this object after max attempts, skip it
        if not placed:
            continue

    # Combine all shapes
    all_shapes = [desk_shape]
    all_shapes.extend(placed_objects)

    return concat_shapes(*all_shapes)


@register()
def table_with_random_objects() -> Shape:
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_random_objects_on_desk(desk_shape)


@register()
def root_scene() -> Shape:
    return table_with_random_objects()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    # 资产ID到名称的映射
    OBJECT_NAMES = {
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
        "benchmark_garage_kit_064": {"zh": "小火龙摆件", "en": "charmander figure"}
    }

    # 初始化objects_list - 所有资产都有交互数据
    objects_list = {}
    for obj_id in OBJECT_NAMES.keys():
        objects_list[obj_id] = {"has_interactions": True}

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. 查找桌子(table_000)
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

    # 2. 查找桌面上的所有物体
    desktop_objects = []
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        # 排除桌子本身
        if obj_id == table_id:
            continue

        # 检查是否在桌面上（根据场景生成代码，桌面物体有关键词"on_desk"）
        if any("on_desk" in kw for kw in keywords):
            pos, rot = robot.get_object_pose(obj_id)
            if pos is not None:
                # 获取名称
                names = OBJECT_NAMES.get(usd, {"zh": usd, "en": usd})

                desktop_objects.append({
                    "id": obj_id,
                    "usd": usd,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "chinese_name": names["zh"],
                    "english_name": names["en"]
                })

    # 如果没有找到桌面物体，尝试其他方法
    if len(desktop_objects) == 0:
        print("No objects found on desk using 'on_desk' keyword, trying alternative search...")
        table_pos, _ = robot.get_object_pose(table_id)
        if table_pos is not None:
            for obj_id, obj_info in scene_info["layout"].items():
                if obj_id == table_id:
                    continue

                pos, rot = robot.get_object_pose(obj_id)
                if pos is not None:
                    # 检查物体是否在桌子附近（z坐标接近）
                    if abs(pos[2] - table_pos[2]) < 0.2:
                        usd = obj_info.get("usd", "")
                        names = OBJECT_NAMES.get(usd, {"zh": usd, "en": usd})

                        desktop_objects.append({
                            "id": obj_id,
                            "usd": usd,
                            "position": pos,
                            "rotation": rot,
                            "keywords": obj_info.get("keywords", []),
                            "chinese_name": names["zh"],
                            "english_name": names["en"]
                        })

    if len(desktop_objects) < 2:
        print(f"Found only {len(desktop_objects)} objects on desk, need at least 2")
        return False

    print(f"Found {len(desktop_objects)} objects on desk:")
    for i, obj in enumerate(desktop_objects):
        print(f"  {i+1}. {obj['id']} - {obj['chinese_name']} at position: {obj['position']}")

    # 3. 辅助函数

    def check_object_already_in_direction(obj_a_pos, obj_b_pos, direction):
        """检查物体A是否已经在物体B的对应方位（考虑正交方向偏移）"""
        dx = obj_a_pos[0] - obj_b_pos[0]  # x方向差值
        dy = obj_a_pos[1] - obj_b_pos[1]  # y方向差值

        main_threshold = 0.05  # 主要方向阈值5cm
        ortho_threshold = 0.03  # 正交方向阈值3cm

        if direction == "up":
            # 上边：x差值必须为正，且y差值不能太大
            return dx > main_threshold and abs(dy) < ortho_threshold

        elif direction == "down":
            # 下边：x差值必须为负，且y差值不能太大
            return dx < -main_threshold and abs(dy) < ortho_threshold

        elif direction == "left":
            # 左边：y差值必须为正，且x差值不能太大
            return dy > main_threshold and abs(dx) < ortho_threshold

        elif direction == "right":
            # 右边：y差值必须为负，且x差值不能太大
            return dy < -main_threshold and abs(dx) < ortho_threshold

        elif direction == "top_left":
            # 左上方：x和y差值都为正
            return dx > main_threshold and dy > main_threshold

        elif direction == "bottom_left":
            # 左下方：x差值为负，y差值为正
            return dx < -main_threshold and dy > main_threshold

        elif direction == "top_right":
            # 右上方：x差值为正，y差值为负
            return dx > main_threshold and dy < -main_threshold

        elif direction == "bottom_right":
            # 右下方：x和y差值都为负
            return dx < -main_threshold and dy < -main_threshold

        return False

    def get_object_half(position_y):
        """根据y坐标确定物体所在的半区"""
        if position_y > -0.1:  # y > -0.1 在左半区
            return "left"
        elif position_y < 0.1:  # y < 0.1 在右半区
            return "right"
        else:  # -0.1 <= y <= 0.1，两个半区都可以
            return "both"

    def check_same_half(half1, half2):
        """检查两个半区是否相同（考虑both的情况）"""
        if half1 == "both" or half2 == "both":
            return True
        return half1 == half2

    def check_position_collision(position, excluded_objects=None, min_distance=0.1):
        """检查位置是否与其他物体碰撞"""
        if excluded_objects is None:
            excluded_objects = []

        for obj in desktop_objects:
            if obj["id"] in excluded_objects:
                continue

            obj_pos = obj["position"]
            distance = np.linalg.norm(position[:2] - obj_pos[:2])

            if distance < min_distance:
                return True

        return False

    def check_position_in_bounds(position):
        """检查位置是否在边界内"""
        # 检查x范围(-0.2, 0.15)
        if position[0] < -0.2 or position[0] > 0.15:
            return False

        return True

    def generate_base_positions(obj_b_pos, obj_a_size, obj_b_size, direction, margin=0.05):
        """根据方向生成基础位置点"""
        base_offset_x = (obj_a_size[0] + obj_b_size[0]) / 2.0 + margin
        base_offset_y = (obj_a_size[1] + obj_b_size[1]) / 2.0 + margin

        base_positions = []

        if direction == "up":
            # 上边（x增大）：多个x位置
            for dx in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x + dx
                # 保持y基本不变
                base_positions.append(pos)

        elif direction == "down":
            # 下边（x减小）：多个x位置
            for dx in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x + dx
                # 保持y基本不变
                base_positions.append(pos)

        elif direction == "left":
            # 左边（y增大）：多个y位置
            for dy in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[1] += base_offset_y + dy
                # 保持x基本不变
                base_positions.append(pos)

        elif direction == "right":
            # 右边（y减小）：多个y位置
            for dy in np.linspace(-0.02, 0.02, 5):
                pos = obj_b_pos.copy()
                pos[1] -= base_offset_y + dy
                # 保持x基本不变
                base_positions.append(pos)

        elif direction == "top_left":
            # 左上方：多个斜向位置
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] += base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "bottom_left":
            # 左下方：多个斜向位置
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] += base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "top_right":
            # 右上方：多个斜向位置
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] += base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] -= base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        elif direction == "bottom_right":
            # 右下方：多个斜向位置
            for i in range(5):
                pos = obj_b_pos.copy()
                pos[0] -= base_offset_x * 0.7 + np.random.uniform(-0.02, 0.02)
                pos[1] -= base_offset_y * 0.7 + np.random.uniform(-0.02, 0.02)
                base_positions.append(pos)

        return base_positions

    def generate_candidate_poses(base_positions, base_rotation, obj_a_size, table_top_z):
        """从基础位置生成候选姿态"""
        candidate_poses = []

        for base_pos in base_positions:
            # 设置z坐标（放在桌子表面）
            margin_z = 0.01  # 安全边距
            position = base_pos.copy()
            position[2] = table_top_z + margin_z + obj_a_size[2] / 2.0

            # 使用quat2mat_wxyz转换旋转
            base_rot_matrix = quat2mat_wxyz(base_rotation)

            # 基础姿态
            pose = np.eye(4)
            pose[:3, :3] = base_rot_matrix
            pose[:3, 3] = position
            candidate_poses.append(pose)

            # 添加轻微旋转变化的姿态
            for angle in [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6]:
                rot_z = euler2mat([0, 0, angle])
                final_rot = np.dot(base_rot_matrix, rot_z)

                pose_variation = np.eye(4)
                pose_variation[:3, :3] = final_rot
                pose_variation[:3, 3] = position
                candidate_poses.append(pose_variation)

            # 添加小位置偏移
            for dx in np.linspace(-0.005, 0.005, 3):
                for dy in np.linspace(-0.005, 0.005, 3):
                    if dx == 0 and dy == 0:
                        continue

                    offset_position = position + np.array([dx, dy, 0])

                    pose_offset = np.eye(4)
                    pose_offset[:3, :3] = base_rot_matrix
                    pose_offset[:3, 3] = offset_position
                    candidate_poses.append(pose_offset)

        return np.array(candidate_poses)

    # 4. 尝试生成可行的指令
    directions = [
        "up",           # 上边
        "down",         # 下边
        "left",         # 左边
        "right",        # 右边
        "top_left",     # 左上方
        "bottom_left",  # 左下方
        "top_right",    # 右上方
        "bottom_right"  # 右下方
    ]

    random.seed(os.getpid())

    # 最大尝试次数
    max_attempts = 100
    found_valid_instruction = False
    valid_base_positions = []
    selected_objects = {}
    selected_direction = ""

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt+1} to find valid instruction...")

        # 随机选择物体B作为基准物体
        obj_b = random.choice(desktop_objects)

        # 随机选择方位
        selected_direction = random.choice(directions)

        # 选择物体A（不同于物体B）
        other_objects = [obj for obj in desktop_objects if obj["id"] != obj_b["id"]]
        if len(other_objects) == 0:
            continue

        obj_a = random.choice(other_objects)

        obj_a_id = obj_a["id"]
        obj_b_id = obj_b["id"]
        obj_a_pos = obj_a["position"]
        obj_b_pos = obj_b["position"]

        print(f"Testing: Object A={obj_a['chinese_name']}, Object B={obj_b['chinese_name']}, Direction={selected_direction}")
        print(f"  Position differences: dx={obj_a_pos[0]-obj_b_pos[0]:.3f}, dy={obj_a_pos[1]-obj_b_pos[1]:.3f}")

        # 检查物体A是否已经在物体B的对应方位
        if check_object_already_in_direction(obj_a_pos, obj_b_pos, selected_direction):
            print(f"  Skipped: Object A is already in the {selected_direction} of Object B")
            continue

        # 获取物体尺寸
        obj_a_size = robot.get_object_size_in_world_frame(obj_a_id)
        obj_b_size = robot.get_object_size_in_world_frame(obj_b_id)

        if obj_a_size is None or obj_b_size is None:
            continue

        # 确定物体A的起始半区
        obj_a_start_half = get_object_half(obj_a_pos[1])

        # 生成基础位置点
        base_positions = generate_base_positions(obj_b_pos, obj_a_size, obj_b_size, selected_direction, margin=0.05)

        if len(base_positions) == 0:
            continue

        # 筛选有效的基础位置点
        valid_positions = []
        for pos in base_positions:
            # 检查边界条件
            if not check_position_in_bounds(pos):
                continue

            # 检查碰撞
            if check_position_collision(pos, excluded_objects=[obj_a_id, obj_b_id], min_distance=0.1):
                continue

            # 检查半区一致性
            target_half = get_object_half(pos[1])
            if not check_same_half(obj_a_start_half, target_half):
                continue

            # 通过所有检查
            valid_positions.append(pos)

        print(f"  Generated {len(base_positions)} base positions, {len(valid_positions)} valid positions")

        # 如果有足够的有效位置点，保存指令
        if len(valid_positions) >= 3:  # 至少需要3个有效位置
            selected_objects = {
                "obj_a": obj_a,
                "obj_b": obj_b,
                "obj_a_id": obj_a_id,
                "obj_b_id": obj_b_id,
                "obj_a_pos": obj_a_pos,
                "obj_b_pos": obj_b_pos,
                "obj_a_size": obj_a_size,
                "obj_b_size": obj_b_size,
                "obj_a_name_zh": obj_a["chinese_name"],
                "obj_b_name_zh": obj_b["chinese_name"],
                "obj_a_name_en": obj_a["english_name"],
                "obj_b_name_en": obj_b["english_name"]
            }
            valid_base_positions = valid_positions
            found_valid_instruction = True
            print(f"✓ Found valid instruction with {len(valid_positions)} valid positions!")
            break

    if not found_valid_instruction:
        print(f"\n❌ Failed to find valid instruction after {max_attempts} attempts")
        return False

    # 5. 获取桌子信息
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 6. 从有效的基础位置生成候选姿态
    candidate_poses = generate_candidate_poses(
        valid_base_positions,
        selected_objects["obj_a"]["rotation"],
        selected_objects["obj_a_size"],
        table_top_z
    )

    print(f"Generated {len(candidate_poses)} candidate poses from {len(valid_base_positions)} valid base positions")

    # 方向描述映射
    direction_descriptions_chinese = {
        "up": "上边",
        "down": "下边",
        "left": "左边",
        "right": "右边",
        "top_left": "左上方",
        "bottom_left": "左下方",
        "top_right": "右上方",
        "bottom_right": "右下方"
    }

    direction_descriptions_english = {
        "up": "top",
        "down": "bottom",
        "left": "left",
        "right": "right",
        "top_left": "top-left",
        "bottom_left": "bottom-left",
        "top_right": "top-right",
        "bottom_right": "bottom-right"
    }

    dir_desc_zh = direction_descriptions_chinese.get(selected_direction, "旁边")
    dir_desc_en = direction_descriptions_english.get(selected_direction, "side")

    # MANDATORY: Set task description dynamically before first action (after determining objects and direction)
    task_description = {
        "task_name": f"将{selected_objects['obj_a_name_zh']}放置到{selected_objects['obj_b_name_zh']}的{dir_desc_zh}",
        "english_task_name": f"place the {selected_objects['obj_a_name_en']} to the {dir_desc_en} of the {selected_objects['obj_b_name_en']}",
        "init_scene_text": "The robot is in front of the table, there are objects on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 7. 执行移动操作
    arm = robot.judge_arm_to_use(selected_objects["obj_a_pos"])
    print(f"\nUsing arm: {arm}")

    # 准备不可移动对象列表
    unmovable_objects = [table_id, selected_objects["obj_b_id"]]

    # 创建抓取配置
    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={
            "zh": f"拿起{selected_objects['obj_a_name_zh']}",
            "en": f"pick up the {selected_objects['obj_a_name_en']}"
        }
    )

    # 定义检查函数
    def create_check_function(obj_a_id, obj_b_id, selected_direction):
        def check_placement_success():
            current_pos, _ = robot.get_object_pose(obj_a_id)
            obj_b_pos, _ = robot.get_object_pose(obj_b_id)

            if current_pos is None or obj_b_pos is None:
                return False

            # 检查是否在桌子表面
            if current_pos[2] < table_top_z:
                return False

            # 检查相对于物体B的位置关系
            dx = current_pos[0] - obj_b_pos[0]  # x方向差值
            dy = current_pos[1] - obj_b_pos[1]  # y方向差值

            main_threshold = 0.02  # 主要方向阈值2cm
            ortho_threshold = 0.03  # 正交方向阈值3cm

            if selected_direction == "up":
                return dx > main_threshold and abs(dy) < ortho_threshold
            elif selected_direction == "down":
                return dx < -main_threshold and abs(dy) < ortho_threshold
            elif selected_direction == "left":
                return dy > main_threshold and abs(dx) < ortho_threshold
            elif selected_direction == "right":
                return dy < -main_threshold and abs(dx) < ortho_threshold
            elif selected_direction == "top_left":
                return dx > main_threshold and dy > main_threshold
            elif selected_direction == "bottom_left":
                return dx < -main_threshold and dy > main_threshold
            elif selected_direction == "top_right":
                return dx > main_threshold and dy < -main_threshold
            elif selected_direction == "bottom_right":
                return dx < -main_threshold and dy < -main_threshold

            return False

        return check_placement_success

    # 创建放置配置
    place_config = PlaceConfig(
        unmovable_objects=unmovable_objects,
        extra_checkers=[create_check_function(
            selected_objects["obj_a_id"],
            selected_objects["obj_b_id"],
            selected_direction
        )],
        action_description={
            "zh": f"将{selected_objects['obj_a_name_zh']}放置到{selected_objects['obj_b_name_zh']}的{dir_desc_zh}",
            "en": f"place the {selected_objects['obj_a_name_en']} to the {dir_desc_en} of the {selected_objects['obj_b_name_en']}"
        }
    )

    # 执行抓取和放置
    print(f"\nExecuting: Move {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")
    current_success = robot.pick_and_place_actor_to_target_pose(
        actor=selected_objects["obj_a_id"],
        target_pose=candidate_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if not current_success:
        print(f"Failed to place {selected_objects['obj_a_name_zh']}")
        success = False
    else:
        print(f"Successfully placed {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")

    # 8. 最终验证
    if success:
        final_pos, _ = robot.get_object_pose(selected_objects["obj_a_id"])
        if final_pos is not None:
            print(f"\nFinal position: {final_pos}")
            print(f"Task completed successfully!")

    # 返回手臂到初始位置
    robot.back_to_origin(arm=arm)

    print(f"\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Moved {selected_objects['obj_a_name_zh']} to {dir_desc_zh} of {selected_objects['obj_b_name_zh']}")
    else:
        print(f"❌ FAILED: Could not complete the task")

    return success
```
