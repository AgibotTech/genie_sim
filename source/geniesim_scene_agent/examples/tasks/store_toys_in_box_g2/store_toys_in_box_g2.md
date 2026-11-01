# Store toys in a box

**Tags:** `place_into_container` `table_with_containers` `multi_object_scatter` `place_actor_to_target_pose` `pick_actor` `extra_checkers` `check_box_containment` `get_object_size_in_local_frame` `ignore_axes_z` `keywords` `toy_in_box` `toy_on_desk` `unmovable_objects` `multiple_candidate_poses` `GraspConfig` `PlaceConfig` `action_description` `overlap_threshold`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find storage box (usd/keywords). Find toys on desk: objects with toy-related usd/keywords but not "toy_in_box" or "inside_storage". Skip toys already in box.
2. **Target poses:** Generate multiple candidate poses inside/above box (box interior or top + height offsets). Use get_object_size_in_local_frame for box to build target positions.
3. **Verification:** extra_checkers: check_box_containment(box_pos, box_rot, box_size_local, toy_pos, toy_rot, toy_size_local, ignore_axes=["z"], overlap_threshold) to confirm toy inside box after place.
4. **Execution:** For each toy on desk: pick_actor then place_actor_to_target_pose (or pick_and_place_actor_to_target_pose) with unmovable_objects=[table_id, box_id] + already placed toys.

---

# INPUT

**scene_description:** A table with one storage box and several toys on the table. Some toys may already be inside the box (e.g. marked by keywords like "toy_in_box" or "inside_storage"); only toys on the desk need to be moved. The robot must put all toys that are on the table into the storage box.

**task description:** Put all toys on the table into the storage box. Identify toys by keywords/usd (e.g. toy, building_blocks, toy_car, etc.) and the box by storage_box; skip toys already in the box. After each placement, verify the toy is inside the box (e.g. with check_box_containment, ignore_axes=["z"]).

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: desk_with_storage_box_and_toys
description: A desk with a randomly selected storage box containing 0-3 toys, and 1-3 additional toys placed on the rest of the desktop. Total toys <= 4, with at least one outside the box. Toys are non-repeating, spaced at least 0.08m apart, and avoid a 5cm margin around the box.
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        desk_shape (Shape): The shape of the desk object.

    Returns:
        Tuple[P, P]: A tuple containing the desktop's top surface center position and its size.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def is_position_valid(candidate_pos: P, existing_positions: List[P], min_dist: float = 0.08) -> bool:
    """
    Check if a candidate position is valid given existing positions and a minimum distance.
    """
    for pos in existing_positions:
        if np.linalg.norm(np.array(candidate_pos[:2]) - np.array(pos[:2])) < min_dist:
            return False
    return True


def sample_position_in_region(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> P:
    """
    Sample a random (x, y) position within the specified ranges, z=0 for now (will be set later).
    """
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    return np.array([x, y, 0.0])


@register()
def create_toy(toy_oid: str, keywords: List[str]) -> Shape:
    """
    Creates a single toy with a small random rotation around Z-axis.
    """
    toy_shape = library_call("usd", oid=toy_oid, keywords=keywords)
    toy_center = compute_shape_center(toy_shape)
    angle = np.random.uniform(-0.2, 0.2)  # approx +/- 11.5 degrees, more than enough for "random"
    return transform_shape(toy_shape, rotation_matrix(angle=angle, direction=(0, 0, 1), point=toy_center))


@register()
def place_toys_on_desktop(desk_shape: Shape, storage_box_shape: Shape, num_toys_outside: int) -> Shape:
    """
    Places a specified number of toys on the desktop outside the storage box, respecting constraints.
    """
    # Get desktop info
    desktop_pos, desktop_size = find_desktop_top_surface(desk_shape)
    desktop_min = desktop_pos - desktop_size / 2.0
    desktop_max = desktop_pos + desktop_size / 2.0

    # Get storage box info to define exclusion zone
    box_info = get_object_info(storage_box_shape)
    box_center = box_info["center"]
    box_size = box_info["size"]
    margin = 0.05  # 5cm exclusion zone
    box_exclusion_min = box_center - box_size / 2.0 - margin
    box_exclusion_max = box_center + box_size / 2.0 + margin

    # Define the allowed region for outside toys: x(-0.25, 0.0), y(-0.25, 0.25) relative to desk center
    # Since `desktop_pos` is the absolute center of the desktop top, we add the offsets to it.
    allowed_x_min = desktop_pos[0] - 0.25
    allowed_x_max = desktop_pos[0] + 0.0
    allowed_y_min = desktop_pos[1] - 0.25
    allowed_y_max = desktop_pos[1] + 0.25

    # Toy asset list
    toy_candidates = [
        "benchmark_building_blocks_057",
        "benchmark_toy_car_018",
        "benchmark_toy_plane_029",
        "benchmark_building_blocks_029",
        "benchmark_building_blocks_028",
        "benchmark_rubik_cube_001",
        "benchmark_garage_kit_064",
        "benchmark_chess_001",
    ]
    selected_toys = random.sample(toy_candidates, k=len(toy_candidates))  # Shuffle the list

    placed_shapes = []
    placed_positions = []  # Keep track of (x, y, z) for distance checks
    toy_index = 0

    for _ in range(num_toys_outside):
        placed = False
        attempts = 0
        max_attempts = 100

        while not placed and attempts < max_attempts:
            attempts += 1
            # Sample a candidate position in the allowed region
            candidate_xy = sample_position_in_region((allowed_x_min, allowed_x_max), (allowed_y_min, allowed_y_max))
            candidate_pos = np.array([candidate_xy[0], candidate_xy[1], desktop_pos[2]])

            # Check if it's inside the box exclusion zone
            if (
                box_exclusion_min[0] <= candidate_pos[0] <= box_exclusion_max[0]
                and box_exclusion_min[1] <= candidate_pos[1] <= box_exclusion_max[1]
            ):
                continue

            # Check distance from already placed toys
            if not is_position_valid(candidate_pos, placed_positions):
                continue

            # If we get here, the position is valid
            toy_oid = selected_toys[toy_index]
            toy_index += 1
            toy_shape = library_call(
                "create_toy", toy_oid=toy_oid, keywords=[f"toy_{toy_index}", "small", "plaything", "randomly_placed"]
            )

            # Get toy size for correct vertical placement
            toy_min = compute_shape_min(toy_shape)
            toy_height = compute_shape_sizes(toy_shape)[2]
            # Place the toy so its bottom is on the desktop
            final_translation = translation_matrix(
                [
                    candidate_pos[0],
                    candidate_pos[1],
                    desktop_pos[2] - toy_min[2],  # toy_min[2] is negative, so this adds height
                ]
            )

            placed_shape = transform_shape(toy_shape, final_translation)
            placed_shapes.append(placed_shape)
            placed_positions.append(candidate_pos)
            placed = True

        if not placed:
            # If we can't find a spot, just break to avoid infinite loop
            break

    return concat_shapes(*placed_shapes)


@register()
def place_toys_in_storage_box(storage_box_shape: Shape, num_toys_inside: int) -> Shape:
    """
    Places a specified number of toys inside the storage box.
    """
    box_info = get_object_info(storage_box_shape)
    box_center = box_info["center"]
    box_size = box_info["size"]

    # Define the inner volume for placing toys, leave a small margin
    inner_margin = 0.01
    inner_x_range = (box_center[0] - box_size[0] / 2.0 + inner_margin, box_center[0] + box_size[0] / 2.0 - inner_margin)
    inner_y_range = (box_center[1] - box_size[1] / 2.0 + inner_margin, box_center[1] + box_size[1] / 2.0 - inner_margin)
    inner_z_base = box_center[2] - box_size[2] / 2.0 + inner_margin  # Start placing from the bottom

    toy_candidates = [
        "benchmark_building_blocks_057",
        "benchmark_toy_car_018",
        "benchmark_toy_plane_029",
        "benchmark_building_blocks_029",
        "benchmark_building_blocks_028",
        "benchmark_rubik_cube_001",
        "benchmark_garage_kit_064",
        "benchmark_chess_001",
    ]
    selected_toys = random.sample(toy_candidates, k=len(toy_candidates))

    placed_shapes = []
    current_z = inner_z_base
    toy_index = 0

    for i in range(num_toys_inside):
        toy_oid = selected_toys[toy_index]
        toy_index += 1
        toy_shape = library_call(
            "create_toy", toy_oid=toy_oid, keywords=[f"toy_in_box_{i+1}", "small", "plaything", "inside_storage"]
        )

        toy_size = compute_shape_sizes(toy_shape)
        # Center the toy in X and Y within the box
        tx = box_center[0]
        ty = box_center[1]
        tz = current_z + toy_size[2] / 2.0  # Place on top of previous or base
        current_z = tz + toy_size[2] / 2.0  # Update z for next toy

        placed_shape = transform_shape(toy_shape, translation_matrix([tx, ty, tz]))
        placed_shapes.append(placed_shape)

    return concat_shapes(*placed_shapes)


@register()
def table_with_storage_box_and_toys() -> Shape:
    # Load the desk
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    desktop_pos, _ = find_desktop_top_surface(desk_shape)

    # Select a random storage box
    storage_box_options = [
        "benchmark_storage_box_001",
        "benchmark_storage_box_004",
        "benchmark_storage_box_005",
        "benchmark_storage_box_003",
    ]
    chosen_box = random.choice(storage_box_options)
    storage_box_shape = library_call("usd", oid=chosen_box, keywords=["storage_container", "plastic", "box", "on_desk"])

    # Determine toy distribution
    # Total toys must be <= 4, and at least 1 outside.
    total_toys = random.randint(2, 4)  # Min 2 (1 in, 1 out), max 4
    num_toys_outside = random.randint(1, min(3, total_toys - 0))  # At least 1 outside, max 3 outside
    num_toys_inside = total_toys - num_toys_outside
    # Ensure inside count is between 0 and 3
    num_toys_inside = max(0, min(3, num_toys_inside))
    num_toys_outside = total_toys - num_toys_inside

    # Position the storage box on the desk within its specified region
    # Box region: x(-0.1, 0.05), y(-0.03, 0.03) relative to desk center
    box_tx = np.random.uniform(-0.1, 0.05)
    box_ty = np.random.uniform(-0.03, 0.03)
    box_tz = desktop_pos[2]  # Place directly on the desktop

    # Apply a small random rotation around Z-axis (up to 5 degrees = ~0.087 rad)
    box_angle = np.random.uniform(-0.087, 0.087)
    box_center_initial = compute_shape_center(storage_box_shape)
    storage_box_transformed = transform_shape(
        storage_box_shape, translation_matrix([desktop_pos[0] + box_tx, desktop_pos[1] + box_ty, box_tz])
    )
    box_center_final = compute_shape_center(storage_box_transformed)
    storage_box_transformed = transform_shape(
        storage_box_transformed, rotation_matrix(angle=box_angle, direction=(0, 0, 1), point=box_center_final)
    )

    # Create toys inside and outside
    toys_inside_shape = library_call(
        "place_toys_in_storage_box", storage_box_shape=storage_box_transformed, num_toys_inside=num_toys_inside
    )
    toys_outside_shape = library_call(
        "place_toys_on_desktop",
        desk_shape=desk_shape,
        storage_box_shape=storage_box_transformed,
        num_toys_outside=num_toys_outside,
    )

    # Combine all shapes
    return concat_shapes(desk_shape, storage_box_transformed, toys_inside_shape, toys_outside_shape)


@register()
def root_scene() -> Shape:
    return table_with_storage_box_and_toys()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 基于提供的交互数据初始化objects_list
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

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "将所有玩具放进收纳盒里",
        "english_task_name": "put all toys into the storage box",
        "init_scene_text": "The robot is in front of the table, there is a storage box and several toys on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. 查找桌子
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

    # 2. 查找收纳盒（storage box）
    storage_box_id = None
    storage_box_usd = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是收纳盒（通过usd判断）
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

    # 3. 查找所有玩具（排除收纳盒本身）
    all_toys = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 排除收纳盒和桌子
        if obj_id == storage_box_id or obj_id == table_id:
            continue

        # 检查是否是玩具（通过usd判断）
        is_toy = False
        for toy_usd in list(objects_list.keys()):
            if toy_usd in ["benchmark_storage_box_001", "benchmark_storage_box_004",
                          "benchmark_storage_box_005", "benchmark_storage_box_003"]:
                continue  # 跳过收纳盒
            if toy_usd in usd:
                is_toy = True
                break

        # 也可以通过keywords判断（根据场景生成代码，玩具可能有"toy"关键词）
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
        return True  # 没有玩具需要处理，任务成功

    print(f"Total toys found: {len(all_toys)}")

    # 4. 根据keyword判断玩具是否已经在收纳盒内
    # 根据场景生成代码，盒子内的玩具keywords包含"toy_in_box_{i}"
    # 桌面上的玩具keywords包含"toy_on_desk_{i}"或"small", "toy"
    toys_in_box = []
    toys_on_desk = []

    for toy in all_toys:
        toy_id = toy["id"]
        keywords = toy["keywords"]

        # 检查是否在盒子内
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

    # 5. 如果需要处理的玩具数量为0，直接返回成功
    if len(toys_on_desk) == 0:
        print("All toys are already in the storage box")
        return True

    print(f"Toys already in box: {len(toys_in_box)}")
    print(f"Toys to process (on desk): {len(toys_on_desk)}")

    # 6. 获取收纳盒的位置和尺寸信息
    storage_box_pos, storage_box_rot = robot.get_object_pose(storage_box_id)
    storage_box_size = robot.get_object_size_in_world_frame(storage_box_id)

    if storage_box_pos is None or storage_box_size is None:
        print("Cannot get storage box position/size")
        return False

    # 计算收纳盒顶部高度
    storage_box_top_z = storage_box_pos[2] + storage_box_size[2] / 2.0
    print(f"Storage box top z: {storage_box_top_z:.3f}")

    # 7. 处理每个需要放入收纳盒的玩具
    last_arm = ""

    for i, toy in enumerate(toys_on_desk):
        toy_id = toy["id"]
        toy_usd = toy["usd"]
        toy_pos = toy["position"]

        print(f"\n=== Processing toy {i+1}/{len(toys_on_desk)}: {toy_id} ===")

        # 判断使用哪只手臂
        arm = robot.judge_arm_to_use(toy_pos)
        print(f"Using arm: {arm}")

        # 切换手臂时需要返回原点
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        # 准备不可移动对象列表（保护桌子、收纳盒和已经在盒子里的玩具）
        unmovable_objects = [table_id, storage_box_id]

        # 根据玩具类型确定描述
        toy_desc_zh = "玩具"
        toy_desc_en = "toy"

        # 提取玩具类型信息（从usd或keywords）
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

        # 抓取配置
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects.copy(),
            action_description={"zh": f"拿起{toy_desc_zh}", "en": f"picks up the {toy_desc_en}"}  # REQUIRED
        )

        # 执行抓取操作
        print(f"Attempting to pick up toy {toy_id}")
        pick_success = robot.pick_actor(toy_id, arm=arm, grasp_config=grasp_config)

        if not pick_success:
            print(f"Failed to pick up toy {toy_id}")
            success = False
            continue

        # 获取玩具的尺寸信息
        toy_size = robot.get_object_size_in_world_frame(toy_id)
        if toy_size is None:
            print(f"Cannot get toy size for {toy_id}")
            success = False
            # 尝试放置回原位置
            robot.back_to_origin(arm=arm)
            continue

        # 计算放置位置
        # 任务要求：物体的底和收纳盒的顶部至少有0.03cm（应该是0.03m = 3cm）
        # 注意：0.03cm = 0.0003m，这太小了，应该是0.03m = 3cm
        # 根据上下文，应该是3cm的安全距离
        min_clearance = 0.03  # 3cm clearance

        # 玩具底部到其中心的距离 = toy_size[2] / 2.0
        # 放置位置的z坐标 = storage_box_top_z + min_clearance + toy_size[2] / 2.0
        placement_z = storage_box_top_z + min_clearance + toy_size[2] / 2.0

        # 生成多个候选放置位置（考虑x,y的小范围偏移）
        target_poses = []

        # 任务要求：尽量放置在中心区域，允许小范围偏移
        x_offsets = np.linspace(-0.02, 0.02, 5)  # ±2cm x方向偏移
        y_offsets = np.linspace(-0.02, 0.02, 5)  # ±2cm y方向偏移

        # 获取收纳盒的旋转矩阵作为基础方向
        base_rot_matrix = quat2mat_wxyz(storage_box_rot)

        # 任务要求：玩具可以沿三个轴任意旋转
        # 生成多个随机旋转姿态
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

        # 定义检查函数：验证玩具是否成功放入收纳盒
        def create_toy_in_box_checker(toy_id, box_id):
            def checker():
                toy_pos, toy_rot = robot.get_object_pose(toy_id)
                toy_size = robot.get_object_size_in_local_frame(toy_id)
                box_pos, box_rot = robot.get_object_pose(box_id)
                box_size = robot.get_object_size_in_local_frame(box_id)

                if toy_pos is None or box_pos is None:
                    return False

                # 检查玩具是否在收纳盒内（忽略z轴，只检查xy平面投影）
                box_contains_toy, _ = check_box_containment(
                    box_pos, box_rot, box_size,
                    toy_pos, toy_rot, toy_size,
                    ignore_axes=["z"],
                    overlap_threshold=0.4
                )

                return box_contains_toy

            return checker

        # 放置配置
        place_desc_zh = f"将{toy_desc_zh}放置到收纳盒"
        place_desc_en = f"places the {toy_desc_en} in the storage box"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects.copy(),
            extra_checkers=[create_toy_in_box_checker(toy_id, storage_box_id)],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # 执行放置操作（使用pose-based placement）
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

    # 8. 最终验证
    print("\n=== Performing final verification ===")

    # 重新检查所有玩具的物理位置
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

        # 使用物理检查
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

    # 返回手臂到初始位置
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if len(final_toys_not_in_box) == 0:
        print("✅ SUCCESS: All toys placed into storage box")
        return True
    else:
        print(f"❌ FAILED: {len(final_toys_not_in_box)} toys are not in the storage box")
        return False
```
