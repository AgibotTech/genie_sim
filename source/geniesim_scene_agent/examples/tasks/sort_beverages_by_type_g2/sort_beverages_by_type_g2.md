# Sort beverages by type

**Tags:** `sort` `beverage` `two_piles` `by_type` `beverage_type` `keywords_usd` `place_actor_to_target_pose` `left_right_regions` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose` `table_000`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table and all beverage cans (usd/keywords coca_cola, sprite, fanta, wang_lao_ji). Group by beverage_type (e.g. type_1 and type_2).
2. **Regions:** Define left and right regions (e.g. left: x < -0.05, right: x > 0.05). Assign one type to left pile, the other to right. Target positions in each region with spacing.
3. **Execution:** For each can: determine target region by type; build target_poses in that region (table_top_z + margin + size/2). pick_and_place_actor_to_target_pose(can_id, target_poses, arm, grasp_config, place_config).
4. **Order:** Process cans in stable order; unmovable_objects = [table_id] + already placed cans.

---

# INPUT

**scene_description:** A table with 3–5 canned beverages of two types (e.g. coca_cola, sprite, fanta, wang_lao_ji) randomly placed. The robot must sort them into two piles by type (e.g. one type on the left, the other on the right).

**task description:** Sort the beverages on the table into two piles by type. Identify each can's type from usd/keywords; place all cans of type 1 in one region (e.g. left) and type 2 in another (e.g. right) with spacing.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: sort_beverages_by_type_g2
description: A table with 3-5 canned beverages of exactly two types, randomly placed and rotated within specified bounds
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface and its size.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.1

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def generate_random_positions(
    num_items: int, x_range: Tuple[float, float], y_range: Tuple[float, float], min_distance: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Generate random positions within given ranges ensuring minimum distance between any two points.
    """
    positions = []
    max_attempts = 1000

    for i in range(num_items):
        attempt = 0
        while attempt < max_attempts:
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])

            # Check distance to all existing positions
            valid = True
            for existing_x, existing_y in positions:
                distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
                if distance < min_distance:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                break
            attempt += 1

        if attempt >= max_attempts:
            # If we can't find a valid position, just use the last attempted one
            positions.append((x, y))

    return positions


@register()
def canned_beverage(oid: str, keywords: List[str]) -> Shape:
    """
    Create a single canned beverage with random rotation.
    """
    beverage_shape = library_call("usd", oid=oid, keywords=keywords)

    # Get the current center for rotation
    center = compute_shape_center(beverage_shape)

    # Apply random rotation around z-axis
    rotation_angle = random.uniform(0, 2 * math.pi)
    rotated_shape = transform_shape(beverage_shape, rotation_matrix(rotation_angle, direction=(0, 0, 1), point=center))

    return rotated_shape


@register()
def place_beverages_on_desktop(desk_shape: Shape) -> Shape:
    """
    Place 3-5 canned beverages of exactly two types on the desktop within specified bounds.
    """
    workspace, _ = find_desktop_top_surface(desk_shape)

    # Define the placement bounds relative to workspace center
    x_min, x_max = -0.05, 0.15
    y_min, y_max = -0.25, 0.25

    # Available canned beverage types (only benchmark assets with interaction)
    beverage_types = [
        ("benchmark_beverage_bottle_001", ["coca_cola_can", "red", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_002", ["sprite_can", "green", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_003", ["fanta_can", "orange", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_029", ["wang_lao_ji_can", "red", "aluminum_can", "herbal_tea"]),
    ]

    # Select exactly two types randomly
    selected_types = random.sample(beverage_types, 2)

    # Determine number of beverages (3-5 total)
    num_beverages = random.randint(3, 5)

    # Distribute beverages between the two types
    num_type1 = random.randint(1, num_beverages - 1)
    num_type2 = num_beverages - num_type1

    # Generate random positions ensuring minimum distance
    positions = generate_random_positions(num_beverages, (x_min, x_max), (y_min, y_max), min_distance=0.1)

    beverage_shapes = []

    # Create beverages of first type
    for i in range(num_type1):
        oid, keywords = selected_types[0]
        beverage = library_call("canned_beverage", oid=oid, keywords=keywords)
        pos_x, pos_y = positions[i]
        beverage_placed = transform_shape(
            beverage, translation_matrix([workspace[0] + pos_x, workspace[1] + pos_y, workspace[2]])
        )
        beverage_shapes.append(beverage_placed)

    # Create beverages of second type
    for i in range(num_type2):
        oid, keywords = selected_types[1]
        beverage = library_call("canned_beverage", oid=oid, keywords=keywords)
        pos_x, pos_y = positions[num_type1 + i]
        beverage_placed = transform_shape(
            beverage, translation_matrix([workspace[0] + pos_x, workspace[1] + pos_y, workspace[2]])
        )
        beverage_shapes.append(beverage_placed)

    return concat_shapes(desk_shape, *beverage_shapes)


@register()
def sort_beverages_by_type_g2() -> Shape:
    """
    Main scene function: table with randomly placed and rotated canned beverages.
    """
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular_desk"]
    )
    return place_beverages_on_desktop(desk_shape)


@register()
def root_scene() -> Shape:
    return sort_beverages_by_type_g2()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## 初始化objects_list
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

    ## 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "把桌子上的饮料按种类分成两堆",
        "english_task_name": "sort the beverages on the table into two piles by type",
        "init_scene_text": "The robot is in front of the table, there are 3-5 canned beverages of two types randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. 查找桌子
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

    ## 2. 查找所有饮料
    beverages = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        ## 检查是否是饮料
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

            ## 确定饮料类型
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

    ## 3. 按类型分类饮料
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

    ## 4. 规划两种饮料的摆放位置
    ## 区域1: x(-0.2 ~ -0.1) y(-0.15, -0.03)
    region1 = {
        "x_range": (-0.2, -0.1),
        "y_range": (-0.15, -0.03),
        "center": np.array([-0.15, -0.09])
    }

    ## 区域2: x(-0.2 ~ -0.1) y(0.03, 0.15)
    region2 = {
        "x_range": (-0.2, -0.1),
        "y_range": (0.03, 0.15),
        "center": np.array([-0.15, 0.09])
    }

    ## 获取桌子高度
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    def plan_positions(num_items, region):
        """
        规划饮料摆放位置：
        - 1个：摆在区域中间
        - 2个：摆成一排（x相同）
        - 3个：摆成三角形
        - 4个：摆成两排（四边形）
        """
        positions = []
        min_spacing = 0.1

        if num_items == 1:
            ## 摆在区域中间，x靠前（x越小越靠前），y靠区域中间
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y = region["center"][1]
            positions.append(np.array([x, y]))

        elif num_items == 2:
            ## 摆成一排（x相同），y分开
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]
            y_offset = min_spacing / 2.0
            positions.append(np.array([x, y_center - y_offset]))
            positions.append(np.array([x, y_center + y_offset]))

        elif num_items == 3:
            ## 摆成三角形
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            ## 下方两个
            positions.append(np.array([x, y_center - min_spacing / 2.0]))
            positions.append(np.array([x, y_center + min_spacing / 2.0]))
            ## 上方一个
            positions.append(np.array([x + min_spacing * 0.866, y_center]))

        elif num_items == 4:
            ## 摆成两排（四边形）
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            ## 第一排（前方）
            positions.append(np.array([x, y_center - min_spacing / 2.0]))
            positions.append(np.array([x, y_center + min_spacing / 2.0]))
            ## 第二排（后方）
            positions.append(np.array([x + min_spacing, y_center - min_spacing / 2.0]))
            positions.append(np.array([x + min_spacing, y_center + min_spacing / 2.0]))

        else:
            ## 更多的情况，按网格排列
            x = region["x_range"][0] + (region["x_range"][1] - region["x_range"][0]) * 0.3
            y_center = region["center"][1]

            for i in range(num_items):
                row = i // 2
                col = i % 2
                x_pos = x + row * min_spacing
                y_pos = y_center + (col - 0.5) * min_spacing
                positions.append(np.array([x_pos, y_pos]))

        return positions

    ## 为每种饮料类型规划目标位置
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

    ## 5. 按抓取顺序排序所有饮料（x越小越靠前）
    all_beverages_sorted = sorted(beverages, key=lambda b: b["position"][0])

    print("\nBeverages sorted by x (front to back):")
    for i, bev in enumerate(all_beverages_sorted):
        print(f"  {i+1}. {bev['id']} (type: {bev['type']}), x={bev['position'][0]:.3f}")

    ## 6. 执行抓取和摆放
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

        ## 判断使用哪只手臂
        arm = robot.judge_arm_to_use(bev_pos)
        print(f"Using arm: {arm}")

        ## 切换手臂时需要返回原点
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        ## 获取初始旋转角度
        initial_rot_matrix = quat2mat_wxyz(bev_rot)

        ## 规划目标位置
        if bev_type not in type_positions:
            print(f"Warning: Unknown beverage type {bev_type}, skipping")
            continue

        target_positions = type_positions[bev_type]

        ## 找到当前饮料在列表中的索引
        type_beverages = [b for b in beverages if b["type"] == bev_type and b["id"] not in processed_beverages]
        if len(type_beverages) == 0:
            continue

        target_xy = target_positions.pop(0)

        ## 计算目标z高度（比初始z值高0.03~0.05）
        height_offset = np.random.uniform(0.03, 0.05)
        target_z = table_top_z + height_offset + bev_size[2] / 2.0

        ## 生成多个候选旋转角度（0~2π采样12个）
        rotation_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        ## 生成候选姿态
        target_poses = []

        ## 位置变化（小范围偏移）
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
                        ## 先获取初始旋转角度，然后绕z轴旋转任意角度
                        rot_z = euler2mat([0, 0, rot_angle])
                        final_rot = np.dot(rot_z, initial_rot_matrix)

                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = position
                        target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for beverage {bev_id}")

        ## 准备抓取配置
        unmovable_objects = [table_id]
        unmovable_objects.extend(processed_beverages)

        ## 根据饮料类型确定描述
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

        ## 定义检查函数：验证饮料是否成功放置在正确区域
        def create_beverage_checker(bev_id, target_xy, region):
            def checker():
                current_pos, current_rot = robot.get_object_pose(bev_id)
                if current_pos is None:
                    return False

                ## 检查xy位置是否在目标区域附近
                x_range = region["x_range"]
                y_range = region["y_range"]

                x_in_range = x_range[0] - 0.05 <= current_pos[0] <= x_range[1] + 0.05
                y_in_range = y_range[0] - 0.05 <= current_pos[1] <= y_range[1] + 0.05

                if not x_in_range or not y_in_range:
                    print(f"  Beverage {bev_id} not in target region")
                    print(f"    Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
                    print(f"    Target region: x[{x_range[0]:.3f}, {x_range[1]:.3f}], y[{y_range[0]:.3f}, {y_range[1]:.3f}]")
                    return False

                ## 检查z高度是否正确
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

        ## 执行抓取和放置
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

    ## 7. 最终验证
    print("\n=== Performing final verification ===")

    ## 验证所有饮料是否都被正确分类和放置
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

    ## 返回手臂到初始位置
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: All beverages sorted into two piles by type")
    else:
        print("⚠️ PARTIAL SUCCESS: Some beverages not correctly placed")

    return success
```
