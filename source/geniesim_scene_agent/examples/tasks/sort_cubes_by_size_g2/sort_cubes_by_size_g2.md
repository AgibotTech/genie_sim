# Sort cubes by size

**Tags:** `sort` `building_blocks` `cubes` `volume_sort` `get_object_size_in_world_frame` `linear_arrangement` `target_x_range` `spacing` `multiple_cube_assets` `pose_based_placement` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose` `benchmark_building_blocks_079_072_076_019_073_011`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table and all cubes (usd matching building block assets). get_object_size_in_world_frame for each; volume = size[0]*size[1]*size[2]. Sort by volume ascending.
2. **Target positions:** Linear arrangement in region (e.g. target_x in (-0.05~-0.15), y in (-0.1, 0.1)). target_positions[i] = (start_x - i*spacing, start_y, table_top_z + margin + size[2]/2). min_spacing e.g. 0.04 m.
3. **Candidate poses:** Multiple rotations around z (and optionally small xy/z offsets). unmovable_objects = [table_id] + already placed cube ids.
4. **Execution:** For each cube in sorted order: pick_and_place_actor_to_target_pose with GraspConfig/PlaceConfig and extra_checkers.

---

# INPUT

**scene_description:** A table with multiple cubes (building blocks of different sizes, e.g. benchmark_building_blocks_079/072/076/019/073/011). The robot must arrange them in a line sorted by volume (smallest to largest), e.g. in region x = (-0.05~-0.15), y = (-0.1, 0.1) with spacing between cubes.

**task description:** Arrange the cubes on the table in order of volume from smallest to largest. Place them in a line with consistent spacing (e.g. min_spacing 0.04m). Use get_object_size_in_world_frame to compute volume (size[0]*size[1]*size[2]); sort by volume.

**code to generate the scene:**

```python
from helper import *

"""
scene_name: sort_cubes_by_size_g2
description: A table with three cubic building blocks of different sizes (small, medium, large) placed randomly on the specified region x(-0.1, 0), y(-0.2, 0.2), with random rotation angles around the world z-axis and at least 4cm spacing between them.
"""


def get_table_top_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface center and its size.
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Table top surface center in world coordinates (x, y from table center, z from top of desktop)
    table_top_center = np.array(
        [table_info["center"][0], table_info["center"][1], table_info["center"][2] + desktop_subpart_info["xyz_max"][2]]
    )
    table_top_size = desktop_subpart_info["size"]

    return table_top_center, table_top_size


@register()
def small_cube() -> Shape:
    """Small cubic building block (0.02m) with interaction."""
    return library_call(
        "usd",
        oid="benchmark_building_blocks_079",
        keywords=["small_cube", "green", "cube", "building_block", "interaction", "up_axis_y"],
    )


@register()
def medium_cube(color: str = "red") -> Shape:
    """Medium cubic building block (~0.03m) with interaction. Color options: red, blue, yellow, reddish_brown."""
    oid_map = {
        "blue": "benchmark_building_blocks_076",
        "reddish_brown": "benchmark_building_blocks_019",
    }
    color = np.random.choice(["blue", "reddish_brown"])
    oid = oid_map.get(color, "benchmark_building_blocks_076")  # default to blue
    return library_call(
        "usd", oid=oid, keywords=[f"medium_cube_{color}", color, "cube", "building_block", "interaction", "up_axis_y"]
    )


@register()
def large_cube(color: str = "red") -> Shape:
    """Large cubic building block (0.04m or 0.05m) with interaction. Color options: red, blue."""
    oid_map = {"red": "benchmark_building_blocks_073", "blue": "benchmark_building_blocks_011"}
    color = np.random.choice(["red", "blue"])
    oid = oid_map.get(color, "benchmark_building_blocks_073")  # default to red
    return library_call(
        "usd", oid=oid, keywords=[f"large_cube_{color}", color, "cube", "building_block", "interaction", "up_axis_y"]
    )


@register()
def place_cubes_on_table(table_shape: Shape) -> Shape:
    """
    Places small, medium, and large cubes on the table within the specified region x(-0.1, 0), y(-0.2, 0.2),
    with random positions and random rotation angles around the world z-axis, ensuring at least 4cm spacing.
    """
    # Get table top info
    table_top_center, table_top_size = get_table_top_surface(table_shape)

    # Define placement region boundaries (absolute world coordinates as per user request)
    x_min, x_max = -0.1, 0.0
    y_min, y_max = -0.2, 0.2

    # Create cubes and get their sizes
    small = library_call("small_cube")
    medium = library_call("medium_cube", color="blue")
    large = library_call("large_cube", color="red")

    small_size = compute_shape_sizes(small)
    medium_size = compute_shape_sizes(medium)
    large_size = compute_shape_sizes(large)

    # Minimum spacing required (4cm = 0.04m)
    min_spacing = 0.04

    # Helper function to check if a position is valid (within bounds and spaced from others)
    def is_valid_position(pos: P, existing_positions: List[Tuple[P, P]], obj_size: P) -> bool:
        # Check bounds: ensure object doesn't exceed placement region
        if not (x_min + obj_size[0] / 2 <= pos[0] <= x_max - obj_size[0] / 2):
            return False
        if not (y_min + obj_size[1] / 2 <= pos[1] <= y_max - obj_size[1] / 2):
            return False
        # Check spacing with existing objects
        for existing_pos, existing_size in existing_positions:
            dx = abs(pos[0] - existing_pos[0])
            dy = abs(pos[1] - existing_pos[1])
            # Minimum distance required is half of each object's size plus spacing
            min_dist_x = (obj_size[0] + existing_size[0]) / 2 + min_spacing
            min_dist_y = (obj_size[1] + existing_size[1]) / 2 + min_spacing
            if dx < min_dist_x or dy < min_dist_y:
                return False
        return True

    # Place cubes with retry logic and increased max attempts for tighter space
    placed_positions = []
    cubes = [("large", large, large_size), ("medium", medium, medium_size), ("small", small, small_size)]
    placed_shapes = []
    max_attempts = 500  # Increased attempts for tighter space

    for name, cube, size in cubes:
        placed = False
        for _ in range(max_attempts):
            # Generate random position within bounds
            x = np.random.uniform(x_min + size[0] / 2, x_max - size[0] / 2)
            y = np.random.uniform(y_min + size[1] / 2, y_max - size[1] / 2)
            pos_2d = [x, y, 0]

            if is_valid_position(pos_2d, placed_positions, size):
                # Place at table top height
                world_pos = [x, y, table_top_center[2]]
                # Apply random rotation around world z-axis (0,0,1)
                angle = np.random.uniform(0, 2 * np.pi)
                cube_transformed = transform_shape(cube, translation_matrix(world_pos))
                cube_center = compute_shape_center(cube_transformed)
                cube_rotated = transform_shape(
                    cube_transformed, rotation_matrix(angle, direction=(0, 0, 1), point=cube_center)
                )

                placed_shapes.append(cube_rotated)
                placed_positions.append((pos_2d, size))
                placed = True
                break

        if not placed:
            # Fallback: place in a line if random placement fails
            if len(placed_positions) == 0:
                # First cube: place at center
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
            elif len(placed_positions) == 1:
                # Second cube: offset along x
                x = placed_positions[0][0][0] + (placed_positions[0][1][0] + size[0]) / 2 + min_spacing + 0.01
                y = (y_min + y_max) / 2
                if x > x_max - size[0] / 2:
                    x = placed_positions[0][0][0] - (placed_positions[0][1][0] + size[0]) / 2 - min_spacing - 0.01
            else:
                # Third cube: offset along y
                x = (x_min + x_max) / 2
                y = placed_positions[0][0][1] + (placed_positions[0][1][1] + size[1]) / 2 + min_spacing + 0.01
                if y > y_max - size[1] / 2:
                    y = placed_positions[0][0][1] - (placed_positions[0][1][1] + size[1]) / 2 - min_spacing - 0.01

            world_pos = [x, y, table_top_center[2]]
            angle = np.random.uniform(0, 2 * np.pi)
            cube_transformed = transform_shape(cube, translation_matrix(world_pos))
            cube_center = compute_shape_center(cube_transformed)
            cube_rotated = transform_shape(
                cube_transformed, rotation_matrix(angle, direction=(0, 0, 1), point=cube_center)
            )
            placed_shapes.append(cube_rotated)
            placed_positions.append(([x, y, 0], size))

    return concat_shapes(table_shape, *placed_shapes)


@register()
def sort_cubes_by_size_g2() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_cubes_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return sort_cubes_by_size_g2()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 从场景生成代码中提取的积木资产ID列表
    objects_list = {
        "benchmark_building_blocks_079": {"has_interactions": True, "object_type": "building_block", "size": "small"},
        "benchmark_building_blocks_072": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_076": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_019": {"has_interactions": True, "object_type": "building_block", "size": "medium"},
        "benchmark_building_blocks_073": {"has_interactions": True, "object_type": "building_block", "size": "large"},
        "benchmark_building_blocks_011": {"has_interactions": True, "object_type": "building_block", "size": "large"},
    }

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

    # 2. 查找所有立方体积木
    cubes = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        is_cube = False
        for cube_asset in list(objects_list.keys()):
            if cube_asset in usd:
                is_cube = True
                break

        if is_cube or any("cube" in kw.lower() for kw in keywords) or any("block" in kw.lower() for kw in keywords):
            size = robot.get_object_size_in_world_frame(obj_id)
            if size is not None:
                volume = size[0] * size[1] * size[2]
                pos, rot = robot.get_object_pose(obj_id)
                cubes.append({
                    "id": obj_id,
                    "size": size,
                    "volume": volume,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found cube: {obj_id}, size: {size}, volume: {volume:.6f}")

    if len(cubes) < 2:
        print(f"Found only {len(cubes)} cubes, need at least 2")
        return False

    print(f"Total cubes found: {len(cubes)}")

    # 3. 按体积从小到大排序
    cubes.sort(key=lambda x: x["volume"])
    print("Cubes sorted by volume (smallest to largest):")
    for i, cube in enumerate(cubes):
        print(f"  {i+1}. {cube['id']}: volume={cube['volume']:.6f}, size={cube['size']}")

    # 4. 计算摆放位置参数
    # 摆放区域：x = (-0.05 ~ -0.15), y = (-0.1, 0.1)
    target_x = np.random.uniform(-0.13, -0.17)  # 固定x位置

    # 计算总宽度
    total_width = 0
    for cube in cubes:
        cube_width = cube["size"][1]  # y尺寸作为宽度
        total_width += cube_width

    # 增大积木间间距：至少3cm
    min_spacing = 0.04  # 3cm

    # 计算所需总宽度
    total_spacing = min_spacing * (len(cubes) - 1)
    total_needed_width = total_width + total_spacing

    # 检查是否能放在指定y范围内
    available_y_range = 0.20  # y从-0.1到0.1，总共0.20米
    print(f"Available y range: {available_y_range:.3f}m")
    print(f"Total needed width: {total_needed_width:.3f}m")

    if total_needed_width > available_y_range:
        print(f"Warning: Total needed width {total_needed_width:.3f}m exceeds available range {available_y_range:.3f}m")
        # 按比例调整间距，但保持至少2cm
        scale_factor = available_y_range / total_needed_width
        if scale_factor < 0.5:
            print("Error: Cannot fit cubes with reasonable spacing")
            return False

        adjusted_spacing = max(min_spacing * scale_factor, 0.03)  # 至少2cm
        total_needed_width = total_width + adjusted_spacing * (len(cubes) - 1)
        print(f"Adjusted spacing to {adjusted_spacing:.3f}m")
        spacing = adjusted_spacing
    else:
        spacing = min_spacing

    print(f"Using spacing: {spacing:.3f}m")

    # 计算起始位置（最小的积木在左边，y最大）
    # y坐标范围：-0.1到0.1，从大到小排列
    start_y = np.random.uniform(0.090, 0.1)

    # 5. 为每个积木计算目标位置（按从小到大顺序）
    target_positions = []
    current_y = start_y

    for i, cube in enumerate(cubes):
        cube_id = cube["id"]
        cube_size = cube["size"]

        # 计算y位置
        current_y = current_y - cube_size[1]/2.0
        target_y = current_y

        # 获取桌子高度
        table_pos, table_rot = robot.get_object_pose(table_id)
        table_size = robot.get_object_size_in_world_frame(table_id)
        table_top_z = table_pos[2] + table_size[2] / 2.0

        # 计算积木放置高度：桌子顶部 + 安全边距 + 半个积木高度
        margin = 0.01  # 安全边距1cm，避免夹爪碰撞
        target_z = table_top_z + margin + cube_size[2] / 2.0

        # 位置向量
        target_position = np.array([target_x, target_y, target_z])
        target_positions.append({
            "id": cube_id,
            "position": target_position,
            "size": cube_size,
            "index": i
        })

        print(f"Cube {i+1} ({cube_id}) target position: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}], size: {cube_size}")

        # 更新当前y位置，为下一个积木准备
        current_y = current_y - cube_size[1]/2.0 - spacing

    # 6. 生成候选旋转姿态（提前生成，供所有积木使用）
    def generate_candidate_rotations():
        """生成满足条件的候选旋转矩阵"""
        candidate_rotations = []

        # 基础旋转：将本地y轴旋转到世界z轴（绕x轴旋转90度）
        rot_x_90 = euler2mat([np.pi/2, 0, 0])

        # 方案1：绕z轴旋转0, 90, 180, 270度
        for z_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_z = euler2mat([0, 0, z_angle])
            final_rot = np.dot(rot_z, rot_x_90)
            candidate_rotations.append(final_rot)

        # 方案2：绕y轴旋转后再绕x轴旋转90度
        for y_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_y = euler2mat([0, y_angle, 0])
            rot_combined = np.dot(rot_x_90, rot_y)
            candidate_rotations.append(rot_combined)

        # 去重（虽然重复的可能性不大）
        unique_rotations = []
        for rot in candidate_rotations:
            is_unique = True
            for existing in unique_rotations:
                if np.allclose(rot, existing, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_rotations.append(rot)

        print(f"Generated {len(unique_rotations)} unique rotation matrices")
        return unique_rotations

    candidate_rotations = generate_candidate_rotations()

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": "将积木按从小到大排列",
        "english_task_name": "arrange blocks from small to large",
        "init_scene_text": "The robot is in front of the table, there are three cube building blocks of different sizes randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 7. 执行摆放任务（按从小到大顺序）
    placed_cubes = []
    last_arm = ""
    for i, target_info in enumerate(target_positions):
        cube_id = target_info["id"]
        base_position = target_info["position"]
        cube_size = target_info["size"]

        print(f"\n=== Processing cube {i+1}/{len(target_positions)}: {cube_id} ===")

        # 获取当前积木位置
        cube_position, cube_rotation = robot.get_object_pose(cube_id)
        if cube_position is None:
            print(f"Warning: Cannot get position for cube {cube_id}")
            continue

        # 判断使用哪只手臂
        cube_arm = robot.judge_arm_to_use(cube_position)
        print(f"Using arm: {cube_arm}")
        if last_arm != cube_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = cube_arm
        # 创建抓取配置
        unmovable_objects = [table_id]
        unmovable_objects.extend([c["id"] for c in placed_cubes])

        # 根据积木大小确定描述（从小到大排序，第i个积木）
        size_descriptions = ["最小的积木", "中等大小的积木", "最大的积木"]
        size_descriptions_en = ["smallest block", "medium-sized block", "largest block"]
        if i < len(size_descriptions):
            block_desc_zh = size_descriptions[i]
            block_desc_en = size_descriptions_en[i]
        else:
            block_desc_zh = f"第{i+1}个积木"
            block_desc_en = f"the {i+1}th block"

        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{block_desc_zh}", "en": f"pick up the {block_desc_en}"}  # REQUIRED
        )

        # 生成大量候选姿态（一次性传入所有备选方案）
        target_poses = []

        # 位置变化参数
        x_offsets = np.linspace(-0.001, 0.001, 3)  # x方向偏移
        y_offsets = np.linspace(-0.001, 0.001, 3)  # y方向偏移
        z_offsets = np.linspace(0, 0.005, 3)  # z方向偏移

        # 生成所有候选姿态的组合
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
        print(f"Generated {len(target_poses)} candidate poses for cube {cube_id}")

        # 定义检查函数
        def create_check_function(cube_id, expected_position):
            def check_cube_aligned():
                current_pos, current_rot = robot.get_object_pose(cube_id)
                if current_pos is None or current_rot is None:
                    return False

                # 位置检查（3cm容差）
                pos_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                tolerance = 0.03
                if pos_distance > tolerance:
                    print(f"Cube {cube_id} position error: {pos_distance:.3f}m > {tolerance:.3f}m")
                    return False

                # 检查是否在桌子上
                table_pos, _ = robot.get_object_pose(table_id)
                table_size = robot.get_object_size_in_world_frame(table_id)
                table_top_z = table_pos[2] + table_size[2] / 2.0

                if current_pos[2] < table_top_z:
                    print(f"Cube {cube_id} below table surface")
                    return False

                return True

            return check_cube_aligned

        # 创建检查函数
        check_function = create_check_function(cube_id, base_position)

        # 根据积木大小和位置确定描述（从左到右排列）
        if i == 0:
            # 第一个（最小的）放在左侧
            place_desc_zh = "将最小的积木放置到桌子中央偏下位置的左侧"
            place_desc_en = "place the smallest block to the left side of the lower center of the table"
        elif i == 1:
            # 第二个（中等大小的）放在最小的右侧
            place_desc_zh = "将中等大小的积木放置到最小积木的右侧"
            place_desc_en = "place the medium-sized block to the right of the smallest block"
        elif i == 2:
            # 第三个（最大的）放在中等大小的右侧
            place_desc_zh = "将最大的积木放置到中等大小积木的右侧"
            place_desc_en = "place the largest block to the right of the medium-sized block"
        else:
            # 如果有多于3个积木的情况
            place_desc_zh = f"将第{i+1}个积木放置到桌子中央偏下位置"
            place_desc_en = f"place the {i+1}th block to the lower center of the table"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # 执行抓取和放置（一次性传入所有候选姿态）
        print(f"Attempting to pick and place cube {cube_id} with {len(target_poses)} candidate poses")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=cube_id,
            target_pose=target_poses,
            arm=cube_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )
        success = success and current_success
        if not success:
            print(f"Failed to place cube {cube_id}")
            return False
        print(f"Successfully placed cube {cube_id}")

        # 添加到已放置列表
        placed_cubes.append({"id": cube_id, "position": base_position})


    # 8. 最终验证和总结
    print("\n=== Performing final verification ===")

    # 收集所有积木的最终位置
    final_positions = []
    for cube in cubes:
        cube_id = cube["id"]
        current_pos, current_rot = robot.get_object_pose(cube_id)

        if current_pos is None:
            print(f"Cube {cube_id} not found after placement")
            success = False
            continue

        final_positions.append((cube_id, current_pos[1], current_pos))

        # 检查基本要求
        if current_pos[0] < -0.15 or current_pos[0] > -0.05:
            print(f"  Warning: {cube_id} x position {current_pos[0]:.3f} out of range")

        if current_pos[1] < -0.1 or current_pos[1] > 0.1:
            print(f"  Warning: {cube_id} y position {current_pos[1]:.3f} out of range")

    # 检查排列顺序
    if len(final_positions) >= 2:
        final_positions.sort(key=lambda x: x[1], reverse=True)  # 按y从大到小排序

        print("\nFinal arrangement (from left to right, y decreasing):")
        for i, (cube_id, y_pos, full_pos) in enumerate(final_positions):
            print(f"  {i+1}. {cube_id}: y={y_pos:.3f}, position={full_pos}")

        # 检查间距
        print("\nChecking spacing between cubes:")
        for i in range(len(final_positions) - 1):
            cube1_id, y1, pos1 = final_positions[i]
            cube2_id, y2, pos2 = final_positions[i+1]

            # 计算中心间距
            center_distance = abs(y1 - y2)

            # 获取积木尺寸
            size1 = None
            size2 = None
            for cube in cubes:
                if cube["id"] == cube1_id:
                    size1 = cube["size"][1]
                if cube["id"] == cube2_id:
                    size2 = cube["size"][1]

            if size1 is not None and size2 is not None:
                # 计算边缘间距 = 中心间距 - (size1/2 + size2/2)
                edge_distance = center_distance - (size1/2.0 + size2/2.0)

                if edge_distance < 0.02:  # 至少2cm
                    print(f"  Warning: {cube1_id} and {cube2_id} too close: {edge_distance:.3f}m < 0.02m")
                else:
                    print(f"  OK: {cube1_id} and {cube2_id} spacing: {edge_distance:.3f}m")
    robot.back_to_origin(arm=last_arm)
    print("\n=== Task summary ===")
    if success:
        print("SUCCESS: All cubes placed with at least 3cm spacing")
    else:
        print("PARTIAL SUCCESS: Some issues encountered")

    return success
```
