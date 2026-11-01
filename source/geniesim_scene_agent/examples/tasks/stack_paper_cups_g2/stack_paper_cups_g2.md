# Stack paper cups

**Tags:** `stack` `paper_cup` `benchmark_cup_105` `table_only` `sort_by_distance` `up_axis_detection` `quat2mat_wxyz` `dot_with_world_z` `pose_based_placement` `multiple_candidate_poses` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose` `interaction_data_active_passive_place`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table and all paper cups (usd "benchmark_cup_105" or "paper_cup" in keywords). For each cup compute distance_to_center = np.linalg.norm(pos[:2] - table_pos[:2]).
2. **Order:** cups.sort(key=lambda x: x["distance"]). Stack nearest to center first. Detect up-axis: initial_rot_matrix = quat2mat_wxyz(rot); up_axis_idx = np.argmax([abs(np.dot(col, [0,0,1])) for col in rot_matrix.T]).
3. **Target poses:** First cup at table center; subsequent cups on top of previous (target_z = prev_top_z + margin + size[2]/2). Generate target_poses with position and orientation variations (preserve or align up-axis).
4. **Execution:** pick_and_place_actor_to_target_pose with GraspConfig/PlaceConfig and extra_checkers (e.g. xy within tolerance).

---

# INPUT

**scene_description:** A table with multiple paper cups (benchmark_cup_105). Cups have interaction data (e.g. active place "侧放"/"正放", passive place "正放"). The robot must stack the cups (e.g. by distance from table center: nearest first, then stack outward).

**task description:** Stack the paper cups. Sort cups by distance from table center; stack in order (nearest to center first). Preserve or align cup orientation (up-axis).

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: stack_paper_cups_g2
description: 3 to 5 paper cups randomly placed on a table within x(-0.2, 0.2) and y(-0.3, 0.3),
avoiding the central forbidden zone x(-0.03, 0.03) y(-0.03, 0.03). Cups must be at least 0.07m apart (center-to-center).
"""


def get_tabletop_center(table_shape: Shape) -> P:
    """
    Retrieves the world coordinates of the table's top surface center.

    Args:
        table_shape (Shape): The shape of the table.

    Returns:
        P: The table's top center position in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    return np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])


@register()
def paper_cup() -> Shape:
    """
    Creates a single paper cup object.
    """
    return library_call(
        "usd",
        oid="benchmark_cup_105",
        keywords=["paper_cup", "white", "cylinder", "disposable", "beverage"],
    )


@register()
def arrange_cups_on_table(table_shape: Shape) -> Shape:
    """
    Arranges 3-5 paper cups on the given table, respecting all placement constraints.
    """
    table_center = get_tabletop_center(table_shape)

    # Define the allowed placement area relative to table center
    allowed_x_range = (-0.3, 0.1)
    allowed_y_range = (-0.3, 0.3)
    forbidden_x_range = (-0.03, 0.03)
    forbidden_y_range = (-0.03, 0.03)
    min_center_distance = 0.15

    num_cups = np.random.randint(3, 6)  # 3 to 5 inclusive
    placed_positions = []

    cup_template = library_call("paper_cup")
    cup_info = get_object_info(cup_template)
    cup_radius = max(cup_info["size"][0], cup_info["size"][1]) / 2.0

    for i in range(num_cups):
        placed = False
        for _ in range(200):  # Max attempts to find a valid spot
            # Generate random position in the allowed area
            x_offset = np.random.uniform(*allowed_x_range)
            y_offset = np.random.uniform(*allowed_y_range)

            # Skip if in forbidden zone
            if (
                forbidden_x_range[0] <= x_offset <= forbidden_x_range[1]
                and forbidden_y_range[0] <= y_offset <= forbidden_y_range[1]
            ):
                continue

            world_x = table_center[0] + x_offset
            world_y = table_center[1] + y_offset

            # Check distance from already placed cups
            too_close = False
            for pos in placed_positions:
                dist = np.linalg.norm(np.array([world_x, world_y]) - np.array(pos[:2]))
                if dist < min_center_distance:
                    too_close = True
                    break

            if not too_close:
                placed_positions.append((world_x, world_y, table_center[2]))
                placed = True
                break

        if not placed:
            # If we can't place all cups, just place what we have
            break

    # Create and position the cups
    cup_shapes = []
    for i, (x, y, z) in enumerate(placed_positions):
        # Determine position tag for keywords
        pos_tag = "left" if x < table_center[0] else "right"

        cup = library_call(
            "usd",
            oid="benchmark_cup_105",
            keywords=[f"paper_cup_{i}", "white", "cylinder", "disposable", "beverage", pos_tag],
        )

        # Get cup center for rotation
        cup_center = compute_shape_center(cup)

        # Apply random rotation around z-axis
        rotated_cup = transform_shape(
            cup, rotation_matrix(angle=np.random.uniform(0, 2 * np.pi), direction=(0, 0, 1), point=cup_center)
        )

        # Translate to final position
        final_cup = transform_shape(rotated_cup, translation_matrix((x, y, z)))
        cup_shapes.append(final_cup)

    return concat_shapes(*cup_shapes)


@register()
def stack_paper_cups_g2() -> Shape:
    """
    Creates the complete scene with a table and randomly placed paper cups.
    """
    table = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "rectangular", "furniture"])
    cups = arrange_cups_on_table(table)
    return concat_shapes(table, cups)


@register()
def root_scene() -> Shape:
    return stack_paper_cups_g2()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 初始化objects_list - 从场景生成代码和交互数据中提取
    objects_list = {
        "benchmark_cup_105": {
            "has_interactions": True,
            "object_type": "paper_cup",
            "interaction_data": {
                "active": {
                    "place": ["侧放", "正放"]
                },
                "passive": {
                    "place": ["正放"],
                    "grasp": ["default"]
                }
            }
        },
        "table_000": {
            "has_interactions": False,
            "object_type": "table"
        }
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
        if usd == "table_000" or "table" in keywords or "dining_table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. 查找所有纸杯 (benchmark_cup_105) 并记录初始状态
    cups = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是纸杯benchmark_cup_105
        if "benchmark_cup_105" in usd or any("paper_cup" in kw.lower() for kw in keywords):
            pos, rot = robot.get_object_pose(obj_id)
            if pos is not None:
                # 计算到桌子中心的距离
                table_pos, _ = robot.get_object_pose(table_id)
                if table_pos is None:
                    distance_to_center = np.linalg.norm(pos[:2])
                else:
                    distance_to_center = np.linalg.norm(pos[:2] - table_pos[:2])

                # 获取初始旋转矩阵和朝上轴
                initial_rot_matrix = quat2mat_wxyz(rot)

                # 确定哪个轴是朝上轴（与世界坐标z轴最接近的轴）
                world_z = np.array([0, 0, 1])
                col_0 = initial_rot_matrix[:, 0]
                col_1 = initial_rot_matrix[:, 1]
                col_2 = initial_rot_matrix[:, 2]

                dot_0 = np.abs(np.dot(col_0, world_z))
                dot_1 = np.abs(np.dot(col_1, world_z))
                dot_2 = np.abs(np.dot(col_2, world_z))

                # 选择最接近世界z轴的轴作为朝上轴
                dots = [dot_0, dot_1, dot_2]
                up_axis_idx = np.argmax(dots)
                initial_up_axis = initial_rot_matrix[:, up_axis_idx]

                # 获取杯子尺寸
                cup_size = robot.get_object_size_in_world_frame(obj_id)
                if cup_size is None:
                    print(f"Warning: Cannot get size for cup {obj_id}")
                    cup_size = np.array([0.08, 0.08, 0.12])  # 默认尺寸

                cups.append({
                    "id": obj_id,
                    "position": pos,
                    "rotation": rot,
                    "size": cup_size,
                    "initial_rot_matrix": initial_rot_matrix,
                    "initial_up_axis": initial_up_axis,
                    "up_axis_idx": up_axis_idx,
                    "distance": distance_to_center,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found cup: {obj_id}, distance={distance_to_center:.3f}m, size={cup_size}")

    if len(cups) < 3:
        print(f"Found only {len(cups)} cups, need at least 3")
        return False

    print(f"\nTotal cups found: {len(cups)}")

    # 3. 按距离由近到远排序
    cups.sort(key=lambda x: x["distance"])
    print("Cups sorted by distance from center (nearest to farthest):")
    for i, cup in enumerate(cups):
        print(f"  {i+1}. {cup['id']}: distance={cup['distance']:.3f}m, size={cup['size']}")

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": "将纸杯堆叠",
        "english_task_name": "stack paper cups",
        "init_scene_text": "The robot is in front of the table, there are paper cups on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 4. 获取桌子信息用于计算堆叠位置
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    # 桌子顶部高度
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 5. 执行堆叠任务（按距离从近到远）
    placed_cups = []
    last_arm = ""

    # 目标堆叠位置：桌子中心(0,0)
    base_x = 0.0
    base_y = 0.0

    # 记录第一个杯子的最终位置（用于后续对齐）
    first_cup_final_pos = None
    first_cup_final_size = None
    last_cup_final_pos = None


    for i, cup_info in enumerate(cups):
        cup_id = cup_info["id"]
        cup_pos = cup_info["position"]
        cup_size = cup_info["size"]
        cup_distance = cup_info["distance"]
        initial_rot_matrix = cup_info["initial_rot_matrix"]
        initial_up_axis = cup_info["initial_up_axis"]
        up_axis_idx = cup_info["up_axis_idx"]

        print(f"\n=== Processing cup {i+1}/{len(cups)}: {cup_id} ===")
        print(f"  Distance: {cup_distance:.3f}m")
        print(f"  Size: {cup_size}")
        print(f"  Initial up axis: {initial_up_axis} (idx: {up_axis_idx})")

        # 判断使用哪只手臂
        cube_arm = robot.judge_arm_to_use(cup_pos)
        print(f"Using arm: {cube_arm}")

        # 切换手臂时需要返回原点
        if last_arm != cube_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = cube_arm

        # 计算堆叠位置 - 插入式堆叠
        if i == 0:
            # 第一个杯子：放在桌子中心附近，允许小范围偏移
            margin = 0.01  # 1cm安全边距
            base_z = table_top_z + margin + cup_size[2] / 2.0

            # 第一个杯子使用随机小偏移
            x_offset = np.random.uniform(-0.02, 0.02)
            y_offset = np.random.uniform(-0.02, 0.02)
            base_position = np.array([base_x + x_offset, base_y + y_offset, base_z])

            print(f"  First cup offset: x={x_offset:.3f}m, y={y_offset:.3f}m")
        else:
            # 后续杯子：插入到前一个杯子中
            # 需要计算插入深度
            margin = 0.01
            insertion_depth = cup_size[2] * 0.3
            base_z = last_cup_final_pos[2] - insertion_depth + cup_size[2] + margin
            base_position = last_cup_final_pos.copy()
            base_position[2] = base_z

        print(f"  Target position: [{base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f}]")

        # 创建抓取配置
        unmovable_objects = [table_id]
        unmovable_objects.extend([c["id"] for c in placed_cups])

        # 根据距离和堆叠顺序确定描述
        if i == 0:
            cup_desc_zh = "离中心最近的纸杯"
            cup_desc_en = "cup closest to the center"
        elif i == 1:
            cup_desc_zh = "第二近的纸杯"
            cup_desc_en = "second closest cup"
        else:
            cup_desc_zh = f"第{i+1}个纸杯"
            cup_desc_en = f"{i+1}th cup"

        grasp_config = GraspConfig(
            passive_primitive="default",  # 使用纸杯的默认抓取点[1]
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起{cup_desc_zh}", "en": f"pick up the {cup_desc_en}"}
        )

        # 生成候选姿态：初始旋转矩阵绕世界z轴旋转任意角度
        target_poses = []

        # 位置变化参数（插入式堆叠需要更精确的位置）
        if i == 0:
            # 第一个杯子：小范围位置变化
            x_offsets = np.linspace(-0.003, 0.003, 3)
            y_offsets = np.linspace(-0.003, 0.003, 3)
            position_offsets = []
            for dx in x_offsets:
                for dy in y_offsets:
                    position_offsets.append(np.array([dx, dy, 0]))
        else:
            # 后续杯子：插入需要非常精确的对齐
            position_offsets = [
                np.array([0, 0, 0]),
            ]

        # z方向微调（对于插入很重要）
        if i == 0:
            z_offsets = np.linspace(0, 0.002, 3)
        else:
            # 插入时的z方向调整范围更小
            z_offsets = np.linspace(-0.001, 0.001, 3)

        # 生成绕世界z轴的多个旋转角度
        rotation_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)

        # 生成所有候选姿态的组合
        for pos_offset in position_offsets:
            for z_offset in z_offsets:
                position = base_position + pos_offset + np.array([0, 0, z_offset])

                for z_angle in rotation_angles:
                    # 绕世界z轴旋转
                    rot_z = euler2mat([0, 0, z_angle])

                    # 组合旋转：初始旋转矩阵绕z轴旋转
                    final_rot = np.dot(rot_z, initial_rot_matrix)

                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"  Generated {len(target_poses)} candidate poses")

        # 定义检查函数：针对插入式堆叠的特殊检查
        def create_check_function(cup_id, expected_position, cup_size, initial_up_axis,
                                  up_axis_idx, placed_cup_ids, is_first=False):
            def check_cup_stacked():
                # 获取当前杯子状态
                current_pos, current_rot = robot.get_object_pose(cup_id)
                if current_pos is None or current_rot is None:
                    print(f"    Error: Cannot get cup state")
                    return False

                # 获取当前旋转矩阵
                current_rot_matrix = quat2mat_wxyz(current_rot)
                current_up_axis = current_rot_matrix[:, up_axis_idx]

                # 1. 检查朝上轴是否仍然朝上（与世界z轴的夹角）
                world_z = np.array([0, 0, 1])
                current_up_dot = np.abs(np.dot(current_up_axis, world_z))

                # 检查与初始朝上轴的偏差
                up_axis_dot = np.dot(current_up_axis, initial_up_axis)
                up_axis_dot = np.clip(up_axis_dot, -1.0, 1.0)
                up_axis_angle = np.arccos(abs(up_axis_dot))

                # 检查是否朝上（与z轴夹角小于30度）
                if current_up_dot < np.cos(np.deg2rad(30)):
                    print(f"    Error: Cup not upright enough, dot={current_up_dot:.3f}")
                    return False

                # 检查朝上轴变化（与初始相比小于45度）
                if up_axis_angle > np.deg2rad(45):
                    print(f"    Error: Up axis changed too much: {np.rad2deg(up_axis_angle):.1f}°")
                    return False

                # 2. 检查位置是否正确（XY平面）
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                xy_tolerance = 0.03 if is_first else 0.01  # 插入需要更高精度

                if xy_distance > xy_tolerance:
                    print(f"    Error: XY position off by {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    return False

                # 3. 如果是插入式堆叠（不是第一个杯子），检查是否真的插入了
                if not is_first and len(placed_cup_ids) > 0:
                    # 获取前一个杯子（被插入的杯子）
                    prev_cup_id = placed_cup_ids[-1]
                    prev_cup_pos, prev_cup_rot = robot.get_object_pose(prev_cup_id)
                    prev_cup_size = robot.get_object_size_in_world_frame(prev_cup_id)

                    if prev_cup_pos is not None and prev_cup_size is not None:
                        # 检查中心对齐（插入需要良好的中心对齐）
                        center_distance_xy = np.linalg.norm(current_pos[:2] - prev_cup_pos[:2])
                        if center_distance_xy > 0.015:  # 1.5cm容差
                            print(f"    Error: Not centered for insertion")
                            print(f"      Center offset: {center_distance_xy:.3f}m")
                            return False
                        # 高度检查
                        height_diff = current_pos[2] - prev_cup_pos[2]
                        if height_diff < 0.001:
                            print(f"    Error: Not inserted properly - current cup should be below the previous cup")
                            print(f"      Current cup top: {current_pos[2]:.3f}m")
                            print(f"      Previous cup top: {prev_cup_pos[2]:.3f}m")
                            print(f"      Difference: {height_diff:.3f}m")
                            return False

                print(f"    Check passed:")
                print(f"      XY error: {xy_distance:.3f}m")
                # print(f"      Z error: {abs(current_pos[2] - expected_z):.3f}m")
                print(f"      Up axis deviation: {np.rad2deg(up_axis_angle):.1f}°")
                print(f"      Uprightness (dot): {current_up_dot:.3f}")
                return True

            return check_cup_stacked

        # 创建检查函数
        is_first = (i == 0)
        placed_cup_ids = [c["id"] for c in placed_cups]
        check_function = create_check_function(cup_id, base_position, cup_size,
                                              initial_up_axis, up_axis_idx,
                                              placed_cup_ids, is_first)

        # 确定放置描述
        if i == 0:
            place_desc_zh = "将离中心最近的纸杯放置到桌子中心附近"
            place_desc_en = "place the cup closest to the center near the table center"
        elif i == 1:
            place_desc_zh = "将第二近的纸杯插入第一个纸杯中"
            place_desc_en = "insert the second closest cup into the first cup"
        else:
            place_desc_zh = f"将第{i+1}个纸杯插入纸杯塔中"
            place_desc_en = f"insert the {i+1}th cup into the cup tower"

        # 放置配置：使用pre_insert_offset实现插入动作
        place_config = PlaceConfig(
            active_primitive="正放",  # 使用"正放"放置点[1]
            pre_insert_offset=0.1,    # 插入动作使用8cm预放置偏移
            unmovable_objects=unmovable_objects,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}
        )

        # 执行抓取和放置
        print(f"  Executing pick and place...")
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=cup_id,
            target_pose=target_poses,
            arm=cube_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"  ❌ Failed to place cup {cup_id}")
            success = False
            break
        else:
            print(f"  ✅ Successfully placed cup {cup_id}")
            cup_final_pos, _ = robot.get_object_pose(cup_id)
            last_cup_final_pos = cup_final_pos

            # 如果是第一个杯子，记录最终位置和尺寸
            if i == 0:
                first_cup_final_pos, _ = robot.get_object_pose(cup_id)
                first_cup_final_size = robot.get_object_size_in_world_frame(cup_id)
                print(f"  First cup final position: {first_cup_final_pos}")

        # 添加到已放置列表
        placed_cups.append({
            "id": cup_id,
            "position": base_position,
            "size": cup_size,
            "initial_up_axis": initial_up_axis,
            "up_axis_idx": up_axis_idx
        })

    # 6. 最终详细验证
    print("\n" + "="*60)
    print("DETAILED FINAL VERIFICATION - INSERTION STACKING")
    print("="*60)

    all_checks_passed = True

    if len(placed_cups) > 0:
        print(f"\nChecking {len(placed_cups)} inserted cups:")

        for i, cup_info in enumerate(placed_cups):
            cup_id = cup_info["id"]
            cup_size = cup_info["size"]
            initial_up_axis = cup_info["initial_up_axis"]
            up_axis_idx = cup_info["up_axis_idx"]

            current_pos, current_rot = robot.get_object_pose(cup_id)
            if current_pos is None:
                print(f"Cup {i+1} ({cup_id}): ❌ Not found")
                all_checks_passed = False
                continue

            # 获取当前旋转矩阵
            current_rot_matrix = quat2mat_wxyz(current_rot)
            current_up_axis = current_rot_matrix[:, up_axis_idx]

            # 检查朝上轴
            world_z = np.array([0, 0, 1])
            current_up_dot = np.abs(np.dot(current_up_axis, world_z))

            # 检查与初始朝上轴的偏差
            up_axis_dot = np.dot(current_up_axis, initial_up_axis)
            up_axis_dot = np.clip(up_axis_dot, -1.0, 1.0)
            up_axis_angle = np.arccos(abs(up_axis_dot))

            # 检查位置
            center_distance = np.linalg.norm(current_pos[:2])

            # 检查插入状态
            insertion_ok = True
            insertion_info = ""

            if i > 0:
                # 检查是否插入到前一个杯子中
                prev_cup_id = placed_cups[i-1]["id"]
                prev_cup_pos, _ = robot.get_object_pose(prev_cup_id)
                prev_cup_size = robot.get_object_size_in_world_frame(prev_cup_id)

                if prev_cup_pos is not None and prev_cup_size is not None:
                    # 计算高度关系
                    current_cup_top = current_pos[2] + cup_size[2] / 2.0
                    prev_cup_top = prev_cup_pos[2] + prev_cup_size[2] / 2.0

                    # 对于插入式堆叠，当前杯子顶部应该接近或低于前一个杯子顶部
                    top_diff = current_cup_top - prev_cup_top

                    # 检查中心对齐
                    center_offset = np.linalg.norm(current_pos[:2] - prev_cup_pos[:2])

                    insertion_ok = (top_diff <= 0.05) and (center_offset <= 0.02)
                    insertion_info = f"Top diff: {top_diff:.3f}m, Center offset: {center_offset:.3f}m"

            # 综合评估
            upright_ok = current_up_dot > np.cos(np.deg2rad(30))
            axis_ok = up_axis_angle < np.deg2rad(45)
            position_ok = center_distance < 0.03

            cup_status = "✅" if (upright_ok and axis_ok and position_ok and (insertion_ok if i > 0 else True)) else "⚠️"

            print(f"Cup {i+1} ({cup_id}): {cup_status}")
            print(f"  Position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
            print(f"  Height above table: {current_pos[2] - table_top_z:.3f}m")
            print(f"  Center distance: {center_distance:.3f}m {'OK' if position_ok else 'TOO FAR'}")
            print(f"  Uprightness (dot): {current_up_dot:.3f} {'OK' if upright_ok else 'NOT UPRIGHT'}")
            print(f"  Up axis deviation: {np.rad2deg(up_axis_angle):.1f}° {'OK' if axis_ok else 'TOO LARGE'}")

            if i > 0:
                print(f"  Insertion: {'OK' if insertion_ok else 'NOT PROPERLY INSERTED'}")
                if insertion_info:
                    print(f"    {insertion_info}")

            if not (upright_ok and axis_ok and position_ok and (insertion_ok if i > 0 else True)):
                all_checks_passed = False

    # 返回手臂到初始位置
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n" + "="*60)
    print("TASK SUMMARY - INSERTION STACKING")
    print("="*60)

    if success and len(placed_cups) == len(cups) and all_checks_passed:
        print("✅ SUCCESS: All cups properly inserted and stacked!")
        print(f"   Stacking order (by distance from center):")
        for i, cup in enumerate(cups):
            print(f"     {i+1}. {cup['id']}: {cup['distance']:.3f}m")
        print(f"   Total cups inserted: {len(placed_cups)}")
    elif success and len(placed_cups) == len(cups) and not all_checks_passed:
        print("⚠️ PARTIAL: All cups placed but some insertion checks failed")
        print(f"   This may indicate shallow insertion or alignment issues")
    else:
        print(f"❌ PARTIAL: Inserted {len(placed_cups)}/{len(cups)} cups")

    return success and (len(placed_cups) == len(cups)) and all_checks_passed
```
