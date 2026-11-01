# Stack bowls

**Tags:** `stack` `bowls` `table_only` `pose_based_placement` `sort_by_height` `generate_bowl_rotations` `rotate_around_z` `quat2mat_wxyz` `multiple_candidate_poses` `safety_margin` `unmovable_objects` `extra_checkers` `benchmark_bowl_025` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keywords "table"/"dining_table". Find all bowls by usd "benchmark_bowl_025" or "bowl_" in keywords; collect id, size, position, rotation.
2. **Order:** Sort bowls by z (height): bowls.sort(key=lambda x: x["position"][2]). Stack from lowest to highest.
3. **Target position:** First bowl: table center (0, 0), target_z = table_top_z + safety_margin + bowl_size[2]/2. Subsequent bowls: target = previous bowl top; target_z = prev_bowl_top_z - height_offset + bowl_bottom_to_center (height_offset 0.02–0.03 m).
4. **Candidate poses:** generate_bowl_rotations(base_rotation): quat2mat_wxyz + 36 rotations around world z; combine with small xy/z offsets. unmovable_objects = [table_id] + already stacked bowl ids.
5. **Verification:** extra_checkers: current bowl xy within 5 cm of expected position.
6. **Execution:** pick_and_place_actor_to_target_pose(actor=bowl_id, target_pose=target_poses, arm, grasp_config, place_config). Back to origin on arm switch.

---

# INPUT

**scene_description:** A table with several bowls (benchmark_bowl_025). Bowls may have keywords like "bowl_". The robot must stack all bowls (e.g. at table center), with the lowest bowl first and then stacking upward.

**task description:** Stack all bowls. Sort bowls by z (height); place the lowest first at table center, then stack each next bowl on top of the previous one.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_three_bowls
description: Three bowls randomly placed on a table, avoiding an expanded central forbidden zone (x: -0.13 to 0.13, y: -0.13 to 0.13), with minimum distance of 0.25m between any two bowls, and random rotation.
"""


def get_desktop_workspace_info(desk_shape: Shape) -> Tuple[P, P]:
    """
    Retrieves the top surface center and size of the desktop.

    Args:
        desk_shape (Shape): The shape of the desk.

    Returns:
        Tuple[P, P]: (desktop_center_on_top, desktop_size)
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def point_in_forbidden_zone(
    x: float, y: float, forbidden_x_min: float, forbidden_x_max: float, forbidden_y_min: float, forbidden_y_max: float
) -> bool:
    """Check if a point (x, y) is inside the central forbidden rectangle."""
    return (forbidden_x_min <= x <= forbidden_x_max) and (forbidden_y_min <= y <= forbidden_y_max)


@register()
def single_bowl(keywords: List[str]) -> Shape:
    """Creates a single bowl with given keywords."""
    return library_call("usd", oid="benchmark_bowl_025", keywords=keywords)


@register()
def three_well_spaced_rotated_bowls_on_table(desk_shape: Shape) -> Shape:
    desktop_center, _ = get_desktop_workspace_info(desk_shape)

    # Define the placement zones relative to the desktop center
    outer_x_min, outer_x_max = -0.2, 0.2
    outer_y_min, outer_y_max = -0.3, 0.3
    # Further expanded forbidden central zone
    forbidden_x_min, forbidden_x_max = -0.15, 0.15
    forbidden_y_min, forbidden_y_max = -0.15, 0.15

    min_separation = 0.25  # Increased minimum distance between any two bowls

    placed_positions = []
    bowl_shapes = []

    for i in range(3):
        max_attempts = 1000  # Significantly increased attempts due to very tight constraints
        for attempt in range(max_attempts):
            # Sample a candidate position in the world frame
            x_offset = random.uniform(outer_x_min, outer_x_max)
            y_offset = random.uniform(outer_y_min, outer_y_max)

            # Skip if in the forbidden central zone
            if point_in_forbidden_zone(
                x_offset, y_offset, forbidden_x_min, forbidden_x_max, forbidden_y_min, forbidden_y_max
            ):
                continue

            candidate_pos = np.array([desktop_center[0] + x_offset, desktop_center[1] + y_offset, desktop_center[2]])

            # Check for collision with already placed bowls
            collision = False
            for placed_pos in placed_positions:
                dist = np.linalg.norm(candidate_pos[:2] - placed_pos[:2])
                if dist < min_separation:
                    collision = True
                    break

            if not collision:
                placed_positions.append(candidate_pos)
                break
        else:
            print(f"Failed to place bowl {i+1} after {max_attempts} attempts")
            # Fallback: place at safe default locations that are guaranteed to be outside the forbidden zone and far apart
            # These positions are carefully chosen to be near the corners/edges of the allowed area
            fallback_offsets = [(-0.145, -0.245), (0.095, -0.245), (-0.025, 0.245)]
            fallback_pos = np.array(
                [
                    desktop_center[0] + fallback_offsets[i][0],
                    desktop_center[1] + fallback_offsets[i][1],
                    desktop_center[2],
                ]
            )
            placed_positions.append(fallback_pos)

        # Create the bowl
        bowl = library_call(
            "single_bowl",
            keywords=[f"bowl_{i+1}", "white", "ceramic", "kitchenware", "on_table"],
        )

        # Apply random rotation around the Z-axis (vertical)
        random_angle = random.uniform(0, 2 * math.pi)
        bowl = transform_shape(bowl, rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=(0, 0, 0)))

        # Then translate to the final position
        bowl = transform_shape(bowl, translation_matrix(placed_positions[-1]))
        bowl_shapes.append(bowl)

    return concat_shapes(desk_shape, *bowl_shapes)


@register()
def table_with_three_bowls() -> Shape:
    desk = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "furniture"])
    return three_well_spaced_rotated_bowls_on_table(desk)


@register()
def root_scene() -> Shape:
    return table_with_three_bowls()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 从交互数据中初始化objects_list
    objects_list = {
        "benchmark_bowl_025": {
            "has_interactions": True,
            "object_type": "bowl",
            "interaction_type": "container"
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

    # MANDATORY: Set task description before first action
    task_description = {
        "task_name": "叠碗",
        "english_task_name": "stack bowls",
        "init_scene_text": "The robot is in front of the table, there are several bowls on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

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

    # 2. 查找所有碗 (benchmark_bowl_025)
    bowls = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_bowl_025" in usd or any("bowl_" in kw for kw in keywords):
            size = robot.get_object_size_in_world_frame(obj_id)
            if size is not None:
                pos, rot = robot.get_object_pose(obj_id)
                bowls.append({
                    "id": obj_id,
                    "size": size,
                    "position": pos,
                    "rotation": rot,
                    "keywords": keywords,
                    "usd": usd
                })
                print(f"Found bowl: {obj_id}, size: {size}, position: {pos}")

    if len(bowls) < 2:
        print(f"Found only {len(bowls)} bowls, need at least 2")
        return False

    print(f"Total bowls found: {len(bowls)}")

    # 3. 获取桌子信息用于计算堆叠位置
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    # 桌子顶部高度
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 4. 计算堆叠位置参数
    # 目标位置：桌子正中心 (0, 0)
    target_x = 0.0
    target_y = 0.0

    # 5. 按z坐标从低到高排序（从最接近桌面的开始）
    bowls.sort(key=lambda x: x["position"][2])
    print("Bowls sorted by height (lowest to highest):")
    for i, bowl in enumerate(bowls):
        print(f"  {i+1}. {bowl['id']}: height={bowl['position'][2]:.3f}")

    # 6. 提取碗的初始旋转并生成绕z轴的候选旋转
    def generate_bowl_rotations(base_rotation):
        """生成绕世界z轴旋转的候选姿态"""
        candidate_rotations = []

        # 提取基础旋转矩阵
        base_rot_mat = quat2mat_wxyz(base_rotation)
        candidate_rotations.append(base_rot_mat)

        # 生成绕z轴的不同角度
        for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_z = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])

            # 组合旋转：原始旋转 + 绕z轴随机旋转
            final_rot = np.dot(rot_z, base_rot_mat)
            candidate_rotations.append(final_rot)

        print(f"Generated {len(candidate_rotations)} rotation variations around z-axis")
        return candidate_rotations

    # 7. 执行堆叠任务（从最低的碗开始）
    stacked_bowls = []
    last_arm = ""

    for i, bowl in enumerate(bowls):
        bowl_id = bowl["id"]
        bowl_size = bowl["size"]
        bowl_position = bowl["position"]
        bowl_rotation = bowl["rotation"]

        print(f"\n=== Processing bowl {i+1}/{len(bowls)}: {bowl_id} ===")

        # 判断使用哪只手臂
        bowl_arm = robot.judge_arm_to_use(bowl_position)
        print(f"Using arm: {bowl_arm}")

        # 切换手臂时需要返回原点
        if last_arm != bowl_arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = bowl_arm

        # 计算堆叠高度
        if i == 0:
            # 第一个碗：放在桌子中心
            # 根据任务要求，碗底应比另一个碗口高0.02~0.03米
            # 由于这是第一个碗，我们放在桌子上，需要考虑碗的高度
            margin = 0.025  # 取中间值0.025米
            # 碗在桌子上的高度 = 桌子顶部 + 安全边距 + 半个碗高度
            safety_margin = 0.01  # 避免夹爪碰撞的安全边距
            target_z = table_top_z + safety_margin + bowl_size[2] / 2.0
        else:
            # 后续碗：放在前一个碗上面
            prev_bowl_id = stacked_bowls[-1]["id"]
            prev_bowl_pos, prev_bowl_rot = robot.get_object_pose(prev_bowl_id)
            prev_bowl_size = robot.get_object_size_in_world_frame(prev_bowl_id)

            if prev_bowl_pos is None or prev_bowl_size is None:
                print(f"Cannot get previous bowl {prev_bowl_id} info")
                success = False
                break

            # 计算前一个碗的碗口高度
            # 碗口大致在碗的顶部
            prev_bowl_top_z = prev_bowl_pos[2] + prev_bowl_size[2] / 2.0

            # 当前碗底应比前一个碗口高0.02~0.03米
            # 碗底到碗中心的高度大约是半个碗高度
            bowl_bottom_to_center = bowl_size[2] / 2.0
            height_offset = np.random.uniform(0.02, 0.03)  # 随机在0.02-0.03米之间

            safety_margin = 0.01  # 避免夹爪碰撞的安全边距
            target_z = prev_bowl_top_z - height_offset + bowl_bottom_to_center
            target_x = prev_bowl_pos[0]
            target_y = prev_bowl_pos[1]

        base_position = np.array([target_x, target_y, target_z])

        # 创建抓取配置
        unmovable_objects = [table_id]
        # 保护已经堆叠好的碗
        unmovable_objects.extend([b["id"] for b in stacked_bowls])

        # 根据交互数据，碗有grasp注释点
        grasp_config = GraspConfig(
            pick_up_distance= 0.1 + i * 0.05,
            passive_primitive="default",
            unmovable_objects=unmovable_objects,
            action_description={"zh": "拿起碗", "en": "pick up the bowl"}  # REQUIRED
        )

        # 生成候选旋转姿态
        candidate_rotations = generate_bowl_rotations(bowl_rotation)

        # 生成大量候选姿态
        target_poses = []

        # 位置变化参数（小范围偏移）
        x_offsets = np.linspace(-0.001, 0.001, 3)
        y_offsets = np.linspace(-0.001, 0.001, 3)
        z_offsets = np.linspace(0, 0.005, 3)  # 高度微调

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
        print(f"Generated {len(target_poses)} candidate poses for bowl {bowl_id}")

        # 定义检查函数：验证碗是否成功堆叠
        def create_check_function(bowl_id, expected_position, expected_height, is_bottom=False):
            def check_bowl_stacked():
                current_pos, current_rot = robot.get_object_pose(bowl_id)
                if current_pos is None or current_rot is None:
                    return False

                # 位置检查（xy平面容差5cm）
                xy_distance = np.linalg.norm(current_pos[:2] - expected_position[:2])
                xy_tolerance = 0.05
                if xy_distance > xy_tolerance:
                    print(f"Bowl {bowl_id} xy position error: {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    return False

                return True

            return check_bowl_stacked

        # 创建检查函数
        is_bottom = (i == 0)
        check_function = create_check_function(bowl_id, base_position, target_z, is_bottom)

        # 根据堆叠位置确定放置描述
        if i == 0:
            place_desc_zh = "将第一个碗放置到桌子正中心"
            place_desc_en = "place the bottom bowl at the center of the table"
        elif i == 1:
            place_desc_zh = "将第二个碗放置到第一个碗上方"
            place_desc_en = "place the second bowl on top of the first bowl"
        elif i == 2:
            place_desc_zh = "将第三个碗放置到碗塔的顶部"
            place_desc_en = "place the third bowl on top of the bowl tower"
        else:
            place_desc_zh = f"将第{i+1}个碗放置到碗塔的顶部"
            place_desc_en = f"place the {i+1}th bowl on top of the bowl tower"

        # 根据交互数据，碗有place注释点
        place_config = PlaceConfig(
            active_primitive="default",
            passive_primitive="default",
            unmovable_objects=unmovable_objects,
            pre_insert_offset=0.1,
            extra_checkers=[check_function],
            action_description={"zh": place_desc_zh, "en": place_desc_en}  # REQUIRED
        )

        # 执行抓取和放置（使用pick_and_place_actor_to_target_pose，因为碗有交互数据但需要精确控制堆叠高度）
        print(f"Attempting to pick and place bowl {bowl_id} with {len(target_poses)} candidate poses")

        # 使用交互数据中的抓取点
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=bowl_id,
            target_pose=target_poses,
            arm=bowl_arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not current_success:
            print(f"Failed to place bowl {bowl_id}")
            success = False
            break
        else:
            print(f"Successfully placed bowl {bowl_id}")

        # 添加到已堆叠列表
        stacked_bowls.append({"id": bowl_id, "position": base_position})

    # 8. 最终验证
    print("\n=== Performing final verification ===")

    if len(stacked_bowls) > 0:
        # 检查塔的稳定性
        tower_height = 0
        for i, bowl_info in enumerate(stacked_bowls):
            bowl_id = bowl_info["id"]
            current_pos, current_rot = robot.get_object_pose(bowl_id)

            if current_pos is None:
                print(f"Bowl {bowl_id} not found after placement")
                success = False
                continue

            # 检查是否在桌子中心附近
            center_distance = np.linalg.norm(current_pos[:2])
            if center_distance > 0.05:  # 允许5cm偏差
                print(f"  Warning: Bowl {bowl_id} is {center_distance:.3f}m from center")

            # 计算塔高
            if i == 0:
                tower_height = current_pos[2] - table_top_z
            else:
                # 检查上层碗是否直接在下层上面
                prev_bowl_id = stacked_bowls[i-1]["id"]
                prev_bowl_pos, _ = robot.get_object_pose(prev_bowl_id)
                if prev_bowl_pos is not None:
                    height_diff = current_pos[2] - prev_bowl_pos[2]
                    # 碗之间的高度差应该在合理范围内
                    if height_diff < 0.01 or height_diff > 0.1:
                        print(f"  Warning: Height difference between bowls {i} and {i-1}: {height_diff:.3f}m")

        print(f"Bowl tower height: {tower_height:.3f}m")
        print(f"Number of bowls in tower: {len(stacked_bowls)}")

    # 返回手臂到初始位置
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success and len(stacked_bowls) == len(bowls):
        print("✅ SUCCESS: All bowls stacked into a tower at the center of the table")
    else:
        print(f"⚠️ PARTIAL SUCCESS: Stacked {len(stacked_bowls)} out of {len(bowls)} bowls")

    return success and (len(stacked_bowls) == len(bowls))
```
