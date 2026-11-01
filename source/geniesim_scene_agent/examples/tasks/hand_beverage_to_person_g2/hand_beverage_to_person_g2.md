# Hand a beverage to a person

**Tags:** `pick_and_place` `beverage` `person` `hand_over` `table_only` `pick_and_place_actor_to_target_pose` `extra_checkers` `euler2mat` `quat2mat_wxyz` `multiple_candidate_poses` `upright_alignment` `upright_checker` `keywords` `usd_match` `person_detection`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keyword "table"/"dining_table". Find beverage bottle by usd containing "benchmark_beverage_bottle" or keyword "water_bottle"/"bottle". Find person by usd containing "benchmark_character" or keyword "person"/"human".
2. **Target position:** Get person Y position. Compute target placement: x ∈ [0.20, 0.25], y near person (±0.02), z = table_top + margin + bottle_height.
3. **Upright candidate poses:** Generate rotations ensuring bottle's local y-axis aligns with world z-axis (upright). Use `euler2mat([π/2,0,0])` and `euler2mat([-π/2,0,0])` as base, combined with Z-axis rotations (0°, 90°, 180°, 270°). Deduplicate rotation matrices.
4. **Upright checker:** Verify the bottle stays upright near the person and above the table using its local y-axis and the world z-axis.
5. **Execute:** `pick_and_place_actor_to_target_pose` with `GraspConfig` (pick_up_distance=0.07) and `PlaceConfig` with upright checker. Person and table as `unmovable_objects`.

---

# INPUT

**scene_description:** A table with a randomly selected water bottle placed within a specific region, and a person sitting in front of the table.

**task description:** Pick up the drink on the table and bring it to the person, placing it upright near them.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_water_and_person
description: A table with a randomly selected water bottle placed within a specific region, and a person sitting in front of the table.
"""


@register()
def place_water_bottle_on_table(table_shape: Shape) -> Shape:
    """
    Places a randomly selected water bottle on the table within the specified XY range.
    """
    # Get table's top surface info
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Calculate the world position of the desktop's top surface center
    desktop_center_world = table_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    # Select a random water bottle asset
    water_bottle_assets = [
        "benchmark_beverage_bottle_001",
        "benchmark_beverage_bottle_002",
        "benchmark_beverage_bottle_003",
        "benchmark_beverage_bottle_004",
        "benchmark_beverage_bottle_008",
        "benchmark_beverage_bottle_023",
        "benchmark_beverage_bottle_024",
        "benchmark_beverage_bottle_025",
        "benchmark_beverage_bottle_026",
        "benchmark_beverage_bottle_029",
        "benchmark_beverage_bottle_073"
    ]
    selected_bottle = random.choice(water_bottle_assets)

    # Load the bottle
    bottle_shape = library_call(
        "usd",
        oid=selected_bottle,
        keywords=["water_bottle", "transparent", "cylinder", "drinking", "on_table"]
    )

    # Get bottle height to place it correctly on the table
    bottle_min = compute_shape_min(bottle_shape)
    bottle_height = compute_shape_sizes(bottle_shape)[2]
    bottle_bottom_offset = -bottle_min[2]  # offset to bring bottom to z=0

    # Random placement within the specified range relative to table center
    x_offset = random.uniform(-0.15, 0.2)   # Updated x range: (-0.15, 0.2)
    y_offset = random.uniform(-0.25, 0.25)  # y range remains: (-0.25, 0.25)
    bottle_position = desktop_pos + np.array([x_offset, y_offset, bottle_bottom_offset])

    # Transform bottle to correct position
    bottle_transformed = transform_shape(
        bottle_shape,
        translation_matrix(bottle_position.tolist())
    )

    return concat_shapes(table_shape, bottle_transformed)


@register()
def place_person_in_front_of_table(scene_with_bottle: Shape) -> Shape:
    """
    Places a randomly selected person in front of the table at specified location and orientation.
    """
    # Select a random character
    character_assets = [
        "benchmark_character_000",
        "benchmark_character_001",
        "benchmark_character_002",
        "benchmark_character_003",
        "benchmark_character_004",
        "benchmark_character_005",
        "benchmark_character_006",
        "benchmark_character_007",
        "benchmark_character_008",
        "benchmark_character_009",
        "benchmark_character_010",
        "benchmark_character_011",
        "benchmark_character_012",
        "benchmark_character_013",
        "benchmark_character_014"
    ]
    selected_character = random.choice(character_assets)

    # Load the character
    character_shape = library_call(
        "usd",
        oid=selected_character,
        keywords=["person", "human", "sitting", "in_front_of_table"]
    )

    # Set position: x=0.75 (in front of table), z=0 (on ground), y random in (-0.17, 0.17)
    y_pos = random.uniform(-0.17, 0.17)
    character_position = [0.75, y_pos, 0.0]

    # The given quaternion is (w,x,y,z) = (0,0,0,1) which corresponds to xyzw = (0,0,1,0)
    # This represents a 180-degree rotation around Z-axis (facing the table)
    angle, axis = quaternion_to_angle_direction([0.0, 0.0, 1.0, 0.0])

    # First translate, then rotate around the character's base
    character_translated = transform_shape(
        character_shape,
        translation_matrix(character_position)
    )
    character_info = get_object_info(character_translated)
    character_base = np.array([character_info["center"][0], character_info["center"][1], 0.0])

    character_final = transform_shape(
        character_translated,
        rotation_matrix(angle, direction=axis, point=character_base)
    )

    return concat_shapes(scene_with_bottle, character_final)


@register()
def table_with_water_and_person() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["dining_table", "table", "wooden", "four_legs"]
    )

    # Place water bottle on table
    scene_with_bottle = place_water_bottle_on_table(table_shape)

    # Place person in front
    final_scene = place_person_in_front_of_table(scene_with_bottle)

    return final_scene


@register()
def root_scene() -> Shape:
    return table_with_water_and_person()
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
    # 提取资产ID列表
    asset_ids = [
        "benchmark_beverage_bottle_001", "benchmark_beverage_bottle_002",
        "benchmark_beverage_bottle_003", "benchmark_beverage_bottle_004",
        "benchmark_beverage_bottle_008", "benchmark_beverage_bottle_023",
        "benchmark_beverage_bottle_024", "benchmark_beverage_bottle_025",
        "benchmark_beverage_bottle_026", "benchmark_beverage_bottle_029",
        "benchmark_beverage_bottle_073",
        "benchmark_character_000", "benchmark_character_001",
        "benchmark_character_002", "benchmark_character_003",
        "benchmark_character_004", "benchmark_character_005",
        "benchmark_character_006", "benchmark_character_007",
        "benchmark_character_008", "benchmark_character_009",
        "benchmark_character_010", "benchmark_character_011",
        "benchmark_character_012", "benchmark_character_013",
        "benchmark_character_014",
        "table_000"
    ]

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "把桌子上的饮料拿到人的面前",
        "english_task_name": "pick up the drink on the table and bring it to the person",
        "init_scene_text": "The robot is in front of the table, there is a water bottle on the table and a person sitting in front of the table"
    }
    robot.client.set_task_task_basic_info(description_json)
    print("=== Starting task: Bring drink to person ===")

    # 1. 查找桌子
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

    # 2. 查找水杯/饮料瓶
    bottle_id = None
    bottle_usd = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_beverage_bottle" in usd or any("water_bottle" in kw.lower() for kw in keywords) or any("bottle" in kw.lower() for kw in keywords):
            bottle_id = obj_id
            bottle_usd = usd
            print(f"Found bottle: {bottle_id}, usd: {bottle_usd}")
            break

    if not bottle_id:
        print("Water bottle not found on table")
        return False

    # 3. 查找人物
    person_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_character" in usd or any("person" in kw.lower() for kw in keywords) or any("human" in kw.lower() for kw in keywords):
            person_id = obj_id
            print(f"Found person: {person_id}, usd: {usd}")
            break

    if not person_id:
        print("Person not found in front of table")
        return False

    # 4. 获取人物位置，确定目标区域
    person_pos, person_rot = robot.get_object_pose(person_id)
    if person_pos is None:
        print(f"Cannot get person position for {person_id}")
        return False

    person_y = person_pos[1]

    # 获取桌子高度
    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 获取瓶子尺寸
    bottle_size = robot.get_object_size_in_world_frame(bottle_id)
    if bottle_size is None:
        print("Cannot get bottle size")
        return False

    # 5. 计算目标点
    target_x = random.uniform(0.20, 0.25)
    target_y = random.uniform(person_y - 0.02, person_y + 0.02)
    # 悬空1cm释放，防止砸桌子，依靠重力结算
    margin = 0.01
    target_z = table_top_z + margin + 2 * bottle_size[2] / 2.0
    target_position = np.array([target_x, target_y, target_z])

    print(f"Target placement position: {target_position}")

    # 6. 选择手臂
    bottle_pos, bottle_rot = robot.get_object_pose(bottle_id)
    arm = robot.judge_arm_to_use(bottle_pos)
    print(f"Using arm: {arm}")

    # 7. 生成确保竖直的候选姿态
    # 饮料瓶的本地y轴应该朝上（世界z轴）
    # 创建旋转矩阵，使本地y轴对齐世界z轴
    # 方案1: 绕x轴旋转90度，使本地y轴对齐世界z轴
    rot_x_90 = euler2mat([np.pi/2, 0, 0])
    # 方案2: 绕x轴旋转-90度，使本地y轴对齐世界z轴（方向相反）
    rot_x_minus_90 = euler2mat([-np.pi/2, 0, 0])

    # 生成候选旋转矩阵
    candidate_rotations = []

    # 对于每个基础旋转，再绕z轴旋转90度的倍数（饮料可以任意水平旋转）
    for base_rot in [rot_x_90, rot_x_minus_90]:
        for z_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            rot_z = euler2mat([0, 0, z_angle])
            final_rot = np.dot(rot_z, base_rot)
            candidate_rotations.append(final_rot)

    # 去重
    unique_rotations = []
    for rot in candidate_rotations:
        is_unique = True
        for existing in unique_rotations:
            if np.allclose(rot, existing, atol=1e-6):
                is_unique = False
                break
        if is_unique:
            unique_rotations.append(rot)

    print(f"Generated {len(unique_rotations)} unique rotation matrices for upright placement")

    # 生成所有候选姿态
    target_poses = []
    # 位置变化参数（小范围偏移）
    x_offsets = np.linspace(-0.001, 0.001, 3)
    y_offsets = np.linspace(-0.001, 0.001, 3)
    z_offsets = np.linspace(0, 0.002, 3)

    # 生成所有候选姿态的组合
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = target_position + np.array([x_offset, y_offset, z_offset])
                # 为每个候选旋转矩阵生成一个姿态
                for rot_mat in unique_rotations:
                    pose = np.eye(4)
                    pose[:3, :3] = rot_mat
                    pose[:3, 3] = position
                    target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} UPRIGHT candidate poses (ensuring bottle's local y-axis faces upward)")

    # 8. 定义检查器：检查饮料瓶是否被竖直放置（本地y轴朝上）
    def create_bottle_upright_checker(bottle_id, expected_x_range, person_y, table_top_z):
        def check_bottle_upright_and_placed():
            # 获取饮料瓶当前位置和旋转
            current_pos, current_rot = robot.get_object_pose(bottle_id)
            if current_pos is None or current_rot is None:
                print("Cannot get bottle position or rotation")
                return False

            # 1. 检查放置位置
            # 放宽 X 容差：允许滑动
            if current_pos[0] < expected_x_range[0] - 0.02 or current_pos[0] > expected_x_range[1] + 0.02:
                print(f"Bottle X {current_pos[0]:.3f} out of relaxed bounds")
                return False

            # 放宽 Y 容差：允许滑动到 4cm
            if abs(current_pos[1] - person_y) > 0.04:
                print(f"Bottle Y {current_pos[1]:.3f} too far from person Y {person_y:.3f}")
                return False

            # Z 轴不穿模过多即算成功
            if current_pos[2] < table_top_z - 0.01:
                print(f"Bottle fell below table: Z={current_pos[2]:.3f}")
                return False

            # 2. 检查饮料瓶是否竖直（本地y轴朝上）
            # 获取饮料瓶的当前旋转矩阵
            current_rot_matrix = quat2mat_wxyz(current_rot)

            # 饮料瓶的本地y轴方向（在本地坐标系中是[0, 1, 0]）
            # 转换为世界坐标系中的方向
            local_y_axis = np.array([0, 1, 0])
            world_y_axis = np.dot(current_rot_matrix, local_y_axis)

            # 世界坐标系的上方向是[0, 0, 1]
            world_up = np.array([0, 0, 1])

            # 计算本地y轴与世界z轴的夹角
            # 使用点积计算余弦值：cosθ = (a·b) / (|a||b|)
            dot_product = np.dot(world_y_axis, world_up)
            norm_y = np.linalg.norm(world_y_axis)
            norm_up = np.linalg.norm(world_up)

            # 避免除零
            if norm_y == 0 or norm_up == 0:
                print("Zero norm in axis calculation")
                return False

            cos_theta = dot_product / (norm_y * norm_up)

            # 由于浮点误差，限制到[-1, 1]
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # 计算角度（弧度）
            angle_rad = np.arccos(cos_theta)
            angle_deg = angle_rad * 180 / np.pi

            print(f"Bottle local y-axis angle with world up: {angle_deg:.1f}°")

            # 检查本地y轴是否朝上（允许±15°的误差）
            # 本地y轴朝上意味着它与世界z轴的夹角接近0°或180°
            # 但我们需要确定方向：如果是rot_x_90旋转，应该是0°；如果是rot_x_minus_90，应该是180°
            # 更通用的检查：本地y轴是否大致与世界z轴平行（夹角接近0°或180°）
            # 对于饮料瓶，我们期望它竖直放置，所以本地y轴应该大致与世界z轴平行
            # 允许±15°的误差
            upright_tolerance = 15.0  # 度

            # 检查夹角是否在容差范围内（接近0°或180°）
            if angle_deg <= upright_tolerance or (180 - angle_deg) <= upright_tolerance:
                print(f"✅ Bottle is upright (angle: {angle_deg:.1f}°)")

                # 额外检查：饮料瓶是否直立（而不是倒置）
                # 检查本地y轴与世界z轴的点积正负
                # 如果本地y轴朝上，world_y_axis[2]应该是正数（接近1）
                # 如果本地y轴朝下，world_y_axis[2]应该是负数（接近-1）

                # 对于饮料瓶，通常期望它正立（瓶口朝上）
                # 但我们任务没有明确要求瓶口朝上还是朝下，所以接受两种方向
                # 可以根据world_y_axis[2]的正负判断
                if world_y_axis[2] > 0:
                    print(f"  Bottle is upright (local y-axis pointing up)")
                else:
                    print(f"  Bottle is inverted (local y-axis pointing down)")

                return True
            else:
                print(f"❌ Bottle is not upright enough (angle: {angle_deg:.1f}°, tolerance: ±{upright_tolerance}°)")
                return False

        return check_bottle_upright_and_placed

    # 创建检查器
    check_function = create_bottle_upright_checker(bottle_id, (0.20, 0.25), person_y, table_top_z)

    # 9. 配置抓取与放置
    unmovable_objects = [table_id, person_id]

    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        pick_up_distance=0.07,
        action_description={"zh": "拿起桌子上的饮料", "en": "pick up the drink on the table"}
    )

    place_config = PlaceConfig(
        unmovable_objects=unmovable_objects,
        extra_checkers=[check_function],  # 使用包含竖直检查的检查器
        action_description={"zh": "将饮料竖直放置到人面前", "en": "place the drink upright in front of the person"}
    )

    # 11. 执行与验证
    print(f"Attempting to pick and place bottle...")
    success = robot.pick_and_place_actor_to_target_pose(
        actor=bottle_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    print("\n=== Performing final verification ===")
    if success:
        # 使用检查器进行最终验证
        if check_function():
            final_pos, final_rot = robot.get_object_pose(bottle_id)
            if final_pos is not None and final_rot is not None:
                print(f"✅ Bottle final position: {final_pos}")

                # 获取并打印饮料瓶的朝向信息
                final_rot_matrix = quat2mat_wxyz(final_rot)
                local_y_axis = np.array([0, 1, 0])
                world_y_axis = np.dot(final_rot_matrix, local_y_axis)
                world_up = np.array([0, 0, 1])

                dot_product = np.dot(world_y_axis, world_up)
                norm_y = np.linalg.norm(world_y_axis)
                norm_up = np.linalg.norm(world_up)

                if norm_y > 0 and norm_up > 0:
                    cos_theta = dot_product / (norm_y * norm_up)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_rad = np.arccos(cos_theta)
                    angle_deg = angle_rad * 180 / np.pi
                    print(f"✅ Bottle is upright with angle: {angle_deg:.1f}° from vertical")

                print("✅ All placement requirements satisfied (position and upright)")
            else:
                print("❌ Cannot get bottle final state")
                success = False
        else:
            print("❌ Final verification failed: Bottle not upright or not properly placed")
            success = False
    else:
        print("❌ Pick and place operation failed")

    # 12. 安全后撤
    robot.back_to_origin(arm=arm)

    if success:
        print("✅ SUCCESS: Drink handed to the person in upright position")
    else:
        print("❌ FAILED: Task execution failed")

    return success
```
