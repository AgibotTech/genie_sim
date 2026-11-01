# Disassemble a gear or bearing

**Tags:** `disassembly` `gearbox` `gear` `bearing` `keywords_mounted` `on_cylinder` `pick_actor` `GraspConfig` `extra_checkers` `lift_verification`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keywords "main_table"/"table". Find gearbox by usd "benchmark_gearbox_000" or keywords "gearbox_center". Find mounted part: object whose usd is one of benchmark_gear_000/001, benchmark_bearing_000/001 and keywords contain "mounted" or "on_cylinder".
2. **Part type:** From part usd set part_type_zh/en and part_size_category (small/large). gear_000→large, gear_001→small; bearing_001→large, bearing_000→small (per data_gen logic).
3. **Pick config:** pick_up_distance = 0.20 for large else 0.15. GraspConfig with passive_primitive="default", set_grasp_along_axis="y", unmovable_objects=[table_id, gearbox_id]. extra_checkers: verify part z increased by at least 0.1 m after pick (create_pick_height_checker).
4. **Execution:** Set task description; robot.pick_actor(actor=mounted_part_id, arm=arm, grasp_config=grasp_config). No place step in this task (unmount = lift only).

---

# INPUT

**scene_description:** A table with a gearbox that has a gear or bearing already mounted on one of its cylinders. The mounted part has keywords like "mounted" or "on_cylinder". The robot must pick up the assembled part (unmount it).

**task description:** Pick up the assembled part (the gear or bearing that is mounted on the gearbox).

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: table_with_gearbox_and_mounted_part
description: A table with a gearbox and a gear/bearing already mounted on a specific cylinder.
             The part is placed according to specific rules: gear_000 and bearing_001 on right/back cylinders,
             others on the small cylinder. Coordinates are relative to the gearbox.
"""


@register()
def table_with_gearbox_and_mounted_part() -> Shape:
    # 1. Load the table
    table_shape = library_call("usd", oid="table_000", keywords=["main_table", "furniture", "table_000"])
    table_info = get_object_info(table_shape)

    # Calculate desktop center and height
    # The table center (x, y) is table_info["center"][:2]
    # The top surface height is table_info["max"][2]
    desktop_center_x = table_info["center"][0]
    desktop_center_y = table_info["center"][1]
    desktop_z = table_info["max"][2] + 0.1

    # 2. Load and Place Gearbox
    # Gearbox parameters: Random rotation within 5 degrees around Z
    g_rot_deg = np.random.uniform(0, 360)
    g_rot_rad = np.radians(g_rot_deg)

    # Gearbox position: Centered on the desktop
    g_x = np.random.uniform(-0.15, 0.1)
    g_y = np.random.uniform(-0.15, 0.15)
    g_pos = np.array([desktop_center_x + g_x, desktop_center_y + g_y, desktop_z])

    # Create gearbox transformation matrices
    # Rotation around Z axis at origin
    g_rot_mat = rotation_matrix(g_rot_rad, (0, 0, 1), (0, 0, 0))
    # Translation to desktop position
    g_trans_mat = translation_matrix(g_pos)
    # Combine: Rotation then Translation
    g_pose = g_trans_mat @ g_rot_mat

    # Load gearbox shape
    gearbox_shape = library_call(
        "usd",
        oid="benchmark_gearbox_000",
        keywords=["gearbox_center", "mechanical_part", "fixed"],
    )
    # Apply transformation
    gearbox_shape = transform_shape(gearbox_shape, g_pose)

    # 3. Select Part and Determine Target Cylinder
    # Define local coordinates for cylinders (relative to gearbox center)
    # Based on user provided data
    cyl_local_coords = {
        "small_cylinder": np.array([0.049, -0.029, 0.03872 - 0.02]),
        "right_cylinder": np.array([0.019, 0.06, 0.11479 - 0.06]),
        "back_cylinder": np.array([-0.061, -0.019, 0.11479 - 0.06]),
    }

    # List of available parts
    parts_pool = [
        "benchmark_gear_000",  # Large Gear
        "benchmark_bearing_001",  # Small Bearing
        "benchmark_gear_001",  # Small Gear
        "benchmark_bearing_000",  # Large Bearing
    ]

    # Randomly select one part to place in the scene
    part_oid = np.random.choice(parts_pool)

    # Determine target cylinder based on user rules:
    # "gear_000 and bearing_001 on right or back, others on small"
    target_cylinders = []
    if part_oid in ["benchmark_gear_000", "benchmark_bearing_001"]:
        target_cylinders = ["right_cylinder", "back_cylinder"]
    else:
        target_cylinders = ["small_cylinder"]

    target_cyl_name = np.random.choice(target_cylinders)
    target_local_pos = cyl_local_coords[target_cyl_name]

    # 4. Calculate Part World Position and Rotation
    # Calculate world position: P_world = T_gearbox * P_local
    # We need to treat P_local as a homogeneous point [x, y, z, 1]
    p_local_h = np.append(target_local_pos, 1.0)
    p_world_h = g_pose @ p_local_h
    part_pos = p_world_h[:3]

    # Calculate Part Rotation
    # The part is mounted on the cylinder.
    # The cylinder axis is rotated by g_rot_rad relative to world Z.
    # To align the part's hole with the cylinder, the part must rotate by g_rot_rad.
    # Additionally, the user allows "arbitrary random rotation around Z".
    # This means the part can spin around the cylinder axis.
    # Total Rotation = Gearbox_Rotation + Random_Spin
    p_spin_rad = np.random.uniform(0, 2 * np.pi)
    p_total_rot_rad = g_rot_rad + p_spin_rad

    # 5. Load and Place Part
    part_shape = library_call(
        "usd",
        oid=part_oid,
        keywords=[f"mounted_{part_oid}", "mechanical", "on_cylinder", target_cyl_name],
    )

    # Create part transformation
    # Rotate around Z, then translate to calculated world position
    part_rot_mat = rotation_matrix(p_total_rot_rad, (0, 0, 1), (0, 0, 0))
    part_trans_mat = translation_matrix(part_pos)
    part_pose = part_trans_mat @ part_rot_mat

    part_shape = transform_shape(part_shape, part_pose)

    # 6. Combine shapes
    return concat_shapes(table_shape, gearbox_shape, part_shape)


@register()
def root_scene() -> Shape:
    return library_call("table_with_gearbox_and_mounted_part")
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # 基于场景生成代码和交互数据初始化objects_list
    objects_list = {
        "benchmark_gearbox_000": {
            "has_interactions": True,
            "object_type": "gearbox",
            "interaction_type": "container_with_cylinders"
        },
        "benchmark_gear_000": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "large",
            "interaction_type": "mechanical_part"
        },
        "benchmark_gear_001": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "small",
            "interaction_type": "mechanical_part"
        },
        "benchmark_bearing_000": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "large",
            "interaction_type": "mechanical_part"
        },
        "benchmark_bearing_001": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "small",
            "interaction_type": "mechanical_part"
        }
    }

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: 在第一个动作前设置任务描述
    description_json = {
        "task_name": "把装配好的零件拿起来",
        "english_task_name": "pick up the assembled part",
        "init_scene_text": "The robot is in front of the table, there is a gearbox with a gear or bearing mounted on one of its cylinders"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. 查找桌子
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "main_table" in keywords or "table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. 查找齿轮盒
    gearbox_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_gearbox_000" in usd or "gearbox_center" in keywords:
            gearbox_id = obj_id
            break

    if not gearbox_id:
        print("Gearbox not found")
        return False

    print(f"Found gearbox: {gearbox_id}")

    # 3. 查找已装配的齿轮/轴承
    mounted_part_id = None
    mounted_part_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是齿轮或轴承
        is_part = False
        for part_usd in ["benchmark_gear_000", "benchmark_gear_001",
                        "benchmark_bearing_000", "benchmark_bearing_001"]:
            if part_usd in usd:
                is_part = True
                break

        if is_part and (any("mounted" in kw for kw in keywords) or any("on_cylinder" in kw for kw in keywords)):
            mounted_part_id = obj_id
            mounted_part_info = obj_info
            break

    if not mounted_part_id:
        print("Mounted part not found")
        return False

    part_usd = mounted_part_info.get("usd", "")
    print(f"Found mounted part: {mounted_part_id}, usd: {part_usd}")

    # 4. 确定零件的具体类型和大小
    part_type_zh = ""
    part_type_en = ""
    part_size_category = None  # "small" or "large"

    if "benchmark_gear_000" in part_usd:
        part_type_zh = "大齿轮"
        part_type_en = "large gear"
        part_size_category = "large"
    elif "benchmark_gear_001" in part_usd:
        part_type_zh = "小齿轮"
        part_type_en = "small gear"
        part_size_category = "small"
    elif "benchmark_bearing_001" in part_usd:
        part_type_zh = "大轴承"
        part_type_en = "large bearing"
        part_size_category = "large"
    elif "benchmark_bearing_000" in part_usd:
        part_type_zh = "小轴承"
        part_type_en = "small bearing"
        part_size_category = "small"
    else:
        # 尝试通过关键词判断
        keywords = mounted_part_info.get("keywords", [])
        if "gear" in part_usd.lower() or any("gear" in kw.lower() for kw in keywords):
            base_zh = "齿轮"
            base_en = "gear"
        elif "bearing" in part_usd.lower() or any("bearing" in kw.lower() for kw in keywords):
            base_zh = "轴承"
            base_en = "bearing"
        else:
            base_zh = "零件"
            base_en = "part"

        if any("large" in kw.lower() for kw in keywords):
            part_type_zh = "大" + base_zh
            part_type_en = "large " + base_en
            part_size_category = "large"
        elif any("small" in kw.lower() for kw in keywords):
            part_type_zh = "小" + base_zh
            part_type_en = "small " + base_en
            part_size_category = "small"
        else:
            part_type_zh = base_zh
            part_type_en = base_en
            # 默认为小
            part_size_category = "small"

    print(f"Part identified: {part_type_zh} / {part_type_en}, size: {part_size_category}")

    # 5. 设置 pick_up_distance
    if part_size_category == "large":
        pick_up_distance = 0.20  # 大零件 0.2m
    else:
        pick_up_distance = 0.15  # 小零件 0.15m

    print(f"Pick up distance: {pick_up_distance}m")

    # 6. 获取零件的初始位置（用于checker）
    initial_part_position, _ = robot.get_object_pose(mounted_part_id)
    if initial_part_position is None:
        print(f"Cannot get initial position for part {mounted_part_id}")
        return False

    print(f"Initial part position: {initial_part_position}")

    # 7. 定义检查函数：判断零件位置是否至少增加10cm
    def create_pick_height_checker(part_id, initial_position):
        def checker():
            current_position, _ = robot.get_object_pose(part_id)
            if current_position is None:
                return False

            # 计算Z轴位置差
            z_diff = current_position[2] - initial_position[2]

            # 阈值：10cm = 0.1m
            threshold = 0.10

            if z_diff >= threshold:
                print(f"Pick checker passed: Z increased by {z_diff:.3f}m >= {threshold}m")
                return True
            else:
                print(f"Pick checker failed: Z increased by {z_diff:.3f}m < {threshold}m")
                return False

        return checker

    pick_checker = create_pick_height_checker(mounted_part_id, initial_part_position)

    # 8. 获取零件位置以确定使用哪只手臂
    part_position, _ = robot.get_object_pose(mounted_part_id)
    if part_position is None:
        print(f"Cannot get position for part {mounted_part_id}")
        return False

    arm = robot.judge_arm_to_use(part_position)
    print(f"Using arm: {arm}")

    # 9. 配置抓取参数
    grasp_config = GraspConfig(
        passive_primitive="default",
        pick_up_distance=pick_up_distance,  # 根据零件大小设置抬升距离
        set_grasp_along_axis="y",
        unmovable_objects=[table_id, gearbox_id],
        extra_checkers=[pick_checker],  # 添加高度检查
        action_description={
            "zh": f"把{part_type_zh}卸下来",
            "en": f"remove the {part_type_en}"
        }
    )

    # 10. 执行抓取操作
    print(f"Attempting to pick {part_type_zh} ({mounted_part_id})")
    success = robot.pick_actor(
        actor=mounted_part_id,
        arm=arm,
        grasp_config=grasp_config
    )

    if not success:
        print(f"Failed to pick {part_type_zh}")
        return False

    print(f"Successfully picked {part_type_zh}")

    # 11. 最终验证
    final_position, _ = robot.get_object_pose(mounted_part_id)
    if final_position is not None and initial_part_position is not None:
        z_diff = final_position[2] - initial_part_position[2]
        print(f"Final Z increase: {z_diff:.3f}m")

    print(f"\n=== Task summary ===")
    print(f"✅ SUCCESS: {part_type_zh} removed and lifted")

    return success
```
