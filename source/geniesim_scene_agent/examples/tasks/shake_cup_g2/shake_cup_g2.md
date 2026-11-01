# Shake a cup

**Tags:** `shake` `shake_config` `ShakeConfig` `cup` `beer_mug` `pick_actor` `move_actor_to_target_pose` `robot_shake` `judge_arm_to_use` `GraspConfig` `MoveConfig` `extra_checkers` `quat2mat_wxyz` `rotate_along_axis` `multiple_candidate_poses`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find the cup by usd (benchmark_cup_043/105, benchmark_beer_mug_007/001/002) or keywords containing \"cup\".
2. **Pick:** arm = `judge_arm_to_use(cup_pos)`. Use `GraspConfig(set_grasp_along_axis="y", set_grasp_pose_center=True)` then `pick_actor`.
3. **Move to chest position:** Build a horizontal gripper rotation (different quat for left/right), generate many target poses near chest with z/y jitter and small z rotations. Use `extra_checkers` to ensure cup remains in hand.
4. **Shake:** Build small `angle_ranges` (5–10 deg in rad) and call `robot.shake(actor=cup_id, arm=arm, horizontal=True, times=3, shake_config=ShakeConfig(...))`.

---

# INPUT

**scene_description:** A table with a randomly selected cup (cup or beer mug) placed on it with random rotation.

**task description:** Pick up the cup on the table, move to chest position and shake horizontally by a small angle.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import random

"""
scene_name: table_with_random_cup
description: A table with a randomly selected cup placed on top with random rotation
"""


def get_table_top_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface center and its size.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Tuple[P, P]: A tuple containing the table top's center position and its size.
    """
    table_info = get_object_info(table_shape)

    # Get subpart info for desktop (assuming "desktop" is the subpart name for table top)
    try:
        desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    except Exception:
        # Fallback: use the table's max z coordinate as the top surface
        desktop_subpart_info = {
            "center": np.array([0.0, 0.0, 0.0]),
            "size": np.array([0.8, 0.6, 0.05]),  # Approximate desktop size
            "xyz_max": np.array([0.4, 0.3, 0.025]),
            "xyz_min": np.array([-0.4, -0.3, -0.025])
        }

    # The desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def random_cup() -> Shape:
    """
    Randomly selects and returns one cup from the available cup options.

    Returns:
        Shape: The selected cup shape.
    """
    # List of available cup object IDs
    cup_options = [
        "benchmark_cup_043",
        "benchmark_cup_105",
        "benchmark_beer_mug_007",
        "benchmark_beer_mug_001",
        "benchmark_beer_mug_002"
    ]

    # Randomly select one cup
    selected_cup_id = random.choice(cup_options)

    # Create keywords for the selected cup
    keywords = ["cup", "random_selection", f"{selected_cup_id}"]

    # Load the cup using usd
    cup_shape = library_call(
        "usd",
        oid=selected_cup_id,
        keywords=keywords
    )

    return cup_shape


@register()
def place_cup_on_table(table_shape: Shape) -> Shape:
    """
    Places a randomly selected cup on the table with random rotation at specified position range.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined table and cup shapes.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Get cup shape
    cup_shape = library_call("random_cup")

    # Get cup size to ensure it fits on the table
    cup_size = compute_shape_sizes(cup_shape)

    # Define position ranges (relative to table top center)
    x_range = (-0.15, 0.15)
    y_range = (-0.2, 0.2)

    # Generate random position within the specified range
    random_x = np.random.uniform(x_range[0], x_range[1])
    random_y = np.random.uniform(y_range[0], y_range[1])

    # Calculate final cup position (on table top surface)
    # Place cup bottom on table surface (z = table_top_z)
    cup_position = np.array([
        table_top_pos[0] + random_x,
        table_top_pos[1] + random_y,
        table_top_pos[2]  # Cup sits on the table surface
    ])

    # Translate cup to position
    cup_transformed = transform_shape(
        cup_shape,
        translation_matrix(cup_position)
    )

    # Apply random rotation around z-axis (vertical axis)
    random_rotation_angle = np.random.uniform(-np.pi, np.pi)
    cup_center = compute_shape_center(cup_transformed)

    cup_final = transform_shape(
        cup_transformed,
        rotation_matrix(
            angle=random_rotation_angle,
            direction=(0, 0, 1),
            point=cup_center
        )
    )

    # Combine table and cup
    return concat_shapes(table_shape, cup_final)


@register()
def table_with_random_cup() -> Shape:
    """
    Main scene function that creates a table with a randomly selected cup on top.

    Returns:
        Shape: The complete scene with table and cup.
    """
    # Load table using usd
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )

    # Place cup on table
    return place_cup_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_random_cup()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## 从交互数据中提取资产信息
    objects_list = {
        "benchmark_cup_043": {"has_interactions": True, "object_type": "cup"},
        "benchmark_cup_105": {"has_interactions": True, "object_type": "cup"},
        "benchmark_beer_mug_007": {"has_interactions": True, "object_type": "beer_mug"},
        "benchmark_beer_mug_001": {"has_interactions": True, "object_type": "beer_mug"},
        "benchmark_beer_mug_002": {"has_interactions": True, "object_type": "beer_mug"},
    }

    ## 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "抓起桌子上的杯子，移动到胸前并水平小幅度摇晃",
        "english_task_name": "pick up the cup on the table, move to chest position and shake horizontally by a small angle",
        "init_scene_text": "The robot is in front of the table with a randomly selected cup (cup or beer mug) placed on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. 查找杯子对象（通过usd或keywords判断）
    cup_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是支持的杯子类型
        if any(cup_usd in usd for cup_usd in objects_list.keys()):
            cup_id = obj_id
            print(f"Found cup: {cup_id}, usd: {usd}")
            break

        # 或者通过keywords检查
        elif any("cup" in kw.lower() for kw in keywords):
            cup_id = obj_id
            print(f"Found cup by keyword: {cup_id}")
            break

    if not cup_id:
        print("Cup not found")
        return False

    ## 获取杯子对象的信息
    cup_position, _ = robot.get_object_pose(cup_id)
    cup_size = robot.get_object_size_in_world_frame(cup_id)

    if cup_position is None or cup_size is None:
        print("Cannot get cup position/size")
        return False

    ## 2. 判断使用哪只手臂
    arm = robot.judge_arm_to_use(cup_position)
    print(f"Using arm: {arm}")

    ## 3. 抓取杯子
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",  # 沿杯子轴线抓取
        set_grasp_pose_center=True,
        unmovable_objects=[],
        action_description={"zh": "抓起桌上的杯子", "en": "picks up the cup on the table"}  # REQUIRED
    )

    if not robot.pick_actor(cup_id, arm=arm, grasp_config=grasp_config):
        print("Failed to pick up cup")
        return False

    print("Successfully picked up cup")

    ## 4. 将夹爪移动到胸前水平位置（为摇晃做准备）
    ## 获取当前夹爪高度作为参考
    gripper_pose = robot.get_ee_pose(arm)
    shake_z = gripper_pose[2, 3]

    ## 构建水平姿态（绕世界Z轴旋转90度，使夹爪水平）
    # 左臂和右臂的水平姿态不同
    if arm == "left":
        gripper_direction_quat = np.array([0, 0, 0.70711, -0.70711])  # wxyz
    else:
        gripper_direction_quat = np.array([0, 0, 0.70711, 0.70711])  # wxyz

    base_rot_mat = quat2mat_wxyz(gripper_direction_quat)

    ## 生成多个候选pose（胸前位置，水平方向 + 抖动）
    shake_poses = []
    for z_off in np.linspace(-0.05, 0.10, 5):  # z方向偏移
        for y_off in np.linspace(-0.02, 0.02, 5):  # y方向偏移
            for angle in np.linspace(-15, 15, 5):  # 小角度旋转抖动
                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                # 位置：在胸部前方，z高度变化
                pose[:3, 3] = [np.random.uniform(-0.1, 0.1), y_off, shake_z + z_off]
                # 绕z轴小角度旋转
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                shake_poses.append(pose)

    shake_poses = np.array(shake_poses)
    print(f"Generated {len(shake_poses)} candidate poses for chest position")

    def bottle_in_hand_checker():
        """确保杯子仍在手中"""
        cup_pos, _ = robot.get_object_pose(cup_id)
        gripper_pos = robot.get_ee_pose(arm)[:3, 3]
        return np.linalg.norm(cup_pos - gripper_pos) < 0.08

    move_config = MoveConfig(
        extra_checkers=[bottle_in_hand_checker],
        action_description={"zh": "移动杯子到胸前位置", "en": "moves the cup to chest position"},
    )

    if not robot.move_actor_to_target_pose("gripper", shake_poses, arm=arm, move_config=move_config):
        print("Failed to move to chest position")
        success = False

    print("Successfully moved to chest position")

    ## 5. 摇晃杯子（水平摇晃，角度范围5-10度）
    ## 将角度转换为弧度
    angle_ranges = []
    for angle_deg in np.linspace(5, 10, 5):
        angle_rad = angle_deg * np.pi / 180
        angle_ranges.append([-angle_rad, angle_rad])

    print(f"Shaking with angle ranges (radians): {angle_ranges}")

    shake_config = ShakeConfig(
        action_description={"zh": "水平小幅度摇晃杯子", "en": "shakes the cup horizontally by a small angle"},
        extra_checkers=[bottle_in_hand_checker],
    )

    ## horizontal=True 使用joint6（水平摇晃），horizontal=False 使用joint7（垂直摇晃）
    ## 根据需求，这里使用horizontal=True进行水平摇晃
    if not robot.shake(
        actor=cup_id,
        arm=arm,
        horizontal=True,  # 水平摇晃（joint6）
        angle_ranges=angle_ranges,
        times=3,  # 每个角度范围重复3次
        shake_config=shake_config
    ):
        print("Failed to shake the cup")
        success = False

    print("Successfully shook the cup")


    return success
```
