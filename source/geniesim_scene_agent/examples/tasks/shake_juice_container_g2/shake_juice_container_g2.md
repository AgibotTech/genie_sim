# Shake a juice container

**Tags:** `shake` `shake_config` `ShakeConfig` `beverage_bottle` `juice_bottle` `pick_actor` `move_actor_to_target_pose` `robot_shake` `judge_arm_to_use` `GraspConfig` `MoveConfig` `extra_checkers` `quat2mat_wxyz` `rotate_along_axis` `multiple_candidate_poses`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table (`table_000`) and juice bottle (`benchmark_beverage_bottle_008` or `genie_beverage_bottle_009`) by usd/keywords.
2. **Pick (side grasp):** arm = `judge_arm_to_use(bottle_pos)`. Use `GraspConfig(set_grasp_along_axis="y", set_grasp_pose_center=True, grasp_lower_percentile=40, grasp_upper_percentile=60, unmovable_objects=[table_id])` then `pick_actor`.
3. **Move to chest position:** Generate multiple gripper target poses around a chest-ish position with small jitter and small z-axis rotations. `extra_checkers`: ensure bottle stays in hand (distance to gripper < threshold).
4. **Shake:** Build `angle_ranges` (deg 15–30 → rad, shuffled), then call `robot.shake(actor=bottle_id, arm=arm, horizontal=False, times=4, shake_config=ShakeConfig(..., extra_checkers=[bottle_in_hand_checker]))`.

---

# INPUT

**scene_description:** A table with a juice bottle placed randomly on the tabletop (random XY within bounds and random rotation).

**task description:** Pick up the juice bottle on the table and shake it.

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: table_with_juice_bottle
description: A juice bottle placed randomly on a table with specific bounds and random rotation
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
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def juice_bottle() -> Shape:
    """
    Creates a juice bottle with random rotation around z-axis.
    Uses genie_beverage_bottle_009 as the default beverage bottle.
    """
    # Choose one of the available beverage bottles
    bottle_shape = library_call(
        "usd",
        oid="genie_beverage_bottle_009",
        keywords=["juice_bottle", "beverage", "bottle", "glass", "transparent"],
    )

    # Get bottle size for positioning
    bottle_size = compute_shape_sizes(bottle_shape)
    bottle_center = compute_shape_center(bottle_shape)

    # Apply random rotation around z-axis (vertical axis)
    random_angle = np.random.uniform(-np.pi, np.pi)
    rotated_bottle = transform_shape(
        bottle_shape,
        rotation_matrix(random_angle, direction=(0, 0, 1), point=bottle_center),
    )

    return rotated_bottle


@register()
def place_bottle_on_table(table_shape: Shape) -> Shape:
    """
    Places a juice bottle on the table within specified bounds.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined shape of table and bottle.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Define placement bounds as specified by user
    x_min, x_max = -0.15, 0.15
    y_min, y_max = -0.2, 0.2

    # Get bottle size to ensure it fits within bounds
    bottle_shape = library_call("juice_bottle")
    bottle_size = compute_shape_sizes(bottle_shape)

    # Calculate safe placement range (accounting for bottle half-size)
    safe_x_range = (
        table_top_pos[0] + x_min + bottle_size[0] / 2.0,
        table_top_pos[0] + x_max - bottle_size[0] / 2.0,
    )
    safe_y_range = (
        table_top_pos[1] + y_min + bottle_size[1] / 2.0,
        table_top_pos[1] + y_max - bottle_size[1] / 2.0,
    )

    # Random position within bounds
    random_x = np.random.uniform(safe_x_range[0], safe_x_range[1])
    random_y = np.random.uniform(safe_y_range[0], safe_y_range[1])

    # Bottle sits on table top surface
    bottle_z = table_top_pos[2]

    # Position the bottle
    positioned_bottle = transform_shape(
        bottle_shape,
        translation_matrix((random_x, random_y, bottle_z)),
    )

    return concat_shapes(table_shape, positioned_bottle)


@register()
def table_with_juice_bottle() -> Shape:
    """
    Main scene function: creates a table with a juice bottle on it.
    """
    # Load table from assets library
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "desktop_with_legs"],
    )

    # Place bottle on table
    return place_bottle_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return table_with_juice_bottle()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    """
    任务：从侧面抓起桌子上的果汁瓶，并摇晃果汁
    果汁瓶有两个可能类型：benchmark_beverage_bottle_008 或 genie_beverage_bottle_009
    """

    # 基于提供的场景生成代码初始化objects_list
    # 果汁瓶有两种可能的资产ID，都需要检查是否有交互数据
    objects_list = {
        "benchmark_beverage_bottle_008": {
            "has_interactions": True,  # 假设有交互数据
            "object_type": "beverage_bottle",
            "description": "果汁瓶"
        },
        "genie_beverage_bottle_009": {
            "has_interactions": True,  # 假设有交互数据
            "object_type": "beverage_bottle",
            "description": "果汁瓶"
        },
        "table_000": {
            "has_interactions": False,  # 桌子通常没有交互数据
            "object_type": "table"
        }
    }

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    # 任务描述：抓起桌子上的果汁瓶
    description_json = {
        "task_name": "抓起桌子上的果汁瓶，并摇晃果汁",
        "english_task_name": "pick up the juice bottle on the table and shake it",
        "init_scene_text": "The robot is in front of the table, there is a juice bottle placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

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

    # 2. 查找果汁瓶
    juice_bottle_id = None
    juice_bottle_usd = None
    juice_bottle_info = None

    # 通过usd查找果汁瓶（两种可能的资产ID）
    bottle_usds = ["benchmark_beverage_bottle_008", "genie_beverage_bottle_009"]
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是果汁瓶
        is_bottle = False
        for bottle_usd in bottle_usds:
            if bottle_usd in usd:
                is_bottle = True
                juice_bottle_usd = bottle_usd
                break

        if is_bottle or any("juice_bottle" in kw.lower() for kw in keywords) or any("beverage" in kw.lower() for kw in keywords):
            juice_bottle_id = obj_id
            juice_bottle_info = obj_info
            print(f"Found juice bottle: {obj_id}, usd: {usd}, keywords: {keywords}")
            break

    if not juice_bottle_id:
        print("Juice bottle not found")
        return False

    print(f"Using juice bottle: {juice_bottle_id}")

    # 3. 获取果汁瓶的位置和大小
    juice_bottle_position, juice_bottle_rotation = robot.get_object_pose(juice_bottle_id)
    if juice_bottle_position is None:
        print("Cannot get juice bottle position")
        return False

    # 获取果汁瓶的大小（用于计算抓取位置）
    juice_bottle_size = robot.get_object_size_in_world_frame(juice_bottle_id)
    if juice_bottle_size is None:
        print("Cannot get juice bottle size")
        return False

    print(f"Juice bottle position: {juice_bottle_position}")
    print(f"Juice bottle size: {juice_bottle_size}")

    # 4. 判断使用哪只手臂
    arm = robot.judge_arm_to_use(juice_bottle_position)
    print(f"Using arm: {arm}")

    # 5. 检查交互数据，确定是否有侧面抓取点
    # 果汁瓶是容器，可能没有交互数据或只有特定抓取点
    # 如果输入没有提供interaction.json，需要检查运行时数据
    bottle_interaction = interactions.get(juice_bottle_usd, {})
    passive_grasp = bottle_interaction.get("passive", {}).get("grasp", {})

    print(f"Bottle has passive.grasp primitives: {bool(passive_grasp)}")
    if passive_grasp:
        print(f"Available grasp primitives: {list(passive_grasp.keys())}")

    # 6. 准备抓取配置
    # 根据任务要求"从侧面抓起"，我们使用默认的抓取点（系统会自动选择最佳抓取点）
    # 也可以尝试使用侧面抓取的特殊参数配置

    # 不可移动对象：桌子应该保持不动
    unmovable_objects = [table_id]

    # 为抓取配置设置action_description（必须提供）
    grasp_action_desc = {
        "zh": "抓起果汁瓶",
        "en": "picks up the juice bottle"
    }

    # 创建抓取配置
    # 由于任务要求从侧面抓起，我们可以尝试配置grasp参数
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",
        set_grasp_pose_center=True,
        # 使用中间的抓取点（避免抓取顶部或底部）
        grasp_lower_percentile=40,
        grasp_upper_percentile=60,
        # 不可移动对象
        unmovable_objects=unmovable_objects,
        # 必需的action_description
        action_description=grasp_action_desc
    )

    # 7. 执行抓取
    print("Attempting to pick up juice bottle from the side...")
    current_success = robot.pick_actor(
        actor=juice_bottle_id,
        arm=arm,
        grasp_config=grasp_config
    )

    success = success and current_success
    if not current_success:
        print("Failed to pick up juice bottle")
        return False

     # 6. Move to shake Pose (Hand A)
    gripper_pose = robot.get_ee_pose(arm)
    # Position: x(-0.1 ~ -0.05), y(0.0). Z set to a reasonable chest height (e.g., 0.3m)
    shake_x = np.random.uniform(-0.1, 0.0)
    shake_y = 0.0
    shake_z = gripper_pose[2, 3] # Chest height approximation

    gripper_direction_quaternion = np.array([0,0,0.70711, -0.70711]) if arm == "left" else np.array([0,0,0.70711, 0.70711])
    base_rot_mat = quat2mat_wxyz(gripper_direction_quaternion)
    # Generate candidates with small Z rotation (±5 deg)
    shake_poses = []
    z_jitter = np.linspace(-0.05, 0.15, 5)
    y_jitter = np.linspace(-0.02, 0.02, 5) # Small y jitter
    angle_jitter = np.linspace(-15, 15, 5)

    for z_off in z_jitter:
        for y_off in y_jitter:
            for angle in angle_jitter:
                # Apply rotation around World Z

                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                pose[:3, 3] = [shake_x, shake_y + y_off, shake_z + z_off]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                shake_poses.append(pose)

    shake_poses = np.array(shake_poses)

    def bottle_in_hand_checker():
        bottle_pos, bottle_rot = robot.get_object_pose(juice_bottle_id)
        gripper_pos = robot.get_ee_pose(arm)
        gripper_position = gripper_pos[:3, 3]
        distance = np.linalg.norm(bottle_pos - gripper_position)
        if distance > 0.08:
            print("Bottle is not in hand: distance = ", distance)
            return False
        return True
    move_config = MoveConfig(
        extra_checkers=[bottle_in_hand_checker],
        action_description={"zh": "移动到胸前位置", "en": "moves to chest position"}
    )
    success = robot.move_actor_to_target_pose("gripper", shake_poses, arm=arm, move_config=move_config)
    if not success:
        print("Failed to move to shake pose")
        return False

    # shake
    angle_ranges = []
    for angle in np.linspace(15, 30, 5):
        angle = angle * np.pi / 180
        angle_ranges.append([-angle, angle])
    import random
    random.shuffle(angle_ranges)
    horizontal = False
    shake_config = ShakeConfig(
        action_description={"zh": "摇晃果汁瓶", "en": "shakes the juice bottle"},
        extra_checkers=[bottle_in_hand_checker]
    )
    success = robot.shake(actor=juice_bottle_id, arm=arm, horizontal=horizontal, angle_ranges=angle_ranges, times=4, shake_config=shake_config)

    print("\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: Juice bottle picked up and shaken")
    else:
        print("❌ FAILED: Could not pick up juice bottle or shake it")

    return success
```
