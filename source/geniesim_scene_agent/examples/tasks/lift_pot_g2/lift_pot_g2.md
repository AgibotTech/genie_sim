# Lift a pot

**Tags:** `bimanual` `dual_arm` `pot` `table_only` `move_actor_to_target_pose` `MoveConfig` `BimanualRigidMoveConfig` `bimanual_rigid_move` `attach_obj` `lift_verification` `extra_checkers` `get_world_position_and_orientation_from_relative` `local_grasp_points`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find pot by keywords containing "pot"; find table by keywords "table" or usd "table_000".
2. **Local grasp points:** Define left/right grasp positions and quaternions in pot's local frame (handles at ±x offset). Convert to world frame using `get_world_position_and_orientation_from_relative`.
3. **Right arm grasp:** Generate grasp candidates around the right handle (z offset + small angle perturbations around pot local z/y axes). Use `move_actor_to_target_pose("gripper", candidates, arm="right", move_config)` with `gripper_action="close"`.
4. **Re-read pot pose:** After right arm contacts, pot may shift slightly; re-query pose before left arm.
5. **Left arm grasp:** Same candidate generation for left handle; move left gripper and close.
6. **Attach pot:** `attach_obj` pot prim to both arms.
7. **Bimanual lift:** Compute lift targets from right EE pose + z offset. Use `bimanual_rigid_move` with `leader_arm="right"` and `extra_checkers` verifying pot z increased by ≥ half of `LIFT_HEIGHT`.

---

# INPUT

**scene_description:** A table with a pot (benchmark_pot_001) on it. The pot has two handles on left and right sides.

**task description:** The robot uses both arms to grasp the pot's two handles simultaneously and lift it up from the table.

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: table_with_pot
description: A pot placed on a table with specific position offsets and rotation
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
def pot_on_table(table_shape: Shape) -> Shape:
    """
    Places a pot on the table with specified position offsets and rotation.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined shape of table and pot.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Random x offset within (-0.2, -0.1) range
    x_offset = np.random.uniform(-0.2, -0.05)

    # Random y offset within (-0.02, 0.02) range
    y_offset = np.random.uniform(-0.1, 0.1)

    # Random rotation around z-axis between 85-95 degrees (convert to radians)
    rotation_angle_deg = np.random.uniform(80, 100)
    rotation_angle_rad = np.radians(rotation_angle_deg)

    # Load the pot
    pot_shape = library_call(
        "usd",
        oid="benchmark_pot_001",
        keywords=["pot", "kitchenware", "cooking_pot", "on_table"],
    )

    # Get pot size to calculate proper placement height
    pot_info = get_object_info(pot_shape)
    pot_height = compute_shape_sizes(pot_shape)[2]

    # Calculate pot bottom position (z = table top z)
    pot_bottom_z = table_top_pos[2]

    # Final pot center position
    pot_center_pos = np.array(
        [
            table_top_pos[0] + x_offset,
            table_top_pos[1] + y_offset,
            pot_bottom_z + pot_height / 2.0,  # Center is half height above bottom
        ]
    )

    # Translate pot to position
    pot_translated = transform_shape(
        pot_shape,
        translation_matrix(pot_center_pos),
    )

    # Rotate pot around z-axis
    pot_rotated = transform_shape(
        pot_translated,
        rotation_matrix(rotation_angle_rad, direction=(0, 0, 1), point=pot_center_pos),
    )

    # Combine table and pot
    return concat_shapes(table_shape, pot_rotated)


@register()
def table_with_pot() -> Shape:
    """
    Main scene function that creates a table with a pot on it.

    Returns:
        Shape: Combined shape of table and pot.
    """
    # Load table on ground (z=0.0 by default)
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "furniture", "ground_level"],
    )

    # Place pot on table
    return pot_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_pot()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


LEFT_GRASP_POS = np.array([0.17591, 0.02774, 0.00044])
LEFT_GRASP_QUAT = np.array([0.5, -0.5, -0.5, 0.5])  # wxyz
RIGHT_GRASP_POS = np.array([-0.17591, 0.02774, 0.00044])
RIGHT_GRASP_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # wxyz

LIFT_HEIGHT = 0.14


def _compute_world_grasp_pose(local_pos, local_quat_wxyz, pot_pos, pot_quat_wxyz):
    world_pos, world_quat = get_world_position_and_orientation_from_relative(
        local_pos, local_quat_wxyz, pot_pos, pot_quat_wxyz
    )
    return pose_from_position_quaternion(world_pos, world_quat)


def _gen_grasp_candidates(base_pose, pot_pose_mat, z_range=0.02, angle_range=5, n_z=5, n_angle=3):
    pot_rot = pot_pose_mat[:3, :3]
    local_z = pot_rot[:, 2]
    local_y = pot_rot[:, 1]

    poses = []
    for dz in np.linspace(-z_range, z_range, n_z):
        shifted = base_pose.copy()
        shifted[:3, 3] += local_z * dz
        for az_deg in np.linspace(-angle_range, angle_range, n_angle):
            for ay_deg in np.linspace(-angle_range, angle_range, n_angle):
                p = shifted.copy()
                if az_deg != 0:
                    R_z = rotation_matrix(np.deg2rad(az_deg), local_z, point=p[:3, 3])
                    p = R_z @ p
                if ay_deg != 0:
                    R_y = rotation_matrix(np.deg2rad(ay_deg), local_y, point=p[:3, 3])
                    p = R_y @ p
                poses.append(p)
    return np.array(poses)


def _gen_move_targets(base_pose, offset):
    target = base_pose.copy()
    target[:3, 3] += offset
    targets = []
    for dz in np.linspace(-0.05, 0.05, 8):
                t = target.copy()
                t[2, 3] += dz
                targets.append(t)
    targets = np.array(targets)
    np.random.shuffle(targets)
    return targets


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_pot_001": {"has_interactions": False, "object_type": "pot"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info

    description_json = {
        "task_name": "双臂协同抓取并抬起锅",
        "english_task_name": "bimanual grasp and lift the pot",
        "init_scene_text": "The robot is in front of the table, there is a pot on the table",
    }
    robot.client.set_task_task_basic_info(description_json)

    pot_id = None
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")
        if any("pot" in kw.lower() for kw in keywords):
            pot_id = obj_id
        if any("table" in kw.lower() for kw in keywords) or usd == "table_000":
            table_id = obj_id
    if pot_id is None:
        print("Pot not found")
        return False
    if table_id is None:
        print("Table not found")
        return False
    print(f"Found pot: {pot_id}, table: {table_id}")

    unmovable_objects = [pot_id]

    pot_pos, pot_quat = robot.get_object_pose(pot_id)
    pot_mat = pose_from_position_quaternion(pot_pos, pot_quat)
    initial_pot_z = pot_pos[2]
    print(f"Pot pose: pos={pot_pos}, quat={pot_quat}")

    right_base = _compute_world_grasp_pose(RIGHT_GRASP_POS, RIGHT_GRASP_QUAT, pot_pos, pot_quat)
    right_candidates = _gen_grasp_candidates(right_base, pot_mat)
    move_right_cfg = MoveConfig(
        gripper_action="close",
        unmovable_objects=unmovable_objects,
        action_description={"zh": "抓取锅右把手", "en": "grasps pot's right handle"},
    )
    if not robot.move_actor_to_target_pose("gripper", right_candidates, arm="right", move_config=move_right_cfg):
        print("Failed to move right arm to pot")
        return False
    print("Right arm grasped")

    pot_pos, pot_quat = robot.get_object_pose(pot_id)
    pot_mat = pose_from_position_quaternion(pot_pos, pot_quat)
    print(f"Pot pose after right grasp: pos={pot_pos}, quat={pot_quat}")

    left_base = _compute_world_grasp_pose(LEFT_GRASP_POS, LEFT_GRASP_QUAT, pot_pos, pot_quat)
    left_candidates = _gen_grasp_candidates(left_base, pot_mat)
    move_left_cfg = MoveConfig(
        gripper_action="close",
        unmovable_objects=unmovable_objects,
        action_description={"zh": "抓取锅左把手", "en": "grasps pot's left handle"},
    )
    if not robot.move_actor_to_target_pose("gripper", left_candidates, arm="left", move_config=move_left_cfg):
        print("Failed to move left arm to pot")
        return False
    print("Left arm grasped")

    pot_prim = robot.get_object_prim_path(pot_id)
    robot.attach_obj([pot_prim], arm="right")
    robot.attach_obj([pot_prim], arm="left")
    print("Pot attached to both arms")

    def check_pot_lifted():
        current_pos, _ = robot.get_object_pose(pot_id)
        if current_pos is None:
            return False
        lift_delta = current_pos[2] - initial_pot_z
        min_lift = LIFT_HEIGHT * 0.5
        print(f"Pot lift delta: {lift_delta:.3f}m (min required: {min_lift:.3f}m)")
        if lift_delta < min_lift:
            print(f"Pot not lifted enough: {lift_delta:.3f}m < {min_lift:.3f}m")
            return False
        return True

    ee_right = robot.get_ee_pose(arm="right")
    lift_targets = _gen_move_targets(ee_right, np.array([0, 0, LIFT_HEIGHT]))
    bimanual_lift_cfg = BimanualRigidMoveConfig(
        leader_arm="right",
        extra_checkers=[check_pot_lifted],
        action_description={"zh": "抬起锅", "en": "lift the pot"},
        max_consecutive_failures=3,
        offset_and_constraint_in_goal_frame=False,
    )
    if not robot.bimanual_rigid_move(target_pose=lift_targets, config=bimanual_lift_cfg):
        print("Failed to bimanual lift")
        return False

    print("Successfully lifted the pot with both arms")
    return True
```
