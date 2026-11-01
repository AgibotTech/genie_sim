# Pour a small ball into a box

**Tags:** `pour` `table_with_containers` `pick_actor` `move_actor_to_target_pose` `rotate_actor` `RotateConfig` `constraint_axises` `check_box_containment` `get_object_size_in_local_frame` `extra_checkers` `ignore_axes_z` `GraspConfig` `set_grasp_along_axis` `action_description` `run_step` `world_frame` `rotate_around_x`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find cup (with ball) and storage box. Get cup and box poses/sizes. Identify ball (e.g. small object inside cup or by keywords).
2. **Pick cup:** pick_actor(cup_id, arm, GraspConfig with set_grasp_along_axis). Move cup above box: move_actor_to_target_pose(cup_id, pose_above_box).
3. **Pour:** rotate_actor(cup_id, RotateConfig with constraint_axises so tilt is around one axis, e.g. world x or y). Run steps until tilt angle sufficient. extra_checkers: ball inside box via check_box_containment(box_pos, box_rot, box_size_local, ball_pos, ball_rot, ball_size_local, ignore_axes=["z"]).
4. **Verification:** After pour, verify ball position is inside box (overlap_threshold).

---

# INPUT

**scene_description:** A table with a paper cup containing a small ball, and a storage box on the table. The cup can be grasped and tilted; the ball must end up inside the box after pouring.

**task description:** Pick up the cup, move it above the box, then tilt (rotate) the cup to pour the ball from the cup into the box. Verify the ball is inside the box after the pour.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import math

"""
scene_name: table_with_cup_and_box
description: A table with a paper cup containing a small ball and a storage box placed on top.
             The cup is placed in specific ranges on the table (left-back or right-back),
             and the box is near the center. Constraints on distance and rotation are applied.
"""


@register()
def table() -> Shape:
    """
    Create the base table.
    """
    return library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "furniture"],
    )


@register()
def storage_box() -> Shape:
    """
    Create the storage box.
    """
    return library_call(
        "usd",
        oid="benchmark_storage_box_007",
        keywords=["storage_box", "rectangular", "plastic", "container"],
    )


@register()
def paper_cup() -> Shape:
    """
    Create the paper cup.
    """
    return library_call(
        "usd",
        oid="benchmark_cup_112",
        keywords=["paper_cup", "cylinder", "white", "container"],
    )


@register()
def small_ball() -> Shape:
    """
    Create the small ball to be placed inside the cup.
    """
    return library_call(
        "usd",
        oid="benchmark_building_blocks_111",
        keywords=["ball", "sphere", "small", "toy"],
    )


@register()
def root_scene() -> Shape:
    """
    Main function to assemble the scene.
    """
    # 1. Setup Table
    table_shape = table()
    table_info = get_object_info(table_shape)

    # Get desktop height relative to table center
    desktop_info = get_subpart_info("table_000", "desktop")
    table_center = table_info["center"]
    surface_z = table_center[2] + desktop_info["xyz_max"][2] - 0.125

    # 2. Setup Storage Box
    box_shape = storage_box()

    # Box position: x in (-0.05, 0.01), y in (-0.01, 0.01)
    box_pos_x = np.random.uniform(-0.05, 0.01)
    box_pos_y = np.random.uniform(-0.01, 0.01)

    # Box rotation: small angle around Z (< 7 degrees)
    box_angle_rad = np.random.uniform(np.radians(-7), np.radians(7))

    # Apply transformations
    box_shape = transform_shape(
        box_shape,
        rotation_matrix(box_angle_rad, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    box_shape = transform_shape(
        box_shape,
        translation_matrix((box_pos_x, box_pos_y, surface_z)),
    )

    # Store box center for distance calculation
    box_center_xy = np.array([box_pos_x, box_pos_y])

    # 3. Setup Paper Cup with constraints
    cup_shape = paper_cup()

    valid_cup_pos = False
    cup_pos_x = 0.0
    cup_pos_y = 0.0
    cup_side_tag = ""

    # Find a valid position for the cup satisfying distance constraint
    while not valid_cup_pos:
        region_choice = np.random.choice(["A", "B"])

        if region_choice == "A":
            # Region A: x(-0.2, 0.1), y(-0.2, -0.1) -> Right side
            cup_pos_x = np.random.uniform(-0.1, 0.2)
            cup_pos_y = np.random.uniform(-0.2, -0.1)
            cup_side_tag = "right"
        else:
            # Region B: x(-0.2, 0.1), y(0.1, 0.2) -> Left side
            cup_pos_x = np.random.uniform(-0.1, 0.2)
            cup_pos_y = np.random.uniform(0.1, 0.2)
            cup_side_tag = "left"

        # Check distance constraint: distance between centers >= 0.15
        dist = np.linalg.norm(np.array([cup_pos_x, cup_pos_y]) - box_center_xy)
        if dist >= 0.2:
            valid_cup_pos = True

    # Cup rotation: random around Z axis
    cup_angle_rad = np.random.uniform(0, 2 * np.pi)

    # Apply transformations
    cup_shape = transform_shape(
        cup_shape,
        rotation_matrix(cup_angle_rad, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    cup_shape = transform_shape(
        cup_shape,
        translation_matrix((cup_pos_x, cup_pos_y, surface_z)),
    )

    # 4. Setup Small Ball inside Cup
    ball_shape = small_ball()

    # Calculate position inside the cup
    cup_info = get_object_info(cup_shape)
    cup_min = cup_info["min"]
    cup_max = cup_info["max"]

    # Place ball at the center of the cup's xy base and half height up
    ball_x = (cup_min[0] + cup_max[0]) / 2.0
    ball_y = (cup_min[1] + cup_max[1]) / 2.0
    ball_z = cup_min[2] + (cup_max[2] - cup_min[2]) * 0.5

    ball_shape = transform_shape(
        ball_shape,
        translation_matrix((ball_x, ball_y, ball_z)),
    )

    # 5. Combine all shapes
    return concat_shapes(table_shape, box_shape, cup_shape, ball_shape)
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided interaction data
    objects_list = {
        "benchmark_cup_112": {
            "has_interactions": True,
            "object_type": "cup",
            "interaction_type": "container"
        },
        "benchmark_storage_box_007": {
            "has_interactions": True,
            "object_type": "storage_box",
            "interaction_type": "container"
        },
        "benchmark_building_blocks_111": {
            "has_interactions": True,
            "object_type": "ball",
            "interaction_type": "small_object"
        }
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "拿起杯子把杯中的小球倒入盒子中",
        "english_task_name": "pick up the cup and pour the ball from the cup into the box",
        "init_scene_text": "The robot is in front of the table, there is a paper cup with a ball inside, and a storage box on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find objects (Box, Cup, Ball)
    box_id = None
    cup_id = None
    ball_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_storage_box_007" in usd or any("storage_box" in kw for kw in keywords):
            box_id = obj_id
        elif "benchmark_cup_112" in usd or any("paper_cup" in kw for kw in keywords):
            cup_id = obj_id
        elif "benchmark_building_blocks_111" in usd or any("ball" in kw for kw in keywords):
            ball_id = obj_id

    if not box_id:
        print("Storage box not found")
        return False
    if not cup_id:
        print("Cup not found")
        return False
    if not ball_id:
        print("Ball not found")
        return False

    print(f"Found box: {box_id}, cup: {cup_id}, ball: {ball_id}")

    ## 2. Get positions and sizes
    box_pos, box_rot = robot.get_object_pose(box_id)
    box_size = robot.get_object_size_in_world_frame(box_id)
    cup_pos, _ = robot.get_object_pose(cup_id)

    if box_pos is None or box_size is None or cup_pos is None:
        print("Failed to get object info")
        return False

    ## 3. Determine arm
    arm = robot.judge_arm_to_use(cup_pos)
    print(f"Using arm: {arm}")

    def check_ball_in_cup():
        """Check if ball is inside the cup"""
        ball_pos, ball_rot = robot.get_object_pose(ball_id)
        ball_size = robot.get_object_size_in_local_frame(ball_id)
        cup_pos, cup_rot = robot.get_object_pose(cup_id)
        cup_size = robot.get_object_size_in_local_frame(cup_id)
        if ball_pos is None or cup_pos is None:
            return False

        # Check containment ignoring z-axis (xy plane projection)
        # For a ball in an open box, xy containment is sufficient
        cup_contains_ball, _ = check_box_containment(
            cup_pos, cup_rot, cup_size,
            ball_pos, ball_rot, ball_size,
            ignore_axes=["z"]
        )
        return cup_contains_ball

    ## 4. Pick up the cup
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",
        set_grasp_pose_center=True,
        grasp_lower_percentile=40,
        grasp_upper_percentile=60,
        unmovable_objects=[box_id], # Protect box
        action_description={"zh": "拿起杯子", "en": "picks up the cup"},
        extra_checkers=[check_ball_in_cup]
    )

    print("Picking up cup...")
    if not robot.pick_actor(cup_id, arm=arm, grasp_config=grasp_config):
        print("Failed to pick up cup")
        return False

    ## 5. Move the cup above the box
    ## Calculate target position: Above the box center
    target_x = box_pos[0]
    target_y = box_pos[1]
    box_top_z = box_pos[2] + box_size[2] / 2.0

    ## Add height to hold cup above box (avoid collision)
    hold_height = 0.15  # 15cm above box top
    target_z = box_top_z + hold_height

    target_position = np.array([target_x, target_y, target_z])

    ## Generate candidate poses for moving (small variations in xy and z)
    move_poses = []
    cup_pos, cup_rot = robot.get_object_pose(cup_id)
    cup_rotation_matrix = quat2mat_wxyz(cup_rot)
    arm_offset = 0.05 if arm == "left" else -0.05
    for dx in np.linspace(-0.01, 0.01, 3):
        for dy in np.linspace(-0.01, 0.01, 3):
            for dz in np.linspace(-0.02, 0.02, 5):
                pos = target_position + np.array([dx, dy + arm_offset, dz])
                pose = np.eye(4)
                pose[:3, 3] = pos
                pose[:3, :3] = cup_rotation_matrix
                move_poses.append(pose)

    move_poses = np.array(move_poses)

    move_config = MoveConfig(
        action_description={"zh": "移动杯子到盒子上方", "en": "moves the cup above the box"},
        extra_checkers=[check_ball_in_cup]
    )

    print(f"Moving cup to above box (target z: {target_z})...")
    if not robot.move_actor_to_target_pose(cup_id, move_poses, arm=arm, move_config=move_config):
        print("Failed to move cup above box")
        return False

    ## 6. Rotate the cup to pour (around World X-axis)
    ## Constraint: Only rotate around X axis
    rotate_config = RotateConfig(
        constraint_axises=["y", "z"],  # Constrain Y and Z, only X is free
        offset_and_constraint_in_goal_frame=False,  # World frame
        action_description={"zh": "倒入小球", "en": "pours the ball"}
    )

    ## Define checker function to verify ball is in box after rotation
    def check_ball_in_box():
        """Check if ball is inside the box"""
        robot.client.run_step(30)
        ball_pos, ball_rot = robot.get_object_pose(ball_id)
        ball_size = robot.get_object_size_in_local_frame(ball_id)
        box_size_local = robot.get_object_size_in_local_frame(box_id)
        box_pos, box_rot = robot.get_object_pose(box_id)
        if ball_pos is None or box_pos is None:
            return False
        # Check containment ignoring z-axis (xy plane projection)
        # For a ball in an open box, xy containment is sufficient
        box_contains_ball, _ = check_box_containment(
            box_pos, box_rot, box_size_local,
            ball_pos, ball_rot, ball_size,
            ignore_axes=["z"]
        )
        if not box_contains_ball:
            print("Ball not in box")
        if not ball_pos[2] < box_pos[2] + box_size[2] / 2.0:
            print("Ball not below box")
            print(ball_pos[2], box_pos[2] + box_size[2] / 2.0)
        return box_contains_ball and ball_pos[2] < box_pos[2] + box_size[2] / 2.0

    ## Add checker to rotate config
    rotate_config.extra_checkers = [check_ball_in_box]

    ## Generate target poses for rotation
    rotate_poses = []

    ## Generate poses with different tilt angles around X
    cup_pos, cup_rot = robot.get_object_pose(cup_id)
    cup_pose = make_pose(cup_pos, quat2mat_wxyz(cup_rot))
    for angle_deg in [100, 110, 120, 130]:
        angle_deg = angle_deg if arm == "left" else -angle_deg
        target_pose = rotate_along_axis(cup_pose, angle_deg, "x", use_local=False)
        rotate_poses.append(target_pose)

    rotate_poses = np.array(rotate_poses)

    print("Rotating cup to pour ball...")
    if not robot.rotate_actor(cup_id, rotate_poses, arm=arm, rotate_config=rotate_config):
        print("Failed to rotate/pour cup")
        success = False
    else:
        print("Pour action completed")

    # robot.back_to_origin(arm=arm)

    return success
```
