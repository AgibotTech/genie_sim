# Open a microwave door

**Tags:** `articulation` `microwave` `open_action` `table_only` `move_actor_to_target_pose` `OpenConfig` `MoveConfig` `extra_checkers` `articulation_joint_info`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find microwave by keywords "microwave" or "kitchen_appliance" in scene_info. Find table for reference.
2. **Handle positioning:** Get handle position using `get_object_pose(microwave_id, "part_01")`. Calculate grasp position and orientation relative to handle.
3. **Move gripper to handle:** Use `move_actor_to_target_pose` to move gripper to handle position with gripper_action="close" to grasp.
4. **Remove obstacles:** Remove microwave door parts from obstacle list to allow articulation movement.
5. **Open action:** Use `robot.open()` with OpenConfig specifying target_limit="min", solvable_threshold, and extra_checkers for joint angle verification.
6. **Verification:** Check articulation joint angle reached the threshold (opened state).

---

# INPUT

**scene_description:** A microwave oven placed on a table. The microwave has a door that can be opened by pulling the handle.

**task description:** Open the microwave door by grasping the handle and pulling it open.

**code to generate the scene:**

```python
from helper import *
import numpy as np

"""
scene_name: table_with_microwave
description: A microwave placed on a table with specific rotation and position constraints
"""

def get_desktop_top_position(desk_shape: Shape) -> Tuple[P, P]:
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]
    return desktop_pos, desktop_size

@register()
def microwave_on_table(microwave_oid: str) -> Shape:
    microwave_shape = library_call("usd", oid=microwave_oid, keywords=["microwave", "kitchen_appliance", "gray", "rectangular"])
    microwave_size = compute_shape_sizes(microwave_shape)
    rotation_angle_deg = np.random.uniform(175, 185)
    rotation_angle_rad = np.radians(rotation_angle_deg)
    x_pos = np.random.uniform(-0.12459, -0.07)
    y_pos = np.random.uniform(-0.05, 0.03)
    z_pos = 0.87465 - 0.14135
    translated_shape = transform_shape(microwave_shape, translation_matrix((x_pos, y_pos, z_pos)))
    microwave_center = compute_shape_center(translated_shape)
    rotated_shape = transform_shape(translated_shape, rotation_matrix(rotation_angle_rad, direction=(0, 0, 1), point=microwave_center))
    return rotated_shape

@register()
def table_with_microwave() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"])
    microwave_shape = library_call("microwave_on_table", microwave_oid="benchmark_microwave_oven_003")
    return concat_shapes(table_shape, microwave_shape)

@register()
def root_scene() -> Shape:
    return table_with_microwave()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    objects_list = {
        "table_000": {"has_interactions": False},
        "benchmark_microwave_oven_003": {"has_interactions": False},
    }

    scene_info = robot.scene_info
    interactions = robot.interactions

    description_json = {
        "task_name": "打开微波炉门",
        "english_task_name": "open the microwave door",
        "init_scene_text": "The robot is in front of the table, there is a microwave on the table",
    }
    robot.client.set_task_task_basic_info(description_json)

    microwave_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if "microwave" in keywords or "kitchen_appliance" in keywords:
            microwave_id = obj_id
            break

    if microwave_id is None:
        print("Microwave not found in scene")
        return False

    microwave_pos, microwave_rot = robot.get_object_pose(microwave_id)
    handle_pos, handle_rot = robot.get_object_pose(microwave_id, "part_01")

    grasp_orientation_local = np.array([-0.5, -0.5, -0.5, 0.5])
    local_position = np.array([0.0, 0.02, 0.0])
    grasp_position, grasp_orientation = get_world_position_and_orientation_from_relative(
        local_position, grasp_orientation_local, handle_pos, handle_rot
    )

    move_config = MoveConfig(
        action_description={"zh": "抓取微波炉门把手", "en": "grasp the microwave door handle"},
        gripper_action="close",
    )

    target_poses = []
    for z in np.linspace(-0.03, 0.03, 10):
        for rotate_z in np.linspace(-10, 10, 10):
            target_pose = np.eye(4)
            target_pose[:3, 3] = grasp_position
            target_pose[:3, :3] = quat2mat_wxyz(grasp_orientation)
            target_pose[2, 3] += z
            target_pose = rotate_along_axis(target_pose, rotate_z, "z", use_local=False)
            target_poses.append(target_pose)
    target_poses = np.array(target_poses)

    move_success = robot.move_actor_to_target_pose(
        "gripper", target_poses, arm="left", move_config=move_config
    )

    if not move_success:
        print("Move actor to target pose failed")
        return False

    robot.remove_objs_from_obstacle([
        robot.get_object_prim_path(microwave_id, "part_01"),
        robot.get_object_prim_path(microwave_id, "part_00"),
    ])

    def check_open_success() -> bool:
        current_joint_info = robot.get_articulation_joint_info(
            robot.get_object_prim_path(microwave_id)
        )
        current_joint_angle = current_joint_info["current_angle_deg"]
        upper_limit_deg = current_joint_info["upper_limit_deg"]
        lower_limit_deg = current_joint_info["lower_limit_deg"]
        open_threshold_ratio = 0.3
        if current_joint_angle <= upper_limit_deg - open_threshold_ratio * (upper_limit_deg - lower_limit_deg):
            return True
        return False

    open_success = robot.open(
        actor=microwave_id,
        open_config=OpenConfig(
            target_limit="min",
            solvable_threshold=0.4,
            action_description={"zh": "打开微波炉门", "en": "open the microwave door"},
            extra_checkers=[check_open_success],
        ),
        arm="left",
    )

    if not open_success:
        print("Open microwave door failed")
        return False
    return True
```
