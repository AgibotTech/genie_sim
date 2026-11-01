# Pass a chips bucket between hands and place it on the opposite side

**Tags:** `hand_over` `dual_arm` `chips_bucket` `snack` `keywords_chips_bucket` `pick_actor` `move_actor_to_target_pose` `handover_ownership_of_attached_object` `place_actor_to_target_pose` `GraspConfig` `MoveConfig` `PlaceConfig` `grasp_pose_filter` `quat2mat_wxyz` `rotate_along_axis` `left_right_by_y` `extra_checkers`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find chips bucket by keywords "chips_bucket" or "snack". Get initial pose; determine side: is_on_left = (init_pos[1] > 0). arm_a = hand on same side, arm_b = other hand.
2. **First grasp:** GraspConfig with set_grasp_pose_center, set_grasp_along_axis='y', grasp_lower/upper_percentile (lower or upper part). pick_actor(chips_id, arm=arm_a, grasp_config).
3. **Handoff:** move_actor_to_target_pose("gripper", handoff_poses, arm=arm_a). Handoff pose: x(-0.1,-0.05), y=0, z from current gripper; rotation for hand-over orientation; multiple candidates with z/y/angle jitter.
4. **Second grasp:** pick_actor(chips_id, arm=arm_b, grasp_config with opposite percentile and grasp_pose_filter so gripper faces other hand). handover_ownership_of_attached_object(from_arm=arm_a); back_to_origin(arm_a).
5. **Place:** Target opposite side (left→y(-0.25,-0.15), right→y(0.15,0.25)); x(-0.15,0.1). place_actor_to_target_pose(chips_id, place_poses, arm=arm_b, place_config with extra_checkers for height).

---

# INPUT

**scene_description:** A table with a bucket of chips (benchmark_snack_241 or 242) placed randomly on the left or right side of the table. Keywords include "chips_bucket", "snack", and position tag "left" or "right".

**task description:** Pick up the chips bucket with one hand (the hand on the same side as the bucket), move it to a hand-off pose in front of the robot, pass it to the other hand, then place it on the opposite side of the table.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import random

"""
scene_name: table_with_chips_bucket
description: A bucket of chips placed randomly on left or right side of a table
"""


def get_table_top_surface(table_shape: Shape) -> Tuple[P, P]:
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center = table_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]
    return desktop_pos, desktop_size


@register()
def random_chips_bucket(position_tag: str) -> Shape:
    snack_options = ["benchmark_snack_241", "benchmark_snack_242"]
    selected_snack = random.choice(snack_options)
    chips_shape = library_call(
        "usd",
        oid=selected_snack,
        keywords=["chips_bucket", "snack", "cylinder", "food", position_tag],
    )
    return chips_shape


@register()
def place_chips_on_table(table_shape: Shape) -> Shape:
    table_top_center, table_top_size = get_table_top_surface(table_shape)
    is_left = random.choice([True, False])
    if is_left:
        pos_x = np.random.uniform(-0.15, 0.1)
        pos_y = np.random.uniform(0.15, 0.25)
        position_tag = "left"
    else:
        pos_x = np.random.uniform(-0.15, 0.1)
        pos_y = np.random.uniform(-0.25, -0.15)
        position_tag = "right"
    chips_shape = library_call("random_chips_bucket", position_tag=position_tag)
    chips_size = compute_shape_sizes(chips_shape)
    chips_bottom_z = table_top_center[2]
    chips_center_z = chips_bottom_z + chips_size[2] / 2.0
    chips_position = np.array([pos_x, pos_y, chips_center_z])
    random_rotation_angle = np.random.uniform(0, 2 * np.pi)
    chips_center = compute_shape_center(chips_shape)
    chips_transformed = transform_shape(chips_shape, translation_matrix(chips_position))
    chips_final = transform_shape(
        chips_transformed,
        rotation_matrix(random_rotation_angle, direction=(0, 0, 1), point=chips_center),
    )
    return concat_shapes(table_shape, chips_final)


@register()
def table_with_chips_bucket() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    return place_chips_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_chips_bucket()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info
    interactions = robot.interactions

    chips_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "chips_bucket" in keywords or "snack" in keywords:
            chips_id = obj_id
            break

    if not chips_id:
        print("Chips bucket not found")
        return False

    init_pos, init_rot = robot.get_object_pose(chips_id)
    init_pos = np.array(init_pos)
    init_rot = np.array(init_rot)
    is_on_left = init_pos[1] > 0
    arm_a = "left" if is_on_left else "right"
    arm_b = "right" if is_on_left else "left"

    print(f"Chips on {'Left' if is_on_left else 'Right'}. First hand: {arm_a}, Second hand: {arm_b}")

    description_json = {
        "task_name": "双手传递薯片桶并放置到对侧",
        "english_task_name": "pick up chips bucket with one hand, pass to the other, and place on the opposite side",
        "init_scene_text": "A bucket of chips is on the table."
    }
    robot.client.set_task_task_basic_info(description_json)

    choose_lower = np.random.choice([True, False])
    UPPER_PART_LOWER_PERCENTILE = 70
    UPPER_PART_UPPER_PERCENTILE = 90
    LOWER_PART_LOWER_PERCENTILE = 10
    LOWER_PART_UPPER_PERCENTILE = 30
    grasp_a_config = GraspConfig(
        set_grasp_pose_center=True,
        set_grasp_along_axis='y',
        grasp_lower_percentile=LOWER_PART_LOWER_PERCENTILE if choose_lower else UPPER_PART_LOWER_PERCENTILE,
        grasp_upper_percentile=LOWER_PART_UPPER_PERCENTILE if choose_lower else UPPER_PART_UPPER_PERCENTILE,
        action_description={"zh": "抓起薯片桶", "en": "grasps the chips bucket"}
    )

    print(f"Hand {arm_a} grasping ({'lower' if choose_lower else 'upper'} part)...")
    success = robot.pick_actor(chips_id, arm=arm_a, grasp_config=grasp_a_config)
    if not success:
        return False

    gripper_pose = robot.get_ee_pose(arm_a)
    handoff_x = np.random.uniform(-0.1, -0.05)
    handoff_y = 0.0
    handoff_z = gripper_pose[2, 3]
    gripper_direction_quaternion = np.array([0, 0, 0.70711, -0.70711]) if is_on_left else np.array([0, 0, 0.70711, 0.70711])
    base_rot_mat = quat2mat_wxyz(gripper_direction_quaternion)
    handoff_poses = []
    z_jitter = np.linspace(-0.05, 0.15, 5)
    y_jitter = np.linspace(-0.02, 0.02, 5)
    angle_jitter = np.linspace(-15, 15, 5)
    for z_off in z_jitter:
        for y_off in y_jitter:
            for angle in angle_jitter:
                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                pose[:3, 3] = [handoff_x, handoff_y + y_off, handoff_z + z_off]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                handoff_poses.append(pose)
    handoff_poses = np.array(handoff_poses)

    print(f"Hand {arm_a} moving to handoff pose...")
    move_config = MoveConfig(action_description={"zh": "移动到胸前位置", "en": "moves to chest position"})
    success = robot.move_actor_to_target_pose("gripper", handoff_poses, arm=arm_a, move_config=move_config)
    if not success:
        return False

    def grasp_pose_filter(grasp_poses, obj_pose, grasp_widths):
        gripper_normal_direction_quat = np.array([0,0,0.70711, -0.70711]) if not is_on_left else np.array([0,0,0.70711, 0.70711])
        gripper_normal_direction_mat = quat2mat_wxyz(gripper_normal_direction_quat)
        gripper_poses_mat = grasp_poses[:, :3, :3]
        for i in range(grasp_poses.shape[0]):
            grasp_poses[i, :3, :3] = gripper_normal_direction_mat
        return grasp_poses, np.array([0.8] * grasp_poses.shape[0])

    grasp_b_config = GraspConfig(
        set_grasp_pose_center=True,
        set_grasp_along_axis='y',
        grasp_lower_percentile=UPPER_PART_LOWER_PERCENTILE if choose_lower else LOWER_PART_LOWER_PERCENTILE,
        grasp_upper_percentile=UPPER_PART_UPPER_PERCENTILE if choose_lower else LOWER_PART_UPPER_PERCENTILE,
        pick_up_distance=0.0,
        action_description={"zh": "接过薯片桶", "en": "receives the chips bucket"},
        extra_params={"grasp_pose_filter": grasp_pose_filter},
    )
    print(f"Hand {arm_b} grasping ({'upper' if choose_lower else 'lower'} part)...")
    success = robot.pick_actor(chips_id, arm=arm_b, grasp_config=grasp_b_config)
    if not success:
        return False

    print(f"Hand {arm_a} releasing and resetting...")
    robot.handover_ownership_of_attached_object(from_arm=arm_a)
    robot.back_to_origin(arm=arm_a)

    target_x_range = np.linspace(-0.15, 0.1, 10)
    if is_on_left:
        target_y_range = np.linspace(-0.25, -0.15, 10)
    else:
        target_y_range = np.linspace(0.15, 0.25, 10)
    margin = 0.01
    placement_z_base = init_pos[2] + margin
    init_rot_mat = quat2mat_wxyz(init_rot)
    place_poses = []
    z_jitter = np.linspace(0, 0.02, 2)
    angle_jitter = np.linspace(0, 360, 9)
    for tx in target_x_range:
        for ty in target_y_range:
            for tz_off in z_jitter:
                for angle in angle_jitter:
                    pose = np.eye(4)
                    pose[:3, :3] = init_rot_mat
                    pose = rotate_along_axis(pose, angle, "z", use_local=False)
                    pose[:3, 3] = [tx, ty, placement_z_base + tz_off]
                    place_poses.append(pose)
    place_poses = np.array(place_poses)

    print(f"Hand {arm_b} placing to opposite side...")
    def place_checker():
        robot.wait(0.5)
        cur_pose, cur_rot = robot.get_object_pose(chips_id)
        cur_pos = np.array(cur_pose)
        if np.abs(cur_pos[2] - init_pos[2]) > 0.02:
            return False
        return True
    place_config = PlaceConfig(
        extra_checkers=[place_checker],
        action_description={"zh": "将薯片放置到对侧", "en": "places the chips to the opposite side"}
    )
    success = robot.place_actor_to_target_pose(chips_id, target_pose=place_poses, arm=arm_b, place_config=place_config)
    return success
```
