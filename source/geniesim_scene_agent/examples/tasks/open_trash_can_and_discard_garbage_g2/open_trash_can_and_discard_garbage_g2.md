# Open a trash can and discard garbage

**Tags:** `dual_arm` `trash_can` `trash` `press` `articulation` `pick_actor` `place_actor_to_target_pose` `place_into_container` `check_box_containment` `judge_arm_to_use` `move_actor_to_target_pose` `path_constraint` `goal_offset` `articulation_part` `extra_checkers` `unmovable_objects` `GraspConfig` `PlaceConfig` `MoveConfig` `local_to_world` `quat2mat_wxyz`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table, trash can (benchmark_trash_can_015), and trash item (benchmark_trash_021/024/028). Get poses and sizes.
2. **Arm assignment:** trash_arm = judge_arm_to_use(trash_pos) for picking trash; press_arm = the other hand for pressing pedal.
3. **STEP 1 – Press pedal:** Same as open_trash_can_g2: local pedal at x≈0.1, y=-0.08847, z∈[-0.02,0.02]; build candidate poses above pedal; close_gripper(press_arm); move_actor_to_target_pose(gripper, arm=press_arm); remove part_00 from obstacle; press with from_current_pose + goal_offset [0,0,-0.045], path_constraint; extra_checkers verify that the cover (part_01) opens.
4. **STEP 2 – Pick trash:** pick_actor(trash_id, arm=trash_arm, GraspConfig with unmovable_objects=[trash_can_id, table_id]).
5. **STEP 3 – Place into can:** Generate placement poses inside trash can (xy offsets, z = trash_can_top_z + margin + trash_size[2]/2). place_actor_to_target_pose(trash_id, arm=trash_arm, target_pose, PlaceConfig with extra_checkers=check_trash_in_can). check_trash_in_can uses check_box_containment to verify placement inside the can.

---

# INPUT

**scene_description:** A table with a trash can (random position/rotation) and one piece of trash on the table in one of two regions. The robot must use one hand to press the trash can foot pedal to open the lid, and the other hand to pick the trash and throw it into the trash can.

**task description:** First determine which hand picks up the trash; use the other hand to press the trash can foot pedal. Then use the first hand to pick up the trash and throw it into the trash can. Verify that pressing opens the lid and the trash is placed inside the can.

**code to generate the scene:**

```python
from helper import *
import random
"""
scene_name: open_trash_can_and_discard_garbage_g2
description: A trash can on a table with random rotation and position, plus a piece of trash on the table in one of two regions.
"""


@register()
def trash_can() -> Shape:
    return library_call(
        "usd",
        oid="benchmark_trash_can_015",
        keywords=["trash_can", "black", "cylinder", "waste_container"],
    )


@register()
def trash_item() -> Shape:
    trash_options = ["benchmark_trash_021", "benchmark_trash_028", "benchmark_trash_024"]
    selected_trash = random.choice(trash_options)
    return library_call(
        "usd",
        oid=selected_trash,
        keywords=["trash", "waste", "litter", "on_table"],
    )


@register()
def place_trash_can_on_table(table_shape: Shape) -> Shape:
    table_info = get_object_info(table_shape)
    trash_can_subpart_info = get_subpart_info(
        object_id="benchmark_trash_can_015", subpart_id="body"
    )
    table_top_z = table_info["center"][2] + trash_can_subpart_info["xyz_max"][2] + 0.4
    trash_can_x = random.uniform(-0.06, 0.02)
    trash_can_y = random.uniform(-0.02, 0.02)
    trash_can_pos = np.array([trash_can_x, trash_can_y, table_top_z])
    angle_degrees = random.uniform(175, 185)
    angle_radians = angle_degrees * math.pi / 180
    trash_can_shape = library_call("trash_can")
    trash_can_transformed = transform_shape(
        trash_can_shape, translation_matrix(trash_can_pos)
    )
    trash_can_center = compute_shape_center(trash_can_transformed)
    trash_can_transformed = transform_shape(
        trash_can_transformed,
        rotation_matrix(angle_radians, direction=(0, 0, 1), point=trash_can_center)
    )
    return concat_shapes(table_shape, trash_can_transformed)


@register()
def place_trash_on_table(table_shape: Shape, trash_can_shape: Shape) -> Shape:
    table_info = get_object_info(table_shape)
    trash_item_subpart_info = get_subpart_info(
        object_id=random.choice(["benchmark_trash_021", "benchmark_trash_028", "benchmark_trash_024"]),
        subpart_id="body"
    )
    table_top_z = table_info["center"][2] + trash_item_subpart_info["xyz_max"][2] + 0.4
    region_choice = random.choice([1, 2])
    if region_choice == 1:
        trash_x = random.uniform(-0.2, 0.1)
        trash_y = random.uniform(-0.2, -0.15)
    else:
        trash_x = random.uniform(-0.2, 0.1)
        trash_y = random.uniform(0.15, 0.2)
    trash_pos = np.array([trash_x, trash_y, table_top_z])
    trash_shape = library_call("trash_item")
    trash_transformed = transform_shape(trash_shape, translation_matrix(trash_pos))
    trash_min = compute_shape_min(trash_transformed)
    trash_max = compute_shape_max(trash_transformed)
    trash_can_min = compute_shape_min(trash_can_shape)
    trash_can_max = compute_shape_max(trash_can_shape)
    if not (trash_max[0] < trash_can_min[0] or trash_min[0] > trash_can_max[0] or
            trash_max[1] < trash_can_min[1] or trash_min[1] > trash_can_max[1]):
        shift_direction = random.choice([-1, 1])
        trash_transformed = transform_shape(
            trash_transformed,
            translation_matrix((shift_direction * 0.05, 0, 0))
        )
    return concat_shapes(table_shape, trash_can_shape, trash_transformed)


@register()
def table_with_trash_can_and_trash() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    table_with_trash_can = place_trash_can_on_table(table_shape)
    trash_can_shape = library_call("trash_can")
    trash_can_subpart_info = get_subpart_info(
        object_id="benchmark_trash_can_015", subpart_id="body"
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["center"][2] + trash_can_subpart_info["xyz_max"][2] + 0.4
    trash_can_x = random.uniform(-0.06, 0.02)
    trash_can_y = random.uniform(-0.02, 0.02)
    trash_can_pos = np.array([trash_can_x, trash_can_y, table_top_z])
    trash_can_transformed = transform_shape(
        trash_can_shape, translation_matrix(trash_can_pos)
    )
    angle_degrees = random.uniform(175, 185)
    angle_radians = angle_degrees * math.pi / 180
    trash_can_center = compute_shape_center(trash_can_transformed)
    trash_can_transformed = transform_shape(
        trash_can_transformed,
        rotation_matrix(angle_radians, direction=(0, 0, 1), point=trash_can_center)
    )
    final_scene = place_trash_on_table(table_shape, trash_can_transformed)
    return final_scene


@register()
def root_scene() -> Shape:
    return table_with_trash_can_and_trash()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on interaction data
    objects_list = {
        "benchmark_trash_can_015": {"has_interactions": False, "object_type": "trash_can"},
        "benchmark_trash_021": {"has_interactions": True, "object_type": "trash"},
        "benchmark_trash_024": {"has_interactions": True, "object_type": "trash"},
        "benchmark_trash_028": {"has_interactions": True, "object_type": "trash"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "先确定拿垃圾的手，用另一只手按压垃圾桶脚踏板，然后第一只手拿起垃圾丢进垃圾桶",
        "english_task_name": "first determine which hand picks up trash, use the other hand to press the trash can foot pedal, then use the first hand to pick up trash and throw it into the trash can",
        "init_scene_text": "The robot is in front of the table, there is a trash can and a piece of trash on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find the trash can on the table
    trash_can_id = None
    trash_can_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's a trash can by USD or keywords
        if "benchmark_trash_can_015" in usd or any("trash_can" in kw.lower() for kw in keywords):
            trash_can_id = obj_id
            trash_can_info = obj_info
            break

    if not trash_can_id:
        print("Trash can not found")
        return False

    print(f"Found trash can: {trash_can_id}")

    # 3. Find the trash item on the table
    trash_id = None
    trash_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's trash by USD or keywords
        if "benchmark_trash_021" in usd or "benchmark_trash_024" in usd or "benchmark_trash_028" in usd:
            trash_id = obj_id
            trash_info = obj_info
            break

    if not trash_id:
        print("Trash item not found")
        return False

    print(f"Found trash: {trash_id}")

    # 4. Get trash can position and rotation
    trash_can_pos, trash_can_rot = robot.get_object_pose(trash_can_id)
    trash_can_origin_pos = trash_can_pos.copy()
    trash_can_origin_rot = trash_can_rot.copy()

    # Get part_01 position for trash can (cover)
    trash_can_cover_part_pos, trash_can_cover_part_rot = robot.get_object_pose(trash_can_id, articulation_part="part_01")
    trash_can_size = robot.get_object_size_in_world_frame(trash_can_id)

    if trash_can_pos is None or trash_can_size is None:
        print("Cannot get trash can position or size")
        return False

    print(f"Trash can position: {trash_can_pos}")
    print(f"Trash can size: {trash_can_size}")

    # 5. Get trash position and size
    trash_pos, trash_rot = robot.get_object_pose(trash_id)
    trash_size = robot.get_object_size_in_world_frame(trash_id)

    if trash_pos is None or trash_size is None:
        print("Cannot get trash position or size")
        return False

    print(f"Trash position: {trash_pos}")
    print(f"Trash size: {trash_size}")

    # 6. Determine which arm to use for picking up trash
    # 根据垃圾的位置选择拿垃圾的手
    trash_arm = robot.judge_arm_to_use(trash_pos)
    print(f"Arm for picking up trash: {trash_arm}")

    # 7. Determine which arm to use for pressing the pedal
    # 使用另一只手按压脚踏板
    press_arm = "left" if trash_arm == "right" else "right"
    print(f"Arm for pressing pedal: {press_arm}")

    # ========== STEP 1: 按压垃圾桶脚踏板 ==========
    print("\n=== STEP 1: Pressing trash can foot pedal ===")

    # Get trash can rotation matrix
    trash_can_rot_matrix = quat2mat_wxyz(trash_can_rot)

    # 脚踏板的局部坐标位置 (从测试代码中得到)
    x_candidates = np.linspace(0.1, 0.105, 5)  # 5 x positions
    y_value = -0.08847  # fixed y
    z_candidates = np.linspace(-0.02, 0.02, 5)      # 5 z positions

    # Create local to world transformation matrix
    local_to_world = np.eye(4)
    local_to_world[:3, :3] = trash_can_rot_matrix
    local_to_world[:3, 3] = trash_can_pos

    # Base orientation: gripper z-axis downward (world -z direction)
    base_downward_rotation = euler2mat([np.pi, 0, 0])  # Rotate 180 degrees around x

    # Generate target poses above the button
    pedal_target_poses = []

    for x_local in x_candidates:
        for z_local in z_candidates:
            # Local position of button
            local_pos = np.array([x_local, y_value, z_local])

            # Convert to world coordinates
            local_pos_homogeneous = np.append(local_pos, 1.0)
            world_pos_homogeneous = np.dot(local_to_world, local_pos_homogeneous)
            world_pos = world_pos_homogeneous[:3]

            # Target position: above the button (0.02m above)
            target_pos = world_pos.copy()
            target_pos[2] += 0.05  # 5cm above
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_pos
            target_pose[:3, :3] = base_downward_rotation

            # Generate multiple orientations with small rotations
            for angle in np.linspace(180, 270, 12, endpoint=False):
                rotated_target_pose = rotate_along_axis(target_pose, angle, rot_axis="z", use_local=False)
                for angle_x in np.random.uniform(-10, 10, 10):
                    rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_x, rot_axis="x", use_local=False)
                    for angle_y in np.random.uniform(-10, 10, 10):
                        rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_y, rot_axis="y", use_local=False)
                pedal_target_poses.append(rotated_target_pose)

    pedal_target_poses = np.array(pedal_target_poses)
    print(f"Generated {len(pedal_target_poses)} candidate poses above pedal")


    # First, close the gripper (empty grasp) for pressing hand
    robot.close_gripper(arm=press_arm)

    # Define action description for moving above pedal
    move_config = MoveConfig(
        unmovable_objects=[trash_can_id, trash_id, table_id],
        action_description={"zh": "将夹爪移动到垃圾桶脚踏板上方", "en": "moves the gripper above the trash can foot pedal"}
    )

    # Move gripper above the pedal
    move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=pedal_target_poses,
        arm=press_arm,
        move_config=move_config,
    )

    if not move_success:
        print("Move gripper above pedal failed")
        return False

    # Define checker function for pedal press
    def check_cover_opened():
        # Get current trash can part position
        current_trash_can_cover_part_pos, current_trash_can_cover_part_rot = robot.get_object_pose(trash_can_id, articulation_part="part_01")
        if current_trash_can_cover_part_pos is None or current_trash_can_cover_part_rot is None:
            return False

        # Check: local X-axis angle between before/after must be > 70 degrees
        rot_before = quat2mat_wxyz(trash_can_cover_part_rot)
        rot_after = quat2mat_wxyz(current_trash_can_cover_part_rot)
        x_axis_before = rot_before[:3, 0] / np.linalg.norm(rot_before[:3, 0])
        x_axis_after = rot_after[:3, 0] / np.linalg.norm(rot_after[:3, 0])
        x_axis_angle_deg = np.degrees(np.arccos(np.clip(np.dot(x_axis_before, x_axis_after), -1.0, 1.0)))
        if x_axis_angle_deg <= 70:
            print(f"Cover part X-axis angle = {x_axis_angle_deg:.1f} deg (required > 70 deg)")
            return False
        return True
    gripper_pose = robot.get_ee_pose(arm=press_arm)
    gripper_pose[3,3] -= 0.045
    # Press the pedal (move downward)
    press_move_config = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.045, 1, 0, 0, 0],  # Move 0.045m downward
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],  # Allow movement in z direction # Allow movement in z direction
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_cover_opened],
        action_description={"zh": "按压垃圾桶的脚踏板", "en": "presses the foot pedal of the trash can"},
        unmovable_objects=[trash_can_id, trash_id, table_id],
    )

    # Remove part_00 from obstacle list (ignore collision with pedal part)
    prim_paths = [robot.get_object_prim_path(trash_can_id, articulation_part="part_00")]
    robot.client.remove_objs_from_obstacle(prim_paths)
    # # Press the pedal (move downward from current position)
    press_move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=[gripper_pose],
        arm=press_arm,
        move_config=press_move_config,
    )
    if not press_move_success:
        print("Press pedal failed")
        return False
    # ========== STEP 2: 拿起垃圾 ==========
    print("\n=== STEP 2: Picking up trash ===")
    # Create grasp configuration for trash
    grasp_config = GraspConfig(
        pick_up_distance=0.15,
        unmovable_objects=[trash_can_id, table_id],
        action_description={"zh": "拿起垃圾", "en": "picks up the trash"}
    )
    pick_success = robot.pick_actor(trash_id, arm=trash_arm, grasp_config=grasp_config)
    if not pick_success:
        print("Pick up trash failed")
        return False



    # ========== STEP 3: 将垃圾丢进垃圾桶 ==========
    print("\n=== STEP 3: Throwing trash into trash can ===")

    # Define checker function to verify trash is inside trash can
    def check_trash_in_can():
        """Check if trash is inside trash can"""
        # Get current trash position
        current_trash_pos, current_trash_rot = robot.get_object_pose(trash_id)
        if current_trash_pos is None:
            return False

        # Get current trash can position
        current_trash_can_pos, current_trash_can_rot = robot.get_object_pose(trash_can_id)
        if current_trash_can_pos is None:
            return False

        # Get sizes
        current_trash_size = robot.get_object_size_in_local_frame(trash_id)
        current_trash_can_size = robot.get_object_size_in_local_frame(trash_can_id)
        if current_trash_size is None or current_trash_can_size is None:
            return False

        # Check containment using check_box_containment
        from common.base_utils.transform_utils import check_box_containment
        trash_can_contains_trash, _ = check_box_containment(
            current_trash_can_pos, current_trash_can_rot, current_trash_can_size,
            current_trash_pos, current_trash_rot, current_trash_size,
            overlap_threshold=0.7  # 50%重叠即可认为是成功
        )

        return trash_can_contains_trash

    # Generate multiple placement poses inside trash can
    placement_poses = []

    # 计算垃圾桶内部的多个位置
    # 垃圾桶顶部高度 = trash_can_pos[2] + trash_can_size[2] / 2.0
    trash_can_top_z = trash_can_pos[2] + trash_can_size[2] / 2.0

    # 生成垃圾桶内部的多个位置
    x_offsets = np.linspace(-0.01, 0.01, 3)  # x方向偏移
    y_offsets = np.linspace(-0.01, 0.01, 3)  # y方向偏移

    for x_offset in x_offsets:
        for y_offset in y_offsets:
            # 垃圾桶中心位置
            target_position = trash_can_pos.copy()
            target_position[0] += x_offset
            target_position[1] += y_offset

            # 放置高度：垃圾桶内部（略低于顶部）
            # 安全边距：0.01m
            margin = 0.03
            target_position[2] = trash_can_top_z + margin + trash_size[2] / 2.0
            # 垃圾桶的旋转矩阵
            trash_rot_matrix = quat2mat_wxyz(trash_rot)

            # 生成多个随机的垃圾朝向
            for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                # 创建旋转矩阵
                pose = np.eye(4)
                pose[:3, :3] = trash_rot_matrix
                pose[:3, 3] = target_position
                pose = rotate_along_axis(pose, angle, rot_axis="z", use_local=False)
                placement_poses.append(pose)

    placement_poses = np.array(placement_poses)
    print(f"Generated {len(placement_poses)} candidate poses inside trash can")
    # Create place configuration
    place_config = PlaceConfig(
        disable_upside_down=False,
        pre_insert_offset=0.03,
        unmovable_objects=[trash_can_id, table_id],
        extra_checkers=[check_trash_in_can],
        action_description={"zh": "将垃圾扔进垃圾桶", "en": "throws the trash into the trash can"}
    )


    # Place trash into trash can
    place_success = robot.place_actor_to_target_pose(
        actor=trash_id,
        arm=trash_arm,
        target_pose=placement_poses,
        place_config=place_config
    )

    if not place_success:
        print("Place trash into trash can failed")
        return False
    robot.back_to_origin(arm=trash_arm)
    # robot.back_to_origin(arm=press_arm)
    # 任务总结
    print(f"\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: Pedal pressed, trash picked up and thrown into trash can")
    else:
        print("⚠️ PARTIAL SUCCESS: Some operations may have failed")

    return success
```
