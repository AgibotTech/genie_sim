# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on interaction data
    objects_list = {
        "benchmark_trash_can_015_short": {"has_interactions": False, "object_type": "trash_can"},
        "benchmark_trash_021": {"has_interactions": True, "object_type": "trash"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "first determine which hand picks up trash, use the other hand to press the trash can foot pedal, then use the first hand to pick up trash and throw it into the trash can",
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
        if "benchmark_trash_can_015_short" in usd or any("trash_can" in kw.lower() for kw in keywords):
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
        if "paper_ball" in keywords or "benchmark_trash_021" in usd:
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
    # Choose the garbage-grasping arm based on the garbage position.
    trash_arm = robot.judge_arm_to_use(trash_pos)
    print(f"Arm for picking up trash: {trash_arm}")

    # 7. Determine which arm to use for pressing the pedal
    # Use the other hand to press the pedal.
    press_arm = "left" if trash_arm == "right" else "right"
    print(f"Arm for pressing pedal: {press_arm}")

    # ========== STEP 1: Press the trash-can pedal ==========
    print("\n=== STEP 1: Pressing trash can foot pedal ===")

    # Get trash can rotation matrix
    trash_can_rot_matrix = quat2mat_wxyz(trash_can_rot)

    # Pedal-local position derived from the test code.
    x_candidates = np.linspace(0.1, 0.105, 5)  # 5 x positions
    y_value = -0.08847  # fixed y
    z_candidates = np.linspace(0.012, 0.022, 3)      # 5 z positions

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
            for pitch in [-60, -40, -20, 0, 20, 40, 60]:
                for yaw in range(0, 360, 30):
                    candidate=target_pose.copy()
                    candidate[:3,:3]=euler2mat(np.radians([180, pitch, yaw]))
                    pedal_target_poses.append(candidate)

    pedal_target_poses = np.array(pedal_target_poses)
    robot.close_gripper(arm=press_arm)
    # Check both endpoints before choosing an approach. Pedal contact is allowed
    # in the downward move; all other collision constraints remain active.
    press_endpoints = pedal_target_poses.copy()
    press_endpoints[:,2,3] -= 0.055
    prim_paths = [robot.get_object_prim_path(trash_can_id, articulation_part="part_00")]
    robot.client.remove_objs_from_obstacle(prim_paths)
    valid, _ = robot.robot.solve_ik(press_endpoints, arm=press_arm, type="AvoidObs")
    pedal_target_poses = pedal_target_poses[valid]
    print(f"Generated {len(pedal_target_poses)} approaches with reachable press endpoints")
    if not len(pedal_target_poses):
        return False


    # First, close the gripper (empty grasp) for pressing hand

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
        print(f"LID_CHECK angle_deg={x_axis_angle_deg:.2f}")
        if x_axis_angle_deg <= 70:
            print(f"Cover part X-axis angle = {x_axis_angle_deg:.1f} deg (required > 70 deg)")
            return False
        return True
    gripper_pose = robot.get_ee_pose(arm=press_arm)
    print(f"PRESS_START actual_pose={gripper_pose.tolist()}")
    distances = np.linalg.norm(pedal_target_poses[:,:3,3] - gripper_pose[:3,3], axis=1) + 0.1*np.linalg.norm(pedal_target_poses[:,:3,:3]-gripper_pose[:3,:3], axis=(1,2))
    gripper_pose = pedal_target_poses[np.argmin(distances)].copy()
    gripper_pose[2,3] -= 0.055
    print(f"PRESS_END target_pose={gripper_pose.tolist()}")
    # Press the pedal (move downward)
    press_move_config = MoveConfig(
        extra_params={
            "from_current_pose": False,
            "goal_offset": [0, 0, 0, 1, 0, 0, 0],  # Move 0.055m downward
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
    # ========== STEP 2: Pick up the paper ball ==========
    print("\n=== STEP 2: Picking up trash ===")
    # Keep annotated contact pairs near the paper centre and approach above.
    def overhead_paper_grasps(poses, object_pose, widths):
        mask = (poses[:, 2, 2] < -0.7) & (poses[:, 2, 3] > trash_pos[2]) & (np.linalg.norm(poses[:, :2, 3] - trash_pos[:2], axis=1) < 0.02)
        print(f"OVERHEAD_GRASPS {int(mask.sum())}/{len(mask)}")
        return poses[mask], widths[mask]
    grasp_config = GraspConfig(
        grasp_pose_filter=overhead_paper_grasps,
        disable_upside_down=False, flip_grasp=True,
        pick_up_distance=0.0,
        unmovable_objects=[trash_can_id, table_id],
        action_description={"zh": "从上方夹住纸团", "en": "grasps the paper ball from above"},
    )
    pick_success = robot.pick_actor(trash_id, arm=trash_arm, grasp_config=grasp_config)
    if not pick_success:
        print("Pick up trash failed")
        return False



    # Lift with a full-pose plan so the 6-DoF arm can correct its orientation.
    # Verify real paper motion after closing; closing alone is not success.
    lift_start, lift_rotation = robot.get_object_pose(trash_id)
    lift_poses = []
    for dx in [0.0, -0.04]:
        for roll in [-20, 0, 20]:
            for pitch in [-20, 0, 20]:
                for yaw in [-20, 0, 20]:
                    pose = np.eye(4)
                    pose[:3, :3] = euler2mat(np.radians([roll, pitch, yaw])) @ quat2mat_wxyz(lift_rotation)
                    pose[:3, 3] = lift_start + np.array([dx, 0.0, 0.13])
                    lift_poses.append(pose)
    def check_paper_lifted():
        pos, _ = robot.get_object_pose(trash_id)
        valid = pos[2] > trash_pos[2] + 0.08
        print(f"LIFT_CHECK height={pos[2]-trash_pos[2]:.3f} lifted={valid}")
        return valid
    if not robot.move_actor_to_target_pose(
        actor=trash_id, target_pose=np.array(lift_poses), arm=trash_arm,
        move_config=MoveConfig(unmovable_objects=[trash_can_id, table_id],
            extra_checkers=[check_paper_lifted],
            action_description={"zh": "抬起夹住的纸团", "en": "lifts the grasped paper ball"}),
    ):
        return False


    # ========== STEP 3: Discard the garbage into the trash can ==========
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

        # The articulated body pose is its USD origin, not the bounding-box
        # center. Test the actual cavity, excluding the lid and external pedal.
        local_trash = quat2mat_wxyz(current_trash_can_rot).T @ (current_trash_pos-current_trash_can_pos)
        cavity_center = current_trash_can_pos + quat2mat_wxyz(current_trash_can_rot) @ np.array([0.005, -0.046, 0.0])
        inside, _ = check_box_containment(
            cavity_center, current_trash_can_rot, [0.145, 0.128, 0.140],
            current_trash_pos, current_trash_rot, robot.get_object_size_in_local_frame(trash_id),
            overlap_threshold=0.7,
        )
        radial = np.linalg.norm(local_trash[[0,2]] - np.array([0.005,0]))
        valid = bool(inside and radial < 0.065 and -0.112 < local_trash[1] < 0.017)
        print(f"TRASH_CHECK local={local_trash.tolist()} inside={valid}")
        return valid

    # Keep the paper above the opening, with modest rotation freedom.
    _, current_rot = robot.get_object_pose(trash_id)
    placement_poses = []
    for roll in [-30, -15, 0, 15, 30]:
        for pitch in [-30, -15, 0, 15, 30]:
            for yaw in [-30, -15, 0, 15, 30]:
                pose = np.eye(4)
                pose[:3,:3] = euler2mat(np.radians([roll,pitch,yaw])) @ quat2mat_wxyz(current_rot)
                pose[:3,3] = trash_can_pos + [0, 0, 0.0325 + 0.075]
                placement_poses.append(pose)
    placement_poses = np.array(placement_poses)
    # Create place configuration
    place_config = PlaceConfig(
        disable_upside_down=False,
        pre_insert_offset=None,
        unmovable_objects=[trash_can_id, table_id],
        extra_checkers=[check_trash_in_can],
        action_description={"zh": "将纸团扔进垃圾桶", "en": "throws the trash into the trash can"}
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
    robot.client.run_step(60)
    if not check_trash_in_can():
        return False
    # robot.back_to_origin(arm=press_arm)
    # Task summary.
    print(f"\n=== Task summary ===")
    if success:
        print("✅ SUCCESS: Pedal pressed, trash picked up and thrown into trash can")
    else:
        print("⚠️ PARTIAL SUCCESS: Some operations may have failed")

    return success
