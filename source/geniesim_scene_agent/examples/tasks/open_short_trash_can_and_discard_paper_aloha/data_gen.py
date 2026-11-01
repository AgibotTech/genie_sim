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

    # Use the measured left-hand contact in the robot body frame.
    # The articulation-root frame has a different origin.
    body_pos, body_rot = robot.get_object_pose(trash_can_id, articulation_part="body")
    body_pose = np.eye(4)
    body_pose[:3,:3] = quat2mat_wxyz(body_rot)
    body_pose[:3,3] = body_pos
    reference_approach = np.array([[0.6795765214436812, 0.5328156193987729, -0.5042650763509283, 0.10171265833983399], [-0.4982246959861764, -0.16932941885773792, -0.850352691663161, -0.03937537962393478], [-0.5384681084446763, 0.8291170385621096, 0.15038960254621825, 0.0003337429721436691], [0.0, 0.0, 0.0, 1.0]])
    target_poses = np.array([body_pose @ reference_approach])
    print(f"REFERENCE_PEDAL source_episode=14018044 target={target_poses[0].tolist()}")

    # 6. First, close the gripper (empty grasp)
    robot.close_gripper(arm=press_arm)

    # 7. Define action description for moving above button
    move_config = MoveConfig(
        action_description={"zh": "将夹爪移动到垃圾桶按钮上方", "en": "moves the gripper above the trash can button"}
    )

    # Move gripper above the button
    move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=press_arm,
        move_config=move_config
    )

    if not move_success:
        print("Move gripper above button failed")
        return False

    # 8. Define checker function for button press
    def check_cover_opened():
        ee = robot.get_ee_pose(arm=press_arm)
        print(f"PEDAL_PRESS actual_delta={(ee[:3,3]-press_start[:3,3]).tolist()} actual_pose={ee.tolist()}")
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

    # 9. Press the button (move downward)
    gripper_pose = robot.get_ee_pose(arm=press_arm)
    press_start = gripper_pose.copy()
    print(f"PEDAL_HAND downward_angle_deg={np.degrees(np.arcsin(np.clip(-gripper_pose[2,2], -1, 1))):.2f}")
    press_move_config = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.045, 1, 0, 0, 0],  # Move 0.045m downward
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],  # Allow movement in z direction
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_cover_opened],
        action_description={"zh": "按压垃圾桶的脚按钮", "en": "presses the foot pedal of the trash can"},
        unmovable_objects=[trash_can_id],
    )

    # Remove part_00 from obstacle list (ignore collision with button part)
    prim_paths = [robot.get_object_prim_path(trash_can_id, articulation_part="part_00")]
    robot.client.remove_objs_from_obstacle(prim_paths)

    # Press the button (move downward from current position)
    press_move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=[gripper_pose],
        arm=press_arm,
        move_config=press_move_config,
    )

    if not press_move_success:
        print("Press button failed")
        return False


    # ========== STEP 2: Pick up the paper ball ==========
    print("\n=== STEP 2: Picking up trash ===")
    # Use the paper asset's annotated contact pairs, as in the original
    # pick/place task, restricted to approaches from above.
    def overhead_paper_grasps(poses, object_pose, widths):
        mask = (poses[:, 2, 2] < -0.85) & (poses[:, 2, 3] > trash_pos[2])
        print(f"OVERHEAD_GRASPS {int(mask.sum())}/{len(mask)}")
        return poses[mask], widths[mask]

    grasp_config = GraspConfig(
        grasp_pose_filter=overhead_paper_grasps,
        disable_upside_down=False,
        flip_grasp=True,
        pick_up_distance=0.0,
        extra_params={"pre_grasp_distance": 0.07},
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

    if not check_cover_opened():
        print("Lid did not stay open during paper lift")
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

    # Generate multiple placement poses inside trash can
    # Preserve the grasp transform while choosing a verified reachable wrist
    # orientation above the opening (paper orientation is free at release).
    current_pos, current_rot = robot.get_object_pose(trash_id)
    current_paper = np.eye(4)
    current_paper[:3,:3] = quat2mat_wxyz(current_rot)
    current_paper[:3,3] = current_pos
    paper_to_gripper = np.linalg.inv(current_paper) @ robot.get_ee_pose(arm=trash_arm)
    placement_poses = []
    for pitch in [-20, -30, -40, 20, 30, 40]:
        for yaw in list(range(0, 360, 15)):
            for x_offset in [0.0, -0.015, 0.015]:
                pose = np.eye(4)
                wrist_rotation = euler2mat(np.radians([180, pitch, yaw]))
                pose[:3,:3] = wrist_rotation @ paper_to_gripper[:3,:3].T
                pose[:3,3] = trash_can_pos + [x_offset, 0, 0]
                pose[2,3] = trash_can_pos[2] + 0.0325 + 0.02 + trash_size[2]/2
                placement_poses.append(pose)

    placement_poses = np.array(placement_poses)
    print(f"Generated {len(placement_poses)} candidate poses inside trash can")
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
        actor=trash_id, arm=trash_arm, target_pose=placement_poses,
        place_config=place_config,
    )

    if not place_success:
        print("Place trash into trash can failed")
        return False
    robot.client.run_step(120)
    if not check_trash_in_can():
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
