# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import re

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on interaction data
    objects_list = {
        "benchmark_chessboard_001": {"has_interactions": True, "object_type": "chessboard"},
        "benchmark_chess_000": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_001": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_002": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_003": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_004": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_005": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_006": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_007": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_008": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_009": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_010": {"has_interactions": True, "object_type": "chess_piece"},
        "benchmark_chess_011": {"has_interactions": True, "object_type": "chess_piece"}
    }
    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "place the chess piece on the table into the storage box",
        "english_task_name": "place the chess piece on the table into the storage box",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with a target chess piece outside"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find chessboard (benchmark_chessboard_001)
    board_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if "benchmark_chessboard_001" in usd:
            board_id = obj_id
            break

    if not board_id:
        print("Chessboard not found")
        return False

    print(f"Found chessboard: {board_id}")

    ## 2. Find target chess piece (by tag: position_x_{xxx}_y_{xxx})
    target_piece_id = None
    target_position_tag = None

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        for kw in keywords:
            if "position_x_" in kw and "_y_" in kw:
                target_piece_id = obj_id
                target_position_tag = kw
                print(f"Found target piece: {obj_id} with tag: {kw}")
                break
        if target_piece_id:
            break

    if not target_piece_id or not target_position_tag:
        print("Target chess piece not found or no position tag")
        return False

    ## 3. Parse target position from tag
    ## Tag format: position_x_{value}_y_{value}, where value is world coordinate
    try:
        # Extract x and y values using regex
        match = re.search(r"position_x_([-+]?\d*\.?\d+)_y_([-+]?\d*\.?\d+)", target_position_tag)
        if match:
            target_x = float(match.group(1))
            target_y = float(match.group(2))
        else:
            # Fallback: split by underscore
            parts = target_position_tag.split('_')
            target_x = float(parts[2])
            target_y = float(parts[4])

        print(f"Target position from tag: ({target_x:.4f}, {target_y:.4f})")
    except Exception as e:
        print(f"Failed to parse position tag: {e}")
        return False

    ## 4. Get board info to calculate target z height
    board_pos, board_rot = robot.get_object_pose(board_id)
    board_size = robot.get_object_size_in_world_frame(board_id)

    if board_pos is None or board_size is None:
        print("Cannot get board position/size")
        return False

    ## 5. Get piece size and initial pose
    piece_pos, piece_quat = robot.get_object_pose(target_piece_id)
    piece_size = robot.get_object_size_in_world_frame(target_piece_id)

    if piece_pos is None or piece_quat is None or piece_size is None:
        print("Cannot get piece pose/size")
        return False

    ## [NEW] Logic: Find all chess pieces already inside the storage box
    pieces_inside_box = []

    for obj_id, obj_info in scene_info["layout"].items():
        # Skip the target piece and the board itself
        if obj_id == target_piece_id or obj_id == board_id:
            continue

        usd = obj_info.get("usd", "")

        # Check if this object is a chess piece (from our list)
        is_chess_piece = False
        for piece_usd in objects_list.keys():
            if piece_usd == usd and piece_usd != "benchmark_chessboard_001":
                is_chess_piece = True
                break

        if is_chess_piece:
            curr_piece_pos, curr_piece_rot = robot.get_object_pose(obj_id)
            curr_piece_size = robot.get_object_size_in_local_frame(obj_id)
            board_local_size = robot.get_object_size_in_local_frame(board_id)

            if curr_piece_pos is not None and curr_piece_rot is not None:
                # Check if the piece is inside the board's area (XY plane projection)
                board_contains_piece, _ = check_box_containment(
                    board_pos, board_rot, board_local_size,
                    curr_piece_pos, curr_piece_rot, curr_piece_size,
                    ignore_axes=["z"],  # Only check XY plane
                    overlap_threshold=0.6  # 60% overlap required to be considered "inside"
                )

                if board_contains_piece:
                    pieces_inside_box.append(obj_id)
                    print(f"Found piece inside box: {obj_id}")

    # Combine board and internal pieces into unmovable_objects list
    unmovable_objects_list = [board_id] + pieces_inside_box
    print(f"Unmovable objects: {unmovable_objects_list}")

    ## 6. Calculate target z height
    ## Board top surface: board_pos[2] + 0.5 * board_size[2]
    ## CRITICAL: Always add safety margin (0.01m) to avoid gripper collision
    margin = 0.01  # Safety margin in meters (1cm)
    board_top_z = board_pos[2] + 0.5 * board_size[2]
    target_z = board_top_z + margin + 0.5 * piece_size[2]

    target_position = np.array([target_x, target_y, target_z])

    print(f"Target position: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]")

    ## 7. Extract initial rotation and generate candidate poses
    ## Convert quaternion to rotation matrix (wxyz format)
    initial_rot_mat = quat2mat_wxyz(piece_quat)

    ## Generate 36 rotation angles around world z axis (0 to 360 degrees, step 10 degrees)
    target_poses = []
    z_angles = np.linspace(0, 2*np.pi, 72, endpoint=False)  # 72 samples as per user request

    for z_angle in z_angles:
        ## Rotation matrix for z axis rotation (world frame)
        rot_z = euler2mat([0, 0, z_angle])

        ## Apply rotation: new_rot = rot_z * initial_rot (rotate around world z axis)
        final_rot = np.dot(rot_z, initial_rot_mat)

        ## Build pose matrix
        pose = np.eye(4)
        pose[:3, :3] = final_rot
        pose[:3, 3] = target_position
        target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses (72 rotations around world z axis)")

    ## 8. Determine which arm to use
    arm = robot.judge_arm_to_use(piece_pos)
    print(f"Using arm: {arm}")

    ## 9. Execute pick and place
    grasp_config = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects_list,  # Use the updated list
        action_description={"zh": "拿起桌上的棋子", "en": "picks up the chess piece on the table"}
    )

    ## 10. Define checker to verify placement success
    def check_piece_placed_at_target():
        """Check if piece is placed at target position"""
        current_pos, _ = robot.get_object_pose(target_piece_id)
        if current_pos is None:
            return False

        ## Check if position matches target (with 2cm tolerance)
        pos_distance = np.linalg.norm(current_pos[:2] - target_position[:2])

        if pos_distance < 0.03:
            return True
        print(f"Piece not placed at target position: {pos_distance:.4f}m")
        return False

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects_list,  # Use the updated list
        extra_checkers=[check_piece_placed_at_target],
        action_description={"zh": "将棋子放置到收纳盒中", "en": "places the chess piece on the table into the storage box"}
    )

    print("Attempting to pick and place target chess piece...")
    success = robot.pick_and_place_actor_to_target_pose(
        actor=target_piece_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if success:
        print("Successfully placed target chess piece to target position")
    else:
        print("Failed to place target chess piece")

    robot.back_to_origin(arm=arm)

    return success
