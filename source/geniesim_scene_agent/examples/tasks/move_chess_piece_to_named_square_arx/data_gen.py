# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    'Choose a movable piece in the lower half and move it to a legal square.\n    Follow the piece movement rules, allow any legal move length, do not capture, and choose randomly.'

    # Initialize objects_list for the chess pieces and board.
    objects_list = {
        "benchmark_chessboard_000": {"has_interactions": False, "object_type": "chessboard"},
        "benchmark_chess_000": {"has_interactions": True, "object_type": "chess_piece", "name": "pawn_white"},
        "benchmark_chess_001": {"has_interactions": True, "object_type": "chess_piece", "name": "rook_white"},
        "benchmark_chess_002": {"has_interactions": True, "object_type": "chess_piece", "name": "knight_white"},
        "benchmark_chess_003": {"has_interactions": True, "object_type": "chess_piece", "name": "bishop_white"},
        "benchmark_chess_004": {"has_interactions": True, "object_type": "chess_piece", "name": "queen_white"},
        "benchmark_chess_005": {"has_interactions": True, "object_type": "chess_piece", "name": "king_white"},
        "benchmark_chess_006": {"has_interactions": True, "object_type": "chess_piece", "name": "pawn_black"},
        "benchmark_chess_007": {"has_interactions": True, "object_type": "chess_piece", "name": "rook_black"},
        "benchmark_chess_008": {"has_interactions": True, "object_type": "chess_piece", "name": "knight_black"},
        "benchmark_chess_009": {"has_interactions": True, "object_type": "chess_piece", "name": "bishop_black"},
        "benchmark_chess_010": {"has_interactions": True, "object_type": "chess_piece", "name": "queen_black"},
        "benchmark_chess_011": {"has_interactions": True, "object_type": "chess_piece", "name": "king_black"},
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. Find the chessboard (benchmark_chessboard_000).
    chessboard_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_chessboard_000" in usd or any("chessboard" in kw.lower() for kw in keywords):
            chessboard_id = obj_id
            break

    if not chessboard_id:
        print("Chessboard not found")
        return False

    print(f"Found chessboard: {chessboard_id}")

    # 2. Read the chessboard position and dimensions.
    chessboard_pos, chessboard_rot = robot.get_object_pose(chessboard_id)
    chessboard_size = robot.get_object_size_in_world_frame(chessboard_id)

    if chessboard_pos is None or chessboard_size is None:
        print("Cannot get chessboard position/size")
        return False

    print(f"Chessboard position: {chessboard_pos}, size: {chessboard_size}")

    # 3. Find every chess piece, reading square positions directly from keywords.
    chess_pieces = []
    board_state = {}  # Chess square -> piece ID.
    piece_positions = {}  # Piece ID -> chess square.

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether the object is a chess piece.
        is_chess_piece = False
        piece_name = None
        for piece_usd in ["benchmark_chess_000", "benchmark_chess_001", "benchmark_chess_002",
                         "benchmark_chess_003", "benchmark_chess_004", "benchmark_chess_005",
                         "benchmark_chess_006", "benchmark_chess_007", "benchmark_chess_008",
                         "benchmark_chess_009", "benchmark_chess_010", "benchmark_chess_011"]:
            if piece_usd in usd:
                is_chess_piece = True
                piece_name = piece_usd
                break

        if is_chess_piece or any("chess_piece" in kw.lower() for kw in keywords):
            piece_pos, piece_rot = robot.get_object_pose(obj_id)

            # Extract the square from keywords, for example "position_B3".
            position_square = None
            for kw in keywords:
                if "position_" in kw:
                    position_square = kw.replace("position_", "")
                    break

            chess_pieces.append({
                "id": obj_id,
                "usd": usd,
                "position": piece_pos,
                "rotation": piece_rot,  # Save the current rotation.
                "keywords": keywords,
                "piece_type": piece_name,
                "board_square": position_square  # Save the chess square.
            })

            # Update the board state when a square is identified.
            if position_square:
                board_state[position_square] = obj_id
                piece_positions[obj_id] = position_square
                print(f"Found chess piece: {obj_id} at square {position_square}, position: {piece_pos}")
            else:
                print(f"Warning: Chess piece {obj_id} has no position keyword")

    if len(chess_pieces) == 0:
        print("No chess pieces found")
        return False

    print(f"Total chess pieces found: {len(chess_pieces)}")
    print(f"Board state: {board_state}")

    # 4. Map the chessboard coordinate system.
    def square_to_board_coordinates(square):
        'Convert a chess square such as A1 to coordinates relative to the board center.'
        board_coords = {
            "A1": (-0.175, 0.175), "B1": (-0.175, 0.125), "C1": (-0.175, 0.075), "D1": (-0.175, 0.025),
            "E1": (-0.175, -0.025), "F1": (-0.175, -0.075), "G1": (-0.175, -0.125), "H1": (-0.175, -0.175),

            "A2": (-0.125, 0.175), "B2": (-0.125, 0.125), "C2": (-0.125, 0.075), "D2": (-0.125, 0.025),
            "E2": (-0.125, -0.025), "F2": (-0.125, -0.075), "G2": (-0.125, -0.125), "H2": (-0.125, -0.175),

            "A3": (-0.075, 0.175), "B3": (-0.075, 0.125), "C3": (-0.075, 0.075), "D3": (-0.075, 0.025),
            "E3": (-0.075, -0.025), "F3": (-0.075, -0.075), "G3": (-0.075, -0.125), "H3": (-0.075, -0.175),

            "A4": (-0.025, 0.175), "B4": (-0.025, 0.125), "C4": (-0.025, 0.075), "D4": (-0.025, 0.025),
            "E4": (-0.025, -0.025), "F4": (-0.025, -0.075), "G4": (-0.025, -0.125), "H4": (-0.025, -0.175),

            "A5": (0.025, 0.175), "B5": (0.025, 0.125), "C5": (0.025, 0.075), "D5": (0.025, 0.025),
            "E5": (0.025, -0.025), "F5": (0.025, -0.075), "G5": (0.025, -0.125), "H5": (0.025, -0.175),

            "A6": (0.075, 0.175), "B6": (0.075, 0.125), "C6": (0.075, 0.075), "D6": (0.075, 0.025),
            "E6": (0.075, -0.025), "F6": (0.075, -0.075), "G6": (0.075, -0.125), "H6": (0.075, -0.175),

            "A7": (0.125, 0.175), "B7": (0.125, 0.125), "C7": (0.125, 0.075), "D7": (0.125, 0.025),
            "E7": (0.125, -0.025), "F7": (0.125, -0.075), "G7": (0.125, -0.125), "H7": (0.125, -0.175),

            "A8": (0.175, 0.175), "B8": (0.175, 0.125), "C8": (0.175, 0.075), "D8": (0.175, 0.025),
            "E8": (0.175, -0.025), "F8": (0.175, -0.075), "G8": (0.175, -0.125), "H8": (0.175, -0.175)
        }
        return board_coords.get(square)

    # 5. Define the lower half of the board, ranks 1-4.
    lower_half_squares = [f"{file}{rank}" for file in "ABCDEFGH" for rank in "1234"]

    # 6. Helper functions.
    def get_piece_color(piece_id):
        'Return the piece color.'
        for piece in chess_pieces:
            if piece["id"] == piece_id:
                usd = piece.get("usd", "")
                if "benchmark_chess_000" in usd or "benchmark_chess_001" in usd or "benchmark_chess_002" in usd or "benchmark_chess_003" in usd or "benchmark_chess_004" in usd or "benchmark_chess_005" in usd:
                    return "white"
                elif "benchmark_chess_006" in usd or "benchmark_chess_007" in usd or "benchmark_chess_008" in usd or "benchmark_chess_009" in usd or "benchmark_chess_010" in usd or "benchmark_chess_011" in usd:
                    return "black"
        return None

    def get_piece_type(piece_id):
        'Return the piece type.'
        for piece in chess_pieces:
            if piece["id"] == piece_id:
                usd = piece.get("usd", "")
                if "benchmark_chess_000" in usd or "benchmark_chess_006" in usd:
                    return "pawn"
                elif "benchmark_chess_001" in usd or "benchmark_chess_007" in usd:
                    return "rook"
                elif "benchmark_chess_002" in usd or "benchmark_chess_008" in usd:
                    return "knight"
                elif "benchmark_chess_003" in usd or "benchmark_chess_009" in usd:
                    return "bishop"
                elif "benchmark_chess_004" in usd or "benchmark_chess_010" in usd:
                    return "queen"
                elif "benchmark_chess_005" in usd or "benchmark_chess_011" in usd:
                    return "king"
        return None

    def get_legal_non_capture_moves(piece_id, square):
        'Return legal non-capturing moves to empty squares.'
        piece_type = get_piece_type(piece_id)
        piece_color = get_piece_color(piece_id)

        if not piece_type or not piece_color:
            return []

        file = square[0]  # A-H
        rank = int(square[1])  # 1-8

        legal_moves = []

        if piece_type == "pawn":
            # Pawn: move one square forward (white upward, black downward), never backward.
            if piece_color == "white":
                # Move forward.
                forward_rank = rank + 1
                if forward_rank <= 8:
                    forward_square = f"{file}{forward_rank}"
                    if forward_square not in board_state:  # The square must be empty.
                        legal_moves.append(forward_square)
            else:  # black
                forward_rank = rank - 1
                if forward_rank >= 1:
                    forward_square = f"{file}{forward_rank}"
                    if forward_square not in board_state:  # The square must be empty.
                        legal_moves.append(forward_square)

        elif piece_type == "knight":
            # Knight: L-shaped moves to eight possible destinations.
            knight_moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]

            file_index = ord(file) - ord('A')

            for dx, dy in knight_moves:
                new_file_index = file_index + dx
                new_rank = rank + dy

                if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                    new_file = chr(ord('A') + new_file_index)
                    new_square = f"{new_file}{new_rank}"

                    # A knight may jump over pieces, but its destination must be empty for a non-capture.
                    if new_square not in board_state:
                        legal_moves.append(new_square)

        elif piece_type == "king":
            # King: move one square in any of eight directions.
            king_moves = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            file_index = ord(file) - ord('A')

            for dx, dy in king_moves:
                new_file_index = file_index + dx
                new_rank = rank + dy

                if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                    new_file = chr(ord('A') + new_file_index)
                    new_square = f"{new_file}{new_rank}"

                    # The king may move only to an empty square in this non-capture task.
                    if new_square not in board_state:
                        legal_moves.append(new_square)

        elif piece_type == "rook":
            # Rook: straight-line movement.
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            file_index = ord(file) - ord('A')

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"

                        if new_square in board_state:
                            break  # Blocked.
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # Outside the board boundaries.

        elif piece_type == "bishop":
            # Bishop: diagonal movement.
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            file_index = ord(file) - ord('A')

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"

                        if new_square in board_state:
                            break  # Blocked.
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # Outside the board boundaries.

        elif piece_type == "queen":
            # Queen: straight and diagonal movement.
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
            file_index = ord(file) - ord('A')

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"

                        if new_square in board_state:
                            break  # Blocked.
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # Outside the board boundaries.

        return legal_moves

    # 7. Find a movable piece in the board's lower half.
    available_pieces = []
    for piece in chess_pieces:
        piece_id = piece["id"]
        square = piece.get("board_square")

        if square and square in lower_half_squares:
            legal_moves = get_legal_non_capture_moves(piece_id, square)
            if legal_moves:
                piece_type = get_piece_type(piece_id)
                piece_color = get_piece_color(piece_id)

                available_pieces.append({
                    "id": piece_id,
                    "square": square,
                    "legal_moves": legal_moves,
                    "piece_type": piece_type,
                    "color": piece_color,
                    "original_rotation": piece["rotation"]  # Save the original rotation.
                })
                print(f"Lower half piece {piece_id} at {square} ({piece_color} {piece_type}) has {len(legal_moves)} legal non-capture moves")

    if len(available_pieces) == 0:
        print("No movable pieces found in lower half")
        return False

    print(f"Found {len(available_pieces)} movable pieces in lower half")

    # 8. Randomly select a piece as required.
    selected_piece = random.choice(available_pieces)

    piece_id = selected_piece["id"]
    current_square = selected_piece["square"]
    legal_moves = selected_piece["legal_moves"]
    piece_type = selected_piece["piece_type"]
    piece_color = selected_piece["color"]
    original_rotation = selected_piece["original_rotation"]  # Original rotation.

    print(f"\nRandomly selected piece: {piece_id} ({piece_color} {piece_type}) at {current_square}")
    print(f"Legal non-capture moves: {legal_moves}")

    # 9. Randomly select a destination square.
    if legal_moves:
        target_square = random.choice(legal_moves)
    else:
        print(f"No valid target square for piece {piece_id}")
        return False

    print(f"Randomly selected target square: {target_square}")

    # 10. Compute the destination in world coordinates.
    target_board_coords = square_to_board_coordinates(target_square)
    if not target_board_coords:
        print(f"Cannot convert square {target_square} to board coordinates")
        return False

    target_x, target_y = target_board_coords

    # Piece height: board surface + 0.05 m, following the scene generator.
    chessboard_height = chessboard_size[2]
    chessboard_surface_z = chessboard_pos[2] - chessboard_height / 2.0
    piece_height = chessboard_surface_z + 0.05 + 0.03

    target_world_pos = np.array([
        chessboard_pos[0] + target_x,
        chessboard_pos[1] + target_y,
        piece_height
    ])

    print(f"Target world position: {target_world_pos}")

    # 11. Generate pose candidates by adding random Z rotations to the original orientation.
    target_poses = []

    # First read the piece's original rotation matrix.
    original_rot_matrix = quat2mat_wxyz(original_rotation)  # Original rotation matrix.

    # Use small position offsets to improve placement success.
    x_offsets = np.linspace(0.00, 0.00, 3)  # Small X offset.
    y_offsets = np.linspace(0.00, 0.00, 3)  # Small Y offset.
    z_offsets = np.linspace(0, 0.002, 3)  # Small Z offset.

    # Generate random angles about the Z axis.
    num_rotations = 12
    z_rotation_angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)

    # Generate every combination of candidate pose parameters.
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = target_world_pos + np.array([x_offset, y_offset, z_offset])

                # Generate multiple random rotations about Z.
                for angle in z_rotation_angles:
                    # Build the rotation matrix about Z.
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rot_z = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]
                    ])

                    # Compose the original rotation with the random Z-axis rotation.
                    final_rot = np.dot(rot_z, original_rot_matrix)
                    # final_rot = original_rot_matrix

                    # Build a 4x4 pose matrix.
                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses with random z-axis rotations")

    # MANDATORY: Set task description dynamically before first action (after determining piece and target square)
    task_description = {
        "task_name": f"move the {piece_color} {piece_type} from {current_square} to {target_square}",
        "english_task_name": f"move the {piece_color} {piece_type} from {current_square} to {target_square}",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 12. Execute the move.
    piece_position, _ = robot.get_object_pose(piece_id)
    arm = robot.judge_arm_to_use(piece_position)
    print(f"Using arm: {arm}")

    # Protect the chessboard and other pieces.
    unmovable_objects = [chessboard_id]
    for piece in chess_pieces:
        if piece["id"] != piece_id:
            unmovable_objects.append(piece["id"])

    # Choose the description from the piece type and color.
    piece_type_name = {
        "pawn": "兵",
        "rook": "城堡",
        "knight": "骑士",
        "bishop": "主教",
        "queen": "皇后",
        "king": "国王"
    }.get(piece_type, "棋子")

    piece_color_name = {"white": "白", "black": "黑"}.get(piece_color, "")

    # Grasp configuration.
    grasp_config = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects.copy(),
        action_description={
            "zh": f"拿起{piece_color_name}色的{piece_type_name}",
            "en": f"pick up the {piece_color} {piece_type}"
        }
    )

    # Define a checker for successful placement at the destination.
    def create_move_check_function(piece_id, target_pos):
        def check_piece_moved():
            current_pos, current_rot = robot.get_object_pose(piece_id)
            if current_pos is None:
                return False

            # Position check with 0.013 m tolerance.
            distance = np.linalg.norm(current_pos - target_pos)
            tolerance = 0.013
            print(f"distance: {distance:.3f}m, tolerance: {tolerance:.3f}m")
            if distance > tolerance:
                print(f"Piece {piece_id} position error: {distance:.3f}m > {tolerance:.3f}m")
                return False

            # Check that the piece lies on the board.
            if current_pos[2] < chessboard_surface_z - 0.02 or current_pos[2] > chessboard_surface_z + 0.10:
                print(f"Piece {piece_id} z position {current_pos[2]:.3f}m out of range")
                return False

            return True

        return check_piece_moved

    check_function = create_move_check_function(piece_id, target_world_pos)

    # Placement configuration.
    move_desc_zh = f"将{piece_color_name}色的{piece_type_name}从{current_square}移动到{target_square}"
    move_desc_en = f"move the {piece_color} {piece_type} from {current_square} to {target_square}"

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects.copy(),
        extra_checkers=[check_function],
        action_description={"zh": move_desc_zh, "en": move_desc_en}
    )

    # Execute grasping and placement.
    print(f"Attempting to move piece {piece_id} from {current_square} to {target_square}")
    success = robot.pick_and_place_actor_to_target_pose(
        actor=piece_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if success:
        print(f"Successfully moved piece {piece_id} to {target_square}")
    else:
        print(f"Failed to move piece {piece_id}")

    # Return the arm to its initial position.
    robot.back_to_origin(arm=arm)

    print("\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Randomly moved {piece_color} {piece_type} from {current_square} to {target_square}")
    else:
        print("❌ FAILED: Could not move piece")

    return success
