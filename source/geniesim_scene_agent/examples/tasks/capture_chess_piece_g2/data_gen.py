# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    """
    Task: find a lower-half piece able to capture an opponent, white capturing black or vice versa.
    1. Remove captured piece from board and place it in designated table left/right region according to its board position.
    2. Move capturing piece to captured piece's original square.
    """

    # Initialize objects_list from interaction data.
    objects_list = {
        "benchmark_chessboard_000": {"has_interactions": True, "object_type": "chessboard"},
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

    # MANDATORY: set task description; update dynamically later for selected chess pieces.
    description_json = {
        "task_name": "Find pieces in lower half that can capture opponent pieces, move captured piece to specific area on left or right side of table based on position, move capturing piece to opponent's original position",
        "english_task_name": "Find pieces in lower half that can capture opponent pieces, move captured piece to specific area on left or right side of table based on position, move capturing piece to opponent's original position",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find chessboard benchmark_chessboard_000.
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

    # 2. Get chessboard position and size.
    chessboard_pos, chessboard_rot = robot.get_object_pose(chessboard_id)
    chessboard_size = robot.get_object_size_in_world_frame(chessboard_id)

    if chessboard_pos is None or chessboard_size is None:
        print("Cannot get chessboard position/size")
        return False

    print(f"Chessboard position: {chessboard_pos}, size: {chessboard_size}")

    # 3. Find table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if any("table" in kw.lower() for kw in keywords) or obj_info.get("usd", "") == "table_000":
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)

    if table_pos is None or table_size is None:
        print("Cannot get table position/size")
        return False

    print(f"Found table: {table_id}, position: {table_pos}, size: {table_size}")

    # 4. Find all chess pieces and read positions.
    chess_pieces = []
    board_state = {}  # Board square -> piece information.
    piece_positions = {}  # Piece ID -> board square.

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether object is a chess piece.
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

            # Extract board position from keywords.
            position_square = None
            for kw in keywords:
                if "position_" in kw:
                    position_square = kw.replace("position_", "")
                    break

            chess_pieces.append({
                "id": obj_id,
                "usd": usd,
                "position": piece_pos,
                "rotation": piece_rot,
                "keywords": keywords,
                "piece_type": piece_name,
                "board_square": position_square
            })

            if position_square:
                board_state[position_square] = {
                    "id": obj_id,
                    "usd": usd,
                    "position": piece_pos,
                    "rotation": piece_rot
                }
                piece_positions[obj_id] = position_square
                print(f"Found chess piece: {obj_id} at square {position_square}")
            else:
                print(f"Warning: Chess piece {obj_id} has no position keyword")

    if len(chess_pieces) == 0:
        print("No chess pieces found")
        return False

    print(f"Total chess pieces found: {len(chess_pieces)}")

    # 5. Chessboard coordinate mapping.
    def square_to_board_coordinates(square):
        """Convert a board square into coordinates relative to board center."""
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

    # 6. Helper functions.
    def get_piece_color(piece_id):
        """Get piece color."""
        for piece in chess_pieces:
            if piece["id"] == piece_id:
                usd = piece.get("usd", "")
                if "benchmark_chess_000" in usd or "benchmark_chess_001" in usd or "benchmark_chess_002" in usd or "benchmark_chess_003" in usd or "benchmark_chess_004" in usd or "benchmark_chess_005" in usd:
                    return "white"
                elif "benchmark_chess_006" in usd or "benchmark_chess_007" in usd or "benchmark_chess_008" in usd or "benchmark_chess_009" in usd or "benchmark_chess_010" in usd or "benchmark_chess_011" in usd:
                    return "black"
        return None

    def get_piece_type(piece_id):
        """Get piece type."""
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

    def get_piece_at_square(square):
        """Get piece on specified square."""
        return board_state.get(square)

    def get_legal_capture_moves(piece_id, piece_square, piece_color):
        """Get legal captures for a piece, targeting opposing pieces."""
        piece_type = get_piece_type(piece_id)

        if not piece_type:
            return []

        file = piece_square[0]  # A-H
        rank = int(piece_square[1])  # 1-8
        file_index = ord(file) - ord('A')

        legal_captures = []
        target_color = "black" if piece_color == "white" else "white"

        if piece_type == "pawn":
            # Pawn capture: forward diagonals left/right.
            if piece_color == "white":
                # White pawn: forward increases rank number.
                forward_rank = rank + 1
                if forward_rank <= 8:
                    # Left diagonal.
                    left_file_index = file_index - 1
                    if left_file_index >= 0:
                        left_file = chr(ord('A') + left_file_index)
                        left_square = f"{left_file}{forward_rank}"
                        target_piece = get_piece_at_square(left_square)
                        if target_piece and get_piece_color(target_piece["id"]) == target_color:
                            legal_captures.append((left_square, target_piece["id"]))

                    # Right diagonal.
                    right_file_index = file_index + 1
                    if right_file_index < 8:
                        right_file = chr(ord('A') + right_file_index)
                        right_square = f"{right_file}{forward_rank}"
                        target_piece = get_piece_at_square(right_square)
                        if target_piece and get_piece_color(target_piece["id"]) == target_color:
                            legal_captures.append((right_square, target_piece["id"]))
            else:
                # Black pawn: forward decreases rank number.
                forward_rank = rank - 1
                if forward_rank >= 1:
                    # Left diagonal.
                    left_file_index = file_index - 1
                    if left_file_index >= 0:
                        left_file = chr(ord('A') + left_file_index)
                        left_square = f"{left_file}{forward_rank}"
                        target_piece = get_piece_at_square(left_square)
                        if target_piece and get_piece_color(target_piece["id"]) == target_color:
                            legal_captures.append((left_square, target_piece["id"]))

                    # Right diagonal.
                    right_file_index = file_index + 1
                    if right_file_index < 8:
                        right_file = chr(ord('A') + right_file_index)
                        right_square = f"{right_file}{forward_rank}"
                        target_piece = get_piece_at_square(right_square)
                        if target_piece and get_piece_color(target_piece["id"]) == target_color:
                            legal_captures.append((right_square, target_piece["id"]))

        elif piece_type == "knight":
            # Knight: L-shaped capture movement.
            knight_moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]

            for dx, dy in knight_moves:
                new_file_index = file_index + dx
                new_rank = rank + dy

                if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                    new_file = chr(ord('A') + new_file_index)
                    new_square = f"{new_file}{new_rank}"
                    target_piece = get_piece_at_square(new_square)

                    if target_piece and get_piece_color(target_piece["id"]) == target_color:
                        legal_captures.append((new_square, target_piece["id"]))

        elif piece_type == "king":
            # King: capture one square in any direction.
            king_moves = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            for dx, dy in king_moves:
                new_file_index = file_index + dx
                new_rank = rank + dy

                if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                    new_file = chr(ord('A') + new_file_index)
                    new_square = f"{new_file}{new_rank}"
                    target_piece = get_piece_at_square(new_square)

                    if target_piece and get_piece_color(target_piece["id"]) == target_color:
                        legal_captures.append((new_square, target_piece["id"]))

        elif piece_type == "rook":
            # Rook: capture along straight lines.
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"
                        target_piece = get_piece_at_square(new_square)

                        if target_piece:
                            if get_piece_color(target_piece["id"]) == target_color:
                                legal_captures.append((new_square, target_piece["id"]))
                            break  # Blocked by any piece.
                    else:
                        break  # Outside board bounds.

        elif piece_type == "bishop":
            # Bishop: capture along diagonals.
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"
                        target_piece = get_piece_at_square(new_square)

                        if target_piece:
                            if get_piece_color(target_piece["id"]) == target_color:
                                legal_captures.append((new_square, target_piece["id"]))
                            break  # Blocked by any piece.
                    else:
                        break  # Outside board bounds.

        elif piece_type == "queen":
            # Queen: capture along straight lines and diagonals.
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]

            for dx, dy in directions:
                for step in range(1, 8):
                    new_file_index = file_index + dx * step
                    new_rank = rank + dy * step

                    if 0 <= new_file_index < 8 and 1 <= new_rank <= 8:
                        new_file = chr(ord('A') + new_file_index)
                        new_square = f"{new_file}{new_rank}"
                        target_piece = get_piece_at_square(new_square)

                        if target_piece:
                            if get_piece_color(target_piece["id"]) == target_color:
                                legal_captures.append((new_square, target_piece["id"]))
                            break  # Blocked by any piece.
                    else:
                        break  # Outside board bounds.

        return legal_captures

    # 7. Define lower half, ranks 1-4.
    lower_half_squares = [f"{file}{rank}" for file in "ABCDEFGH" for rank in "1234"]

    # 8. Find lower-half pieces that can capture opposing pieces, white capturing black or vice versa.
    pieces_with_captures = []

    for piece in chess_pieces:
        piece_id = piece["id"]
        square = piece.get("board_square")
        piece_color = get_piece_color(piece_id)

        # Consider only lower-half pieces.
        if square and square in lower_half_squares:
            legal_captures = get_legal_capture_moves(piece_id, square, piece_color)

            if legal_captures:
                piece_type = get_piece_type(piece_id)

                pieces_with_captures.append({
                    "id": piece_id,
                    "square": square,
                    "color": piece_color,
                    "legal_captures": legal_captures,
                    "piece_type": piece_type,
                    "original_rotation": piece["rotation"]
                })
                print(f"{piece_color.capitalize()} piece {piece_id} at {square} ({piece_type}) can capture {len(legal_captures)} opponent pieces")

    if len(pieces_with_captures) == 0:
        print("No pieces in lower half can capture opponent pieces")
        return False

    print(f"Found {len(pieces_with_captures)} pieces with capture opportunities")

    # 9. Randomly select capturing piece and its target.
    selected_piece = random.choice(pieces_with_captures)
    capturing_piece_id = selected_piece["id"]
    capturing_square = selected_piece["square"]
    capturing_color = selected_piece["color"]
    legal_captures = selected_piece["legal_captures"]
    capturing_piece_type = selected_piece["piece_type"]
    capturing_original_rotation = selected_piece["original_rotation"]

    print(f"\nRandomly selected capturing piece: {capturing_piece_id} ({capturing_color} {capturing_piece_type}) at {capturing_square}")
    print(f"Legal captures: {legal_captures}")

    # Randomly select one target piece.
    if legal_captures:
        target_square, target_piece_id = random.choice(legal_captures)
    else:
        print(f"No valid captures for piece {capturing_piece_id}")
        return False

    # Get target-piece information.
    target_piece_info = None
    for piece in chess_pieces:
        if piece["id"] == target_piece_id:
            target_piece_info = piece
            break

    if not target_piece_info:
        print(f"Cannot find target piece {target_piece_id}")
        return False

    target_square_name = target_piece_info.get("board_square")
    target_piece_type = get_piece_type(target_piece_id)
    target_piece_color = get_piece_color(target_piece_id)
    target_original_rotation = target_piece_info["rotation"]

    # Get target-piece board y to determine left/right side.
    target_board_coords = square_to_board_coordinates(target_square_name)
    if target_board_coords:
        _, target_board_y = target_board_coords
        print(f"Target piece {target_piece_id} at {target_square_name} has board y-coordinate: {target_board_y}")
    else:
        print(f"Warning: Cannot get board coordinates for {target_square_name}")
        target_board_y = 0  # Default value.

    print(f"Selected target piece to capture: {target_piece_id} ({target_piece_color} {target_piece_type}) at {target_square_name}")

    # Update task description for actual selection.
    task_desc_zh = f"将{target_piece_color}的{target_piece_type}从{target_square_name}移到桌子特定区域，{capturing_color}的{capturing_piece_type}从{capturing_square}移动到{target_square_name}"
    task_desc_en = f"Move {target_piece_color} {target_piece_type} from {target_square_name} to specific area on table, {capturing_color} {capturing_piece_type} from {capturing_square} to {target_square_name}"

    description_json = {
        "task_name": task_desc_en,
        "english_task_name": task_desc_en,
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 10. Compute table placement by target-piece board y, choosing left/right side.

    # Table-surface height.
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # Get target-piece dimensions.
    target_piece_size = robot.get_object_size_in_world_frame(target_piece_id)
    if target_piece_size is None:
        target_piece_size = np.array([0.03, 0.03, 0.05])  # Default dimensions.

    # Target position = table surface + safety margin + piece half-height.
    margin = 0.01
    table_target_z = table_top_z + margin + target_piece_size[2] / 2.0

    # Choose table left/right side according to target's board position.
    # Board left (y>0) -> table left (x<0, 0.35<y<0.4).
    # Board right (y<0) -> table right (x<0, -0.4<y<-0.35).

    if target_board_y > 0:
        # Board left -> table left.
        table_side = "left"
        # Table left region, x<0.
        table_left_x_min = table_pos[0] - table_size[0] / 2.0 + 0.1  # Leave edge margin.
        table_left_x_max = table_pos[0] - 0.1  # Region x<0.

        # Left-table Y range: 0.35-0.4.
        table_left_y_min = 0.35
        table_left_y_max = 0.40

        # Randomly generate position.
        table_target_x = random.uniform(table_left_x_min, table_left_x_max)
        table_target_y = random.uniform(table_left_y_min, table_left_y_max)

        print(f"Target piece is on left side of chessboard (y={target_board_y:.3f}), placing on left side of table")
        print(f"Table left side coordinates: x={table_target_x:.3f}, y={table_target_y:.3f} (range: 0.35-0.4)")

    else:
        # Board right -> table right.
        table_side = "right"
        # Table right region: x<0, same X range as left, different Y.
        table_left_x_min = table_pos[0] - table_size[0] / 2.0 + 0.1  # Leave edge margin.
        table_left_x_max = table_pos[0] - 0.1  # Region x<0.

        # Right-table Y range: -0.4 to -0.35.
        table_right_y_min = -0.40
        table_right_y_max = -0.35

        # Randomly generate position.
        table_target_x = random.uniform(table_left_x_min, table_left_x_max)
        table_target_y = random.uniform(table_right_y_min, table_right_y_max)

        print(f"Target piece is on right side of chessboard (y={target_board_y:.3f}), placing on right side of table")
        print(f"Table right side coordinates: x={table_target_x:.3f}, y={table_target_y:.3f} (range: -0.4 to -0.35)")

    table_target_pos = np.array([table_target_x, table_target_y, table_target_z])

    print(f"Table {table_side} side target position for target piece: {table_target_pos}")

    # 11. Compute target's original position as capturing piece destination.
    target_board_coords = square_to_board_coordinates(target_square_name)
    if not target_board_coords:
        print(f"Cannot convert square {target_square_name} to board coordinates")
        return False

    target_x_chess, target_y_chess = target_board_coords

    # Chessboard-surface height.
    chessboard_height = chessboard_size[2]
    chessboard_surface_z = chessboard_pos[2] + chessboard_height / 2.0  # Top surface.

    # Get capturing-piece dimensions.
    capturing_piece_size = robot.get_object_size_in_world_frame(capturing_piece_id)
    if capturing_piece_size is None:
        capturing_piece_size = np.array([0.03, 0.03, 0.05])  # Default dimensions.

    # Capturing-piece destination = board surface + safety margin + piece half-height.
    capturing_target_z = chessboard_surface_z + margin + capturing_piece_size[2] / 2.0

    capturing_target_pos = np.array([
        chessboard_pos[0] + target_x_chess,
        chessboard_pos[1] + target_y_chess,
        capturing_target_z
    ])

    print(f"Capturing piece target position on chessboard: {capturing_target_pos}")

    # 12. First move target piece to designated table region.
    print(f"\n=== Step 1: Moving target piece {target_piece_id} to table {table_side} side ===")

    target_piece_position, _ = robot.get_object_pose(target_piece_id)
    arm_target = robot.judge_arm_to_use(target_piece_position)
    print(f"Using arm for target piece: {arm_target}")

    # Generate multiple table-placement candidate poses.
    table_target_poses = []

    # Position variation parameters.
    x_offsets = np.linspace(-0.02, 0.02, 5)
    y_offsets = np.linspace(-0.02, 0.02, 5)
    z_offsets = np.linspace(0, 0.005, 3)

    # Generate random rotation angles about z.
    num_rotations = 8
    z_rotation_angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)

    # Original rotation matrix.
    original_rot_matrix = quat2mat_wxyz(target_original_rotation)

    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = table_target_pos + np.array([x_offset, y_offset, z_offset])

                for angle in z_rotation_angles:
                    # Rotation about z.
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rot_z = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]
                    ])

                    # Compose rotations.
                    final_rot = np.dot(rot_z, original_rot_matrix)

                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    table_target_poses.append(pose)

    table_target_poses = np.array(table_target_poses)
    print(f"Generated {len(table_target_poses)} candidate poses for table placement")

    # Immovable objects: protect board, capturing piece, and other pieces.
    unmovable_objects_step1 = [chessboard_id, capturing_piece_id]
    for piece in chess_pieces:
        if piece["id"] not in [target_piece_id, capturing_piece_id]:
            unmovable_objects_step1.append(piece["id"])

    # Grasp configuration.
    grasp_config_target = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects_step1.copy(),
        action_description={
            "zh": f"拿起{target_piece_color}的{target_piece_type}",
            "en": f"pick up the {target_piece_color} {target_piece_type}"
        }
    )

    # Define check: target piece in correct table region.
    def check_target_on_table():
        current_pos, _ = robot.get_object_pose(target_piece_id)
        if current_pos is None:
            return False

        # Check table left side (x<0).
        if current_pos[0] >= 0:
            print(f"Target piece {target_piece_id} not on left side of table (x={current_pos[0]:.3f})")
            return False

        # Check y range according to board position.
        if target_board_y > 0:
            # Expected table left, y between 0.35 and 0.4.
            if not (0.35 <= current_pos[1] <= 0.40):
                print(f"Target piece {target_piece_id} not in left side y-range (y={current_pos[1]:.3f}, expected: 0.35-0.4)")
                return False
        else:
            # Expected table right, y between -0.4 and -0.35.
            if not (-0.40 <= current_pos[1] <= -0.35):
                print(f"Target piece {target_piece_id} not in right side y-range (y={current_pos[1]:.3f}, expected: -0.4 to -0.35)")
                return False

        # Check near table surface.
        if current_pos[2] < table_top_z - 0.02 or current_pos[2] > table_top_z + 0.10:
            print(f"Target piece {target_piece_id} not on table surface (z={current_pos[2]:.3f})")
            return False

        return True

    # Placement configuration.
    table_side_desc = "左侧" if table_side == "left" else "右侧"
    place_config_table = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects_step1.copy(),
        extra_checkers=[check_target_on_table],
        action_description={
            "zh": f"将{target_piece_color}的{target_piece_type}放置到桌子{table_side_desc}特定区域",
            "en": f"place the {target_piece_color} {target_piece_type} to {table_side} side specific area of table"
        }
    )

    # Execute pick and place.
    print(f"Moving target piece {target_piece_id} from {target_square_name} to table {table_side} side")
    success_step1 = robot.pick_and_place_actor_to_target_pose(
        actor=target_piece_id,
        target_pose=table_target_poses,
        arm=arm_target,
        grasp_config=grasp_config_target,
        place_config=place_config_table
    )

    if not success_step1:
        print(f"Failed to move target piece {target_piece_id} to table")
        return False

    print(f"Successfully moved target piece {target_piece_id} to table {table_side} side")

    # Wait briefly for physics settling.
    robot.wait(1.0)

    # 13. Move capturing piece to target's original position.
    print(f"\n=== Step 2: Moving capturing piece {capturing_piece_id} to {target_square_name} ===")

    capturing_piece_position, _ = robot.get_object_pose(capturing_piece_id)
    arm_capturing = robot.judge_arm_to_use(capturing_piece_position)
    print(f"Using arm for capturing piece: {arm_capturing}")

    # Generate multiple board-placement candidate poses.
    chess_target_poses = []

    # Position variation parameters: smaller offsets for precise board placement.
    x_offsets_chess = np.linspace(-0.005, 0.005, 5)
    y_offsets_chess = np.linspace(-0.005, 0.005, 5)
    z_offsets_chess = np.linspace(0, 0.002, 3)

    # Original rotation matrix.
    capturing_original_rot_matrix = quat2mat_wxyz(capturing_original_rotation)

    for x_offset in x_offsets_chess:
        for y_offset in y_offsets_chess:
            for z_offset in z_offsets_chess:
                position = capturing_target_pos + np.array([x_offset, y_offset, z_offset])

                for angle in z_rotation_angles:
                    # Rotation about z.
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rot_z = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]
                    ])

                    # Compose rotations.
                    final_rot = np.dot(rot_z, capturing_original_rot_matrix)

                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    chess_target_poses.append(pose)

    chess_target_poses = np.array(chess_target_poses)
    print(f"Generated {len(chess_target_poses)} candidate poses for chessboard placement")

    # Immovable objects: board and all except capturing piece; captured target is already on table.
    unmovable_objects_step2 = [chessboard_id]
    for piece in chess_pieces:
        if piece["id"] != capturing_piece_id:
            unmovable_objects_step2.append(piece["id"])

    # Grasp configuration.
    grasp_config_capturing = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects_step2.copy(),
        action_description={
            "zh": f"拿起{capturing_color}的{capturing_piece_type}",
            "en": f"pick up the {capturing_color} {capturing_piece_type}"
        }
    )

    # Define check: capturing piece at target position.
    def check_capturing_on_target():
        current_pos, _ = robot.get_object_pose(capturing_piece_id)
        if current_pos is None:
            return False

        # Position check, 0.015 m tolerance.
        distance = np.linalg.norm(current_pos[:2] - capturing_target_pos[:2])
        tolerance = 0.015
        print(f"Capturing piece distance to target: {distance:.3f}m, tolerance: {tolerance:.3f}m")

        if distance > tolerance:
            print(f"Capturing piece {capturing_piece_id} position error: {distance:.3f}m > {tolerance:.3f}m")
            return False

        # Check on chessboard.
        if current_pos[2] < chessboard_surface_z - 0.02 or current_pos[2] > chessboard_surface_z + 0.10:
            print(f"Capturing piece {capturing_piece_id} not on chessboard (z={current_pos[2]:.3f})")
            return False

        return True

    # Placement configuration.
    place_config_chess = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects_step2.copy(),
        extra_checkers=[check_capturing_on_target],
        action_description={
            "zh": f"将{capturing_color}的{capturing_piece_type}移动到{target_square_name}位置",
            "en": f"move the {capturing_color} {capturing_piece_type} to {target_square_name}"
        }
    )

    # Execute pick and place.
    print(f"Moving capturing piece {capturing_piece_id} from {capturing_square} to {target_square_name}")
    success_step2 = robot.pick_and_place_actor_to_target_pose(
        actor=capturing_piece_id,
        target_pose=chess_target_poses,
        arm=arm_capturing,
        grasp_config=grasp_config_capturing,
        place_config=place_config_chess
    )

    if not success_step2:
        print(f"Failed to move capturing piece {capturing_piece_id} to {target_square_name}")
        success = False

    # 14. Final verification.
    print("\n=== Final verification ===")

    # Check target piece in correct table region.
    target_final_pos, _ = robot.get_object_pose(target_piece_id)
    if target_final_pos is not None:
        # Check table left side (x<0).
        if target_final_pos[0] >= 0:
            print(f"❌ Target piece {target_piece_id} not on left side of table (x={target_final_pos[0]:.3f})")
            success = False
        else:
            # Check Y according to board position.
            if target_board_y > 0:
                if 0.35 <= target_final_pos[1] <= 0.40:
                    print(f"✅ Target piece {target_piece_id} correctly placed on table left side (x={target_final_pos[0]:.3f}, y={target_final_pos[1]:.3f})")
                else:
                    print(f"❌ Target piece {target_piece_id} not in left side y-range (y={target_final_pos[1]:.3f}, expected: 0.35-0.4)")
                    success = False
            else:
                if -0.40 <= target_final_pos[1] <= -0.35:
                    print(f"✅ Target piece {target_piece_id} correctly placed on table right side (x={target_final_pos[0]:.3f}, y={target_final_pos[1]:.3f})")
                else:
                    print(f"❌ Target piece {target_piece_id} not in right side y-range (y={target_final_pos[1]:.3f}, expected: -0.4 to -0.35)")
                    success = False

    # Check capturing piece at target position.
    capturing_final_pos, _ = robot.get_object_pose(capturing_piece_id)
    if capturing_final_pos is not None:
        # Compute distance.
        distance = np.linalg.norm(capturing_final_pos[:2] - capturing_target_pos[:2])
        if distance < 0.02:
            print(f"✅ Capturing piece {capturing_piece_id} is at target position (distance={distance:.3f}m)")
        else:
            print(f"❌ Capturing piece {capturing_piece_id} is not at target position (distance={distance:.3f}m)")
            success = False

    # Return arm to initial position.
    if arm_target != arm_capturing:
        robot.back_to_origin(arm=arm_target)
    robot.back_to_origin(arm=arm_capturing)

    print("\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Moved {target_piece_color} {target_piece_type} from {target_square_name} (board y={target_board_y:.3f}) to table {table_side} side")
        print(f"✅ SUCCESS: Moved {capturing_color} {capturing_piece_type} from {capturing_square} to {target_square_name}")
        print(f"✅ Capture completed: {capturing_color} {capturing_piece_type} captures {target_piece_color} {target_piece_type}")
    else:
        print("⚠️ PARTIAL SUCCESS: Some steps may have failed")

    return success
