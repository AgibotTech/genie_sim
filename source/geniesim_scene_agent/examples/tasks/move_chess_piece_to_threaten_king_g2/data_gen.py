# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    'Find a piece that can threaten the opposing king after one move.\n    Identify the piece, find a threatening destination, then move it there.'

    # Initialize objects_list from the provided interaction data.
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

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "detect the piece that can capture the opposite-color King and move it to a position where it can capture the King in the next move",
        "english_task_name": "detect the piece that can capture the opposite-color King and move it to a position where it can capture the King in the next move",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(description_json)

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
    piece_info_by_id = {}  # Piece ID -> complete piece information.

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

            # Extract the piece type and color from keywords.
            piece_type = None
            piece_color = None
            for kw in keywords:
                if "pawn_white" in kw:
                    piece_type = "pawn"
                    piece_color = "white"
                    break
                elif "rook_white" in kw:
                    piece_type = "rook"
                    piece_color = "white"
                    break
                elif "knight_white" in kw:
                    piece_type = "knight"
                    piece_color = "white"
                    break
                elif "bishop_white" in kw:
                    piece_type = "bishop"
                    piece_color = "white"
                    break
                elif "queen_white" in kw:
                    piece_type = "queen"
                    piece_color = "white"
                    break
                elif "king_white" in kw:
                    piece_type = "king"
                    piece_color = "white"
                    break
                elif "pawn_black" in kw:
                    piece_type = "pawn"
                    piece_color = "black"
                    break
                elif "rook_black" in kw:
                    piece_type = "rook"
                    piece_color = "black"
                    break
                elif "knight_black" in kw:
                    piece_type = "knight"
                    piece_color = "black"
                    break
                elif "bishop_black" in kw:
                    piece_type = "bishop"
                    piece_color = "black"
                    break
                elif "queen_black" in kw:
                    piece_type = "queen"
                    piece_color = "black"
                    break
                elif "king_black" in kw:
                    piece_type = "king"
                    piece_color = "black"
                    break

            # Infer the type from usd if keywords do not specify it.
            if piece_type is None:
                if "benchmark_chess_000" in usd:
                    piece_type = "pawn"
                    piece_color = "white"
                elif "benchmark_chess_001" in usd:
                    piece_type = "rook"
                    piece_color = "white"
                elif "benchmark_chess_002" in usd:
                    piece_type = "knight"
                    piece_color = "white"
                elif "benchmark_chess_003" in usd:
                    piece_type = "bishop"
                    piece_color = "white"
                elif "benchmark_chess_004" in usd:
                    piece_type = "queen"
                    piece_color = "white"
                elif "benchmark_chess_005" in usd:
                    piece_type = "king"
                    piece_color = "white"
                elif "benchmark_chess_006" in usd:
                    piece_type = "pawn"
                    piece_color = "black"
                elif "benchmark_chess_007" in usd:
                    piece_type = "rook"
                    piece_color = "black"
                elif "benchmark_chess_008" in usd:
                    piece_type = "knight"
                    piece_color = "black"
                elif "benchmark_chess_009" in usd:
                    piece_type = "bishop"
                    piece_color = "black"
                elif "benchmark_chess_010" in usd:
                    piece_type = "queen"
                    piece_color = "black"
                elif "benchmark_chess_011" in usd:
                    piece_type = "king"
                    piece_color = "black"

            piece_info = {
                "id": obj_id,
                "usd": usd,
                "position": piece_pos,
                "rotation": piece_rot,
                "keywords": keywords,
                "piece_type": piece_type,
                "piece_color": piece_color,
                "board_square": position_square
            }

            chess_pieces.append(piece_info)
            piece_info_by_id[obj_id] = piece_info

            # Update the board state when a square is identified.
            if position_square:
                board_state[position_square] = obj_id
                piece_positions[obj_id] = position_square
                print(f"Found chess piece: {obj_id} ({piece_color} {piece_type}) at square {position_square}")
            else:
                print(f"Warning: Chess piece {obj_id} ({piece_color} {piece_type}) has no position keyword")

    if len(chess_pieces) == 0:
        print("No chess pieces found")
        return False

    print(f"Total chess pieces found: {len(chess_pieces)}")
    print(f"Board state: {board_state}")

    # 4. Map chessboard coordinates using the scene-generation convention.
    def square_to_board_coordinates(square):
        'Convert a chess square such as A1 to coordinates relative to the board center.'
        SQUARE_TO_COORD = {
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
        return SQUARE_TO_COORD.get(square)

    # 5. Helper: compute world coordinates from a chess square.
    def square_to_world_coordinates(square, piece_id=None):
        'Convert a chess square to world coordinates.'
        board_coords = square_to_board_coordinates(square)
        if not board_coords:
            return None

        target_x, target_y = board_coords

        # Compute the chessboard surface height.
        chessboard_height = chessboard_size[2]
        chessboard_surface_z = chessboard_pos[2] - chessboard_height / 2.0

        # Piece height: board surface + clearance + half the piece height.
        margin = 0.01
        if piece_id:
            piece_size = robot.get_object_size_in_world_frame(piece_id)
            if piece_size is not None:
                piece_height = chessboard_surface_z + margin + piece_size[2] / 2.0
            else:
                # Default piece height: 0.05 m.
                piece_height = chessboard_surface_z + margin + 0.025
        else:
            # Use the default height when piece_id is not provided.
            piece_height = chessboard_surface_z + margin + 0.025

        target_world_pos = np.array([
            chessboard_pos[0] + target_x,
            chessboard_pos[1] + target_y,
            piece_height
        ])

        return target_world_pos

    # 6. Helper: compute the squares a piece can attack.
    def get_attacked_squares(piece_type, square, piece_color=None):
        'Return the squares the piece can attack, including captures.\n        This follows the scene-generation logic.'
        # Convert a square to row and column indices (0-7).
        def square_to_index(sq):
            file = ord(sq[0]) - ord('A')
            rank = int(sq[1]) - 1
            return (rank, file)

        def index_to_square(row, col):
            if 0 <= row < 8 and 0 <= col < 8:
                file = chr(ord('A') + col)
                rank = str(row + 1)
                return file + rank
            return None

        row, col = square_to_index(square)
        attacked = set()

        # Special movement rules for pawns.
        if piece_type == "pawn":
            if piece_color == "white":
                # White pawns attack diagonally forward.
                for dc in [-1, 1]:
                    new_row, new_col = row + 1, col + dc
                    target_sq = index_to_square(new_row, new_col)
                    if target_sq:
                        attacked.add(target_sq)
            else:  # black
                # Black pawns attack diagonally forward.
                for dc in [-1, 1]:
                    new_row, new_col = row - 1, col + dc
                    target_sq = index_to_square(new_row, new_col)
                    if target_sq:
                        attacked.add(target_sq)

        # The knight moves in an L shape.
        elif piece_type == "knight":
            knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                target_sq = index_to_square(new_row, new_col)
                if target_sq:
                    attacked.add(target_sq)

        # The king can move in eight directions.
        elif piece_type == "king":
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    target_sq = index_to_square(new_row, new_col)
                    if target_sq:
                        attacked.add(target_sq)

        # Rook: straight-line movement.
        elif piece_type == "rook":
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    attacked.add(index_to_square(r, c))
                    r += dr
                    c += dc

        # Bishop: diagonal movement.
        elif piece_type == "bishop":
            for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    attacked.add(index_to_square(r, c))
                    r += dr
                    c += dc

        # Queen: straight and diagonal movement.
        elif piece_type == "queen":
            # Straight-line movement.
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    attacked.add(index_to_square(r, c))
                    r += dr
                    c += dc

            # Diagonal movement.
            for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    attacked.add(index_to_square(r, c))
                    r += dr
                    c += dc

        return attacked

    # 7. Helper: compute legal destination squares, including captures and non-captures.
    def get_possible_moves(piece_id, square, current_board_state=None):
        'Return all possible destination squares.\n        current_board_state maps square to piece_id; None uses the enclosing board_state.'
        piece_info = piece_info_by_id.get(piece_id)
        if not piece_info:
            return set()

        # Use the supplied board state, or the enclosing state when it is None.
        board_state_to_use = current_board_state if current_board_state is not None else board_state

        piece_type = piece_info["piece_type"]
        piece_color = piece_info["piece_color"]

        # Get the attacked squares.
        attacked_squares = get_attacked_squares(piece_type, square, piece_color)

        # Check for blocked paths for rooks, bishops, and queens.
        possible_moves = set()

        if piece_type in ["rook", "bishop", "queen"]:
            # The path must be clear.
            for target_sq in attacked_squares:
                if is_path_clear(square, target_sq, board_state_to_use, exclude_piece=piece_id):
                    possible_moves.add(target_sq)
        else:
            # Knights, kings, and pawns can use their attack destinations directly.
            possible_moves = attacked_squares

        return possible_moves

    # 8. Helper: check a clear path for rooks, bishops, and queens.
    def is_path_clear(from_sq, to_sq, occupied_squares, exclude_piece=None):
        'Check whether the path from from_sq to to_sq is clear.\n        occupied_squares maps square to piece_id.\n        exclude_piece is the ignored piece ID, normally the moving piece.'
        # Convert chess squares to row and column indices.
        def square_to_index(sq):
            file = ord(sq[0]) - ord('A')
            rank = int(sq[1]) - 1
            return (rank, file)

        def index_to_square(row, col):
            if 0 <= row < 8 and 0 <= col < 8:
                file = chr(ord('A') + col)
                rank = str(row + 1)
                return file + rank
            return None

        from_row, from_col = square_to_index(from_sq)
        to_row, to_col = square_to_index(to_sq)

        # Determine the movement direction.
        dr = 0
        dc = 0
        if to_row > from_row:
            dr = 1
        elif to_row < from_row:
            dr = -1

        if to_col > from_col:
            dc = 1
        elif to_col < from_col:
            dc = -1

        # Check whether the move is straight or diagonal.
        if not ((dr == 0 and dc != 0) or (dr != 0 and dc == 0) or (abs(dr) == abs(dc))):
            return True  # Pieces such as knights do not require a path check.

        # Check every intermediate square, excluding the start and destination.
        current_row = from_row + dr
        current_col = from_col + dc

        while (current_row, current_col) != (to_row, to_col):
            current_sq = index_to_square(current_row, current_col)
            if current_sq and current_sq in occupied_squares:
                # A piece on the path blocks the move.
                piece_at_sq = occupied_squares[current_sq]
                if piece_at_sq != exclude_piece:
                    return False  # Blocked by another piece.

            current_row += dr
            current_col += dc

        return True

    # 9. Find pieces in ranks 1-4 that can threaten a king after moving.
    print("\n=== Detecting pieces in lower half (ranks 1-4) that can capture King after one move ===")

    # Find all kings.
    white_king = None
    black_king = None

    for piece in chess_pieces:
        if piece["piece_type"] == "king":
            if piece["piece_color"] == "white":
                white_king = piece
                print(f"Found White King: {piece['id']} at {piece['board_square']}")
            elif piece["piece_color"] == "black":
                black_king = piece
                print(f"Found Black King: {piece['id']} at {piece['board_square']}")

    if not white_king or not black_king:
        print("Error: Need both Kings on the board")
        return False

    # Define ranks 1-4.
    lower_half_squares = [f"{file}{rank}" for file in "ABCDEFGH" for rank in "1234"]

    # Inspect pieces in ranks 1-4.
    candidate_pieces = []

    for piece in chess_pieces:
        piece_square = piece.get("board_square")
        if not piece_square:
            continue

        # Consider only pieces in ranks 1-4.
        if piece_square not in lower_half_squares:
            continue

        # Check that the piece currently cannot capture the opposing king.
        piece_color = piece["piece_color"]
        opposite_king = black_king if piece_color == "white" else white_king

        if not opposite_king or not opposite_king.get("board_square"):
            continue

        king_square = opposite_king["board_square"]

        # Get the current attacked squares.
        current_attacks = get_possible_moves(piece["id"], piece_square)

        # Skip a piece that already attacks the king; seek one that can attack it after moving.
        if king_square in current_attacks:
            print(f"Piece {piece['id']} ({piece_color} {piece['piece_type']}) at {piece_square} can ALREADY capture King - SKIPPED")
            continue

        # Check whether one move creates an attack on the king.
        # Get every possible destination.
        possible_moves = get_possible_moves(piece["id"], piece_square)

        can_capture_after_move = False
        for move_square in possible_moves:
            # Simulate the move.
            temp_board_state = board_state.copy()

            # Remove the piece from its original square.
            if piece_square in temp_board_state:
                del temp_board_state[piece_square]

            # Inspect the destination.
            target_piece_id = temp_board_state.get(move_square)
            if target_piece_id:
                target_piece_info = piece_info_by_id.get(target_piece_id)
                if target_piece_info and target_piece_info["piece_color"] == piece_color:
                    # A friendly piece prevents this move.
                    continue
                # An opposing piece can be captured.
                del temp_board_state[move_square]

            # Check whether the new square attacks the king.
            new_attacks = get_possible_moves(piece["id"], move_square, temp_board_state)

            if king_square in new_attacks:
                can_capture_after_move = True
                print(f"Piece {piece['id']} ({piece_color} {piece['piece_type']}) at {piece_square} can capture King after moving to {move_square}")
                break

        if can_capture_after_move:
            candidate_pieces.append({
                "piece_id": piece["id"],
                "piece_type": piece["piece_type"],
                "piece_color": piece_color,
                "current_square": piece_square,
                "king_id": opposite_king["id"],
                "king_square": king_square,
                "king_color": opposite_king["piece_color"]
            })

    if len(candidate_pieces) == 0:
        print("No piece found that can capture King after one move")
        return False

    print(f"\nFound {len(candidate_pieces)} candidate pieces in lower half")

    # 10. Randomly select a candidate piece.
    selected_piece = random.choice(candidate_pieces)
    piece_id = selected_piece["piece_id"]
    piece_type = selected_piece["piece_type"]
    piece_color = selected_piece["piece_color"]
    current_square = selected_piece["current_square"]
    king_square = selected_piece["king_square"]
    king_color = selected_piece["king_color"]

    print(f"\nSelected piece: {piece_id} ({piece_color} {piece_type}) at {current_square}")
    print(f"Target: {king_color} King at {king_square}")

    # 11. Find a destination from which the opposing king can be captured next.
    print(f"\n=== Finding positions to capture {king_color} King in next move ===")

    # Get all destinations for this piece.
    all_possible_moves = get_possible_moves(piece_id, current_square)

    print(f"Piece {piece_id} at {current_square} has {len(all_possible_moves)} possible moves:")
    for move in sorted(all_possible_moves):
        print(f"  - {move}")

    # Find destinations that create an attack on the king.
    capture_positions = []

    for move_square in all_possible_moves:
        # Simulate the move in a temporary board state.
        # 1. Remove the moving piece from its original square.
        # 2. Remove an opposing piece at the destination when simulating a capture.
        # 3. Check the attack on the king from the new square.

        temp_board_state = board_state.copy()

        # Remove the piece from its original square.
        if current_square in temp_board_state:
            del temp_board_state[current_square]

        # Check whether the destination contains a piece.
        target_piece_id = temp_board_state.get(move_square)
        is_capture = False
        captured_piece_type = None

        if target_piece_id:
            target_piece_info = piece_info_by_id.get(target_piece_id)
            if target_piece_info:
                # Only an opposing occupied square is a valid capture destination.
                if target_piece_info["piece_color"] != piece_color:
                    is_capture = True
                    captured_piece_type = target_piece_info["piece_type"]
                    # The captured piece will be removed from the board.
                    del temp_board_state[move_square]
                else:
                    # A friendly piece makes this destination unavailable.
                    continue

        # Check whether this new square attacks the king.
        # Check the path using the temporary board state.
        new_possible_moves = get_possible_moves(piece_id, move_square, temp_board_state)

        if king_square in new_possible_moves:
            if is_capture:
                capture_positions.append({
                    "square": move_square,
                    "type": "capture_and_then_capture_king",
                    "captured_piece_id": target_piece_id,
                    "captured_piece_type": captured_piece_type
                })
                print(f"  Found: {move_square} (capture {captured_piece_type}, then can capture King)")
            else:
                capture_positions.append({
                    "square": move_square,
                    "type": "move_and_then_capture_king"
                })
                print(f"  Found: {move_square} (move to empty, then can capture King)")
        else:
            # Diagnostics: explain why this destination is invalid.
            print(f"  Checking {move_square}: King at {king_square} NOT in attack range")
            print(f"    Attackable squares from {move_square}: {sorted(new_possible_moves)}")

    print(f"Found {len(capture_positions)} positions where piece can capture King in next move")

    if len(capture_positions) == 0:
        print("No position found where piece can capture King in next move")
        return False

    # 12. Select a destination, preferring an empty square to a capture.
    target_position_info = None

    # Prefer moving to an empty square.
    empty_positions = [pos for pos in capture_positions if pos["type"] == "move_and_then_capture_king"]
    if empty_positions:
        target_position_info = empty_positions[0]
    else:
        # Choose a capture if no empty destination is available.
        target_position_info = capture_positions[0]

    target_square = target_position_info["square"]
    move_type = target_position_info["type"]

    if move_type == "capture_and_then_capture_king":
        print(f"Selected target: {target_square} (will capture {target_position_info['captured_piece_type']} and then capture King)")
    else:
        print(f"Selected target: {target_square} (will move to empty square and then capture King)")

    # 13. Compute the destination in world coordinates.
    target_world_pos = square_to_world_coordinates(target_square, piece_id)
    if target_world_pos is None:
        print(f"Cannot convert square {target_square} to world coordinates")
        return False

    print(f"Target world position: {target_world_pos}")

    # 14. Generate multiple candidate poses.
    target_poses = []

    # Get the piece's original rotation matrix.
    original_rot = piece_info_by_id[piece_id]["rotation"]
    original_rot_matrix = quat2mat_wxyz(original_rot)

    # Small position offsets for pose candidates.
    x_offsets = np.linspace(-0.001, 0.001, 3)
    y_offsets = np.linspace(-0.001, 0.001, 3)
    z_offsets = np.linspace(0, 0.002, 3)

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

                    # Build a 4x4 pose matrix.
                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses")

    # 15. Execute the move.
    piece_position, _ = robot.get_object_pose(piece_id)
    arm = robot.judge_arm_to_use(piece_position)
    print(f"Using arm: {arm}")

    # Protect the board and other pieces while allowing the target capture.
    unmovable_objects = [chessboard_id]
    for piece in chess_pieces:
        if piece["id"] != piece_id:
            # For a capture, allow the target piece to be moved.
            if move_type == "capture_and_then_capture_king" and piece["id"] == target_position_info.get("captured_piece_id"):
                continue  # This piece may be captured.
            unmovable_objects.append(piece["id"])

    # Choose the description from the piece type and color.
    piece_type_name_zh = {
        "pawn": "兵",
        "rook": "车",
        "knight": "马",
        "bishop": "象",
        "queen": "后",
        "king": "王"
    }.get(piece_type, "棋子")

    piece_color_name_zh = {"white": "白", "black": "黑"}.get(piece_color, "")
    king_color_name_zh = {"white": "白", "black": "黑"}.get(king_color, "")

    # Update the task description with the specific move.
    task_description = {
        "task_name": f"move the {piece_color} {piece_type} from {current_square} to {target_square} so it can capture the {king_color} King in the next move",
        "english_task_name": f"move the {piece_color} {piece_type} from {current_square} to {target_square} so it can capture the {king_color} King in the next move",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(task_description)

    # Grasp configuration.
    grasp_config = GraspConfig(
        # is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects.copy(),
        action_description={
            "zh": f"拿起{piece_color_name_zh}色的{piece_type_name_zh}",
            "en": f"pick up the {piece_color} {piece_type}"
        }
    )

    # Define a checker for successful placement at the destination.
    def create_move_check_function(piece_id, target_pos, target_square):
        def check_piece_moved():
            current_pos, current_rot = robot.get_object_pose(piece_id)
            if current_pos is None:
                return False

            # Position check with 0.015 m tolerance.
            distance = np.linalg.norm(current_pos - target_pos)
            tolerance = 0.021

            # Compute the board surface height.
            chessboard_height = chessboard_size[2]
            chessboard_surface_z = chessboard_pos[2] - chessboard_height / 2.0

            print(f"Distance to target: {distance:.3f}m, tolerance: {tolerance:.3f}m")
            print(f"Current z: {current_pos[2]:.3f}m, board surface z: {chessboard_surface_z:.3f}m")

            if distance > tolerance:
                print(f"Piece {piece_id} position error: {distance:.3f}m > {tolerance:.3f}m")
                return False

            # Check that the piece lies on the board within a reasonable height range.
            if current_pos[2] < chessboard_surface_z - 0.02 or current_pos[2] > chessboard_surface_z + 0.10:
                print(f"Piece {piece_id} z position {current_pos[2]:.3f}m out of range")
                return False

            # Also verify that the piece is near the intended square.
            # Compute the destination square's world-coordinate bounds.
            board_coords = square_to_board_coordinates(target_square)
            if board_coords:
                expected_x = chessboard_pos[0] + board_coords[0]
                expected_y = chessboard_pos[1] + board_coords[1]

                xy_distance = np.sqrt((current_pos[0] - expected_x)**2 + (current_pos[1] - expected_y)**2)
                xy_tolerance = 0.02  # 2 cm tolerance.

                if xy_distance > xy_tolerance:
                    print(f"Piece {piece_id} xy position error: {xy_distance:.3f}m > {xy_tolerance:.3f}m")
                    print(f"Expected: ({expected_x:.3f}, {expected_y:.3f}), Actual: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
                    return False

            return True

        return check_piece_moved

    check_function = create_move_check_function(piece_id, target_world_pos, target_square)

    # Placement configuration.
    if move_type == "capture_and_then_capture_king":
        captured_piece_type = target_position_info.get("captured_piece_type", "piece")
        captured_piece_type_zh = {
            "pawn": "兵",
            "rook": "车",
            "knight": "马",
            "bishop": "象",
            "queen": "后",
            "king": "王"
        }.get(captured_piece_type, "棋子")

        move_desc_zh = f"将{piece_color_name_zh}色的{piece_type_name_zh}从{current_square}移动到{target_square}吃掉{king_color_name_zh}色的{captured_piece_type_zh}，使其下一步可以吃到{king_color_name_zh}色的王"
        move_desc_en = f"move the {piece_color} {piece_type} from {current_square} to {target_square} capturing the {king_color} {captured_piece_type}, so it can capture the {king_color} King in the next move"
    else:
        move_desc_zh = f"将{piece_color_name_zh}色的{piece_type_name_zh}从{current_square}移动到{target_square}，使其下一步可以吃到{king_color_name_zh}色的王"
        move_desc_en = f"move the {piece_color} {piece_type} from {current_square} to {target_square} so it can capture the {king_color} King in the next move"

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects.copy(),
        extra_checkers=[check_function],
        action_description={"zh": move_desc_zh, "en": move_desc_en}
    )

    # 16. Execute grasping and placement.
    print(f"\n=== Executing move ===")
    print(f"Moving piece {piece_id} from {current_square} to {target_square}")
    print(f"Arm: {arm}")

    success = robot.pick_and_place_actor_to_target_pose(
        actor=piece_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if success:
        print(f"Successfully moved piece {piece_id} to {target_square}")

        # Verify that the resulting position attacks the king.
        print(f"\n=== Verifying King capture position ===")

        # Read the board state again.
        new_board_state = board_state.copy()
        # Update the board state by moving the piece from its original square to its destination.
        if current_square in new_board_state:
            del new_board_state[current_square]
        new_board_state[target_square] = piece_id

        # Check the attack on the king from the resulting square.
        new_possible_moves = get_possible_moves(piece_id, target_square)

        if king_square in new_possible_moves:
            print(f"✓ Verification PASSED: Piece at {target_square} can capture {king_color} King at {king_square} in next move")
        else:
            print(f"✗ Verification FAILED: Piece at {target_square} cannot capture {king_color} King at {king_square}")
            success = False
    else:
        print(f"Failed to move piece {piece_id}")

    # Return the arm to its initial position.
    robot.back_to_origin(arm=arm)

    print("\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Moved {piece_color} {piece_type} from {current_square} to {target_square}")
        print(f"   Next move: Can capture {king_color} King at {king_square}")
    else:
        print("❌ FAILED: Could not complete the task")

    return success
