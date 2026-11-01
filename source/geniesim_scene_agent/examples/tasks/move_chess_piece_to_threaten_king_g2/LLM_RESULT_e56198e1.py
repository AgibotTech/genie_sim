# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
"""
scene_name: chessboard_on_table
description: A chessboard placed at the center of a table with 1-5 white and 1-5 black chess pieces (total <=7), arranged in plausible endgame positions with both kings present, lower half constraints, proper height adjustment, and valid piece counts.
"""
# Chess piece mapping from user input
CHESS_PIECE_MAPPING = {
    "pawn_white": "benchmark_chess_000",
    "rook_white": "benchmark_chess_001",
    "knight_white": "benchmark_chess_002",
    "bishop_white": "benchmark_chess_003",
    "queen_white": "benchmark_chess_004",
    "king_white": "benchmark_chess_005",
    "pawn_black": "benchmark_chess_006",
    "rook_black": "benchmark_chess_007",
    "knight_black": "benchmark_chess_008",
    "bishop_black": "benchmark_chess_009",
    "queen_black": "benchmark_chess_010",
    "king_black": "benchmark_chess_011",
}
# Chessboard square to coordinate mapping provided by user
SQUARE_TO_COORD = {
    "A1": (-0.175, 0.175),
    "B1": (-0.175, 0.125),
    "C1": (-0.175, 0.075),
    "D1": (-0.175, 0.025),
    "E1": (-0.175, -0.025),
    "F1": (-0.175, -0.075),
    "G1": (-0.175, -0.125),
    "H1": (-0.175, -0.175),
    "A2": (-0.125, 0.175),
    "B2": (-0.125, 0.125),
    "C2": (-0.125, 0.075),
    "D2": (-0.125, 0.025),
    "E2": (-0.125, -0.025),
    "F2": (-0.125, -0.075),
    "G2": (-0.125, -0.125),
    "H2": (-0.125, -0.175),
    "A3": (-0.075, 0.175),
    "B3": (-0.075, 0.125),
    "C3": (-0.075, 0.075),
    "D3": (-0.075, 0.025),
    "E3": (-0.075, -0.025),
    "F3": (-0.075, -0.075),
    "G3": (-0.075, -0.125),
    "H3": (-0.075, -0.175),
    "A4": (-0.025, 0.175),
    "B4": (-0.025, 0.125),
    "C4": (-0.025, 0.075),
    "D4": (-0.025, 0.025),
    "E4": (-0.025, -0.025),
    "F4": (-0.025, -0.075),
    "G4": (-0.025, -0.125),
    "H4": (-0.025, -0.175),
    "A5": (0.025, 0.175),
    "B5": (0.025, 0.125),
    "C5": (0.025, 0.075),
    "D5": (0.025, 0.025),
    "E5": (0.025, -0.025),
    "F5": (0.025, -0.075),
    "G5": (0.025, -0.125),
    "H5": (0.025, -0.175),
    "A6": (0.075, 0.175),
    "B6": (0.075, 0.125),
    "C6": (0.075, 0.075),
    "D6": (0.075, 0.025),
    "E6": (0.075, -0.025),
    "F6": (0.075, -0.075),
    "G6": (0.075, -0.125),
    "H6": (0.075, -0.175),
    "A7": (0.125, 0.175),
    "B7": (0.125, 0.125),
    "C7": (0.125, 0.075),
    "D7": (0.125, 0.025),
    "E7": (0.125, -0.025),
    "F7": (0.125, -0.075),
    "G7": (0.125, -0.125),
    "H7": (0.125, -0.175),
    "A8": (0.175, 0.175),
    "B8": (0.175, 0.125),
    "C8": (0.175, 0.075),
    "D8": (0.175, 0.025),
    "E8": (0.175, -0.025),
    "F8": (0.175, -0.075),
    "G8": (0.175, -0.125),
    "H8": (0.175, -0.175),
}
# Define lower half (ranks 1-4) and upper half (ranks 5-8)
LOWER_HALF_SQUARES = [sq for sq in SQUARE_TO_COORD.keys() if sq[1] in "1234"]
UPPER_HALF_SQUARES = [sq for sq in SQUARE_TO_COORD.keys() if sq[1] in "5678"]

def get_desktop_center_and_size(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the center and size of the desktop's top surface in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    # Desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    # Top surface z-coordinate
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]
    return desktop_pos, desktop_size

def place_chessboard_on_desktop(desk_shape: Shape) -> Shape:
    """
    Place the chessboard at the center of the desktop, lowered by 0.12m as requested.
    """
    desktop_center, _ = get_desktop_center_and_size(desk_shape)
    chessboard_shape = library_call(
        "usd",
        oid="benchmark_chessboard_000",
        keywords=["chessboard", "brown", "square", "game_board", "centered_on_desk"],
    )
    # Get chessboard height to place it correctly on the desktop
    chessboard_info = get_object_info(chessboard_shape)
    chessboard_height = chessboard_info["size"][2]
    # Place chessboard at desktop center, sitting on top but lowered by 0.12m
    chessboard_placed = transform_shape(
        chessboard_shape,
        translation_matrix(
            [
                desktop_center[0],
                desktop_center[1],
                desktop_center[2] + chessboard_height / 2.0 - 0.13,  # Lowered by 0.12m
            ]
        ),
    )
    return concat_shapes(desk_shape, chessboard_placed)

def are_squares_adjacent(square1: str, square2: str) -> bool:
    """
    Check if two squares are adjacent (including diagonally).
    Returns True if they are adjacent, False otherwise.
    """
    file1, rank1 = ord(square1[0]), int(square1[1])
    file2, rank2 = ord(square2[0]), int(square2[1])
    file_diff = abs(file1 - file2)
    rank_diff = abs(rank1 - rank2)
    # Adjacent if both differences are <= 1 and not both zero
    return (file_diff <= 1 and rank_diff <= 1) and not (file_diff == 0 and rank_diff == 0)

def get_attacking_squares(piece_type: str, square: str) -> set:
    """
    Get all squares that a piece can attack from the given square.
    Only handles king, queen, rook, bishop, knight for simplicity.
    """
    file_idx = ord(square[0]) - ord('A')
    rank = int(square[1])

    attacking = set()

    if 'king' in piece_type:
        # King attacks all adjacent squares
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0:
                    continue
                new_file = file_idx + df
                new_rank = rank + dr
                if 0 <= new_file < 8 and 1 <= new_rank <= 8:
                    new_square = chr(ord('A') + new_file) + str(new_rank)
                    attacking.add(new_square)

    elif 'queen' in piece_type or 'rook' in piece_type or 'bishop' in piece_type:
        directions = []
        if 'rook' in piece_type or 'queen' in piece_type:
            directions.extend([(1,0), (-1,0), (0,1), (0,-1)])  # orthogonal
        if 'bishop' in piece_type or 'queen' in piece_type:
            directions.extend([(1,1), (1,-1), (-1,1), (-1,-1)])  # diagonal

        for df, dr in directions:
            for i in range(1, 8):
                new_file = file_idx + df * i
                new_rank = rank + dr * i
                if 0 <= new_file < 8 and 1 <= new_rank <= 8:
                    new_square = chr(ord('A') + new_file) + str(new_rank)
                    attacking.add(new_square)
                else:
                    break

    elif 'knight' in piece_type:
        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        for df, dr in knight_moves:
            new_file = file_idx + df
            new_rank = rank + dr
            if 0 <= new_file < 8 and 1 <= new_rank <= 8:
                new_square = chr(ord('A') + new_file) + str(new_rank)
                attacking.add(new_square)

    # Pawns attack diagonally forward (but we don't need this for our use case since pawns can't deliver checkmate easily)
    # We'll skip pawn logic for simplicity

    return attacking

def get_reachable_squares_in_one_move(piece_type: str, square: str) -> set:
    """
    Get all squares reachable by the piece in exactly one move (same as attacking squares for non-pawns).
    """
    return get_attacking_squares(piece_type, square)

def get_squares_reachable_in_two_moves(piece_type: str, square: str) -> set:
    """
    Get squares that can be reached in exactly two moves (for our purpose: squares that are NOT reachable in 1 move,
    but the piece can move to an intermediate square and then reach the target).
    However, for simplicity, we'll interpret the requirement as: the attacker should NOT be able to capture the king in 1 move,
    but SHOULD be able to capture it in 2 moves (i.e., there exists some intermediate position from which it can attack the king).
    """
    # First, get all squares reachable in 1 move from current position
    one_move_squares = get_reachable_squares_in_one_move(piece_type, square)

    # Now, for each of those intermediate squares, get what they can attack
    two_move_targets = set()
    for intermediate in one_move_squares:
        intermediate_attacks = get_attacking_squares(piece_type, intermediate)
        two_move_targets.update(intermediate_attacks)

    return two_move_targets

def is_valid_attacker_position_for_two_move_check(attacker_square: str, target_king_square: str, attacker_type: str) -> bool:
    """
    Check if:
    1. The attacker CANNOT capture the king in 1 move (not currently attacking)
    2. The attacker CAN capture the king in 2 moves (there exists an intermediate move that enables the capture)
    """
    # Condition 1: Not attacking in 1 move
    if target_king_square in get_attacking_squares(attacker_type, attacker_square):
        return False

    # Condition 2: Can attack in 2 moves
    two_move_targets = get_squares_reachable_in_two_moves(attacker_type, attacker_square)
    return target_king_square in two_move_targets

def get_path_squares(attacker_square: str, target_square: str, attacker_type: str) -> set:
    """
    Get all squares on the path between attacker and target (for blocking purposes).
    For sliding pieces only (queen, rook, bishop). For others, just return empty set.
    """
    if not ('queen' in attacker_type or 'rook' in attacker_type or 'bishop' in attacker_type):
        return set()

    file1 = ord(attacker_square[0]) - ord('A')
    rank1 = int(attacker_square[1])
    file2 = ord(target_square[0]) - ord('A')
    rank2 = int(target_square[1])

    df = file2 - file1
    dr = rank2 - rank1

    if df == 0 and dr == 0:
        return set()

    # Normalize direction
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)

    # Check if it's a valid sliding direction
    if not ((step_f != 0 and step_r == 0) or  # horizontal
            (step_f == 0 and step_r != 0) or  # vertical
            (abs(step_f) == 1 and abs(step_r) == 1)):  # diagonal
        return set()

    path = set()
    current_f = file1 + step_f
    current_r = rank1 + step_r

    while current_f != file2 or current_r != rank2:
        if 0 <= current_f < 8 and 1 <= current_r <= 8:
            square = chr(ord('A') + current_f) + str(current_r)
            path.add(square)
        else:
            break
        current_f += step_f
        current_r += step_r

    return path

@register()
def create_chess_pieces() -> Shape:
    """
    Create a set of chess pieces with specific constraints:
    - Both kings must be present (exactly one per color)
    - One "attacker" piece that CANNOT capture the opposite king in one move,
      but CAN capture it in two moves (by moving to an intermediate position first)
    - The attacker must be placed on ranks 1-4 (lower half)
    - 1-2 additional "scene" pieces that don't interfere with the potential attack path
    - Total pieces <= 7
    """
    import random

    # Initialize with both kings
    white_king_square = None
    black_king_square = None
    all_pieces_info = []

    # First, place the two kings on non-adjacent squares
    available_squares = list(SQUARE_TO_COORD.keys())
    white_king_square = random.choice(available_squares)
    all_pieces_info.append(("king_white", white_king_square))
    available_squares.remove(white_king_square)

    # Find non-adjacent square for black king
    non_adjacent_for_black = [sq for sq in available_squares if not are_squares_adjacent(white_king_square, sq)]
    if non_adjacent_for_black:
        black_king_square = random.choice(non_adjacent_for_black)
    else:
        black_king_square = random.choice(available_squares)
    all_pieces_info.append(("king_black", black_king_square))
    available_squares.remove(black_king_square)

    # Determine which color is attacking which
    attack_color = random.choice(["white", "black"])
    target_king_square = black_king_square if attack_color == "white" else white_king_square
    attacker_color = attack_color

    # Choose attacker type (avoid pawns for simplicity)
    possible_attackers = ["queen", "rook", "bishop", "knight"]
    attacker_type = random.choice(possible_attackers)
    attacker_piece_key = f"{attacker_type}_{attacker_color}"

    # Find valid attacker positions in lower half (ranks 1-4) that satisfy two-move condition
    valid_attacker_positions = []
    for square in LOWER_HALF_SQUARES:
        if square in [white_king_square, black_king_square]:
            continue
        if is_valid_attacker_position_for_two_move_check(square, target_king_square, attacker_piece_key):
            valid_attacker_positions.append(square)

    # If no valid positions found, try to find any position that at least satisfies the two-move condition
    if not valid_attacker_positions:
        for square in available_squares:
            if square in [white_king_square, black_king_square]:
                continue
            if is_valid_attacker_position_for_two_move_check(square, target_king_square, attacker_piece_key):
                valid_attacker_positions.append(square)

    # If still none found, fall back to placing attacker anywhere in lower half (this should rarely happen)
    if not valid_attacker_positions:
        valid_attacker_positions = [sq for sq in LOWER_HALF_SQUARES if sq not in [white_king_square, black_king_square]]
        if not valid_attacker_positions:
            valid_attacker_positions = available_squares[:1] if available_squares else [list(SQUARE_TO_COORD.keys())[0]]

    attacker_square = random.choice(valid_attacker_positions)
    all_pieces_info.append((attacker_piece_key, attacker_square))
    if attacker_square in available_squares:
        available_squares.remove(attacker_square)

    # Get potential attack paths (from possible intermediate positions to the target king)
    # For simplicity, we'll consider the direct path from attacker to king as forbidden for scene pieces
    # even though the attacker isn't currently on that path
    attack_path_squares = get_path_squares(attacker_square, target_king_square, attacker_piece_key)
    # Also avoid squares adjacent to kings (to maintain valid position)
    forbidden_squares = {white_king_square, black_king_square, attacker_square}
    forbidden_squares.update(attack_path_squares)

    # Add 1-2 scene pieces that don't interfere
    num_scene_pieces = random.randint(1, 2)
    scene_piece_types = ["pawn", "rook", "knight", "bishop", "queen"]
    scene_pieces_added = 0

    safe_squares = [sq for sq in available_squares if sq not in forbidden_squares]

    for _ in range(num_scene_pieces):
        if scene_pieces_added >= num_scene_pieces or not safe_squares:
            break

        # Choose a random scene piece type and color
        scene_piece_type = random.choice(scene_piece_types)
        scene_color = random.choice(["white", "black"])
        scene_piece_key = f"{scene_piece_type}_{scene_color}"

        # Pick a safe square
        scene_square = random.choice(safe_squares)
        all_pieces_info.append((scene_piece_key, scene_square))
        safe_squares.remove(scene_square)
        scene_pieces_added += 1

    # Create shapes for all pieces with height adjustment and random rotation
    all_pieces = []
    for piece_type, square in all_pieces_info:
        piece_oid = CHESS_PIECE_MAPPING[piece_type]
        piece_shape = library_call(
            "usd",
            oid=piece_oid,
            keywords=[piece_type, "white" if "white" in piece_type else "black", "chess_piece", f"position_{square}"],
        )
        x, y = SQUARE_TO_COORD[square]
        # First translate to position and raise by 0.05m relative to chessboard surface
        piece_transformed = transform_shape(piece_shape, translation_matrix([x, y, 0.05]))  # Raised by 0.05m
        # Then apply random rotation around z-axis
        piece_center = compute_shape_center(piece_transformed)
        random_angle = random.uniform(-math.pi, math.pi)
        piece_final = transform_shape(
            piece_transformed, rotation_matrix(random_angle, direction=(0, 0, 1), point=piece_center)
        )
        all_pieces.append(piece_final)
    return concat_shapes(*all_pieces) if all_pieces else []

@register()
def move_chess_piece_to_named_square_g2() -> Shape:
    """
    Main scene function: table with chessboard at center (lowered by 0.12m) and chess pieces in endgame position.
    """
    # Load the table
    desk_shape = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "four_legs"])
    # Place chessboard on table (lowered by 0.12m)
    scene_with_board = place_chessboard_on_desktop(desk_shape)
    # Create and position chess pieces
    pieces_shape = library_call("create_chess_pieces")
    # Get the chessboard position to place pieces correctly
    desktop_center, _ = get_desktop_center_and_size(desk_shape)
    chessboard_height = get_object_info(library_call("usd", oid="benchmark_chessboard_000", keywords=[]))["size"][2]
    # Chessboard is lowered by 0.12m from normal position
    chessboard_z = desktop_center[2] + chessboard_height / 2.0 - 0.13
    pieces_on_board = transform_shape(
        pieces_shape, translation_matrix([desktop_center[0], desktop_center[1], chessboard_z])
    )
    return concat_shapes(scene_with_board, pieces_on_board)

@register()
def root_scene() -> Shape:
    return move_chess_piece_to_named_square_g2()
