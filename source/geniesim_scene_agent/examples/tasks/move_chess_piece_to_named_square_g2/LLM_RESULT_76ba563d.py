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


def select_non_adjacent_squares(squares_pool: list, num_squares: int) -> list:
    """
    Select non-adjacent squares from the given pool.
    """
    if num_squares == 0:
        return []

    selected = []
    available = squares_pool.copy()

    for _ in range(num_squares):
        if not available:
            break

        # Try to find a square that's not adjacent to any already selected
        valid_choices = []
        for sq in available:
            is_valid = True
            for sel_sq in selected:
                if are_squares_adjacent(sq, sel_sq):
                    is_valid = False
                    break
            if is_valid:
                valid_choices.append(sq)

        if valid_choices:
            chosen = random.choice(valid_choices)
            selected.append(chosen)
            available.remove(chosen)
        else:
            # If no non-adjacent squares available, pick any remaining
            chosen = random.choice(available)
            selected.append(chosen)
            available.remove(chosen)

    return selected


@register()
def create_chess_pieces() -> Shape:
    """
    Create a set of chess pieces for an endgame:
    - Both kings must be present (exactly one per color)
    - White pieces: 1-5 total (including king)
    - Black pieces: 1-5 total (including king)
    - Total pieces: <=7
    - Lower half (ranks 1-4) must have 2-4 pieces total
    - Lower half pieces must be at least one square apart (non-adjacent)
    - Pieces are raised by 0.05m relative to chessboard and randomly rotated around z-axis
    """
    import random

    # Initialize piece lists with exactly one king each
    white_pieces = ["king_white"]
    black_pieces = ["king_black"]

    # Determine additional pieces for each color (0-4 additional since king is already included)
    max_white_additional = 4  # 5 total - 1 king = 4 additional
    max_black_additional = 4  # 5 total - 1 king = 4 additional
    max_total_additional = 5  # 7 total - 2 kings = 5 additional

    # Randomly determine additional pieces while respecting constraints
    num_white_additional = random.randint(0, max_white_additional)
    num_black_additional = random.randint(0, min(max_black_additional, max_total_additional - num_white_additional))

    # Ensure we don't exceed total limit
    if num_white_additional + num_black_additional > max_total_additional:
        # Adjust if needed
        excess = (num_white_additional + num_black_additional) - max_total_additional
        if num_black_additional >= excess:
            num_black_additional -= excess
        else:
            num_white_additional -= excess - num_black_additional
            num_black_additional = 0

    # Add white additional pieces
    white_piece_types = ["pawn_white", "rook_white", "knight_white", "bishop_white", "queen_white"]
    for _ in range(num_white_additional):
        piece_type = random.choice(white_piece_types)
        white_pieces.append(piece_type)

    # Add black additional pieces
    black_piece_types = ["pawn_black", "rook_black", "knight_black", "bishop_black", "queen_black"]
    for _ in range(num_black_additional):
        piece_type = random.choice(black_piece_types)
        black_pieces.append(piece_type)

    # Combine all pieces with their colors
    all_pieces_with_color = []
    for piece in white_pieces:
        all_pieces_with_color.append((piece, "white"))
    for piece in black_pieces:
        all_pieces_with_color.append((piece, "black"))

    # Now handle lower half constraint: need 2-4 pieces total in lower half
    total_pieces = len(all_pieces_with_color)
    min_lower = 2
    max_lower = min(4, total_pieces)  # Can't have more lower pieces than total pieces

    if total_pieces < min_lower:
        # This shouldn't happen since we have at least 2 kings, but just in case
        num_lower_pieces = total_pieces
    else:
        num_lower_pieces = random.randint(min_lower, max_lower)

    # Select non-adjacent squares for lower half
    lower_squares = select_non_adjacent_squares(LOWER_HALF_SQUARES, num_lower_pieces)

    # Assign pieces to positions
    all_pieces_info = []
    available_squares = set(SQUARE_TO_COORD.keys())

    # First assign lower half pieces
    for i, square in enumerate(lower_squares):
        if i < len(all_pieces_with_color):
            piece_type, color = all_pieces_with_color[i]
            all_pieces_info.append((piece_type, square))
            available_squares.remove(square)

    # Assign remaining pieces to upper half or any remaining squares
    remaining_pieces = all_pieces_with_color[num_lower_pieces:]
    for piece_type, color in remaining_pieces:
        if available_squares:
            square = random.choice(list(available_squares))
            all_pieces_info.append((piece_type, square))
            available_squares.remove(square)

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
