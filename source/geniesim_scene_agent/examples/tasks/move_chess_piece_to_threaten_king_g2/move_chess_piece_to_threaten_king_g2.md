# Move a chess piece to threaten the opposing king

**Tags:** `reasoning` `chess` `detection` `chessboard` `no_manipulation` `scene_analysis` `board_game` `check_detection` `piece_identification` `spatial_reasoning`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find chessboard by usd "chessboard" or keywords "chess"/"board". Find all chess pieces by usd prefix "benchmark_chess_" or keywords matching piece types (king, queen, rook, bishop, knight, pawn).
2. **Board coordinate system:** Determine chessboard orientation from its pose. Map each piece to board squares (a1–h8) based on piece world position relative to board origin.
3. **Piece identification:** Classify each piece by its usd name or keywords (color + type, e.g. "white_king", "black_rook").
4. **Check detection:** Apply chess rules to determine if either king is in check. Trace attack lines for each opposing piece (rook=rank/file, bishop=diagonal, queen=both, knight=L-shape, pawn=diagonal-forward).
5. **Report:** Output which king is in check and by which piece(s), or "no check" if neither king is threatened.

---

# INPUT

**scene_description:** A chessboard with chess pieces arranged in a specific position. Some pieces may be attacking the opponent's king.

**task description:** Analyze the chess position and determine if either king is in check.

**code to generate the scene:**

```python
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

    return attacking

def get_reachable_squares_in_one_move(piece_type: str, square: str) -> set:
    """
    Get all squares reachable by the piece in exactly one move (same as attacking squares for non-pawns).
    """
    return get_attacking_squares(piece_type, square)

def get_squares_reachable_in_two_moves(piece_type: str, square: str) -> set:
    """
    Get squares that can be reached in exactly two moves.
    """
    one_move_squares = get_reachable_squares_in_one_move(piece_type, square)
    two_move_targets = set()
    for intermediate in one_move_squares:
        intermediate_attacks = get_attacking_squares(piece_type, intermediate)
        two_move_targets.update(intermediate_attacks)
    return two_move_targets

def is_valid_attacker_position_for_two_move_check(attacker_square: str, target_king_square: str, attacker_type: str) -> bool:
    """
    Check if:
    1. The attacker CANNOT capture the king in 1 move (not currently attacking)
    2. The attacker CAN capture the king in 2 moves
    """
    if target_king_square in get_attacking_squares(attacker_type, attacker_square):
        return False
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
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    if not ((step_f != 0 and step_r == 0) or
            (step_f == 0 and step_r != 0) or
            (abs(step_f) == 1 and abs(step_r) == 1)):
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
      but CAN capture it in two moves
    - The attacker must be placed on ranks 1-4 (lower half)
    - 1-2 additional "scene" pieces that don't interfere with the potential attack path
    - Total pieces <= 7
    """
    import random

    white_king_square = None
    black_king_square = None
    all_pieces_info = []

    available_squares = list(SQUARE_TO_COORD.keys())
    white_king_square = random.choice(available_squares)
    all_pieces_info.append(("king_white", white_king_square))
    available_squares.remove(white_king_square)

    non_adjacent_for_black = [sq for sq in available_squares if not are_squares_adjacent(white_king_square, sq)]
    if non_adjacent_for_black:
        black_king_square = random.choice(non_adjacent_for_black)
    else:
        black_king_square = random.choice(available_squares)
    all_pieces_info.append(("king_black", black_king_square))
    available_squares.remove(black_king_square)

    attack_color = random.choice(["white", "black"])
    target_king_square = black_king_square if attack_color == "white" else white_king_square
    attacker_color = attack_color

    possible_attackers = ["queen", "rook", "bishop", "knight"]
    attacker_type = random.choice(possible_attackers)
    attacker_piece_key = f"{attacker_type}_{attacker_color}"

    valid_attacker_positions = []
    for square in LOWER_HALF_SQUARES:
        if square in [white_king_square, black_king_square]:
            continue
        if is_valid_attacker_position_for_two_move_check(square, target_king_square, attacker_piece_key):
            valid_attacker_positions.append(square)

    if not valid_attacker_positions:
        for square in available_squares:
            if square in [white_king_square, black_king_square]:
                continue
            if is_valid_attacker_position_for_two_move_check(square, target_king_square, attacker_piece_key):
                valid_attacker_positions.append(square)

    if not valid_attacker_positions:
        valid_attacker_positions = [sq for sq in LOWER_HALF_SQUARES if sq not in [white_king_square, black_king_square]]
        if not valid_attacker_positions:
            valid_attacker_positions = available_squares[:1] if available_squares else [list(SQUARE_TO_COORD.keys())[0]]

    attacker_square = random.choice(valid_attacker_positions)
    all_pieces_info.append((attacker_piece_key, attacker_square))
    if attacker_square in available_squares:
        available_squares.remove(attacker_square)

    attack_path_squares = get_path_squares(attacker_square, target_king_square, attacker_piece_key)
    forbidden_squares = {white_king_square, black_king_square, attacker_square}
    forbidden_squares.update(attack_path_squares)

    num_scene_pieces = random.randint(1, 2)
    scene_piece_types = ["pawn", "rook", "knight", "bishop", "queen"]
    scene_pieces_added = 0

    safe_squares = [sq for sq in available_squares if sq not in forbidden_squares]

    for _ in range(num_scene_pieces):
        if scene_pieces_added >= num_scene_pieces or not safe_squares:
            break
        scene_piece_type = random.choice(scene_piece_types)
        scene_color = random.choice(["white", "black"])
        scene_piece_key = f"{scene_piece_type}_{scene_color}"
        scene_square = random.choice(safe_squares)
        all_pieces_info.append((scene_piece_key, scene_square))
        safe_squares.remove(scene_square)
        scene_pieces_added += 1

    all_pieces = []
    for piece_type, square in all_pieces_info:
        piece_oid = CHESS_PIECE_MAPPING[piece_type]
        piece_shape = library_call(
            "usd",
            oid=piece_oid,
            keywords=[piece_type, "white" if "white" in piece_type else "black", "chess_piece", f"position_{square}"],
        )
        x, y = SQUARE_TO_COORD[square]
        piece_transformed = transform_shape(piece_shape, translation_matrix([x, y, 0.05]))
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
    desk_shape = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "four_legs"])
    scene_with_board = place_chessboard_on_desktop(desk_shape)
    pieces_shape = library_call("create_chess_pieces")
    desktop_center, _ = get_desktop_center_and_size(desk_shape)
    chessboard_height = get_object_info(library_call("usd", oid="benchmark_chessboard_000", keywords=[]))["size"][2]
    chessboard_z = desktop_center[2] + chessboard_height / 2.0 - 0.13
    pieces_on_board = transform_shape(
        pieces_shape, translation_matrix([desktop_center[0], desktop_center[1], chessboard_z])
    )
    return concat_shapes(scene_with_board, pieces_on_board)

@register()
def root_scene() -> Shape:
    return move_chess_piece_to_named_square_g2()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

PIECE_TYPES = {
    "king": "K", "queen": "Q", "rook": "R",
    "bishop": "B", "knight": "N", "pawn": "P"
}

COLORS = ["white", "black"]


def _world_to_board(piece_pos, board_origin, board_x_axis, board_y_axis, square_size):
    relative = piece_pos[:2] - board_origin[:2]
    col = int(round(np.dot(relative, board_x_axis[:2]) / square_size))
    row = int(round(np.dot(relative, board_y_axis[:2]) / square_size))
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    return (row, col)


def _square_name(row, col):
    return f"{chr(ord('a') + col)}{row + 1}"


def _is_valid(r, c):
    return 0 <= r <= 7 and 0 <= c <= 7


def _find_attacks(board, piece_color, king_pos):
    kr, kc = king_pos
    attackers = []
    opponent = "black" if piece_color == "white" else "white"

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p is None or p["color"] != opponent:
                continue
            pt = p["type"]

            if pt == "rook" or pt == "queen":
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    while _is_valid(nr, nc):
                        if nr == kr and nc == kc:
                            attackers.append((p, r, c))
                            break
                        if board[nr][nc] is not None:
                            break
                        nr += dr
                        nc += dc

            if pt == "bishop" or pt == "queen":
                for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    while _is_valid(nr, nc):
                        if nr == kr and nc == kc:
                            attackers.append((p, r, c))
                            break
                        if board[nr][nc] is not None:
                            break
                        nr += dr
                        nc += dc

            if pt == "knight":
                for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                (1, -2), (1, 2), (2, -1), (2, 1)]:
                    if r + dr == kr and c + dc == kc:
                        attackers.append((p, r, c))

            if pt == "pawn":
                direction = 1 if p["color"] == "white" else -1
                for dc in [-1, 1]:
                    if r + direction == kr and c + dc == kc:
                        attackers.append((p, r, c))

    return attackers


def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    description_json = {
        "task_name": "检测棋盘上是否存在将军",
        "english_task_name": "detect check on the chessboard",
        "init_scene_text": "A chessboard with pieces in a specific position"
    }
    robot.client.set_task_task_basic_info(description_json)

    board_id = None
    pieces = []

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        kw_str = " ".join(keywords).lower()

        if "chessboard" in usd or "chess" in kw_str and "board" in kw_str:
            board_id = obj_id
            continue

        if "benchmark_chess_" in usd or any(pt in kw_str for pt in PIECE_TYPES):
            color = None
            piece_type = None
            for c in COLORS:
                if c in usd.lower() or c in kw_str:
                    color = c
                    break
            for pt in PIECE_TYPES:
                if pt in usd.lower() or pt in kw_str:
                    piece_type = pt
                    break
            if color and piece_type:
                pieces.append({"id": obj_id, "usd": usd, "color": color, "type": piece_type})

    if not board_id:
        print("Chessboard not found")
        return False

    print(f"Board: {board_id}, Pieces: {len(pieces)}")

    board_pos, board_rot = robot.get_object_pose(board_id)
    board_size = robot.get_object_size_in_world_frame(board_id)

    if board_pos is None or board_size is None:
        print("Cannot get board pose/size")
        return False

    rot_mat = quat2mat_wxyz(board_rot)
    board_x = rot_mat[:, 0]
    board_y = rot_mat[:, 1]

    square_size = min(board_size[0], board_size[1]) / 8.0
    board_origin = board_pos - board_x * (3.5 * square_size) - board_y * (3.5 * square_size)

    board = [[None for _ in range(8)] for _ in range(8)]
    king_positions = {}

    for piece in pieces:
        pos, _ = robot.get_object_pose(piece["id"])
        if pos is None:
            continue
        row, col = _world_to_board(pos, board_origin, board_x, board_y, square_size)
        piece["row"] = row
        piece["col"] = col
        board[row][col] = piece
        if piece["type"] == "king":
            king_positions[piece["color"]] = (row, col)
        print(f"  {piece['color']} {piece['type']} at {_square_name(row, col)}")

    results = {}
    for color in COLORS:
        if color not in king_positions:
            print(f"{color} king not found")
            continue
        attackers = _find_attacks(board, color, king_positions[color])
        kr, kc = king_positions[color]
        if attackers:
            attacker_strs = [
                f"{a[0]['color']} {a[0]['type']} at {_square_name(a[1], a[2])}"
                for a in attackers
            ]
            print(f"CHECK: {color} king at {_square_name(kr, kc)} is in check by {', '.join(attacker_strs)}")
            results[color] = attackers
        else:
            print(f"{color} king at {_square_name(kr, kc)} is NOT in check")

    if results:
        print("\n✅ Check detected")
    else:
        print("\n✅ No check on the board")

    return True
```
