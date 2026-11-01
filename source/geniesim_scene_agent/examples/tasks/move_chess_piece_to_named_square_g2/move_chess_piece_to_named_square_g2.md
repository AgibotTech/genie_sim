# Move a chess piece to a named square

**Tags:** `chess` `chessboard` `chess_rules` `keywords_position_square` `board_state` `square_to_board_coordinates` `legal_moves` `pick_and_place` `pose_based_placement` `benchmark_chessboard_000` `dynamic_task_description` `GraspConfig` `PlaceConfig`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find chessboard (benchmark_chessboard_000). Collect all pieces with board square from keyword "position_*" (e.g. position_A3). Fill board_state (square -> piece id) and piece_positions (piece id -> square).
2. **Board coordinates:** square_to_board_coordinates(square): map "A1" etc. to (x, y) relative to board center. Use board pose/size to get world position and z for placement.
3. **Legal move:** Select piece (e.g. in lower half). Compute legal_moves for that piece (e.g. pawn forward, rook horizontal/vertical). target_square = random.choice(legal_moves). Target pose = board world position for target_square + piece height.
4. **Execution:** Set dynamic task_description with piece name and target square. pick_and_place_actor_to_target_pose with multiple candidate poses (e.g. rotations around z).

---

# INPUT

**scene_description:** A table with a chessboard (benchmark_chessboard_000) and chess pieces. Each piece has a keyword like "position_A3" (board square). The task is to select one movable piece (e.g. in the lower half) and move it to another legal position according to chess rules (move only, no capture).

**task description:** Select a chess piece that can move (e.g. in the lower half of the board), and move it to a legal position. The move must follow chess rules; only move, no capture. Number of squares moved can be any legal amount. Choice can be random.

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
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    """
    任务：在下半区选择一枚可以移动的棋子，将其移动到另一个合法位置
    注意：1. 棋子移动需要符合规则 2. 仅仅移动，不吃子 3. 移动的格数只要符合规则都可以 4. 随机选择
    """

    # 初始化objects_list - 棋子和棋盘
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

    # 运行时数据
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # 1. 查找棋盘 (benchmark_chessboard_000)
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

    # 2. 获取棋盘位置和尺寸
    chessboard_pos, chessboard_rot = robot.get_object_pose(chessboard_id)
    chessboard_size = robot.get_object_size_in_world_frame(chessboard_id)

    if chessboard_pos is None or chessboard_size is None:
        print("Cannot get chessboard position/size")
        return False

    print(f"Chessboard position: {chessboard_pos}, size: {chessboard_size}")

    # 3. 查找所有棋子，直接从keywords中读取位置信息
    chess_pieces = []
    board_state = {}  # 棋盘方格 -> 棋子ID
    piece_positions = {}  # 棋子ID -> 棋盘方格

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # 检查是否是棋子
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

            # 从keywords中提取棋盘位置（如"position_B3"）
            position_square = None
            for kw in keywords:
                if "position_" in kw:
                    position_square = kw.replace("position_", "")
                    break

            chess_pieces.append({
                "id": obj_id,
                "usd": usd,
                "position": piece_pos,
                "rotation": piece_rot,  # 保存当前旋转
                "keywords": keywords,
                "piece_type": piece_name,
                "board_square": position_square  # 保存棋盘方格
            })

            # 如果找到了棋盘方格，更新棋盘状态
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

    # 4. 棋盘坐标系映射函数
    def square_to_board_coordinates(square):
        """将棋盘方格（如'A1'）转换为相对于棋盘中心的坐标"""
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

    # 5. 定义下半区（排名1-4）
    lower_half_squares = [f"{file}{rank}" for file in "ABCDEFGH" for rank in "1234"]

    # 6. 辅助函数
    def get_piece_color(piece_id):
        """获取棋子颜色"""
        for piece in chess_pieces:
            if piece["id"] == piece_id:
                usd = piece.get("usd", "")
                if "benchmark_chess_000" in usd or "benchmark_chess_001" in usd or "benchmark_chess_002" in usd or "benchmark_chess_003" in usd or "benchmark_chess_004" in usd or "benchmark_chess_005" in usd:
                    return "white"
                elif "benchmark_chess_006" in usd or "benchmark_chess_007" in usd or "benchmark_chess_008" in usd or "benchmark_chess_009" in usd or "benchmark_chess_010" in usd or "benchmark_chess_011" in usd:
                    return "black"
        return None

    def get_piece_type(piece_id):
        """获取棋子类型"""
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
        """获取棋子的合法非吃子移动（仅移动到空方格）"""
        piece_type = get_piece_type(piece_id)
        piece_color = get_piece_color(piece_id)

        if not piece_type or not piece_color:
            return []

        file = square[0]  # A-H
        rank = int(square[1])  # 1-8

        legal_moves = []

        if piece_type == "pawn":
            # 兵：向前移动1格（白色向上，黑色向下），不能后退
            if piece_color == "white":
                # 向前移动
                forward_rank = rank + 1
                if forward_rank <= 8:
                    forward_square = f"{file}{forward_rank}"
                    if forward_square not in board_state:  # 方格必须为空
                        legal_moves.append(forward_square)
            else:  # black
                forward_rank = rank - 1
                if forward_rank >= 1:
                    forward_square = f"{file}{forward_rank}"
                    if forward_square not in board_state:  # 方格必须为空
                        legal_moves.append(forward_square)

        elif piece_type == "knight":
            # 骑士：L形移动（8个可能的位置）
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

                    # 骑士可以跳过其他棋子，但目标方格必须为空（不吃子）
                    if new_square not in board_state:
                        legal_moves.append(new_square)

        elif piece_type == "king":
            # 国王：向任意方向移动1格（8个方向）
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

                    # 国王只能移动到空方格（不吃子）
                    if new_square not in board_state:
                        legal_moves.append(new_square)

        elif piece_type == "rook":
            # 车：直线移动
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
                            break  # 被阻挡
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # 超出棋盘边界

        elif piece_type == "bishop":
            # 主教：对角线移动
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
                            break  # 被阻挡
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # 超出棋盘边界

        elif piece_type == "queen":
            # 皇后：直线和对角线移动
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
                            break  # 被阻挡
                        else:
                            legal_moves.append(new_square)
                    else:
                        break  # 超出棋盘边界

        return legal_moves

    # 7. 在下半区寻找可以移动的棋子
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
                    "original_rotation": piece["rotation"]  # 保存原始旋转
                })
                print(f"Lower half piece {piece_id} at {square} ({piece_color} {piece_type}) has {len(legal_moves)} legal non-capture moves")

    if len(available_pieces) == 0:
        print("No movable pieces found in lower half")
        return False

    print(f"Found {len(available_pieces)} movable pieces in lower half")

    # 8. 随机选择一个棋子（按要求随机选择）
    selected_piece = random.choice(available_pieces)

    piece_id = selected_piece["id"]
    current_square = selected_piece["square"]
    legal_moves = selected_piece["legal_moves"]
    piece_type = selected_piece["piece_type"]
    piece_color = selected_piece["color"]
    original_rotation = selected_piece["original_rotation"]  # 原始旋转

    print(f"\nRandomly selected piece: {piece_id} ({piece_color} {piece_type}) at {current_square}")
    print(f"Legal non-capture moves: {legal_moves}")

    # 9. 随机选择一个目标方格
    if legal_moves:
        target_square = random.choice(legal_moves)
    else:
        print(f"No valid target square for piece {piece_id}")
        return False

    print(f"Randomly selected target square: {target_square}")

    # 10. 计算目标位置的世界坐标
    target_board_coords = square_to_board_coordinates(target_square)
    if not target_board_coords:
        print(f"Cannot convert square {target_square} to board coordinates")
        return False

    target_x, target_y = target_board_coords

    # 棋子高度：棋盘表面 + 0.05m（根据场景代码）
    chessboard_height = chessboard_size[2]
    chessboard_surface_z = chessboard_pos[2] - chessboard_height / 2.0
    piece_height = chessboard_surface_z + 0.05 + 0.03

    target_world_pos = np.array([
        chessboard_pos[0] + target_x,
        chessboard_pos[1] + target_y,
        piece_height
    ])

    print(f"Target world position: {target_world_pos}")

    # 11. 生成多个候选姿态，基于原始旋转进行随机绕z轴旋转
    target_poses = []

    # 首先获取棋子的原始旋转矩阵
    original_rot_matrix = quat2mat_wxyz(original_rotation)  # 原始旋转矩阵

    # 位置变化参数（小范围偏移，增加成功机会）
    x_offsets = np.linspace(0.00, 0.00, 3)  # x方向小偏移
    y_offsets = np.linspace(0.00, 0.00, 3)  # y方向小偏移
    z_offsets = np.linspace(0, 0.002, 3)  # z方向小偏移

    # 生成多个随机绕z轴旋转的角度
    num_rotations = 12
    z_rotation_angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)

    # 生成所有候选姿态的组合
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = target_world_pos + np.array([x_offset, y_offset, z_offset])

                # 生成多个绕z轴的随机旋转
                for angle in z_rotation_angles:
                    # 创建绕z轴的旋转矩阵
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rot_z = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]
                    ])

                    # 组合旋转：原始旋转 + 绕z轴随机旋转
                    final_rot = np.dot(rot_z, original_rot_matrix)
                    # final_rot = original_rot_matrix

                    # 创建4x4姿态矩阵
                    pose = np.eye(4)
                    pose[:3, :3] = final_rot
                    pose[:3, 3] = position
                    target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses with random z-axis rotations")

    # MANDATORY: Set task description dynamically before first action (after determining piece and target square)
    task_description = {
        "task_name": f"将{piece_color}的{piece_type}从{current_square}移动到{target_square}",
        "english_task_name": f"move the {piece_color} {piece_type} from {current_square} to {target_square}",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 12. 执行移动操作
    piece_position, _ = robot.get_object_pose(piece_id)
    arm = robot.judge_arm_to_use(piece_position)
    print(f"Using arm: {arm}")

    # 不可移动对象列表（保护棋盘和其他棋子）
    unmovable_objects = [chessboard_id]
    for piece in chess_pieces:
        if piece["id"] != piece_id:
            unmovable_objects.append(piece["id"])

    # 根据棋子类型和颜色确定描述
    piece_type_name = {
        "pawn": "兵",
        "rook": "城堡",
        "knight": "骑士",
        "bishop": "主教",
        "queen": "皇后",
        "king": "国王"
    }.get(piece_type, "棋子")

    piece_color_name = {"white": "白", "black": "黑"}.get(piece_color, "")

    # 抓取配置
    grasp_config = GraspConfig(
        is_from_up_side=True,
        pick_up_distance=0.12,
        unmovable_objects=unmovable_objects.copy(),
        action_description={
            "zh": f"拿起{piece_color_name}色的{piece_type_name}",
            "en": f"pick up the {piece_color} {piece_type}"
        }
    )

    # 定义检查函数：验证棋子是否成功移动到目标位置
    def create_move_check_function(piece_id, target_pos):
        def check_piece_moved():
            current_pos, current_rot = robot.get_object_pose(piece_id)
            if current_pos is None:
                return False

            # 位置检查（0.013m容差）
            distance = np.linalg.norm(current_pos - target_pos)
            tolerance = 0.013
            print(f"distance: {distance:.3f}m, tolerance: {tolerance:.3f}m")
            if distance > tolerance:
                print(f"Piece {piece_id} position error: {distance:.3f}m > {tolerance:.3f}m")
                return False

            # 检查是否在棋盘上
            if current_pos[2] < chessboard_surface_z - 0.02 or current_pos[2] > chessboard_surface_z + 0.10:
                print(f"Piece {piece_id} z position {current_pos[2]:.3f}m out of range")
                return False

            return True

        return check_piece_moved

    check_function = create_move_check_function(piece_id, target_world_pos)

    # 放置配置
    move_desc_zh = f"将{piece_color_name}色的{piece_type_name}从{current_square}移动到{target_square}"
    move_desc_en = f"move the {piece_color} {piece_type} from {current_square} to {target_square}"

    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable_objects.copy(),
        extra_checkers=[check_function],
        action_description={"zh": move_desc_zh, "en": move_desc_en}
    )

    # 执行抓取和放置
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

    # 返回手臂到初始位置
    robot.back_to_origin(arm=arm)

    print("\n=== Task summary ===")
    if success:
        print(f"✅ SUCCESS: Randomly moved {piece_color} {piece_type} from {current_square} to {target_square}")
    else:
        print("❌ FAILED: Could not move piece")

    return success
```
