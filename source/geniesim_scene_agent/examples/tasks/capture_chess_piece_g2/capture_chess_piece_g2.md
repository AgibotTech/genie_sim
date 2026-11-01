# Capture a chess piece

**Tags:** `chess` `chessboard` `chess_rules` `capture` `table_only` `pick_and_place` `keywords_position_square` `board_state` `legal_moves` `square_to_board_coordinates` `dynamic_task_description` `GraspConfig` `PlaceConfig`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find chessboard (benchmark_chessboard_000) and table. Find all chess pieces by usd matching benchmark_chess_XXX pattern.
2. **Position extraction:** Parse keywords to get position_square (e.g., "position_E4"). Build board_state mapping square -> piece info.
3. **Piece identification:** Determine piece color (white: 000-005, black: 006-011) and type (pawn/rook/knight/bishop/queen/king).
4. **Legal capture search:** For each piece in lower half (ranks 1-4), compute legal capture moves against opponent pieces. Use piece-specific movement rules.
5. **Capture execution:** Randomly select a valid capture. Move captured piece to table side (left if y>0, right if y<0), then move capturing piece to target square.
6. **Verification:** Check both pieces are at correct positions. Verify capturing piece replaced captured piece's position.

---

# INPUT

**scene_description:** A chessboard placed at the center of a table with 1-5 white and 1-5 black chess pieces (total <=7), arranged in plausible endgame positions. Both kings are present. The lower half (ranks 1-4) contains pieces that can capture opponent pieces.

**task description:** Find a piece in the lower half that can capture an opponent piece, move the captured piece to the table side (left or right based on position), and move the capturing piece to the opponent's original position.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: chessboard_on_table
description: A chessboard placed at the center of a table with chess pieces in endgame positions, where captures are possible in the lower half.
"""

CHESS_PIECE_MAPPING = {
    "pawn_white": "benchmark_chess_000", "rook_white": "benchmark_chess_001",
    "knight_white": "benchmark_chess_002", "bishop_white": "benchmark_chess_003",
    "queen_white": "benchmark_chess_004", "king_white": "benchmark_chess_005",
    "pawn_black": "benchmark_chess_006", "rook_black": "benchmark_chess_007",
    "knight_black": "benchmark_chess_008", "bishop_black": "benchmark_chess_009",
    "queen_black": "benchmark_chess_010", "king_black": "benchmark_chess_011",
}

SQUARE_TO_COORD = {
    "A1": (-0.175, 0.175), "B1": (-0.175, 0.125), "C1": (-0.175, 0.075), "D1": (-0.175, 0.025),
    # ... all 64 squares
    "H8": (0.175, -0.175),
}

LOWER_HALF_SQUARES = [sq for sq in SQUARE_TO_COORD.keys() if sq[1] in "1234"]

def get_desktop_center_and_size(desk_shape: Shape) -> Tuple[P, P]:
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    return np.array([desktop_center[0], desktop_center[1], desktop_top_z]), desktop_subpart_info["size"]

def get_attacked_squares(piece_type: str, square: str) -> set:
    """Get squares attacked by a piece of given type on given square."""
    row, col = ord(square[0]) - ord('A'), int(square[1]) - 1
    attacked = set()
    # Movement logic for each piece type...
    if "knight" in piece_type:
        for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                attacked.add(f"{chr(ord('A')+nc)}{nr+1}")
    # ... other piece types
    return attacked

@register()
def create_chess_pieces() -> Shape:
    """Create chess pieces with valid capture opportunities in lower half."""
    all_pieces_info = []
    # Generate pieces ensuring at least one capture opportunity in lower half
    # ... piece generation logic
    all_pieces = []
    for piece_type, square in all_pieces_info:
        piece_oid = CHESS_PIECE_MAPPING[piece_type]
        piece_shape = library_call("usd", oid=piece_oid, keywords=[piece_type, "chess_piece", f"position_{square}"])
        x, y = SQUARE_TO_COORD[square]
        piece_t = transform_shape(piece_shape, translation_matrix([x, y, 0.05]))
        all_pieces.append(piece_t)
    return concat_shapes(*all_pieces) if all_pieces else []

@register()
def move_chess_piece_to_named_square_g2() -> Shape:
    desk_shape = library_call("usd", oid="table_000", keywords=["table"])
    chessboard_shape = library_call("usd", oid="benchmark_chessboard_000", keywords=["chessboard"])
    desktop_center, _ = get_desktop_center_and_size(desk_shape)
    chessboard_info = get_object_info(chessboard_shape)
    chessboard_height = chessboard_info["size"][2]
    chessboard_placed = transform_shape(chessboard_shape, translation_matrix([desktop_center[0], desktop_center[1], desktop_center[2] + chessboard_height/2.0 - 0.13]))
    pieces_shape = library_call("create_chess_pieces")
    pieces_on_board = transform_shape(pieces_shape, translation_matrix([desktop_center[0], desktop_center[1], desktop_center[2]]))
    return concat_shapes(desk_shape, chessboard_placed, pieces_on_board)

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
    objects_list = {
        "benchmark_chessboard_000": {"has_interactions": True, "object_type": "chessboard"},
        "benchmark_chess_000": {"has_interactions": True, "object_type": "chess_piece", "name": "pawn_white"},
        "benchmark_chess_001": {"has_interactions": True, "object_type": "chess_piece", "name": "rook_white"},
        # ... all chess pieces
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    description_json = {
        "task_name": "在下半区找出可以吃掉对方棋子的棋子",
        "english_task_name": "Find pieces in lower half that can capture opponent pieces",
        "init_scene_text": "The robot is in front of the table, there is a chessboard with chess pieces on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    # Find chessboard and table
    chessboard_id = None
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_chessboard_000" in usd:
            chessboard_id = obj_id
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id

    if not chessboard_id or not table_id:
        return False

    chessboard_pos, chessboard_rot = robot.get_object_pose(chessboard_id)
    chessboard_size = robot.get_object_size_in_world_frame(chessboard_id)
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # Find all chess pieces with their positions
    chess_pieces = []
    board_state = {}
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        position_square = None
        for kw in keywords:
            if "position_" in kw:
                position_square = kw.replace("position_", "")
                break
        if position_square and "benchmark_chess" in usd:
            piece_pos, piece_rot = robot.get_object_pose(obj_id)
            chess_pieces.append({"id": obj_id, "usd": usd, "position": piece_pos, "rotation": piece_rot, "board_square": position_square})
            board_state[position_square] = {"id": obj_id, "usd": usd, "position": piece_pos, "rotation": piece_rot}

    def square_to_board_coordinates(square):
        board_coords = {
            "A1": (-0.175, 0.175), "B1": (-0.175, 0.125),
            # ... all 64 squares
        }
        return board_coords.get(square)

    def get_piece_color(piece_id):
        for piece in chess_pieces:
            if piece["id"] == piece_id:
                usd = piece.get("usd", "")
                if any(f"benchmark_chess_00{i}" in usd for i in range(6)):
                    return "white"
                return "black"
        return None

    def get_legal_capture_moves(piece_id, piece_square, piece_color):
        """Compute legal captures for the piece."""
        piece_type = get_piece_type(piece_id)
        target_color = "black" if piece_color == "white" else "white"
        legal_captures = []
        # Piece-specific capture logic...
        return legal_captures

    LOWER_HALF_SQUARES = [f"{f}{r}" for f in "ABCDEFGH" for r in "1234"]

    # Find pieces in lower half with capture opportunities
    pieces_with_captures = []
    for piece in chess_pieces:
        square = piece.get("board_square")
        if square and square in LOWER_HALF_SQUARES:
            piece_color = get_piece_color(piece["id"])
            legal_captures = get_legal_capture_moves(piece["id"], square, piece_color)
            if legal_captures:
                pieces_with_captures.append({"id": piece["id"], "square": square, "color": piece_color, "legal_captures": legal_captures})

    if not pieces_with_captures:
        return False

    # Select random capture to execute
    selected_piece = random.choice(pieces_with_captures)
    target_square, target_piece_id = random.choice(selected_piece["legal_captures"])
    capturing_piece_id = selected_piece["id"]
    capturing_square = selected_piece["square"]

    # Move captured piece to table side
    target_piece_info = board_state[target_square]
    target_board_y = square_to_board_coordinates(target_square)[1]
    table_side = "left" if target_board_y > 0 else "right"
    table_target_y = 0.35 if table_side == "left" else -0.35
    target_piece_size = robot.get_object_size_in_world_frame(target_piece_id)
    table_target_z = table_top_z + 0.01 + target_piece_size[2] / 2.0
    table_target_pos = np.array([table_pos[0] - 0.1, table_target_y, table_target_z])

    # Generate poses and execute pick-place for captured piece
    arm_target = robot.judge_arm_to_use(target_piece_info["position"])
    grasp_config = GraspConfig(is_from_up_side=True, pick_up_distance=0.12, unmovable_objects=[chessboard_id], action_description={"zh": "拿起被吃的棋子", "en": "pick up captured piece"})
    place_config = PlaceConfig(pre_insert_offset=0.1, unmovable_objects=[chessboard_id], action_description={"zh": "放到桌子侧面", "en": "place on table side"})

    target_poses = [np.eye(4) for _ in range(10)]  # Generate candidate poses
    success_step1 = robot.pick_and_place_actor_to_target_pose(target_piece_id, np.array(target_poses), arm_target, grasp_config, place_config)

    if not success_step1:
        return False

    # Move capturing piece to target square
    target_coords = square_to_board_coordinates(target_square)
    chessboard_height = chessboard_size[2]
    chessboard_surface_z = chessboard_pos[2] + chessboard_height / 2.0
    capturing_piece_size = robot.get_object_size_in_world_frame(capturing_piece_id)
    capturing_target_z = chessboard_surface_z + 0.01 + capturing_piece_size[2] / 2.0
    capturing_target_pos = np.array([chessboard_pos[0] + target_coords[0], chessboard_pos[1] + target_coords[1], capturing_target_z])

    arm_capturing = robot.judge_arm_to_use(selected_piece["position"])
    # Execute second pick-place...

    return success
```
