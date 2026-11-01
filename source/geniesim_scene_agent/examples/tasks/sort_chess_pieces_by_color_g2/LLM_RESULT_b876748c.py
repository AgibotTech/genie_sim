# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import math

# White: 000-005 (P,R,N,B,Q,K) | Black: 006-011 (P,R,N,B,Q,K)
CHESS_PIECE_MAPPING = {
    "pawn_white":   "benchmark_chess_000",
    "rook_white":   "benchmark_chess_001",
    "knight_white": "benchmark_chess_002",
    "bishop_white": "benchmark_chess_003",
    "queen_white":  "benchmark_chess_004",
    "king_white":   "benchmark_chess_005",
    "pawn_black":   "benchmark_chess_006",
    "rook_black":   "benchmark_chess_007",
    "knight_black": "benchmark_chess_008",
    "bishop_black": "benchmark_chess_009",
    "queen_black":  "benchmark_chess_010",
    "king_black":   "benchmark_chess_011",
}

PIECE_VALUE = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}

# 8x8 grid; squares are 0.05m on a side, board center at (0,0)
SQUARE_TO_COORD = {
    f"{f}{r}": (-0.175 + 0.05 * (int(r) - 1), 0.175 - 0.05 * (ord(f) - ord("A")))
    for f in "ABCDEFGH" for r in "12345678"
}
ALL_SQUARES = list(SQUARE_TO_COORD.keys())

def square_to_index(square):
    return (int(square[1]) - 1, ord(square[0]) - ord("A"))

def are_squares_adjacent(s1, s2):
    r1, f1 = square_to_index(s1)
    r2, f2 = square_to_index(s2)
    return s1 != s2 and abs(r1 - r2) <= 1 and abs(f1 - f2) <= 1

def select_non_adjacent(pool, n, max_tries=200):
    """Pick n squares, no two adjacent. Falls back to nearest valid if exhausted."""
    pool = list(pool)
    for _ in range(max_tries):
        random.shuffle(pool)
        chosen = []
        for sq in pool:
            if all(not are_squares_adjacent(sq, c) for c in chosen):
                chosen.append(sq)
                if len(chosen) == n:
                    return chosen
    # fallback: greedy
    chosen = []
    for sq in pool:
        if all(not are_squares_adjacent(sq, c) for c in chosen):
            chosen.append(sq)
            if len(chosen) == n:
                break
    return chosen

def get_desktop_pose(desk_shape):
    desk_info = get_object_info(desk_shape)
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center = desk_info["center"] + desktop_subpart["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart["xyz_max"][2]
    return np.array([desktop_center[0], desktop_center[1], desktop_top_z]), desktop_subpart["size"]

def make_chessboard_centered(desk_shape):
    desktop_center, _ = get_desktop_pose(desk_shape)
    board_shape = library_call(
        "usd", oid="benchmark_chessboard_000",
        keywords=["chessboard", "brown", "square", "centered_on_desk"],
    )
    board_info = get_object_info(board_shape)
    board_h = min(board_info["size"])
    board_z = desktop_center[2] + board_h / 2.0 + 0.03
    placed = transform_shape(board_shape, translation_matrix([desktop_center[0], desktop_center[1], board_z]))
    return placed, desktop_center, board_z

def piece_on_square(piece_type, square, board_center_xy, board_top_z, extra_kws=None):
    """Place a single chess piece centered on a chessboard square."""
    extra_kws = extra_kws or []
    color = "white" if "white" in piece_type else "black"
    name_root = piece_type.split("_")[0]
    kws = [f"chess_{piece_type}_{square}", piece_type, color, "chess_piece",
           name_root, f"position_{square}", f"value_{PIECE_VALUE[name_root]}"]
    kws.extend(extra_kws)
    sx, sy = SQUARE_TO_COORD[square]
    wx = board_center_xy[0] + sx
    wy = board_center_xy[1] + sy
    side = "left" if wy >= board_center_xy[1] else "right"
    if side not in kws:
        kws.append(side)
    shape = library_call("usd", oid=CHESS_PIECE_MAPPING[piece_type], keywords=kws)
    shape = transform_shape(shape, translation_matrix([wx, wy, board_top_z]))
    angle = random.uniform(-math.pi, math.pi)
    center = compute_shape_center(shape)
    return transform_shape(shape, rotation_matrix(angle, direction=(0, 0, 1), point=center))

def piece_at_world_xy(piece_type, wx, wy, table_top_z, name_suffix, extra_kws=None):
    """Place a single chess piece freely on the table at (wx, wy)."""
    extra_kws = extra_kws or []
    color = "white" if "white" in piece_type else "black"
    name_root = piece_type.split("_")[0]
    side = "left" if wy >= 0 else "right"
    kws = [f"chess_{piece_type}_{name_suffix}", piece_type, color, "chess_piece",
           name_root, f"value_{PIECE_VALUE[name_root]}", "scattered", "on_desk", side] + extra_kws
    shape = library_call("usd", oid=CHESS_PIECE_MAPPING[piece_type], keywords=kws)
    shape = transform_shape(shape, translation_matrix([wx, wy, table_top_z]))
    angle = random.uniform(-math.pi, math.pi)
    center = compute_shape_center(shape)
    return transform_shape(shape, rotation_matrix(angle, direction=(0, 0, 1), point=center))

def random_scatter_pieces(piece_specs, desk_shape, x_range=None, y_range=None, margin=0.06, max_attempts=200):
    """piece_specs: list of (piece_type, name_suffix, extra_kws). Returns list of Shape."""
    desktop_center, desktop_size = get_desktop_pose(desk_shape)
    table_top_z = desktop_center[2]
    if x_range is None:
        x_range = (desktop_center[0] - 0.10, desktop_center[0] + 0.13)
    if y_range is None:
        y_range = (desktop_center[1] - 0.20, desktop_center[1] + 0.20)
    placed = []  # (cx, cy, hx, hy)
    out = []
    for spec in piece_specs:
        piece_type, suffix, extra = spec
        # estimate half-extents from a sample placement
        sample = library_call("usd", oid=CHESS_PIECE_MAPPING[piece_type], keywords=["__probe__"])
        info = get_object_info(sample)
        hx = max(info["size"][0], 0.025) / 2 + margin
        hy = max(info["size"][1], 0.025) / 2 + margin
        for _ in range(max_attempts):
            cx = random.uniform(x_range[0] + hx, x_range[1] - hx)
            cy = random.uniform(y_range[0] + hy, y_range[1] - hy)
            if all(abs(cx - px) > (hx + phx) or abs(cy - py) > (hy + phy) for px, py, phx, phy in placed):
                placed.append((cx, cy, hx, hy))
                out.append(piece_at_world_xy(piece_type, cx, cy, table_top_z, suffix, extra))
                break
    return out

"""
scene_name: sort_chess_pieces_by_color_g2
description: Scattered chess pieces directly on the table (no chessboard). Mix of black and white pieces (3-4 of each color). Random non-overlapping placement with margin >= 2.5cm to ensure gripper clearance.
"""

@register()
def create_scattered_pieces(desk_shape: Shape) -> Shape:
    n_white = random.randint(3, 4)
    n_black = random.randint(3, 4)
    white_types = ["pawn_white", "knight_white", "bishop_white", "rook_white", "queen_white"]
    black_types = ["pawn_black", "knight_black", "bishop_black", "rook_black", "queen_black"]
    specs = []
    for i in range(n_white):
        specs.append((random.choice(white_types), f"w{i}", ["sortable_by_color"]))
    for i in range(n_black):
        specs.append((random.choice(black_types), f"b{i}", ["sortable_by_color"]))
    random.shuffle(specs)
    return concat_shapes(*random_scatter_pieces(specs, desk_shape))

@register()
def sort_chess_pieces_by_color_g2() -> Shape:
    desk = library_call("usd", oid="table_000", keywords=["workspace_table", "table", "wood"])
    pieces = library_call("create_scattered_pieces", desk_shape=desk)
    return concat_shapes(desk, pieces)

@register()
def root_scene() -> Shape:
    return library_call("sort_chess_pieces_by_color_g2")
