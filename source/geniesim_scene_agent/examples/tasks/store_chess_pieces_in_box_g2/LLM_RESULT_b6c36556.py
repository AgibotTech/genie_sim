# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: chessboard_with_target_piece_v3
description: A chessboard (benchmark_chessboard_001) is placed randomly on a table (table_000).
             A target piece position is reserved inside the board's container area (Local Coords).
             The target piece is placed outside the board (Global Coords).
             The target piece is tagged with the GLOBAL world coordinates of the reserved internal spot.
             Constraints:
             - Board placed on table center: x(-0.1, 0.05), y(-0.1, 0.1).
             - Internal placement area (Local): x(-0.12, 0.12), y(-0.04, 0.04).
             - Target external placement area (Global): x(-0.2, 0.1), y(-0.25, 0.25).
             - Target must be >= 0.05m from board edges and other pieces.
"""

CHESS_PIECES = [
    "benchmark_chess_000", "benchmark_chess_001", "benchmark_chess_002",
    "benchmark_chess_003", "benchmark_chess_004", "benchmark_chess_005",
    "benchmark_chess_006", "benchmark_chess_007", "benchmark_chess_008",
    "benchmark_chess_009", "benchmark_chess_010", "benchmark_chess_011"
]

def get_random_piece_id():
    return random.choice(CHESS_PIECES)

@register()
def chessboard_scene() -> Shape:
    # 1. Setup Table
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workplace"])
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]

    # 2. Setup Chessboard (Container)
    # Random position on table center area: x(-0.1, 0.05), y(-0.1, 0.1)
    board_pos_x = random.uniform(-0.1, 0.05)
    board_pos_y = random.uniform(-0.1, 0.1)
    board_pos = np.array([board_pos_x, board_pos_y, table_top_z])

    # Random rotation around Z axis
    board_rot_z = random.uniform(0, 2 * np.pi)

    # Load board and apply transforms
    base_board_shape = library_call("usd", oid="benchmark_chessboard_001", keywords=["chessboard", "container", "wooden"])

    # Rotation matrix (around Z, at origin, then translate)
    rot_mat = rotation_matrix(board_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
    trans_mat = translation_matrix(board_pos)

    board_shape = transform_shape(base_board_shape, rot_mat)
    board_shape = transform_shape(board_shape, trans_mat)

    # Get board world info
    board_world_info = get_object_info(board_shape)
    board_surface_z = board_world_info["max"][2] # Top surface Z

    # Board AABB in world coordinates (XY plane)
    board_min_x = board_world_info["min"][0]
    board_max_x = board_world_info["max"][0]
    board_min_y = board_world_info["min"][1]
    board_max_y = board_world_info["max"][1]

    # Helper: Local to World conversion
    def local_to_world(lx, ly):
        wx = board_pos_x + lx * np.cos(board_rot_z) - ly * np.sin(board_rot_z)
        wy = board_pos_y + lx * np.sin(board_rot_z) + ly * np.cos(board_rot_z)
        return wx, wy

    # 3. Define Reserved Target Position (Inside the box - Local Coords)
    # Internal range: x(-0.120, 0.120), y(-0.040, 0.040)
    local_x_min, local_x_max = -0.120, 0.120
    local_y_min, local_y_max = -0.040, 0.040

    target_local_x = random.uniform(local_x_min, local_x_max)
    target_local_y = random.uniform(local_y_min, local_y_max)

    # Calculate World Coordinates for the Tag (Tag must be GLOBAL coordinates)
    target_world_x, target_world_y = local_to_world(target_local_x, target_local_y)

    # Tag string using GLOBAL world coordinates
    pos_tag = f"position_x_{target_world_x:.3f}_y_{target_world_y:.3f}"

    # 4. Place Internal Pieces (Other pieces, avoiding the reserved spot)
    # Total pieces 1-5. One is target (placed outside). Remaining are placed inside.
    num_total_pieces = random.randint(1, 5)
    num_internal_pieces = num_total_pieces - 1

    internal_pieces_shapes = []
    internal_piece_world_pos = [] # Store (x, y) for collision check

    piece_width = 0.04
    safe_distance = 0.05 # Minimum distance between pieces

    for _ in range(num_internal_pieces):
        pid = get_random_piece_id()
        piece_shape = library_call("usd", oid=pid, keywords=["chess_piece", "internal", "random"])

        placed = False
        while not placed:
            lx = random.uniform(local_x_min, local_x_max)
            ly = random.uniform(local_y_min, local_y_max)

            wx, wy = local_to_world(lx, ly)

            # Check collision with reserved target spot (must leave this empty)
            dist_to_target = np.sqrt((wx - target_world_x)**2 + (wy - target_world_y)**2)
            if dist_to_target < safe_distance:
                continue

            # Check collision with other internal pieces
            collision = False
            for (ox, oy) in internal_piece_world_pos:
                dist = np.sqrt((wx - ox)**2 + (wy - oy)**2)
                if dist < safe_distance:
                    collision = True
                    break

            if not collision:
                internal_piece_world_pos.append((wx, wy))
                placed = True

        # Transform piece to world
        piece_rot_z = random.uniform(0, 2 * np.pi)
        p_rot_mat = rotation_matrix(piece_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
        p_trans_mat = translation_matrix((wx, wy, board_surface_z))
        final_piece = transform_shape(piece_shape, p_rot_mat)
        final_piece = transform_shape(final_piece, p_trans_mat)
        internal_pieces_shapes.append(final_piece)

    # 5. Place Target Piece (Outside the board - Global Coords)
    # Constraints:
    # a. Global Boundary: x (-0.2, 0.1), y (-0.25, 0.25)
    # b. Must be outside the board (distance to board edges >= 0.05)
    # c. Distance to other pieces (internal) >= 0.05

    target_found = False
    target_wx_final, target_wy_final = 0.0, 0.0

    # Global External Bounds
    global_ext_x_min, global_ext_x_max = -0.2, 0.1
    global_ext_y_min, global_ext_y_max = -0.25, 0.25

    max_attempts = 1000

    while not target_found and max_attempts > 0:
        max_attempts -= 1

        # Sample in Global Coordinates
        wx = random.uniform(global_ext_x_min, global_ext_x_max)
        wy = random.uniform(global_ext_y_min, global_ext_y_max)

        # Check Distance to Board Edges
        # Calculate distance from point (wx, wy) to board AABB
        dx = max(board_min_x - wx, 0, wx - board_max_x)
        dy = max(board_min_y - wy, 0, wy - board_max_y)
        dist_edge = np.sqrt(dx*dx + dy*dy)

        if dist_edge < 0.05:
            continue

        # Check Distance to Internal Pieces
        valid_dist = True
        for (px, py) in internal_piece_world_pos:
            dist = np.sqrt((wx - px)**2 + (wy - py)**2)
            if dist < 0.05:
                valid_dist = False
                break

        if valid_dist:
            target_found = True
            target_wx_final, target_wy_final = wx, wy

    if not target_found:
        # Fallback: place it further out if random search failed
        # Place it at a safe distance along the X axis
        target_wx_final = board_max_x + 0.15
        target_wy_final = board_pos_y

    # Create Target Shape with the Tag (tag points to the internal GLOBAL spot)
    target_pid = get_random_piece_id()
    target_shape = library_call("usd", oid=target_pid, keywords=["chess_piece", "target", "white", "highlighted", pos_tag])

    # Transform Target to its External Position
    target_rot_z = random.uniform(0, 2 * np.pi)
    t_rot_mat = rotation_matrix(target_rot_z, direction=(0, 0, 1), point=(0, 0, 0))
    t_trans_mat = translation_matrix((target_wx_final, target_wy_final, board_surface_z))

    final_target = transform_shape(target_shape, t_rot_mat)
    final_target = transform_shape(final_target, t_trans_mat)

    # 6. Combine Shapes
    return concat_shapes(table_shape, board_shape, *internal_pieces_shapes, final_target)

@register()
def root_scene() -> Shape:
    return chessboard_scene()
