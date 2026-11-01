# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

CHESS_PIECE_MAPPING = {
    "pawn_white": "benchmark_chess_000", "rook_white": "benchmark_chess_001",
    "knight_white": "benchmark_chess_002", "bishop_white": "benchmark_chess_003",
    "queen_white": "benchmark_chess_004", "king_white": "benchmark_chess_005",
    "pawn_black": "benchmark_chess_006", "rook_black": "benchmark_chess_007",
    "knight_black": "benchmark_chess_008", "bishop_black": "benchmark_chess_009",
    "queen_black": "benchmark_chess_010", "king_black": "benchmark_chess_011",
}
PIECE_NAMES = {v: k for k, v in CHESS_PIECE_MAPPING.items()}

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
    "E8": (0.175, -0.025), "F8": (0.175, -0.075), "G8": (0.175, -0.125), "H8": (0.175, -0.175),
}

def square_to_index(sq):
    return (int(sq[1]) - 1, ord(sq[0]) - ord('A'))

def index_to_square(row, col):
    if 0 <= row < 8 and 0 <= col < 8:
        return chr(ord('A') + col) + str(row + 1)
    return None

def get_piece_info(usd):
    for pname, pid in CHESS_PIECE_MAPPING.items():
        if pid in usd:
            color = "white" if "white" in pname else "black"
            ptype = pname.replace("_white", "").replace("_black", "")
            return pname, color, ptype
    return None, None, None

def parse_chess_scene(scene_info):
    table_id, chessboard_id = None, None
    pieces = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
        elif "benchmark_chessboard_000" in usd or "chessboard" in keywords:
            chessboard_id = obj_id
        else:
            pname, color, ptype = get_piece_info(usd)
            if pname:
                square = None
                for kw in keywords:
                    if kw.startswith("position_"):
                        square = kw[9:]
                        break
                pieces.append({"id": obj_id, "name": pname, "color": color, "type": ptype, "square": square, "keywords": keywords})
    return table_id, chessboard_id, pieces

def get_knight_moves(sq):
    r, c = square_to_index(sq)
    moves = []
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        s = index_to_square(r+dr, c+dc)
        if s:
            moves.append(s)
    return moves

def get_king_moves(sq):
    r, c = square_to_index(sq)
    moves = []
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            s = index_to_square(r+dr, c+dc)
            if s:
                moves.append(s)
    return moves

def get_sliding_moves(sq, directions):
    r, c = square_to_index(sq)
    moves = []
    for dr, dc in directions:
        nr, nc = r+dr, c+dc
        while 0<=nr<8 and 0<=nc<8:
            moves.append(index_to_square(nr, nc))
            nr += dr
            nc += dc
    return moves

def get_rook_moves(sq):
    return get_sliding_moves(sq, [(0,1),(0,-1),(1,0),(-1,0)])

def get_bishop_moves(sq):
    return get_sliding_moves(sq, [(1,1),(1,-1),(-1,1),(-1,-1)])

def get_queen_moves(sq):
    return get_rook_moves(sq) + get_bishop_moves(sq)

def get_all_moves(ptype, sq):
    if ptype == "knight": return get_knight_moves(sq)
    if ptype == "king": return get_king_moves(sq)
    if ptype == "rook": return get_rook_moves(sq)
    if ptype == "bishop": return get_bishop_moves(sq)
    if ptype == "queen": return get_queen_moves(sq)
    if ptype == "pawn": return []  # pawn moves handled specially
    return []

def gen_board_target_poses(chessboard_pos, target_sq, piece_rot, num_rotations=8):
    coord = SQUARE_TO_COORD.get(target_sq)
    if not coord:
        return np.array([np.eye(4)])
    tx, ty = coord
    rot_mat = quat2mat_wxyz(piece_rot)
    poses = []
    for dx in np.linspace(-0.005, 0.005, 5):
        for dy in np.linspace(-0.005, 0.005, 5):
            for dz in np.linspace(0, 0.002, 3):
                pos = chessboard_pos + np.array([tx+dx, ty+dy, dz])
                for angle in np.linspace(0, 2*np.pi, num_rotations, endpoint=False):
                    rz = euler2mat([0, 0, angle])
                    pose = np.eye(4)
                    pose[:3, :3] = np.dot(rz, rot_mat)
                    pose[:3, 3] = pos
                    poses.append(pose)
    return np.array(poses)

def is_square_attacked_by(sq, attacker_color, pieces):
    occupied = {p["square"]: p for p in pieces if p["square"]}
    for p in pieces:
        if p["color"] != attacker_color or not p["square"]:
            continue
        if p["type"] == "pawn":
            r, c = square_to_index(p["square"])
            d = 1 if p["color"] == "white" else -1
            for dc in [-1, 1]:
                if index_to_square(r+d, c+dc) == sq:
                    return True
        else:
            moves = get_all_moves(p["type"], p["square"])
            if sq in moves:
                return True
    return False

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": "king is in check, move it to a safe square",
        "english_task_name": "king is in check, move it to a safe square",
        "init_scene_text": "The robot is in front of the table with a chessboard and pieces"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id, chessboard_id, pieces = parse_chess_scene(scene_info)
    if not table_id or not chessboard_id:
        return False

    chessboard_pos, _ = robot.get_object_pose(chessboard_id)
    chessboard_size = robot.get_object_size_in_world_frame(chessboard_id)

    occupied = {p["square"] for p in pieces if p["square"]}
    kings = [p for p in pieces if p["type"] == "king" and p["square"]]

    king = None
    for k in kings:
        enemy_color = "black" if k["color"] == "white" else "white"
        if is_square_attacked_by(k["square"], enemy_color, pieces):
            king = k
            break

    if not king:
        king = kings[0] if kings else None
    if not king:
        return False

    enemy_color = "black" if king["color"] == "white" else "white"
    safe_squares = []
    for m in get_king_moves(king["square"]):
        if m not in occupied and not is_square_attacked_by(m, enemy_color, pieces):
            safe_squares.append(m)

    if not safe_squares:
        safe_squares = [m for m in get_king_moves(king["square"]) if m not in occupied]
    if not safe_squares:
        return False

    target_sq = random.choice(safe_squares)
    pos, rot = robot.get_object_pose(king["id"])
    arm = robot.judge_arm_to_use(pos)
    target_poses = gen_board_target_poses(chessboard_pos, target_sq, rot)
    unmovable = [table_id, chessboard_id] + [p["id"] for p in pieces if p["id"] != king["id"]]

    grasp_config = GraspConfig(
        is_from_up_side=True, pick_up_distance=0.12,
        unmovable_objects=unmovable,
        action_description={"zh": f"拿起{king['color']}国王", "en": f"picks up the {king['color']} king"}
    )
    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=unmovable,
        action_description={"zh": f"将国王移到安全格{target_sq}", "en": f"moves king to safe square {target_sq}"}
    )

    success = robot.pick_and_place_actor_to_target_pose(
        actor=king["id"], target_pose=target_poses, arm=arm,
        grasp_config=grasp_config, place_config=place_config
    )
    robot.back_to_origin(arm=arm)
    return success
