# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

# Asset OIDs
CHESS_OIDS = [f"benchmark_chess_{i:03d}" for i in range(12)]
CHESS_OID_TO_TYPE = {
    "benchmark_chess_000": ("pawn",   "white", 1),
    "benchmark_chess_001": ("rook",   "white", 5),
    "benchmark_chess_002": ("knight", "white", 3),
    "benchmark_chess_003": ("bishop", "white", 3),
    "benchmark_chess_004": ("queen",  "white", 9),
    "benchmark_chess_005": ("king",   "white", 0),
    "benchmark_chess_006": ("pawn",   "black", 1),
    "benchmark_chess_007": ("rook",   "black", 5),
    "benchmark_chess_008": ("knight", "black", 3),
    "benchmark_chess_009": ("bishop", "black", 3),
    "benchmark_chess_010": ("queen",  "black", 9),
    "benchmark_chess_011": ("king",   "black", 0),
}

SQUARE_TO_COORD = {
    f"{f}{r}": (-0.175 + 0.05 * (int(r) - 1), 0.175 - 0.05 * (ord(f) - ord("A")))
    for f in "ABCDEFGH" for r in "12345678"
}

def square_to_board_coords(square):
    return SQUARE_TO_COORD.get(square)

def piece_meta_from_usd(usd):
    for oid, meta in CHESS_OID_TO_TYPE.items():
        if oid in usd:
            return oid, meta
    return None, (None, None, None)

def find_objects(scene_info):
    """Return (chessboard_id_or_None, table_id_or_None, storage_box_ids, chess_piece_records)."""
    chessboard_id = None
    table_id = None
    storage_box_ids = []
    chess_pieces = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        kws = obj_info.get("keywords", [])
        if "benchmark_chessboard" in usd:
            chessboard_id = obj_id
            continue
        if usd == "table_000" or "table" in kws or "workspace_table" in kws:
            table_id = obj_id
            continue
        if "benchmark_storage_box" in usd:
            storage_box_ids.append(obj_id)
            continue
        oid, (ptype, color, value) = piece_meta_from_usd(usd)
        if oid is None:
            continue
        sq = None
        for kw in kws:
            if kw.startswith("position_") and len(kw) >= 11:
                cand = kw.replace("position_", "")
                if cand in SQUARE_TO_COORD:
                    sq = cand
                    break
        chess_pieces.append({
            "id": obj_id, "usd": usd, "oid": oid, "keywords": kws,
            "piece_type": ptype, "color": color, "value": value, "square": sq,
        })
    return chessboard_id, table_id, storage_box_ids, chess_pieces

def make_pose_grid(target_xyz, base_rot_3x3, n_rot=12, xy_offsets=(0.0,), z_offsets=(0.0, 0.005)):
    """Generate >=3 candidate poses around target_xyz with randomized z-axis rotation."""
    poses = []
    angles = np.linspace(0.0, 2 * np.pi, n_rot, endpoint=False)
    for dz in z_offsets:
        for dx in xy_offsets:
            for dy in xy_offsets:
                for a in angles:
                    rz = np.array([[np.cos(a), -np.sin(a), 0],
                                   [np.sin(a),  np.cos(a), 0],
                                   [0,          0,         1]])
                    rot = rz @ base_rot_3x3
                    p = np.eye(4)
                    p[:3, :3] = rot
                    p[:3, 3] = np.array([target_xyz[0] + dx, target_xyz[1] + dy, target_xyz[2] + dz])
                    poses.append(p)
    return np.array(poses)

def make_unrelated_static_checker(robot, unrelated_ids, xy_tol=0.025, z_tol=0.05):
    """Snapshot initial poses of unrelated pieces; returns checker that flags any displacement."""
    snapshots = {}
    for pid in unrelated_ids:
        pos, rot = robot.get_object_pose(pid)
        if pos is not None:
            snapshots[pid] = (np.array(pos, dtype=float), rot)

    def _check():
        for pid, (init_pos, _) in snapshots.items():
            cur_pos, _ = robot.get_object_pose(pid)
            if cur_pos is None:
                continue
            cur = np.array(cur_pos, dtype=float)
            dxy = np.linalg.norm(cur[:2] - init_pos[:2])
            dz = abs(cur[2] - init_pos[2])
            if dxy > xy_tol or dz > z_tol:
                print(f"[unrelated-piece-disturbed] {pid}: dxy={dxy:.3f}, dz={dz:.3f}")
                return False
        return True

    return _check

def piece_zh_name(ptype, color):
    zh_t = {"pawn":"兵","rook":"车","knight":"马","bishop":"象","queen":"后","king":"王"}.get(ptype, "棋子")
    zh_c = {"white":"白","black":"黑"}.get(color, "")
    return f"{zh_c}{zh_t}"

OBJECTS_LIST_BASE = {
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

def execute_task(robot: RobotAPI):
    """Sort scattered chess pieces into two color piles. White pieces go left, black pieces go right (or randomized)."""
    objects_list = dict(OBJECTS_LIST_BASE)
    objects_list.pop("benchmark_chessboard_000", None)

    scene_info = robot.scene_info
    _, table_id, _, pieces = find_objects(scene_info)
    if table_id is None or not pieces:
        print("[sort_color] table or pieces not found")
        return False

    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    description_json = {
        "task_name": "Sort the scattered chess pieces into two piles by color",
        "english_task_name": "Sort the scattered chess pieces into two piles by color",
        "init_scene_text": "Chess pieces of mixed colors are scattered on the table.",
    }
    robot.client.set_task_task_basic_info(description_json)

    # Decide pile centers: white -> +y side, black -> -y side
    white_center = np.array([table_pos[0] + 0.0, table_pos[1] + 0.30])
    black_center = np.array([table_pos[0] + 0.0, table_pos[1] - 0.30])

    def linear_offsets(n, spacing=0.06):
        return [(i - (n - 1) / 2.0) * spacing for i in range(n)]

    whites = [p for p in pieces if p["color"] == "white"]
    blacks = [p for p in pieces if p["color"] == "black"]
    white_xs = linear_offsets(len(whites))
    black_xs = linear_offsets(len(blacks))

    overall_ok = True

    def do_one(piece_list, center, x_offsets, side_zh, side_en):
        nonlocal overall_ok
        for piece, dx in zip(piece_list, x_offsets):
            pid = piece["id"]
            others = [p["id"] for p in pieces if p["id"] != pid]
            psize = robot.get_object_size_in_world_frame(pid)
            target = np.array([center[0] + dx, center[1], table_top_z + psize[2] / 2.0 + 0.01])
            ppos, prot = robot.get_object_pose(pid)
            base_rot = quat2mat_wxyz(prot)
            poses = make_pose_grid(target, base_rot, n_rot=8, z_offsets=(0.0, 0.01))
            arm = robot.judge_arm_to_use(ppos)
            unmovable = [table_id] + others
            static_checker = make_unrelated_static_checker(robot, others)
            zh_name = piece_zh_name(piece["piece_type"], piece["color"])
            grasp_cfg = GraspConfig(
                is_from_up_side=True, pick_up_distance=0.12,
                unmovable_objects=list(unmovable),
                action_description={"zh": f"拿起{zh_name}", "en": f"pick up the {piece['color']} {piece['piece_type']}"},
            )
            place_cfg = PlaceConfig(
                pre_insert_offset=0.05,
                unmovable_objects=list(unmovable),
                extra_checkers=[static_checker],
                action_description={"zh": f"把{zh_name}放到{side_zh}堆",
                                    "en": f"place the {piece['color']} {piece['piece_type']} on the {side_en} pile"},
            )
            ok = robot.pick_and_place_actor_to_target_pose(
                actor=pid, target_pose=poses, arm=arm,
                grasp_config=grasp_cfg, place_config=place_cfg,
            )
            robot.back_to_origin(arm=arm)
            if not ok:
                overall_ok = False
                print(f"[sort_color] failed on {pid}")

    do_one(whites, white_center, white_xs, "白", "white")
    do_one(blacks, black_center, black_xs, "黑", "black")
    return overall_ok
