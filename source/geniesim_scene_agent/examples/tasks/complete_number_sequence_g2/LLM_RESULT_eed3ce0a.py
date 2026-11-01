# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: table_with_digit_sequence_missing_one_with_candidates
description: A table with a sequence of consecutive digit blocks arranged in a row with one gap. 2-3 candidate digit blocks (including the missing one) are placed separately on the table. The task is to find which candidate fills the gap.
"""

DIGIT_TO_ASSET = {
    0: "benchmark_building_blocks_034",
    1: "benchmark_building_blocks_035",
    2: "benchmark_building_blocks_036",
    3: "benchmark_building_blocks_037",
    4: "benchmark_building_blocks_038",
    5: "benchmark_building_blocks_039",
    6: "benchmark_building_blocks_040",
    7: "benchmark_building_blocks_041",
    8: "benchmark_building_blocks_042",
    9: "benchmark_building_blocks_043",
}


@register()
def digit_block(digit: int, kw_extra: list) -> Shape:
    asset_id = DIGIT_TO_ASSET[digit]
    return library_call("usd", oid=asset_id, keywords=[f"digit_{digit}"] + kw_extra)


@register()
def table_with_sequence() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["workspace_table", "table"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]

    SAFE_DIGITS = [0, 1, 2, 3, 4, 5, 7, 8]
    seq_len = random.randint(4, 5)
    valid_starts = [d for d in SAFE_DIGITS if all(d+i in SAFE_DIGITS for i in range(seq_len))]
    start_digit = random.choice(valid_starts)
    full_seq = [start_digit + i for i in range(seq_len)]
    missing_idx = random.randint(0, seq_len - 1)
    missing_digit = full_seq[missing_idx]
    seq_without_missing = [d for i, d in enumerate(full_seq) if i != missing_idx]

    row_x = random.uniform(-0.12, -0.06)
    start_y = 0.12
    spacing = 0.05
    shapes = []

    for i, d in enumerate(seq_without_missing):
        pos_idx = i if i < missing_idx else i + 1
        y = start_y - pos_idx * spacing
        side_tag = "left" if y > 0 else "right"
        block = library_call("digit_block", digit=d, kw_extra=["in_sequence", "building_block", side_tag])
        placed = transform_shape(block, translation_matrix([row_x, y, desktop_top_z]))
        rot_angle = random.uniform(170.0/180*np.pi, 190.0/180*np.pi)
        placed = transform_shape(placed, rotation_matrix(rot_angle, (0,0,1), [row_x, y, desktop_top_z]))
        shapes.append(placed)

    num_distractors = random.randint(1, 2)
    all_digits = set(SAFE_DIGITS)
    used_in_seq = set(full_seq)
    distractor_pool = list(all_digits - used_in_seq - {missing_digit})
    random.shuffle(distractor_pool)
    distractors = distractor_pool[:num_distractors]

    candidates = [missing_digit] + distractors
    random.shuffle(candidates)

    placed_positions = []
    candidate_x_min, candidate_x_max = 0.02, 0.12
    candidate_y_min, candidate_y_max = -0.15, 0.15

    for c_digit in candidates:
        side_tag = "candidate"
        block = library_call("digit_block", digit=c_digit, kw_extra=["candidate", "scattered", "building_block", side_tag])
        for _ in range(200):
            cx = random.uniform(candidate_x_min, candidate_x_max)
            cy = random.uniform(candidate_y_min, candidate_y_max)
            ok = all(
                np.sqrt((cx-px)**2 + (cy-py)**2) >= 0.06
                for px, py in placed_positions
            )
            if ok:
                placed_positions.append((cx, cy))
                t = transform_shape(block, translation_matrix([cx, cy, desktop_top_z]))
                rot_angle = random.uniform(0, 2*np.pi)
                t = transform_shape(t, rotation_matrix(rot_angle, (0,0,1), [cx, cy, desktop_top_z]))
                shapes.append(t)
                break

    return concat_shapes(table_shape, *shapes)


@register()
def root_scene() -> Shape:
    return library_call("table_with_sequence")
