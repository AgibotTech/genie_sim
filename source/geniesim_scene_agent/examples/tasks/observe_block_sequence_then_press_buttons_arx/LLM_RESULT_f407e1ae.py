# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import random

"""
scene_name: block_in_front_of_one_of_three_buttons
description: Three identical-size but different-color buttons placed in a row at
the table center (left / middle / right). A single building block is placed in
front of one randomly chosen button (same y as that button, x further toward the
robot). The chosen button carries a "has_block_in_front" tag and the block carries
a "in_front_of_<position>_button" tag so data_gen.py can recover the pairing.

User-specified coordinate ranges (relative to table center):
  - left   button: x in (-0.05, -0.03), y in (-0.12, -0.08)
  - middle button: x in (-0.05, -0.03), y in (-0.02,  0.02)
  - right  button: x in (-0.05, -0.03), y in ( 0.08,  0.12)
  - block (placed at chosen button's y, on +X side): y = chosen button y, x in (0.18, 0.22)
"""

# Three same-size buttons with different primary colors (dimensions 0.0998 x 0.0675 x 0.0998 m).
BUTTON_CANDIDATES = [
    ("benchmark_button_000", "red"),
    ("benchmark_button_012", "green"),
    ("benchmark_button_013", "blue"),
]

BUILDING_BLOCK_CANDIDATES = [
    "benchmark_building_blocks_073",
    "benchmark_building_blocks_085",
    "benchmark_building_blocks_089",
]

# X range shared by all three button slots (relative to table center).
BUTTON_X_RANGE = (-0.05, -0.03)

# Y range of the middle button (left/right are derived from it + a center
# spacing so that the gap between adjacent buttons is at least 0.05 m).
MIDDLE_Y_RANGE = (-0.02, 0.02)

# Center-to-center spacing between adjacent buttons (m). The buttons have a
# Y-extent of ~0.0998 m, so 0.15–0.18 m of center spacing gives a gap of
# roughly 0.05–0.08 m between neighbours.
SIDE_SPACING_RANGE = (0.15, 0.18)

# Per-button yaw jitter (degrees).
BUTTON_YAW_DEG_RANGE = (-5.0, 5.0)

BLOCK_X_RANGE = (0.18, 0.22)

# Minimum surface-to-surface gap (m) between the hint block and the target
# button after both have been placed (accounts for the buttons' actual rotated
# AABB plus the block's diagonal half-extent under random yaw).
BLOCK_TO_BUTTON_MIN_GAP = 0.02


def _place_button(button_oid: str, color: str, position_tag: str,
                  has_block_in_front: bool,
                  x_local: float, y_local: float,
                  table_center, table_top_z: float) -> Shape:
    keywords = [
        f"{position_tag}_button",
        "button",
        "control",
        color,
        position_tag,
        button_oid,
    ]
    if has_block_in_front:
        keywords.append("has_block_in_front")
        keywords.append("target")
    btn = library_call("usd", oid=button_oid, keywords=keywords)
    # Small yaw jitter (±5°) around the button's own center for visual variety
    # without breaking the row alignment.
    btn_center_local = compute_shape_center(btn)
    yaw = np.deg2rad(random.uniform(*BUTTON_YAW_DEG_RANGE))
    btn = transform_shape(
        btn, rotation_matrix(yaw, (0, 0, 1), tuple(btn_center_local))
    )
    info = get_object_info(btn)
    cur_center = info["center"]
    btn_half_h = info["size"][2] / 2.0
    target = (
        table_center[0] + x_local,
        table_center[1] + y_local,
        table_top_z + btn_half_h,
    )
    delta = (target[0] - cur_center[0],
             target[1] - cur_center[1],
             target[2] - cur_center[2])
    return transform_shape(btn, translation_matrix(delta))


def _place_block(in_front_of_position: str, x_local: float, y_local: float,
                 table_center, table_top_z: float) -> Shape:
    block_oid = random.choice(BUILDING_BLOCK_CANDIDATES)
    block = library_call(
        "usd",
        oid=block_oid,
        keywords=[
            "hint_block",
            "building_block",
            "cube",
            f"in_front_of_{in_front_of_position}_button",
            in_front_of_position,
            "target_indicator",
        ],
    )
    # Random yaw for visual variety.
    block_center = compute_shape_center(block)
    yaw = random.uniform(-np.pi, np.pi)
    block = transform_shape(
        block, rotation_matrix(yaw, (0, 0, 1), tuple(block_center))
    )

    info = get_object_info(block)
    cur_center = info["center"]
    block_half_h = info["size"][2] / 2.0
    target = (
        table_center[0] + x_local,
        table_center[1] + y_local,
        table_top_z + block_half_h,
    )
    delta = (target[0] - cur_center[0],
             target[1] - cur_center[1],
             target[2] - cur_center[2])
    return transform_shape(block, translation_matrix(delta))


@register()
def three_buttons_with_hint_block() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "rectangular", "furniture"],
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    # Pick which slot gets the block (uniform across the three positions).
    target_idx = random.randint(0, 2)

    # Assign the three button assets to the three slots in random order so the
    # color↔position mapping varies across scenes. Order: [left, middle, right].
    button_assignment = random.sample(BUTTON_CANDIDATES, k=3)

    # Place the middle button first, then derive left/right from it using an
    # independent random spacing on each side (guarantees ≥ 0.05 m edge gap
    # between neighbours given the buttons' ~0.0998 m Y-extent).
    middle_y = random.uniform(*MIDDLE_Y_RANGE)
    left_spacing = random.uniform(*SIDE_SPACING_RANGE)
    right_spacing = random.uniform(*SIDE_SPACING_RANGE)
    slots = [
        ("left",   random.uniform(*BUTTON_X_RANGE), middle_y + left_spacing),
        ("middle", random.uniform(*BUTTON_X_RANGE), middle_y),
        ("right",  random.uniform(*BUTTON_X_RANGE), middle_y - right_spacing),
    ]

    shapes = [table_shape]
    chosen_y_local = None
    chosen_position = None
    chosen_button_shape = None
    for idx, ((position_tag, x_local, y_local),
              (button_oid, color)) in enumerate(zip(slots, button_assignment)):
        is_target = (idx == target_idx)
        placed_button = _place_button(
            button_oid, color, position_tag, is_target,
            x_local, y_local, table_center, table_top_z,
        )
        shapes.append(placed_button)
        if is_target:
            chosen_y_local = y_local
            chosen_position = position_tag
            chosen_button_shape = placed_button

    # Use the target button's actual world AABB to derive the block's allowed
    # X range, guaranteeing a ≥ BLOCK_TO_BUTTON_MIN_GAP surface clearance even
    # under the button's yaw jitter and the block's random yaw. The block sits
    # on the +X side of the button (further from the robot), so the constraint
    # is on the block's MINIMUM x, not its maximum.
    btn_world_info = get_object_info(chosen_button_shape)
    btn_max_x_world = btn_world_info["max"][0]
    block_diag_half = (0.04 * np.sqrt(2.0)) / 2.0  # block side 0.04 m
    block_x_min_world = btn_max_x_world + BLOCK_TO_BUTTON_MIN_GAP + block_diag_half
    block_x_min_local = block_x_min_world - table_center[0]
    block_x_lo, block_x_hi = BLOCK_X_RANGE
    block_x_lo = max(block_x_lo, block_x_min_local)
    block_x_hi = max(block_x_hi, block_x_lo)
    block_x_local = random.uniform(block_x_lo, block_x_hi)
    shapes.append(
        _place_block(
            chosen_position, block_x_local, chosen_y_local,
            table_center, table_top_z,
        )
    )

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return library_call("three_buttons_with_hint_block")
