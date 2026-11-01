# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_billiards_and_box
description: Several billiard balls scattered on a table with a storage box, ready to be stored
"""


def generate_random_positions_no_collision(num_items, x_range, y_range, min_distance=0.06):
    positions = []
    for _ in range(num_items):
        for attempt in range(100):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            valid = True
            for px, py in positions:
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < min_distance:
                    valid = False
                    break
            if valid:
                positions.append((x, y))
                break
        else:
            positions.append((x, y))
    return positions


@register()
def table_with_billiards_and_box() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "desktop_with_legs"],
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    x_min = table_center[0] - 0.11
    x_max = table_center[0] + 0.146
    y_min = table_center[1] - 0.22
    y_max = table_center[1] + 0.22

    storage_box_candidates = [
        "benchmark_storage_box_000", "benchmark_storage_box_001", "benchmark_storage_box_002",
        "benchmark_storage_box_003", "benchmark_storage_box_004", "benchmark_storage_box_005",
        "benchmark_storage_box_006", "benchmark_storage_box_007", "benchmark_storage_box_008",
        "benchmark_storage_box_009", "benchmark_storage_box_010", "benchmark_storage_box_011",
        "benchmark_storage_box_012", "benchmark_storage_box_013", "benchmark_storage_box_014",
        "benchmark_storage_box_015", "benchmark_storage_box_016", "benchmark_storage_box_017",
    ]
    selected_box = random.choice(storage_box_candidates)

    box_shape = library_call(
        "usd",
        oid=selected_box,
        keywords=["storage_box", "container", "box", "target_container"],
    )
    box_info = get_object_info(box_shape)
    box_half_x = box_info["size"][0] / 2
    box_half_y = box_info["size"][1] / 2

    box_x = random.uniform(x_min + box_half_x, x_max - box_half_x)
    box_y = random.uniform(table_center[1] - 0.05, table_center[1] + 0.05)

    box_angle = random.uniform(0, 2 * math.pi)
    box_rotated = transform_shape(
        box_shape,
        rotation_matrix(box_angle, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    box_final = transform_shape(box_rotated, translation_matrix([box_x, box_y, table_top_z]))

    billiards_candidates = [
        "benchmark_billiards_000", "benchmark_billiards_001", "benchmark_billiards_002",
        "benchmark_billiards_003", "benchmark_billiards_004", "benchmark_billiards_005",
        "benchmark_billiards_006", "benchmark_billiards_007", "benchmark_billiards_008",
        "benchmark_billiards_009", "benchmark_billiards_010", "benchmark_billiards_011",
        "benchmark_billiards_012", "benchmark_billiards_013", "benchmark_billiards_014",
        "benchmark_billiards_015",
    ]
    num_balls = random.randint(1, 2)
    selected_balls = random.sample(billiards_candidates, num_balls)

    ball_y_min = y_min
    ball_y_max = box_y - box_half_y - 0.04
    ball_y_min2 = box_y + box_half_y + 0.04
    ball_y_max2 = y_max

    ball_shapes = []
    positions = []
    for i, ball_id in enumerate(selected_balls):
        if random.random() < 0.5 and ball_y_max > ball_y_min:
            region = (ball_y_min, ball_y_max)
        else:
            region = (ball_y_min2, ball_y_max2)

        for attempt in range(50):
            bx = random.uniform(x_min + 0.03, x_max - 0.03)
            by = random.uniform(region[0] + 0.03, region[1] - 0.03)
            valid = True
            for px, py in positions:
                if ((bx - px) ** 2 + (by - py) ** 2) ** 0.5 < 0.06:
                    valid = False
                    break
            if valid:
                positions.append((bx, by))
                break
        else:
            positions.append((bx, by))

        position_tag = "left" if by > table_center[1] else "right"
        ball_shape = library_call(
            "usd",
            oid=ball_id,
            keywords=[f"billiard_ball_{i}", "billiard", "ball", "on_desk", position_tag],
        )
        ball_final = transform_shape(ball_shape, translation_matrix([bx, by, table_top_z]))
        ball_shapes.append(ball_final)

    return concat_shapes(table_shape, box_final, *ball_shapes)


@register()
def root_scene() -> Shape:
    return library_call("table_with_billiards_and_box")
