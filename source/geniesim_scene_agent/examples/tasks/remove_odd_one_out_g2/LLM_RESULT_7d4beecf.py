# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: remove_odd_one_out_g2
description: A group of 3-4 similar objects on the table with 1 odd object that does not belong.
The robot must identify and pick up the odd one.
Scenario types: fruits with odd-one, blocks with different color, billiards with odd fruit, pens with odd block, etc.
"""

FRUIT_GROUPS = {
    "apple": ["benchmark_apple_000", "benchmark_apple_001", "benchmark_apple_002", "benchmark_apple_003"],
    "orange": ["benchmark_orange_001", "benchmark_orange_002", "benchmark_orange_004"],
    "lemon": ["benchmark_lemon_027", "benchmark_lemon_028", "benchmark_lemon_029", "benchmark_lemon_030"],
    "peach": ["benchmark_peach_000", "benchmark_peach_019", "benchmark_peach_020", "benchmark_peach_021"],
    "pomegranate": ["benchmark_pomegranate_000", "benchmark_pomegranate_001", "benchmark_pomegranate_046", "benchmark_pomegranate_047"],
    "green_apple": ["benchmark_green_apple_000", "benchmark_green_apple_001"],
}

BLOCK_POOL = {
    "red": "benchmark_building_blocks_073",
    "blue": "benchmark_building_blocks_011",
    "yellow": "benchmark_building_blocks_085",
    "purple": "benchmark_building_blocks_089",
}

BILLIARD_POOL = [
    "benchmark_billiards_001", "benchmark_billiards_002", "benchmark_billiards_003",
    "benchmark_billiards_004", "benchmark_billiards_005", "benchmark_billiards_009",
    "benchmark_billiards_010", "benchmark_billiards_011", "benchmark_billiards_012",
    "benchmark_billiards_013",
]

PEN_POOL = [
    "benchmark_pen_000", "benchmark_pen_001", "benchmark_pen_002",
    "benchmark_pen_003", "benchmark_pen_005", "benchmark_pen_006",
]

MISC_ODD_POOL = [
    ("benchmark_toy_car_018", "toy_car"),
    ("benchmark_toy_plane_029", "toy_plane"),
    ("benchmark_snack_241", "snack"),
    ("benchmark_snack_242", "snack"),
    ("benchmark_cup_043", "cup"),
    ("benchmark_cup_105", "cup"),
    ("benchmark_beverage_bottle_001", "beverage"),
    ("benchmark_beverage_bottle_002", "beverage"),
    ("benchmark_beverage_bottle_003", "beverage"),
]

SCENARIOS = [
    "fruits_with_odd",
    "blocks_with_different",
    "billiards_with_odd",
    "pens_with_odd",
]


@register()
def remove_odd_one_out_g2() -> Shape:
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    scenario = random.choice(SCENARIOS)

    items = []
    if scenario == "fruits_with_odd":
        fruit_type = random.choice(list(FRUIT_GROUPS.keys()))
        type_pool = FRUIT_GROUPS[fruit_type]
        num_normal = random.choice([3, 4])
        normal_items = random.sample(type_pool, min(num_normal, len(type_pool)))
        while len(normal_items) < num_normal:
            normal_items.append(random.choice(type_pool))
        for oid in normal_items:
            items.append((oid, "normal", fruit_type))

        odd_choice = random.choice(["block", "misc"])
        if odd_choice == "block":
            odd_color = random.choice(list(BLOCK_POOL.keys()))
            odd_oid = BLOCK_POOL[odd_color]
            items.append((odd_oid, "odd", odd_color + "_block"))
        else:
            odd_oid, odd_cat = random.choice(MISC_ODD_POOL)
            items.append((odd_oid, "odd", odd_cat))

    elif scenario == "blocks_with_different":
        colors = list(BLOCK_POOL.keys())
        main_color = random.choice(colors)
        odd_color = random.choice([c for c in colors if c != main_color])
        num_normal = random.choice([3, 4])
        main_oid = BLOCK_POOL[main_color]
        for _ in range(num_normal):
            items.append((main_oid, "normal", main_color))
        items.append((BLOCK_POOL[odd_color], "odd", odd_color))

    elif scenario == "billiards_with_odd":
        num_normal = random.choice([3, 4])
        normal_balls = random.sample(BILLIARD_POOL, min(num_normal, len(BILLIARD_POOL)))
        for oid in normal_balls:
            items.append((oid, "normal", "billiard"))

        odd_choice = random.choice(["fruit", "block"])
        if odd_choice == "fruit":
            fruit_type = random.choice(list(FRUIT_GROUPS.keys()))
            odd_oid = random.choice(FRUIT_GROUPS[fruit_type])
            items.append((odd_oid, "odd", fruit_type))
        else:
            odd_color = random.choice(list(BLOCK_POOL.keys()))
            items.append((BLOCK_POOL[odd_color], "odd", odd_color + "_block"))

    else:  # pens_with_odd
        num_normal = random.choice([3, 4])
        normal_pens = random.sample(PEN_POOL, min(num_normal, len(PEN_POOL)))
        for oid in normal_pens:
            items.append((oid, "normal", "pen"))

        odd_choice = random.choice(["fruit", "block", "billiard"])
        if odd_choice == "fruit":
            fruit_type = random.choice(list(FRUIT_GROUPS.keys()))
            odd_oid = random.choice(FRUIT_GROUPS[fruit_type])
            items.append((odd_oid, "odd", fruit_type))
        elif odd_choice == "block":
            odd_color = random.choice(list(BLOCK_POOL.keys()))
            items.append((BLOCK_POOL[odd_color], "odd", odd_color + "_block"))
        else:
            odd_oid = random.choice(BILLIARD_POOL)
            items.append((odd_oid, "odd", "billiard"))

    random.shuffle(items)

    positions = []
    shapes = [table_shape]

    for i, (oid, role, category) in enumerate(items):
        placed = False
        for _ in range(200):
            x = random.uniform(-0.05, 0.18)
            y = random.uniform(-0.20, 0.20)
            ok = True
            for px, py in positions:
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < 0.08:
                    ok = False
                    break
            if ok:
                positions.append((x, y))
                placed = True
                break
        if not placed:
            x = -0.05 + (i % 3) * 0.10
            y = -0.15 + (i // 3) * 0.15
            positions.append((x, y))

        x, y = positions[-1]
        pos_tag = "left" if y > table_center[1] else "right"
        kws = [role, category, "on_desk", pos_tag]
        if "block" in category or "building_blocks" in oid:
            kws.append("building_block")
        else:
            kws.append("fruit")

        obj_shape = library_call("usd", oid=oid, keywords=kws)
        obj_shape = transform_shape(obj_shape, translation_matrix([x, y, table_top_z]))
        if "building_blocks" in oid:
            angle = random.uniform(-math.pi / 6, math.pi / 6)
            center = compute_shape_center(obj_shape)
            obj_shape = transform_shape(obj_shape, rotation_matrix(angle, direction=(0, 0, 1), point=center))
        shapes.append(obj_shape)

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return remove_odd_one_out_g2()
