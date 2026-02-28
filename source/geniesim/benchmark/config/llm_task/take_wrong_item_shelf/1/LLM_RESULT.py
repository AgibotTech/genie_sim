# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: random_drink_placement
description: 在指定区域内随机摆放一个cola或者fanta
"""


@register()
def place_random_drink() -> Shape:
    """
    在指定区域内随机摆放一个cola或者fanta
    x范围: -2.91 到 -2.96
    y范围: -0.65 到 -0.95
    z高度: 0.88
    """
    x_min, x_max = -2.96, -2.93
    y_min, y_max = -0.95, -0.65
    z_height = 0.88

    drink_options = [
        ("benchmark_beverage_bottle_001", "cola", "coke cola"),
        ("benchmark_beverage_bottle_003", "fanta", "fanta"),
    ]
    selected_index = np.random.choice(len(drink_options))
    selected_oid, selected_name, selected_keyword = drink_options[selected_index]

    random_x = np.random.uniform(x_min, x_max)
    random_y = np.random.uniform(y_min, y_max)
    random_position = np.array([random_x, random_y, z_height])

    position_tag = "left" if random_y > 0 else "right"

    drink_shape = library_call(
        "usd",
        oid=selected_oid,
        keywords=[selected_keyword, "beverage", "bottle", "drink", position_tag],
    )

    object_info = get_object_info(drink_shape)
    object_center = object_info["center"]

    final_shape = transform_shape(drink_shape, translation_matrix(random_position - object_center))

    return final_shape


@register()
def root_scene() -> Shape:
    return place_random_drink()
