# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_water_and_person
description: A table with a randomly selected water bottle placed within a specific region, and a person sitting in front of the table.
"""


@register()
def place_water_bottle_on_table(table_shape: Shape) -> Shape:
    """
    Places a randomly selected water bottle on the table within the specified XY range.
    """
    # Get table's top surface info
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Calculate the world position of the desktop's top surface center
    desktop_center_world = table_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    # Select a random water bottle asset
    water_bottle_assets = [
        "benchmark_beverage_bottle_001",
        "benchmark_beverage_bottle_002",
        "benchmark_beverage_bottle_003",
        "benchmark_beverage_bottle_004",
        "benchmark_beverage_bottle_008",
        "benchmark_beverage_bottle_023",
        "benchmark_beverage_bottle_024",
        "benchmark_beverage_bottle_025",
        "benchmark_beverage_bottle_026",
        "benchmark_beverage_bottle_029",
        "benchmark_beverage_bottle_073"
    ]
    selected_bottle = random.choice(water_bottle_assets)

    # Load the bottle
    bottle_shape = library_call(
        "usd",
        oid=selected_bottle,
        keywords=["water_bottle", "transparent", "cylinder", "drinking", "on_table"]
    )

    # Get bottle height to place it correctly on the table
    bottle_min = compute_shape_min(bottle_shape)
    bottle_height = compute_shape_sizes(bottle_shape)[2]
    bottle_bottom_offset = -bottle_min[2]  # offset to bring bottom to z=0

    # Random placement within the specified range relative to table center
    x_offset = random.uniform(-0.15, 0.2)   # Updated x range: (-0.15, 0.2)
    y_offset = random.uniform(-0.25, 0.25)  # y range remains: (-0.25, 0.25)
    bottle_position = desktop_pos + np.array([x_offset, y_offset, bottle_bottom_offset])

    # Transform bottle to correct position
    bottle_transformed = transform_shape(
        bottle_shape,
        translation_matrix(bottle_position.tolist())
    )

    return concat_shapes(table_shape, bottle_transformed)


@register()
def place_person_in_front_of_table(scene_with_bottle: Shape) -> Shape:
    """
    Places a randomly selected person in front of the table at specified location and orientation.
    """
    # Select a random character
    character_assets = [
        "benchmark_character_000",
        "benchmark_character_001",
        "benchmark_character_002",
        "benchmark_character_003",
        "benchmark_character_004",
        "benchmark_character_005",
        "benchmark_character_006",
        "benchmark_character_007",
        "benchmark_character_008",
        "benchmark_character_009",
        "benchmark_character_010",
        "benchmark_character_011",
        "benchmark_character_012",
        "benchmark_character_013",
        "benchmark_character_014"
    ]
    selected_character = random.choice(character_assets)

    # Load the character
    character_shape = library_call(
        "usd",
        oid=selected_character,
        keywords=["person", "human", "sitting", "in_front_of_table"]
    )

    # Set position: x=0.75 (in front of table), z=0 (on ground), y random in (-0.17, 0.17)
    y_pos = random.uniform(-0.17, 0.17)
    character_position = [0.75, y_pos, 0.0]

    # The given quaternion is (w,x,y,z) = (0,0,0,1) which corresponds to xyzw = (0,0,1,0)
    # This represents a 180-degree rotation around Z-axis (facing the table)
    angle, axis = quaternion_to_angle_direction([0.0, 0.0, 1.0, 0.0])

    # First translate, then rotate around the character's base
    character_translated = transform_shape(
        character_shape,
        translation_matrix(character_position)
    )
    character_info = get_object_info(character_translated)
    character_base = np.array([character_info["center"][0], character_info["center"][1], 0.0])

    character_final = transform_shape(
        character_translated,
        rotation_matrix(angle, direction=axis, point=character_base)
    )

    return concat_shapes(scene_with_bottle, character_final)


@register()
def table_with_water_and_person() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["dining_table", "table", "wooden", "four_legs"]
    )

    # Place water bottle on table
    scene_with_bottle = place_water_bottle_on_table(table_shape)

    # Place person in front
    final_scene = place_person_in_front_of_table(scene_with_bottle)

    return final_scene


@register()
def root_scene() -> Shape:
    return table_with_water_and_person()
