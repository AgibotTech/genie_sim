# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math


def place_books() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.78834, -0.15, 1.0])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "benchmark_book_00",
            "keywords": ["right"],
        },
        {
            "oid": "benchmark_book_01",
            "keywords": ["right"],
        },
        {
            "oid": "benchmark_book_02",
            "keywords": ["right"],
        },
        # {
        #     "oid": "benchmark_book_03",
        #     "keywords": ["right"],
        # },
        {
            "oid": "benchmark_book_04",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.0
    range_x_min = workspace_center[0] - 0.03 + margin
    range_x_max = workspace_center[0] + 0.03 - margin
    range_y_min = workspace_center[1] - 0.03 + margin
    range_y_max = workspace_center[1] + 0.03 - margin

    # Generate a random position on the table surface.
    rand_x = np.random.uniform(range_x_min, range_x_max)
    rand_y = np.random.uniform(range_y_min, range_y_max)

    # The z position is the top of the table.
    placement_pos = (rand_x, rand_y, workspace_center[2] + 0.01)

    # First, translate the block to the random position on the table.
    books = []
    for p in [-0.06, -0.03, 0, 0.03]:
        # choose one item
        item_chosen = random.choice(items_to_place)
        # Load the block shape.
        book_shape = library_call("usd", oid=item_chosen["oid"], keywords=item_chosen["keywords"])

        translated_book_shape = transform_shape(
            book_shape,
            translation_matrix((0.0, p, 0.18)),
        )
        rotated_book_shape = transform_shape(
            translated_book_shape,
            rotation_matrix(
                angle=-math.pi / 2.0 + math.pi / 30,
                direction=(1, 0, 0),
                point=compute_shape_center(translated_book_shape),
            ),
        )
        rotated_book_shape2 = transform_shape(
            rotated_book_shape,
            rotation_matrix(
                angle=math.pi,
                direction=(0, 0, 1),
                point=compute_shape_center(rotated_book_shape),
            ),
        )
        books.append(rotated_book_shape2)

    holder_with_books = concat_shapes(*books)

    translated_holder_with_books = transform_shape(holder_with_books, translation_matrix(placement_pos))

    # Then, apply a random rotation around its new center's Z-axis.
    # random_angle = np.random.uniform(0, math.pi / 90)
    # final_block = transform_shape(
    #     translated_holder_with_books,
    #     rotation_matrix(
    #         angle=random_angle,
    #         direction=(0, 0, 1),
    #         point=compute_shape_center(translated_holder_with_books),
    #     ),
    # )

    return concat_shapes(translated_holder_with_books)


@register()
def place_magzine() -> Shape:
    # Define the valid placement
    placement_pos = np.array([0.63831, 0.08116, 0.75])
    block_shape = library_call(
        "usd",
        oid="sft_book_000",
        keywords=["magzine"],
    )
    translated_block = transform_shape(block_shape, translation_matrix(placement_pos))
    # magzine_block_r = transform_shape(
    #     magzine_block_t,
    #     rotation_matrix(
    #         angle=math.pi, direction=(0, 0, 1), point=compute_shape_center(magzine_block_t)
    #     ),
    # )

    block_center = compute_shape_center(translated_block)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=-math.pi / 2, direction=(0, 0, 1), point=block_center),
    )

    return final_block


@register()
def study_room_pick_and_place_book_on_shelf2() -> Shape:

    placed_blocks = []

    holder_shape = library_call(
        "usd",
        oid="benchmark_bookshelf_000",
        keywords=["book_holder"],
    )
    translated_holder_shape = transform_shape(holder_shape, translation_matrix((0.55832, -0.3, 0.75)))
    placed_blocks.append(translated_holder_shape)

    placed_blocks.append(place_magzine())
    placed_blocks.append(place_books())

    return concat_shapes(*placed_blocks)


@register()
def root_scene() -> Shape:
    """
    The root function that generates the entire scene.
    """
    return study_room_pick_and_place_book_on_shelf2()
