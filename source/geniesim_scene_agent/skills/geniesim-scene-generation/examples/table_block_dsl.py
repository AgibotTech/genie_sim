# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Minimal foreground GenieSim-style DSL recipe for the bundled runtime."""

from helper import *


@register()
def root_scene(params=None) -> Shape:
    params = params or {}
    table = library_call("usd", oid="table_000", keywords=["table"])
    block = library_call("usd", oid="benchmark_building_blocks_007",
                         keywords=["building_block"])
    tabletop_z = get_object_info(table)["max"][2]
    block = transform_shape(
        block,
        translation_matrix((params.get("block_x", 0.0), 0.0, tabletop_z)),
    )
    return concat_shapes(table, block)
