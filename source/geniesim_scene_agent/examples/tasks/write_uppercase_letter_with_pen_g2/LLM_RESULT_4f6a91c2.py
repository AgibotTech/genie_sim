# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

import numpy as np


"""
scene_name: writing_scene
description: A writing desk with kinematic paper and a dynamic pen holder
             containing benchmark_pen_009.
"""


PAPER_XY = (0.02, 0.12)
PEN_HOLDER_XY = (0.10, -0.38)


def _place_on_table(shape: Shape, xy: tuple[float, float], table_top_z: float, z_offset: float = 0.0) -> Shape:
    """Place a library shape on the table; library_call already lifts its bbox bottom to z=0."""
    return transform_shape(
        shape,
        translation_matrix((xy[0], xy[1], table_top_z + z_offset)),
    )


@register()
def writing_scene() -> Shape:
    # No XY transform: table_000 is centred at (0, 0) and its bbox bottom is
    # automatically aligned to world z=0 by library_call.
    table = library_call(
        "usd",
        oid="table_000",
        keywords=["writing_table", "table", "workspace", "center"],
    )
    table_info = get_object_info(table)
    desktop = get_subpart_info(object_id="table_000", subpart_id="desktop")
    table_top_z = float(table_info["center"][2] + desktop["xyz_max"][2])

    # benchmark_paper_000 carries physics:kinematicEnabled=1 in its asset.
    paper = library_call(
        "usd",
        oid="benchmark_paper_000",
        keywords=["paper", "writing_surface", "kinematic", "on_table"],
    )
    paper = _place_on_table(paper, PAPER_XY, table_top_z)

    # This pen holder is an explicit dynamic rigid body and is intentionally
    # left non-kinematic. Place it near one side of the tabletop.
    pen_holder = library_call(
        "usd",
        oid="benchmark_stationery_021",
        keywords=["pen_holder", "dynamic", "non_kinematic", "table_side"],
    )
    pen_holder = _place_on_table(pen_holder, PEN_HOLDER_XY, table_top_z)

    # benchmark_pen_009 is already vertical after its asset up-axis transform.
    # A small lift avoids starting inside the holder's solid bottom; gravity
    # will settle the dynamic pen into the holder.
    pen = library_call(
        "usd",
        oid="benchmark_pen_009",
        keywords=["writing_pen", "pen", "inside_pen_holder", "dynamic"],
    )
    pen = _place_on_table(pen, PEN_HOLDER_XY, table_top_z, z_offset=0.006)

    return concat_shapes(table, paper, pen_holder, pen)


@register()
def root_scene() -> Shape:
    return writing_scene()
