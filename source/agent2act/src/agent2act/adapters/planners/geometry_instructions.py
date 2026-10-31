# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Prompt references to supplied gripper calibration, independent of task strategies."""

TABLE_CONTACT = "For flat or low objects resting on the table, first assess whether both inner pad faces can engage the object's opposing sides above the support surface. Do not assume that straddling the outline establishes a grasp. For a straight-down approach: Never command TCP below table_surface_z_m + fingertip_beyond_tcp_m. For a direct tabletop side pinch, keep the open/close axis parallel to the table so both pads share one height; also check the full finger clearance during closure. A different orientation requires checking its rotated finger geometry against the support surface. When present, grasp_geometry.tcp_axes names the jaw open/close axis, aperture_by_command maps gripper commands to pad gap, and gripper_notes must be followed before closing."

TABLE_CONTACT_WITHOUT_HEIGHT = "For flat or low objects, use current depth and views to assess object and finger-pad height. Do not infer a metric table or object height from RGB alone; if depth is missing or invalid, keep the open gripper clear and gather a useful observation. For a direct tabletop side pinch, keep the open/close axis parallel to the visible support surface so both pads share one height; check full finger clearance during closure. Follow grasp_geometry.gripper_notes before closing."


def with_grasp_geometry(instructions: str, table_height_available: bool = True) -> str:
    instructions = instructions.replace(
        "Use supplied calibration and height_prior when present; do not invent table/base heights.",
        "Use supplied calibration, height_prior and grasp_geometry when present; do not invent "
        "table/base heights or finger geometry.",
    )
    table_contact = TABLE_CONTACT if table_height_available else TABLE_CONTACT_WITHOUT_HEIGHT
    return instructions + "\n\nGRASP GEOMETRY\n" + table_contact
