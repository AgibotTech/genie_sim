# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Background-room requirements and measured floor heights for benchmark tasks.

Tasks whose foreground lacks a support surface or manipulated background
object require the room. The explicit task table avoids guessing from asset
names. When a room is loaded, ``resolve_base_z`` uses its measured floor
height to avoid spawning the robot inside the collider.

Foreground-only rendering supplies a dome light; its lighting still differs
from the benchmark room even when the room is unnecessary for physics.
"""

from __future__ import annotations


# Value: does this subtask need the background. Comment: the reason (the
# support present in the foreground, or what is missing).
TASK_NEEDS_BACKGROUND: dict[str, bool] = {
    # ---- Needs the background: the foreground has no supporting surface ----
    "clean_the_desktop": True,  # laptop, mouse, pen holder, bin; no desk
    "hold_pot": True,  # only a pot; the stove is in the room
    "open_door": True,  # the door IS background geometry
    "pour_workpiece": True,  # workpiece and box only; bench is in the room
    "scoop_popcorn": True,  # the machine IS background geometry
    "sorting_packages": True,  # cartons only; the sorting table is in the room
    "sorting_packages_continuous": True,  # same, cartons hang in mid-air without it
    "stock_and_straighten_shelf": True,  # bottles only; the shelf is in the room
    "take_wrong_item_shelf": True,  # bottles only; the shelf is in the room
    # ---- No background needed: the foreground brings its own support ----
    # These share one static table (collision geometry, no rigid body), with
    # a surface at about z=0.88.
    "bimanual_chip_handover": False,
    "pack_in_supermarket": False,
    "pick_billiards_color": False,
    "pick_block_color": False,
    "pick_block_number": False,
    "pick_block_shape": False,
    "pick_block_size": False,
    "pick_common_sense": False,
    "pick_follow_logic_or": False,
    "pick_object_absolute_position": False,
    "pick_object_type": False,
    "pick_specific_object": False,
    "place_block_into_box": False,
    "place_object_into_box_color": False,
    "place_object_relative_position": False,
    "sort_cubes_by_size": False,
    "sort_fruit": False,
    "sort_number": False,
    "stack_bowls": False,
    "stack_three_building_blocks": False,
    "straighten_object": False,
    # A different support, but still part of the foreground:
    "pick_object_relative_position": False,  # edited/table_000
    "place_beverage_to_anothers_position": False,  # edited/table_000
    "place_block_into_drawer": False,  # drawer cabinet with a fixed joint
    "geniesim_place_workpiece": False,  # the workstation is sublayered into the foreground
}


def resolve_use_background(
    sub_task_name: str,
    default: bool = False,
) -> tuple[bool, str]:
    """Return (load_background, source) for a task.

    The explicit task table takes precedence because it records required scene
    support. Use default only for tasks absent from the table."""
    if sub_task_name in TASK_NEEDS_BACKGROUND:
        return TASK_NEEDS_BACKGROUND[sub_task_name], "table"
    return bool(default), "default"


# ---------------------------------------------------------------------------
# Second fact: how high the background room's floor is
# ---------------------------------------------------------------------------
#
# The robot asset is authored with the floor at z=0, while the benchmark's
# ``robot_init_pose`` carries only XY and yaw: exactly one task authors a
# non-zero z, and it equals that room's floor height. The field was meant to
# encode the floor and mostly went unfilled.
#
# That is harmless under PhysX, which clamps depenetration velocity. Under
# mjwarp the same overlap becomes an impulse: a few millimetres is enough to
# fling an object, and a floor tens of millimetres off diverges the robot.
#
# Value: the highest top surface among colliding meshes below the robot's
# spawn XY within 0.3 m, capped at 0.30 m. ``None`` means the room has no
# floor collider there and the robot stands on the engine's own z=0 ground.
#
# Not computed at runtime: measuring requires the collision API and a
# "touching the ground" threshold, and getting it wrong is silent — the robot
# sinks or floats. A test pins the table against the assets.
ROOM_FLOOR_Z: dict[str, "float | None"] = {
    "background/home/home_3": 0.03869,  # open_door, the worst case
    "background/study_room/study_4": 0.02128,  # clean_the_desktop
    "background/warehouse/warehouse_1": 0.00245,  # sorting_packages(_continuous)
    "background/popcorn/popcorn_1": 0.00245,  # scoop_popcorn; the benchmark got this one right
    "background/laboratory/laboratory_2": 0.00095,  # pour_workpiece
    "background/kitchen/kitchen_1": 0.00000,  # hold_pot, an exact match
    "background/market/market_1": None,  # take_wrong_item_shelf: visual floor only
    "background/market/market_2": None,  # stock_and_straighten_shelf: same
}

# Clearance left above the floor after lifting. Same value as
# ``depenetrate_spawn``: both solve the same problem, so one number.
BASE_CLEARANCE = 0.002


def room_key(background_usda: str) -> str:
    """Derive the ``ROOM_FLOOR_Z`` key from a background USD path.

    ``/geniesim_assets/background/home/home_3/background.usda``
    Only the part after ``/background/`` is used, since the asset root
    differs between container and host. Returns an empty string when the path
    does not match.
    """
    parts = [x for x in str(background_usda or "").replace("\\", "/").split("/") if x]
    if "background" not in parts:
        return ""
    # The last path SEGMENT named background — not a substring, since a file
    # named background.usda would otherwise be taken for the directory.
    i = len(parts) - 1 - parts[::-1].index("background")
    rest = parts[i + 1 : -1]  # drop the trailing file name
    return "/".join(["background", *rest]) if rest else ""


def resolve_base_z(room: str, benchmark_z: float) -> "tuple[float, str]":
    """Return a nonpenetrating robot-base height and its source.

    Keep benchmark_z when it is above the measured room floor; otherwise lift the
    base. Never lower a pose that already clears the floor."""
    if room not in ROOM_FLOOR_Z:
        return float(benchmark_z), (
            f"room {room!r} is not in ROOM_FLOOR_Z; keeping the benchmark z "
            f"of {benchmark_z:.5f} rather than guessing an unmeasured floor"
        )
    floor = ROOM_FLOOR_Z[room]
    if floor is None:
        return float(benchmark_z), (
            f"{room} has no floor collider (visual only); the robot stands "
            f"on the engine's z=0 ground and z stays {benchmark_z:.5f}"
        )
    if float(benchmark_z) >= float(floor):
        return float(benchmark_z), (
            f"{room} floor top is z={floor:.5f}; the benchmark z of "
            f"{benchmark_z:.5f} does not penetrate, leaving it unchanged"
        )
    new_z = float(floor) + BASE_CLEARANCE
    return new_z, (
        f"{room} floor top is z={floor:.5f}; the benchmark z of "
        f"{benchmark_z:.5f} sinks {(floor - benchmark_z) * 1000:.1f}mm, "
        f"lifting to {new_z:.5f}"
        f"({BASE_CLEARANCE * 1000:.0f}mm clearance)"
    )


__all__ = [
    "BASE_CLEARANCE",
    "ROOM_FLOOR_Z",
    "TASK_NEEDS_BACKGROUND",
    "resolve_base_z",
    "resolve_use_background",
    "room_key",
]
