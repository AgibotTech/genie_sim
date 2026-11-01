# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: pick_object_by_relative_position_g2
description: A table with 3-6 randomly selected objects placed within a specified range.
One type of object appears twice and is marked as 'target'. All objects are randomly rotated,
non-colliding (min distance 0.15m), and labeled with both Chinese and English names.
"""


def get_desktop_workspace(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the desktop in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def sample_non_colliding_positions(
    num_objects: int,
    workspace_center: P,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_dist: float = 0.15,
) -> List[P]:
    """
    Sample non-colliding positions for objects on the table.
    """
    positions = []
    max_attempts = 1000

    for i in range(num_objects):
        for _ in range(max_attempts):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            z = workspace_center[2]
            candidate = np.array([x, y, z])

            # Check distance to all previously placed objects
            collision = False
            for pos in positions:
                if np.linalg.norm(candidate[:2] - pos[:2]) < min_dist:
                    collision = True
                    break

            if not collision:
                positions.append(candidate)
                break
        else:
            # If we can't find a non-colliding position, just place it (should rarely happen with given constraints)
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            positions.append(np.array([x, y, workspace_center[2]]))

    return positions


@register()
def create_tabletop_object(oid: str, zh_name: str, en_name: str, is_target: bool = False) -> Shape:
    """
    Create a single object with proper labeling.
    """
    keywords = [en_name.replace(" ", "_"), "object", en_name, zh_name]
    if is_target:
        keywords.append("target")

    return library_call("usd", oid=oid, keywords=keywords)


@register()
def place_objects_on_table(desk_shape: Shape) -> Shape:
    # Available objects with their Chinese and English names
    available_objects = {
        "benchmark_beverage_bottle_001": {"zh": "可口可乐", "en": "coke cola"},
        "benchmark_beverage_bottle_003": {"zh": "芬达", "en": "fanta"},
        "benchmark_beverage_bottle_004": {"zh": "乌龙茶饮料", "en": "oolong tea beverage"},
        "genie_beverage_bottle_009": {"zh": "葡萄汁", "en": "grape juice"},
        "benchmark_toy_car_018": {"zh": "玩具车", "en": "toy car"},
        "benchmark_building_blocks_072": {"zh": "积木", "en": "building blocks"},
        "benchmark_cup_043": {"zh": "马克杯", "en": "mug"},
        "benchmark_cup_105": {"zh": "纸杯", "en": "paper cup"},
        "benchmark_toiletries_009": {"zh": "洗面奶", "en": "face cleanser"},
        "benchmark_stationery_009": {"zh": "黄色订书机", "en": "yellow stapler"},
        "benchmark_mouse_000": {"zh": "鼠标", "en": "mouse"},
        "benchmark_bottle_017": {"zh": "保温杯", "en": "water bottle"},
        "benchmark_apple_002": {"zh": "苹果", "en": "apple"},
        "benchmark_orange_004": {"zh": "橙子", "en": "orange"},
        "benchmark_lemon_030": {"zh": "柠檬", "en": "lemon"},
        "benchmark_peach_021": {"zh": "桃子", "en": "peach"},
        "benchmark_green_apple_001": {"zh": "青苹果", "en": "green apple"},
        "benchmark_bowl_sponge_000": {"zh": "洗碗海绵", "en": "bowl sponge"},
        "benchmark_clock_044": {"zh": "闹钟", "en": "clock"},
        "benchmark_tape_measure_007": {"zh": "卷尺", "en": "tape measure"},
        "benchmark_toy_plane_029": {"zh": "玩具飞机", "en": "toy plane"},
        "benchmark_rubik_cube_001": {"zh": "魔方", "en": "rubik cube"},
        "benchmark_medicine_bottle_000": {"zh": "药瓶", "en": "medicine bottle"},
        "benchmark_garage_kit_064": {"zh": "小火龙摆件", "en": "charmander figure"},
    }

    # Determine total number of objects (3-6)
    total_objects = random.randint(3, 6)

    # Select which object will appear twice (target object)
    target_oid = random.choice(list(available_objects.keys()))
    target_info = available_objects[target_oid]

    # Create list of object IDs to place
    object_ids = [target_oid, target_oid]  # Two instances of target object

    # Fill remaining slots with unique objects (excluding the target)
    remaining_slots = total_objects - 2
    other_objects = [oid for oid in available_objects.keys() if oid != target_oid]
    for _ in range(remaining_slots):
        object_ids.append(random.choice(other_objects))

    # Get desktop workspace
    workspace_center, workspace_size = get_desktop_workspace(desk_shape)

    # Define placement range (x: -0.2 to 0.2, y: -0.25 to 0.25 relative to workspace center)
    x_min = workspace_center[0] - 0.2
    x_max = workspace_center[0] + 0.2
    y_min = workspace_center[1] - 0.25
    y_max = workspace_center[1] + 0.25

    # Sample non-colliding positions
    positions = sample_non_colliding_positions(len(object_ids), workspace_center, (x_min, x_max), (y_min, y_max))

    # Create and place all objects
    placed_objects = []
    for i, oid in enumerate(object_ids):
        is_target = oid == target_oid
        obj_shape = library_call(
            "create_tabletop_object",
            oid=oid,
            zh_name=available_objects[oid]["zh"],
            en_name=available_objects[oid]["en"],
            is_target=is_target,
        )

        # Get object size for proper placement
        obj_info = get_object_info(obj_shape)
        obj_height = obj_info["size"][2]

        # Position at sampled location (adjusting for object height if needed)
        placement_pos = positions[i].copy()
        placement_pos[2] = workspace_center[2] - 0.1  # Place on table surface

        # Apply translation
        obj_shape = transform_shape(obj_shape, translation_matrix(placement_pos))

        # Apply random rotation around Z-axis
        obj_center = compute_shape_center(obj_shape)
        rotation_angle = random.uniform(0, 2 * math.pi)
        obj_shape = transform_shape(obj_shape, rotation_matrix(rotation_angle, direction=(0, 0, 1), point=obj_center))

        placed_objects.append(obj_shape)

    return concat_shapes(desk_shape, *placed_objects)


@register()
def pick_object_by_relative_position_g2() -> Shape:
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_objects_on_table(desk_shape)


@register()
def root_scene() -> Shape:
    return pick_object_by_relative_position_g2()
