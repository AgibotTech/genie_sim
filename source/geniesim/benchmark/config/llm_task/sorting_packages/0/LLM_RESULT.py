# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: left_side_sorting_table
description: A large benchmark table (scaled 3x) on the left side of the scene, rotated 90 degrees counter-clockwise, with various cartons randomly selected from all available types (000-011, 016-030) scattered naturally on its surface in a single layer without stacking. Cartons come in various colors (brown, white, blue) with some shrink-wrapped in black plastic, arranged densely to mimic warehouse sorting operations.
"""


def find_tabletop_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface center and its size.
    """
    table_info = get_object_info(table_shape)

    # The tabletop center in world coordinates is the same as the table's center
    tabletop_center = table_info["center"]

    # The top surface z-coordinate of the tabletop in world coordinates
    tabletop_top_z = table_info["center"][2] + table_info["size"][2] / 2.0

    # The final tabletop position (center_x, center_y, top_z)
    tabletop_pos = np.array([tabletop_center[0], tabletop_center[1], tabletop_top_z])

    # Use the table's x and y dimensions as the tabletop size
    tabletop_size = np.array([table_info["size"][0], table_info["size"][1]])

    return tabletop_pos, tabletop_size


@register()
def random_carton() -> Shape:
    """
    Creates a single random carton with various types and colors.
    Includes all available carton types (000-011, 016-030) with different colors (brown, white, blue)
    and some shrink-wrapped with black plastic.
    Cartons are placed in a single layer without stacking.
    """
    # All available carton types (000-011, 016-030)
    carton_types = [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010",
        "011",
        "016",
        "017",
        "018",
        "019",
        "020",
        "021",
        "022",
        "023",
        "024",
        "025",
        "026",
        "027",
        "028",
        "029",
        "030",
    ]
    carton_type = np.random.choice(carton_types)
    oid = f"benchmark_carton_{carton_type}"

    # Random color selection - matches the image variety
    colors = ["brown", "white", "blue"]
    color = np.random.choice(colors, p=[0.5, 0.3, 0.2])  # 50% brown, 30% white, 20% blue

    # 20% chance of being shrink-wrapped in black plastic
    is_shrink_wrapped = np.random.random() < 0.2

    if is_shrink_wrapped:
        keywords = [
            "express_carton",
            color,
            "cardboard",
            "sealed",
            "shrink_wrapped",
            "black_plastic",
            f"type_{carton_type}",
        ]
    else:
        keywords = ["express_carton", color, "cardboard", "sealed", f"type_{carton_type}"]

    return library_call("usd", oid=oid, keywords=keywords)


@register()
def scattered_cartons_on_table(num_cartons: int, table_shape: Shape, excluded_zones=None, base_z_offset=0.0) -> Shape:
    """
    Places 4 small cartons (types: 030/029/028/020) in a single layer on the table surface.
    - Positions range stays the same as current code: x in [x_min, x_max] (original +X half), y full width with safety margin
    - Avoids excluded zones (large cartons)
    - base_z_offset: additional height offset to place cartons above table surface (keep your current usage)
    """
    tabletop_pos, tabletop_size = find_tabletop_surface(table_shape)

    safety_margin = 0.02  # 2cm margin

    placed_cartons = []

    # Fixed 4 small carton types, one each
    fixed_small_types = ["030", "029", "028", "020"]
    num_cartons = len(fixed_small_types)  # force 4

    # Y full width
    usable_width = tabletop_size[1] - 2 * safety_margin

    # Keep current range: SMALL cartons on BACK half in local table frame (original +X half)
    """x_min = tabletop_pos[0] + safety_margin
    x_max = tabletop_pos[0] + tabletop_size[0] / 2 - safety_margin
    usable_depth = x_max - x_min"""

    # Split table area along X with ratio SMALL:BIG = 1:2
    table_min_x = tabletop_pos[0] - tabletop_size[0] / 2
    table_max_x = tabletop_pos[0] + tabletop_size[0] / 2
    split_x = table_min_x + (2.0 / 3.0) * (table_max_x - table_min_x)  # big uses left 2/3, small uses right 1/3

    # SMALL cartons occupy the +X side (right 1/3)
    x_min = split_x + safety_margin
    x_max = table_max_x - safety_margin
    usable_depth = x_max - x_min

    # Grid for 4 items
    grid_cols = int(np.ceil(np.sqrt(num_cartons * usable_depth / usable_width)))
    grid_rows = int(np.ceil(num_cartons / max(grid_cols, 1)))
    if grid_cols * grid_rows < num_cartons:
        grid_cols += 1

    spacing_x = usable_depth / grid_cols
    spacing_y = usable_width / grid_rows

    print(f"Table size: {tabletop_size[0]:.2f} x {tabletop_size[1]:.2f}")
    print(f"Usable area: {usable_width:.2f} x {usable_depth:.2f} (x in [{x_min:.3f}, {x_max:.3f}])")
    print(f"Grid: {grid_rows} rows x {grid_cols} cols")
    print(f"Spacing: {spacing_x:.3f} x {spacing_y:.3f}")
    print(f"Placing fixed small cartons: {fixed_small_types} (single layer)")

    # Start positions
    start_y = tabletop_pos[1] - usable_width / 2 + spacing_y / 2
    start_x = x_min + spacing_x / 2

    # Z base (single layer)
    gap_above_table = 0.001  # 1mm
    base_z = tabletop_pos[2] + base_z_offset

    # Jitter
    jitter_amount = 0.3  # same as your current

    for i, carton_type in enumerate(fixed_small_types):
        oid = f"benchmark_carton_{carton_type}"

        # Compute deterministic grid base slot
        col = i // grid_rows
        row = i % grid_rows
        base_x = start_x + col * spacing_x
        base_y = start_y + row * spacing_y

        # Try multiple jitters to avoid excluded zones
        placed = False
        for _ in range(60):
            jitter_x = np.random.uniform(-jitter_amount, jitter_amount) * spacing_x
            jitter_y = np.random.uniform(-jitter_amount, jitter_amount) * spacing_y
            carton_x = base_x + jitter_x
            carton_y = base_y + jitter_y

            # Clamp into range to avoid drifting outside
            carton_x = np.clip(carton_x, x_min, x_max)
            carton_y = np.clip(
                carton_y,
                tabletop_pos[1] - tabletop_size[1] / 2 + safety_margin,
                tabletop_pos[1] + tabletop_size[1] / 2 - safety_margin,
            )

            # Excluded zone check
            if excluded_zones:
                conflict = False
                for zone_center, zone_radius in excluded_zones:
                    dist = np.sqrt((carton_x - zone_center[0]) ** 2 + (carton_y - zone_center[1]) ** 2)
                    if dist < zone_radius + 0.15:  # 15cm buffer
                        conflict = True
                        break
                if conflict:
                    continue

            # Random color + optional shrink wrap (保留你的风格)
            colors = ["brown", "white", "blue"]
            color = np.random.choice(colors, p=[0.5, 0.3, 0.2])
            is_shrink_wrapped = np.random.random() < 0.2

            if is_shrink_wrapped:
                keywords = [
                    "express_carton",
                    color,
                    "cardboard",
                    "sealed",
                    "shrink_wrapped",
                    "black_plastic",
                    f"type_{carton_type}",
                ]
            else:
                keywords = ["express_carton", color, "cardboard", "sealed", f"type_{carton_type}"]

            carton = library_call("usd", oid=oid, keywords=keywords)

            # Random rotation
            """carton_center = compute_shape_center(carton)
            rotation_angle = np.random.uniform(-0.5, 0.5)
            rotated_carton = transform_shape(
                carton,
                rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=carton_center)
            )"""

            # Random pose: yaw (Z) + small roll/pitch (X/Y)
            carton_center = compute_shape_center(carton)

            # yaw: bigger, looks like "randomly tossed"
            yaw = np.random.uniform(-3.1, 3.1)  # ~±210 degrees

            # small roll/pitch: subtle tilt (keep small to avoid crazy penetration)
            max_tilt_deg = 100.0
            roll = np.deg2rad(np.random.uniform(-max_tilt_deg, max_tilt_deg))
            pitch = np.deg2rad(np.random.uniform(-max_tilt_deg, max_tilt_deg))

            rot = rotation_matrix(angle=yaw, direction=(0, 0, 1), point=carton_center)
            rot = rot @ rotation_matrix(angle=roll, direction=(1, 0, 0), point=carton_center)
            rot = rot @ rotation_matrix(angle=pitch, direction=(0, 1, 0), point=carton_center)

            rotated_carton = transform_shape(carton, rot)

            final_center = compute_shape_center(rotated_carton)
            final_info = get_object_info(rotated_carton)

            # Single-layer bottom z
            # carton_bottom_z = base_z + (final_center[2] - final_info["min"][2]) + gap_above_table
            # base_z_offset now means: target TOP height offset above tabletop (e.g. avg big carton height)
            target_top_z = tabletop_pos[2] + base_z_offset

            # height of the rotated carton (world bbox)
            carton_h = final_info["max"][2] - final_info["min"][2]

            # place carton so its TOP is near target_top_z
            carton_bottom_z = target_top_z - carton_h + gap_above_table

            positioned_carton = transform_shape(
                rotated_carton,
                translation_matrix(
                    (carton_x - final_center[0], carton_y - final_center[1], carton_bottom_z - final_center[2])
                ),
            )

            placed_cartons.append(positioned_carton)
            placed = True
            break

        if not placed:
            print(f"[WARN] Failed to place small carton type {carton_type} due to excluded zones.")

    print(f"Successfully placed {len(placed_cartons)} fixed small cartons (single layer)")

    return concat_shapes(*placed_cartons) if placed_cartons else concat_shapes()


@register()
def left_side_sorting_table() -> Shape:
    """
    Creates cartons uniformly distributed in space (no table).
    Large cartons are placed first, then small cartons avoid them.
    """
    # Load a temporary table just to get surface info for positioning
    temp_table = library_call(
        "usd", oid="benchmark_table_021", keywords=["sorting_table", "white", "rectangular", "warehouse", "furniture"]
    )

    # Get tabletop info for placing cartons
    tabletop_pos, tabletop_size = find_tabletop_surface(temp_table)

    # Step 1: Place large cartons on layers 3 and 4
    large_carton_list = []
    excluded_zones = []  # Store (center, radius) for each large carton (for small carton avoidance)

    # Step 1.1: Place 5-8 large cartons from types 004-005 on layers 1 and 2
    # Note: Cartons on different layers can overlap in XY direction
    num_medium_large_cartons = np.random.randint(8, 15)  # 5-8 medium large cartons
    print(f"Placing {num_medium_large_cartons} medium large cartons (benchmark_carton_004-005) on layers 1 and 2")

    medium_large_carton_types = ["005", "008"]

    big_heights = []

    for i in range(num_medium_large_cartons):
        carton_type = np.random.choice(medium_large_carton_types)
        oid = f"benchmark_carton_{carton_type}"

        # Random color for each medium large carton
        colors = ["brown", "white", "blue"]
        color = np.random.choice(colors, p=[0.5, 0.3, 0.2])
        is_shrink_wrapped = np.random.random() < 0.2
        if is_shrink_wrapped:
            keywords = [
                "express_carton",
                color,
                "cardboard",
                "sealed",
                "shrink_wrapped",
                "black_plastic",
                f"type_{carton_type}",
                "large_carton",
            ]
        else:
            keywords = ["express_carton", color, "cardboard", "sealed", f"type_{carton_type}", "large_carton"]

        carton = library_call("usd", oid=oid, keywords=keywords)
        carton_info = get_object_info(carton)
        carton_size = carton_info["size"]
        big_heights.append(carton_size[2])
        carton_radius = max(carton_size[0], carton_size[1]) / 2.0

        # Find position for this medium large carton
        # Place large cartons on the BACK half of the table (away from blue basket)
        # After -90 deg Z rotation: original +X becomes back
        # Random position without checking excluded_zones (different layers can overlap in XY)
        margin = 0.10 + carton_radius
        # Restrict to back half (original +X direction, away from blue basket)

        # Split table area along X with ratio SMALL:BIG = 1:2 (must match small cartons)
        table_min_x = tabletop_pos[0] - tabletop_size[0] / 2
        table_max_x = tabletop_pos[0] + tabletop_size[0] / 2
        split_x = table_min_x + (2.0 / 3.0) * (table_max_x - table_min_x)

        """carton_x = np.random.uniform(
                tabletop_pos[0] - tabletop_size[0]/2 + margin,
                tabletop_pos[0] - margin
            )"""

        """carton_x = np.random.uniform(
            table_min_x + margin,
            split_x - margin
        )
        # Full range in Y (left-right after rotation)
        carton_y = np.random.uniform(
            tabletop_pos[1] - tabletop_size[1]/2 + margin,
            tabletop_pos[1] + tabletop_size[1]/2 - margin
        )"""

        placed_pos = None
        for _ in range(80):
            carton_x = np.random.uniform(table_min_x + margin, split_x - margin)
            carton_y = np.random.uniform(
                tabletop_pos[1] - tabletop_size[1] / 2 + margin, tabletop_pos[1] + tabletop_size[1] / 2 - margin
            )

            # avoid overlap with already placed BIG cartons (excluded_zones)
            ok = True
            for (cx, cy), r in excluded_zones:
                if np.hypot(carton_x - cx, carton_y - cy) < (carton_radius + r + 0.05):
                    ok = False
                    break
            if ok:
                placed_pos = (carton_x, carton_y)
                break

        if placed_pos is None:
            continue  # give up this carton if too crowded

        carton_x, carton_y = placed_pos
        excluded_zones.append(((carton_x, carton_y), carton_radius))  # keep for small-carton avoidance too

        # Add to excluded_zones so small cartons can avoid these areas
        # excluded_zones.append(((carton_x, carton_y), carton_radius))

        # Rotate and position - place on layers 1 or 2
        carton_center = compute_shape_center(carton)
        rotation_angle = np.random.uniform(-0.3, 0.3)
        rotated_carton = transform_shape(
            carton, rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=carton_center)
        )
        rotated_center = compute_shape_center(rotated_carton)
        rotated_info = get_object_info(rotated_carton)
        gap = 0.001

        # Calculate z position based on stacking layer (0 for layer 1, 1 for layer 2)
        """stacking_layer = i % 2
        base_z = tabletop_pos[2]

        # Place directly on table for layers 1 and 2
        medium_carton_height = carton_size[2]  # Actual height of carton
        carton_z = base_z + (medium_carton_height + 0.05) * stacking_layer + (rotated_center[2] - rotated_info["min"][2]) + gap"""

        base_z = tabletop_pos[2]
        carton_z = base_z + (rotated_center[2] - rotated_info["min"][2]) + gap

        positioned_carton = transform_shape(
            rotated_carton,
            translation_matrix(
                (carton_x - rotated_center[0], carton_y - rotated_center[1], carton_z - rotated_center[2])
            ),
        )
        large_carton_list.append(positioned_carton)
        print(f"Successfully placed medium large carton {i+1}/{num_medium_large_cartons}")

    avg_big_h = float(np.mean(big_heights)) if big_heights else 0.25
    print(f"Successfully placed {len(large_carton_list)} large cartons")

    # Step 2: Place small cartons on layers 3 and 4, avoiding the large carton zones
    num_cartons = 4  # Fixed 4 small cartons (types: 030/029/028/020)
    # Calculate the base z for small cartons (on top of layer 2)
    # Layer 2 medium carton center is at: base_z + (medium_carton_height + 0.05) + center_to_bottom_offset
    # Layer 2 medium carton top is at: base_z + (medium_carton_height + 0.05) + center_to_bottom_offset + center_to_top_offset
    #                      = base_z + medium_carton_height + 0.05 + carton_height
    #                      = base_z + 2*medium_carton_height + 0.05
    # For small cartons (which add their own center_to_bottom_offset), we need:
    # small_cartons_base_z + small_carton_center_to_bottom = layer_2_top + spacing
    # So: small_cartons_base_z = base_z + 2*medium_carton_height + 0.05 + spacing - small_carton_center_to_bottom
    # Approximating: small_cartons_base_z ≈ base_z + medium_carton_height + 0.05 + spacing
    avg_medium_carton_height = -0.30  # Average height of medium large cartons (003-005)
    layer_spacing_1_to_2 = 0.05  # 5cm spacing between layer 1 and layer 2 medium cartons
    layer_offset = 0.00  # Additional spacing above medium layer 2
    small_cartons_base_z = tabletop_pos[2] + avg_medium_carton_height + layer_spacing_1_to_2 + layer_offset

    """small_cartons = library_call(
        "scattered_cartons_on_table",
        num_cartons=num_cartons,
        table_shape=temp_table,
        excluded_zones=excluded_zones,
        base_z_offset=small_cartons_base_z
    )"""

    small_cartons_top_offset = avg_big_h

    small_cartons = library_call(
        "scattered_cartons_on_table",
        num_cartons=num_cartons,
        table_shape=temp_table,
        excluded_zones=excluded_zones,
        base_z_offset=small_cartons_top_offset,
    )

    # Combine small cartons and large cartons only (no table)
    all_large_cartons = concat_shapes(*large_carton_list)
    all_cartons = concat_shapes(small_cartons, all_large_cartons)

    # Apply warehouse table 1 rotation: quaternion (0.70710677, 0, 0, -0.70710677)
    # This represents -90 degree rotation around Z axis (clockwise 90 degrees)
    all_cartons = transform_shape(
        all_cartons, rotation_matrix(angle=-math.pi / 2, direction=(0, 0, 1), point=(0, 0, 0))  # -90 degrees in radians
    )

    # Position at warehouse table 1 location
    table_position = (0.3156, 0.9832, 0.3134)
    all_cartons = transform_shape(all_cartons, translation_matrix(table_position))

    return all_cartons


@register()
def root_scene() -> Shape:
    return left_side_sorting_table()
