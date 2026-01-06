# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import random
import math

from shapely import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon


def rotate_point(px, py, cx, cy, angle):
    radians = math.radians(angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    temp_x = px - cx
    temp_y = py - cy
    rotated_x = temp_x * cos_a - temp_y * sin_a
    rotated_y = temp_x * sin_a + temp_y * cos_a
    return rotated_x + cx, rotated_y + cy


def get_rotated_corners(x, y, width, height, angle):
    cx = x
    cy = y
    corners = [
        (x - width / 2, y - height / 2),
        (x + width / 2, y - height / 2),
        (x - width / 2, y + height / 2),
        (x + width / 2, y + height / 2),
    ]
    return [rotate_point(px, py, cx, cy, angle) for px, py in corners]


def is_within_bounds(corners, plane_width, plane_height):
    min_x = min(px for px, _ in corners)
    max_x = max(px for px, _ in corners)
    min_y = min(py for _, py in corners)
    max_y = max(py for _, py in corners)

    return (
        min_x >= -plane_width / 2
        and max_x <= plane_width / 2
        and min_y >= -plane_height / 2
        and max_y <= plane_height / 2
    )


def is_collision(new_corners, placed_objects):
    for _, existing_corners in placed_objects:
        if polygons_intersect(new_corners, existing_corners):
            return True
    return False


def polygons_intersect(p1, p2):
    for polygon in [p1, p2]:
        for i1 in range(len(polygon)):
            i2 = (i1 + 1) % len(polygon)
            projection_axis = (
                -(polygon[i2][1] - polygon[i1][1]),
                polygon[i2][0] - polygon[i1][0],
            )

            min_p1, max_p1 = project_polygon(projection_axis, p1)
            min_p2, max_p2 = project_polygon(projection_axis, p2)

            if max_p1 < min_p2 or max_p2 < min_p1:
                return False

    return True


def project_polygon(axis, polygon):
    min_proj = max_proj = polygon[0][0] * axis[0] + polygon[0][1] * axis[1]
    for x, y in polygon[1:]:
        projection = x * axis[0] + y * axis[1]
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


def generate_object_sizes(num_objects, size_options):
    sizes = []
    max_size = max(size_options, key=lambda s: s[0] * s[1])
    max_count = 0

    for _ in range(num_objects):
        if max_count < 3:
            size = random.choice(size_options)
            if size == max_size:
                max_count += 1
        else:
            size = random.choice([s for s in size_options if s != max_size])

        sizes.append(size)

    sizes.sort(key=lambda s: s[0] * s[1], reverse=True)
    return sizes


def calculate_distance(corners1, corners2):
    center1 = np.mean(corners1, axis=0)
    center2 = np.mean(corners2, axis=0)
    return np.linalg.norm(center1 - center2)


def place_objects(num_objects, plane_width, plane_height, obj_sizes):
    placed_objects = []
    attempts = 0
    max_attempts = num_objects * 100

    while len(placed_objects) < num_objects and attempts < max_attempts:
        width, height = obj_sizes[len(placed_objects)]
        valid_position = False
        best_position = None
        max_distance = -1

        for _ in range(200):
            x = random.uniform(-plane_width / 2 + width / 2, plane_width / 2 - width / 2)
            y = random.uniform(-plane_height / 2 + height / 2, plane_height / 2 - height / 2)
            angle = random.choice(range(0, 360, 15))

            new_corners = get_rotated_corners(x, y, width, height, angle)

            if is_within_bounds(new_corners, plane_width, plane_height) and not is_collision(
                new_corners, placed_objects
            ):
                if not placed_objects:
                    best_position = (x, y, width, height, angle)
                    valid_position = True
                    break

                min_distance = min(calculate_distance(new_corners, obj[1]) for obj in placed_objects)
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_position = (x, y, width, height, angle)
                    valid_position = True

        if valid_position:
            placed_objects.append((best_position, get_rotated_corners(*best_position)))
            attempts = 0
        else:
            attempts += 1

    return [obj[0] for obj in placed_objects]


color_list = [
    "blue",
    "green",
    "orange",
    "purple",
    "red",
    "yellow",
    "pink",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "lavender",
    "brown",
    "beige",
    "maroon",
    "navy",
    "olive",
    "coral",
    "turquoise",
    "silver",
    "gold",
]


def visualize_objects(objects, plane_width, plane_height, filename):
    fig, ax = plt.subplots()
    ax.set_xlim(-plane_width / 2, plane_width / 2)
    ax.set_ylim(-plane_height / 2, plane_height / 2)
    ax.set_aspect("equal")

    for i, (x, y, width, height, angle) in enumerate(objects):
        cx = x
        cy = y
        color = color_list[i % len(color_list)]
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            edgecolor="r",
            facecolor=color,
            alpha=0.5,
        )
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")


def create_grid(plane_width, plane_height, grid_size):
    grid_points = []
    for x in np.arange(-plane_width / 2, plane_width / 2, grid_size):
        for y in np.arange(-plane_height / 2, plane_height / 2, grid_size):
            grid_points.append((x, y))
    return grid_points


def filter_occupied_grids(grid_points, placed_objects):
    available_points = grid_points.copy()

    for obj in placed_objects:
        obj_poly = Polygon(obj[1])
        available_points = [point for point in available_points if not obj_poly.contains(Point(point))]

    return available_points
