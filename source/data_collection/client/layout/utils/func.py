# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Layout utility functions merged from:
- client/layout/solver_2d/multi_add_utils.py
- client/layout/utils/transform_utils.py
"""

import math

import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans

# ===============================================
# Functions from multi_add_utils.py
# ===============================================


def rotate_point(px, py, cx, cy, angle):
    """
    Rotate a 2D point around a center point.

    Args:
        px (float): X coordinate of point to rotate
        py (float): Y coordinate of point to rotate
        cx (float): X coordinate of rotation center
        cy (float): Y coordinate of rotation center
        angle (float): Rotation angle in degrees

    Returns:
        tuple: (rotated_x, rotated_y) - Rotated point coordinates
    """
    radians = math.radians(angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    temp_x = px - cx
    temp_y = py - cy
    rotated_x = temp_x * cos_a - temp_y * sin_a
    rotated_y = temp_x * sin_a + temp_y * cos_a
    return rotated_x + cx, rotated_y + cy


def get_rotated_corners(x, y, width, height, angle):
    """
    Get corners of a rotated rectangle.

    Args:
        x (float): Center X coordinate
        y (float): Center Y coordinate
        width (float): Rectangle width
        height (float): Rectangle height
        angle (float): Rotation angle in degrees

    Returns:
        list: List of corner coordinates [(x, y), ...]
    """
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
    """
    Check if corners are within the plane bounds.

    Args:
        corners (list): List of corner coordinates [(x, y), ...]
        plane_width (float): Width of the plane
        plane_height (float): Height of the plane

    Returns:
        bool: True if all corners are within bounds
    """
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
    """
    Check if new corners collide with any placed objects.

    Args:
        new_corners (list): Corners of the new object
        placed_objects (list): List of placed objects, each is (position, corners)

    Returns:
        bool: True if collision detected
    """
    for _, existing_corners in placed_objects:
        if polygons_intersect(new_corners, existing_corners):
            return True
    return False


def polygons_intersect(p1, p2):
    """
    Check if two polygons intersect using Separating Axis Theorem (SAT).

    Args:
        p1 (list): First polygon corners [(x, y), ...]
        p2 (list): Second polygon corners [(x, y), ...]

    Returns:
        bool: True if polygons intersect
    """
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
    """
    Project a polygon onto an axis.

    Args:
        axis (tuple): Projection axis (x, y)
        polygon (list): Polygon corners [(x, y), ...]

    Returns:
        tuple: (min_projection, max_projection)
    """
    min_proj = max_proj = polygon[0][0] * axis[0] + polygon[0][1] * axis[1]
    for x, y in polygon[1:]:
        projection = x * axis[0] + y * axis[1]
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


def calculate_distance(corners1, corners2):
    """
    Calculate distance between centers of two polygons.

    Args:
        corners1 (list): First polygon corners
        corners2 (list): Second polygon corners

    Returns:
        float: Distance between centers
    """
    center1 = np.mean(corners1, axis=0)
    center2 = np.mean(corners2, axis=0)
    return np.linalg.norm(center1 - center2)


def create_grid(plane_width, plane_height, grid_size, blocked_zone=None):
    """
    Create a grid of points within the plane.

    Args:
        plane_width (float): Width of the plane
        plane_height (float): Height of the plane
        grid_size (float): Size of each grid cell
        blocked_zone (list, optional): Blocked zone as [[x_min, x_max], [y_min, y_max]]

    Returns:
        list: List of grid point coordinates [(x, y), ...]
    """
    grid_points = []
    for x in np.arange(-plane_width / 2, plane_width / 2, grid_size):
        for y in np.arange(-plane_height / 2, plane_height / 2, grid_size):
            if blocked_zone is not None:
                if blocked_zone[0][0] <= x <= blocked_zone[0][1] and blocked_zone[1][0] <= y <= blocked_zone[1][1]:
                    continue
            grid_points.append((x, y))
    return grid_points


def filter_occupied_grids(grid_points, placed_objects):
    """
    Filter out grid points that are inside placed objects.

    Args:
        grid_points (list): List of grid point coordinates
        placed_objects (list): List of placed objects, each is (position, corners)

    Returns:
        list: Filtered grid points
    """
    available_points = grid_points.copy()

    for obj in placed_objects:
        obj_poly = Polygon(obj[1])  # Get polygon of placed object
        available_points = [point for point in available_points if not obj_poly.contains(Point(point))]

    return available_points


# ===============================================
# Functions from transform_utils.py
# ===============================================


def random_point(points, num):
    """
    Randomly select points and compute their mean.

    Args:
        points (np.ndarray): Point cloud of shape (N, 3)
        num (int): Number of points to randomly select

    Returns:
        np.ndarray: Mean point of selected points, shape (3,)
    """
    # Randomly select num different points
    random_indices = np.random.choice(points.shape[0], num, replace=False)
    random_points = points[random_indices]

    # Calculate mean of selected point coordinates
    x_mean = np.mean(random_points[:, 0])
    y_mean = np.mean(random_points[:, 1])
    z_mean = np.mean(random_points[:, 2])
    selected_points = np.array([x_mean, y_mean, z_mean])
    return selected_points


def get_bott_up_point(points, obj_size, descending):
    """
    Get bottom/up surface points from a point cloud using KMeans clustering.

    Args:
        points (np.ndarray): Point cloud of shape (N, 3)
        obj_size (float): Object size for threshold calculation
        descending (bool): If True, get top surface; if False, get bottom surface

    Returns:
        np.ndarray: Selected surface points
    """
    # Sort by Z value from smallest to largest
    ascending_indices = np.argsort(points[:, 2])
    if descending:
        # Reverse indices to achieve descending order
        ascending_indices = ascending_indices[::-1]
    sorted_points = points[ascending_indices]
    threshold = 0.03 * obj_size
    z_m = sorted_points[0][-1]
    while True:
        top_surface_points = sorted_points[np.abs(sorted_points[:, 2] - z_m) < threshold]
        if len(top_surface_points) >= 15:
            break
        # Increase threshold to get more points
        threshold += 0.01 * obj_size
    # Get top/bottom surface points
    top_surface_points = sorted_points[np.abs(sorted_points[:, 2] - z_m) < threshold]

    # Use KMeans clustering to ensure uniform distribution of points
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(top_surface_points[:, :2])  # Only use X and Y coordinates for clustering
    # Get the point closest to each cluster center
    centers = kmeans.cluster_centers_
    selected_points = []

    for center in centers:
        # Calculate distance from each center to all points
        distances = np.linalg.norm(top_surface_points[:, :2] - center, axis=1)
        # Find the closest point
        closest_point_idx = np.argmin(distances)
        selected_points.append(top_surface_points[closest_point_idx])
    selected_points = np.array(selected_points)
    selected_points[:, 2] = z_m
    return selected_points
