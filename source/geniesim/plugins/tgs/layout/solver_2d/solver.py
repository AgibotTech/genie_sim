# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import random
import math

from shapely import Polygon
from .layout import DFS_Solver_Floor
from shapely.geometry import Point, Polygon
from .multi_add_util import *

from geniesim.robot.utils import (
    axis_to_quaternion,
    quaternion_rotate,
    get_rotation_matrix_from_quaternion,
    get_quaternion_from_rotation_matrix,
    get_xyz_euler_from_quaternion,
)

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


def quaternion_multiply(q1, q2):
    """Calculate the product of two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_rotate_z(quaternion, angle):
    """
    Rotate a quaternion around the global z-axis by a given angle.

    Parameters:
    quaternion (numpy array): The input quaternion [w, x, y, z].
    angle (float): The rotation angle in degrees.

    Returns:
    numpy array: The rotated quaternion.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the rotation quaternion for z-axis rotation
    cos_half_angle = np.cos(angle_rad / 2)
    sin_half_angle = np.sin(angle_rad / 2)
    q_z = np.array([cos_half_angle, 0, 0, sin_half_angle])

    # Rotate the input quaternion around the global z-axis
    rotated_quaternion = quaternion_multiply(q_z, quaternion)

    return rotated_quaternion


def rotate_point_ext(px, py, angle, ox, oy):
    s, c = math.sin(angle), math.cos(angle)
    px, py = px - ox, py - oy
    xnew = px * c - py * s
    ynew = px * s + py * c
    return xnew + ox, ynew + oy


def get_corners(pose, size, angle):
    cx, cy = pose
    w, h = size
    corners = [
        rotate_point_ext(cx - w / 2, cy - h / 2, angle, cx, cy),
        rotate_point_ext(cx + w / 2, cy - h / 2, angle, cx, cy),
        rotate_point_ext(cx + w / 2, cy + h / 2, angle, cx, cy),
        rotate_point_ext(cx - w / 2, cy + h / 2, angle, cx, cy),
    ]
    return corners


def compute_bounding_box(objects, expansion=40):
    all_corners = []
    for pose, size, angle in objects:
        all_corners.extend(get_corners(pose, size, angle))

    min_x = min(x for x, y in all_corners)
    max_x = max(x for x, y in all_corners)
    min_y = min(y for x, y in all_corners)
    max_y = max(y for x, y in all_corners)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x + expansion
    height = max_y - min_y + expansion

    return (center_x, center_y, width, height, 0)


def compute_intersection(bbox, plane_center, plane_width, plane_height):
    # Unpacking the minimum external rectangle information
    bbox_center_x, bbox_center_y, bbox_width, bbox_height, _ = bbox

    # Calculate the boundary of the minimum circumscribed rectangle
    min_x = bbox_center_x - bbox_width / 2
    max_x = bbox_center_x + bbox_width / 2
    min_y = bbox_center_y - bbox_height / 2
    max_y = bbox_center_y + bbox_height / 2

    # Calculate the boundary of the plane
    plane_min_x = plane_center[0] - plane_width / 2
    plane_max_x = plane_center[0] + plane_width / 2
    plane_min_y = plane_center[1] - plane_height / 2
    plane_max_y = plane_center[1] + plane_height / 2

    # Calculate the boundary of the intersection part
    intersect_min_x = max(min_x, plane_min_x)
    intersect_max_x = min(max_x, plane_max_x)
    intersect_min_y = max(min_y, plane_min_y)
    intersect_max_y = min(max_y, plane_max_y)

    # Check if the intersection area is valid
    if intersect_min_x < intersect_max_x and intersect_min_y < intersect_max_y:
        # Calculate the center and dimensions of the intersection rectangle
        intersect_center_x = (intersect_min_x + intersect_max_x) / 2
        intersect_center_y = (intersect_min_y + intersect_max_y) / 2
        intersect_width = intersect_max_x - intersect_min_x
        intersect_height = intersect_max_y - intersect_min_y

        return (
            intersect_center_x,
            intersect_center_y,
            intersect_width,
            intersect_height,
            0,
        )
    else:
        # If there is no valid intersection area, return None or other value indicating no intersection
        return None


class LayoutSolver2D:
    """
    1. get room_vertices
    2. Generate Constraint with LLM
    3. Generate layout meet to the constraint
    """

    def __init__(self, workspace_xyz, workspace_size, objects=None, fix_obj_ids=[], obj_infos={}):
        x_half, y_half, z_half = workspace_size / 2
        room_vertices = [
            [-x_half, -y_half],
            [x_half, -y_half],
            [x_half, y_half],
            [-x_half, y_half],
        ]
        self.plane_width = workspace_size[0]
        self.plane_height = workspace_size[1]
        self.room_vertices = room_vertices

        self.objects = objects
        self.obj_infos = obj_infos

        self.cx, self.cy, self.cz = (coord * 1000 for coord in workspace_xyz)
        self.workspace_Z_half = workspace_size[2] / 2.0
        self.z_offset = 20
        self.fix_obj_ids = fix_obj_ids

    def parse_solution(self, solutions, obj_id):
        [obj_cx, obj_cy], rotation, _ = solutions[obj_id][:3]
        obj_xyz = np.array(
            [
                self.cx + obj_cx,
                self.cy + obj_cy,
                self.cz - self.workspace_Z_half + self.objects[obj_id].size[2] / 2.0 + self.z_offset,
            ]
        )
        init_quat = axis_to_quaternion(self.objects[obj_id].up_axis, "z")
        obj_quat = quaternion_rotate(init_quat, self.objects[obj_id].up_axis, -rotation)

        obj_pose = np.eye(4)
        obj_pose[:3, :3] = get_rotation_matrix_from_quaternion(obj_quat)
        obj_pose[:3, 3] = obj_xyz
        self.objects[obj_id].obj_pose = obj_pose

    def old_solution(
        self,
        opt_obj_ids,
        exist_obj_ids,
        object_extent=50,  # Outer extension 5cm
        start_with_edge=False,
        grid_size=0.01,  # 1cm
    ):
        solver = DFS_Solver_Floor(grid_size=int(grid_size * 1000))
        room_poly = Polygon(self.room_vertices)
        grid_points = solver.create_grids(room_poly)

        objs_succ = []
        saved_solutions = {}  # better save the pose of exist_obj_ids into saved_solutions
        for obj_id in opt_obj_ids:
            size = self.objects[obj_id].size
            size_extent = size[:2] + object_extent

            solutions = solver.get_all_solutions(room_poly, grid_points, size_extent)
            if len(solutions) > 0:
                if start_with_edge:
                    solutions = solver.place_edge(room_poly, solutions, size_extent)
                if len(saved_solutions) == 0:
                    saved_solutions[obj_id] = random.choice(solutions)
                else:
                    solutions = solver.filter_collision(saved_solutions, solutions)
                    if len(solutions) > 0:
                        saved_solutions[obj_id] = random.choice(solutions)
                    else:
                        logger.error(f"No valid solutions for apply {obj_id}.")
                        continue

            else:
                logger.error(f"No valid solutions for apply {obj_id}.")
                continue

            self.parse_solution(saved_solutions, obj_id)
            objs_succ.append(obj_id)

        return objs_succ

    def __call__(
        self,
        opt_obj_ids,
        exist_obj_ids,
        object_extent=50,  # Outer extension 5cm
        start_with_edge=False,
        key_obj=True,
        grid_size=0.01,  # 1cm
        initial_angle=0,
    ):

        objs_succ = []
        fail_objs = []
        placed_objects = []
        main_objects = []
        label_flag = "no extra"
        if not key_obj:
            for obj_id in self.objects:
                if obj_id not in opt_obj_ids:
                    if obj_id in self.fix_obj_ids:
                        continue
                    size = self.objects[obj_id].size
                    pose = self.objects[obj_id].obj_pose
                    quaternion_rotate = get_quaternion_from_rotation_matrix(pose[:3, :3])
                    angle_pa = get_xyz_euler_from_quaternion(quaternion_rotate)[2]
                    main_objects.append(
                        (
                            (pose[0, 3] - self.cx, pose[1, 3] - self.cy),
                            (size[0], size[1]),
                            angle_pa,
                        )
                    )
                main_bounding_box_info = compute_bounding_box(main_objects, expansion=object_extent)

            intersection = compute_intersection(main_bounding_box_info, (0, 0), self.plane_width, self.plane_height)
            if intersection is None:
                return objs_succ
            placed_objects.append((intersection, get_rotated_corners(*intersection)))
            label_flag = "add extra"
            object_extent = 0
        obj_sizes = []
        grid_points = create_grid(self.plane_width, self.plane_height, int(grid_size * 1000))
        saved_solutions = {}
        if len(opt_obj_ids) == 1:
            attempts = 800
        else:
            attempts = 400

        for obj_id in opt_obj_ids:
            size = self.objects[obj_id].size
            _extent = self.obj_infos[obj_id].get("extent", object_extent)
            size_extent = size[:2] + _extent
            obj_sizes.append((size_extent[0], size_extent[1]))
            width, height = size_extent[0], size_extent[1]
            area_ratio = (width * height) / (self.plane_width * self.plane_height)

            valid_position = False
            best_position = None
            max_distance = 1
            available_grid_points = filter_occupied_grids(grid_points, placed_objects)
            for idx in range(attempts):
                if not available_grid_points:
                    break
                if len(opt_obj_ids) == 1 and area_ratio > 0.5:
                    if idx < len(available_grid_points):
                        x, y = available_grid_points[idx]  # Get points using index
                        angle = random.choice([0, 90, 180, 270])
                    else:
                        # If the index is out of range, you can choose to exit the loop or take other measures
                        break
                else:
                    x, y = random.choice(available_grid_points)  # Randomly select a point
                    angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270])

                new_corners = get_rotated_corners(x, y, width, height, angle)

                if is_within_bounds(new_corners, self.plane_width, self.plane_height) and not is_collision(
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
                        # break

            if valid_position:
                placed_objects.append((best_position, get_rotated_corners(*best_position)))
                saved_solutions[obj_id] = [
                    [best_position[0], best_position[1]],
                    best_position[4],
                    None,
                ]
                self.parse_solution(saved_solutions, obj_id)
                objs_succ.append(obj_id)
            else:
                fail_objs.append(obj_id)

        if len(fail_objs) > 0:
            logger.error("*******no solution objects************")
            logger.error(fail_objs)
            logger.error("******************")
        return objs_succ
