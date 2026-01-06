# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import random

import numpy as np

from client.layout.utils.func import (
    calculate_distance,
    create_grid,
    filter_occupied_grids,
    get_rotated_corners,
    is_collision,
    is_within_bounds,
)
from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    axis_to_quaternion,
    quat2mat_wxyz,
    quaternion_rotate,
    rotate_along_axis,
    rotate_point_2d,
)


def get_corners(pose, size, angle):
    cx, cy = pose
    w, h = size
    corners = [
        rotate_point_2d(cx - w / 2, cy - h / 2, angle, cx, cy),
        rotate_point_2d(cx + w / 2, cy - h / 2, angle, cx, cy),
        rotate_point_2d(cx + w / 2, cy + h / 2, angle, cx, cy),
        rotate_point_2d(cx - w / 2, cy + h / 2, angle, cx, cy),
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


# compute_intersection moved to common.base_utils.transform_utils as compute_rectangle_intersection


class LayoutSolver2D:
    """
    1. get room_vertices
    2. Generate Constraint with LLM
    3. Generate layout meet to the constraint
    """

    def __init__(
        self,
        workspace_xyz,
        workspace_size,
        objects=None,
        fix_obj_ids=[],
        obj_infos={},
        angle_random_num=24,
        blocked_zone=None,
    ):
        self.angle_random_num = angle_random_num
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
        self.blocked_zone = None
        if blocked_zone is not None and np.array(blocked_zone).shape == (2, 2):
            self.blocked_zone = [
                [blocked_zone[0][0] * 1000, blocked_zone[0][1] * 1000],
                [blocked_zone[1][0] * 1000, blocked_zone[1][1] * 1000],
            ]

        self.cx, self.cy, self.cz = (coord * 1000 for coord in workspace_xyz)
        self.workspace_Z_half = workspace_size[2] / 2.0
        self.z_offset = 20
        self.fix_obj_ids = fix_obj_ids

        # Store sub_workspace boundaries for each object
        self.sub_workspace_bounds = {}
        self._compute_sub_workspace_bounds()

    def parse_solution(self, solutions, obj_id):
        [obj_cx, obj_cy], rotation, _ = solutions[obj_id][:3]
        obj_xyz = np.array(
            [
                self.cx + obj_cx,
                self.cy + obj_cy,
                self.cz
                - self.workspace_Z_half
                + self.objects[obj_id].size[2] / 2.0
                + self.z_offset,
            ]
        )
        if not self.objects[obj_id].up_side_down:
            rotation = -rotation
        up_axis = self.objects[obj_id].up_axis
        upside_down = False
        if "-" in up_axis:
            up_axis = up_axis[1:]
            upside_down = True
            rotation = -rotation
        init_quat = axis_to_quaternion(up_axis, "z", upside_down)
        obj_quat = quaternion_rotate(init_quat, up_axis, rotation)

        obj_pose = np.eye(4)
        obj_pose[:3, :3] = quat2mat_wxyz(obj_quat)
        obj_pose[:3, 3] = obj_xyz
        self.objects[obj_id].obj_pose = obj_pose

    def _compute_sub_workspace_bounds(self):
        """
        Calculate sub_workspace boundaries for each object.
        sub_workspace is represented by min/max ratios in xyz directions (relative to workspace).
        If no sub_workspace is specified, use the entire workspace.

        sub_workspace format supports two types:
        1. {"x": [min_ratio, max_ratio], "y": [min_ratio, max_ratio], "z": [min_ratio, max_ratio]}
        2. {"x": {"min": min_ratio, "max": max_ratio}, ...}
        """
        # Workspace center is at (0, 0), as coordinates are relative to workspace center
        # Workspace range is [-plane_width/2, plane_width/2] x [-plane_height/2, plane_height/2]
        for obj_id in self.obj_infos:
            obj_info = self.obj_infos[obj_id]
            if "sub_workspace" in obj_info:
                sub_ws = obj_info["sub_workspace"]
                # Parse min and max ratios for x direction
                x_range = sub_ws.get("x", [0.0, 1.0])
                if isinstance(x_range, dict):
                    x_min_ratio = x_range.get("min", 0.0)
                    x_max_ratio = x_range.get("max", 1.0)
                elif isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
                    x_min_ratio = x_range[0]
                    x_max_ratio = x_range[1]
                else:
                    x_min_ratio, x_max_ratio = 0.0, 1.0

                # Parse min and max ratios for y direction
                y_range = sub_ws.get("y", [0.0, 1.0])
                if isinstance(y_range, dict):
                    y_min_ratio = y_range.get("min", 0.0)
                    y_max_ratio = y_range.get("max", 1.0)
                elif isinstance(y_range, (list, tuple)) and len(y_range) >= 2:
                    y_min_ratio = y_range[0]
                    y_max_ratio = y_range[1]
                else:
                    y_min_ratio, y_max_ratio = 0.0, 1.0

                # Ensure ratios are within valid range [0, 1]
                x_min_ratio = max(0.0, min(1.0, x_min_ratio))
                x_max_ratio = max(0.0, min(1.0, x_max_ratio))
                y_min_ratio = max(0.0, min(1.0, y_min_ratio))
                y_max_ratio = max(0.0, min(1.0, y_max_ratio))

                # Ensure min <= max
                if x_min_ratio > x_max_ratio:
                    x_min_ratio, x_max_ratio = x_max_ratio, x_min_ratio
                if y_min_ratio > y_max_ratio:
                    y_min_ratio, y_max_ratio = y_max_ratio, y_min_ratio

                # Calculate actual boundaries (relative to workspace center)
                # Workspace range: x from -plane_width/2 to plane_width/2
                min_x = -self.plane_width / 2 + x_min_ratio * self.plane_width
                max_x = -self.plane_width / 2 + x_max_ratio * self.plane_width
                min_y = -self.plane_height / 2 + y_min_ratio * self.plane_height
                max_y = -self.plane_height / 2 + y_max_ratio * self.plane_height

                self.sub_workspace_bounds[obj_id] = {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                }
            else:
                # No sub_workspace, use entire workspace
                self.sub_workspace_bounds[obj_id] = {
                    "min_x": -self.plane_width / 2,
                    "max_x": self.plane_width / 2,
                    "min_y": -self.plane_height / 2,
                    "max_y": self.plane_height / 2,
                }

    def _get_sub_workspace_grid_points(self, obj_id, grid_points):
        """
        Filter grid points based on object's sub_workspace.

        Args:
            obj_id: Object ID
            grid_points: List of all grid points

        Returns:
            Filtered grid points list
        """
        if obj_id not in self.sub_workspace_bounds:
            # If not found, use entire workspace
            return grid_points

        bounds = self.sub_workspace_bounds[obj_id]
        filtered_points = [
            (x, y)
            for x, y in grid_points
            if bounds["min_x"] <= x <= bounds["max_x"] and bounds["min_y"] <= y <= bounds["max_y"]
        ]
        return filtered_points

    def _is_within_sub_workspace(self, corners, obj_id):
        """
        Check if object's corner points are within sub_workspace range.

        Args:
            corners: List of object corner points [(x, y), ...]
            obj_id: Object ID

        Returns:
            bool: Whether within sub_workspace range
        """
        if obj_id not in self.sub_workspace_bounds:
            # If no sub_workspace, use entire workspace
            return is_within_bounds(corners, self.plane_width, self.plane_height)

        bounds = self.sub_workspace_bounds[obj_id]
        min_x = min(px for px, _ in corners)
        max_x = max(px for px, _ in corners)
        min_y = min(py for _, py in corners)
        max_y = max(py for _, py in corners)

        return (
            min_x >= bounds["min_x"]
            and max_x <= bounds["max_x"]
            and min_y >= bounds["min_y"]
            and max_y <= bounds["max_y"]
        )

    def __call__(
        self,
        opt_obj_ids,
        exist_obj_ids,
        object_extent=50,  # Extend 5cm outward
        start_with_edge=False,
        key_obj=True,
        grid_size=0.01,  # 1cm
        initial_angle=0,
    ):

        objs_succ = []
        fail_objs = []
        placed_objects = []
        obj_sizes = []
        grid_points = create_grid(
            self.plane_width,
            self.plane_height,
            int(grid_size * 1000),
            self.blocked_zone,
        )
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
            rotate_fixed_angle = self.obj_infos[obj_id].get("rotate_fixed_angle", -1)

            valid_position = False
            best_position = None
            max_distance = 1
            found_position_count = 0
            # First filter grid points within this object's sub_workspace range
            obj_grid_points = self._get_sub_workspace_grid_points(obj_id, grid_points)
            available_grid_points = filter_occupied_grids(obj_grid_points, placed_objects)
            angle_radom_upper_limit = self.objects[obj_id].angle_upper_limit
            angle_radom_lower_limit = self.objects[obj_id].angle_lower_limit
            angle_step = (angle_radom_upper_limit - angle_radom_lower_limit) / self.angle_random_num
            for idx in range(attempts):
                if not available_grid_points:
                    break
                if (
                    len(opt_obj_ids) == 1
                    and area_ratio > 0.5
                    and (angle_radom_upper_limit - angle_radom_lower_limit > 20)
                ):
                    if idx < len(available_grid_points):
                        x, y = available_grid_points[idx]  # Get point using index
                        if rotate_fixed_angle == -1:
                            angle = random.choice([0, 90, 180, 270])
                        else:
                            angle = rotate_fixed_angle
                    else:
                        # If index is out of range, exit loop or take other measures
                        break
                else:
                    x, y = random.choice(available_grid_points)  # Randomly select a point
                    if rotate_fixed_angle == -1:
                        angle = np.random.choice(
                            np.arange(
                                angle_radom_lower_limit,
                                angle_radom_upper_limit,
                                angle_step,
                            )
                        )
                    else:
                        angle = rotate_fixed_angle

                new_corners = get_rotated_corners(x, y, width, height, angle)

                # Use sub_workspace boundary check
                if self._is_within_sub_workspace(new_corners, obj_id) and not is_collision(
                    new_corners, placed_objects
                ):
                    if not placed_objects:
                        best_position = (x, y, width, height, angle)
                        valid_position = True
                        break

                    min_distance = min(
                        calculate_distance(new_corners, obj[1]) for obj in placed_objects
                    )
                    if min_distance > max_distance:
                        max_distance = min_distance
                        found_position_count += 1
                        best_position = (x, y, width, height, angle)
                        valid_position = True
                        if found_position_count > 3:
                            break

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
            logger.info("*******no solution objects************")
            logger.error(fail_objs)
            logger.info("******************")
        for obj_id in objs_succ:
            angle_radom_upper_limit = self.objects[obj_id].angle_upper_limit
            angle_radom_lower_limit = self.objects[obj_id].angle_lower_limit
            angle_step = (angle_radom_upper_limit - angle_radom_lower_limit) / self.angle_random_num
            self.objects[obj_id].obj_pose = rotate_along_axis(
                self.objects[obj_id].obj_pose,
                random.uniform(-angle_step / 2.0, angle_step / 2.0),
                rot_axis="z",
                use_local=False,
            )
        return objs_succ
