# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import random
import time
import numpy as np
from rtree import index
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, box, LineString

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


class SolutionFound(Exception):
    def __init__(self, solution):
        self.solution = solution


class DFS_Solver_Floor:
    def __init__(self, grid_size, random_seed=0, max_duration=5, constraint_bouns=0.2):
        self.grid_size = grid_size  # grid length (not list size)
        self.random_seed = random_seed
        self.max_duration = max_duration  # maximum allowed time in seconds
        self.constraint_bouns = constraint_bouns
        self.start_time = None
        self.solutions = []

        # Define the functions in a dictionary to avoid if-else conditions
        self.func_dict = {
            "global": {"edge": self.place_edge},
            "relative": self.place_relative,
            "direction": self.place_face,
            "alignment": self.place_alignment_center,
            "distance": self.place_distance,
        }

        self.constraint_type2weight = {
            "global": 1.0,
            "relative": 0.5,
            "direction": 0.5,
            "alignment": 0.5,
            "distance": 1.8,
        }

        self.edge_bouns = 0.0  # worth more than one constraint

    def get_solution(self, bounds, objects_list, constraints, initial_state, use_milp=False):
        self.start_time = time.time()
        if use_milp:
            # iterate through the constraints list
            # for each constraint type "distance", add the same constraint to the target object
            new_constraints = constraints.copy()
            for object_name, object_constraints in constraints.items():
                for constraint in object_constraints:
                    if constraint["type"] == "distance":
                        target_object_name = constraint["target"]
                        if target_object_name in constraints.keys():
                            # if there is already a distance constraint of target object_name, continue
                            if any(
                                constraint["type"] == "distance" and constraint["target"] == object_name
                                for constraint in constraints[target_object_name]
                            ):
                                continue
                            new_constraint = constraint.copy()
                            new_constraint["target"] = object_name
                            new_constraints[target_object_name].append(new_constraint)
            constraints = new_constraints

            logger.debug(f"Time taken: {time.time() - self.start_time}")

        else:
            grid_points = self.create_grids(bounds)
            grid_points = self._remove_points(grid_points, initial_state)
            try:
                self._dfs(bounds, objects_list, constraints, grid_points, initial_state, 30)
            except SolutionFound as e:
                logger.debug(f"Time taken: {time.time() - self.start_time}")

        logger.info(f"Number of solutions found: {len(self.solutions)}")
        max_solution = self._get_max_solution(self.solutions)

        return max_solution

    def _get_max_solution(self, solutions):
        path_weights = []
        for solution in solutions:
            path_weights.append(sum([obj[-1] for obj in solution.values()]))
        max_index = np.argmax(path_weights)
        return solutions[max_index]

    def _dfs(
        self,
        room_poly,
        objects_list,
        constraints,
        grid_points,
        placed_objects,
        branch_factor,
    ):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            return placed_objects

        if time.time() - self.start_time > self.max_duration:
            logger.warning(f"Time limit reached.")
            raise SolutionFound(self.solutions)

        object_name, object_dim = objects_list[0]
        placements = self._get_possible_placements(
            room_poly, object_dim, constraints[object_name], grid_points, placed_objects
        )

        if len(placements) == 0 and len(placed_objects) != 0:
            self.solutions.append(placed_objects)

        paths = []
        if branch_factor > 1:
            random.shuffle(placements)  # shuffle the placements of the first object

        for placement in placements[:branch_factor]:
            placed_objects_updated = copy.deepcopy(placed_objects)
            placed_objects_updated[object_name] = placement
            grid_points_updated = self._remove_points(grid_points, placed_objects_updated)

            sub_paths = self._dfs(
                room_poly,
                objects_list[1:],
                constraints,
                grid_points_updated,
                placed_objects_updated,
                1,
            )
            paths.extend(sub_paths)

        return paths

    def _get_possible_placements(self, room_poly, object_dim, constraints, grid_points, placed_objects):
        solutions = self.filter_collision(placed_objects, self.get_all_solutions(room_poly, grid_points, object_dim))
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), object_dim)

        if len(edge_solutions) == 0:
            return edge_solutions

        global_constraint = next(
            (constraint for constraint in constraints if constraint["type"] == "global"),
            None,
        )

        if global_constraint is None:
            global_constraint = {"type": "global", "constraint": "edge"}

        if global_constraint["constraint"] == "edge":
            candidate_solutions = copy.deepcopy(edge_solutions)  # edge is hard constraint
        else:
            if len(constraints) > 1:
                candidate_solutions = solutions + edge_solutions  # edge is soft constraint
            else:
                candidate_solutions = copy.deepcopy(solutions)  # the first object

        candidate_solutions = self.filter_collision(
            placed_objects, candidate_solutions
        )  # filter again after global constraint

        if candidate_solutions == []:
            return candidate_solutions
        random.shuffle(candidate_solutions)
        placement2score = {tuple(solution[:3]): solution[-1] for solution in candidate_solutions}

        # add a bias to edge solutions
        for solution in candidate_solutions:
            if solution in edge_solutions and len(constraints) >= 1:
                placement2score[tuple(solution[:3])] += self.edge_bouns

        for constraint in constraints:
            if "target" not in constraint:
                continue

            func = self.func_dict.get(constraint["type"])
            valid_solutions = func(
                constraint["constraint"],
                placed_objects[constraint["target"]],
                candidate_solutions,
            )

            weight = self.constraint_type2weight[constraint["type"]]
            if constraint["type"] == "distance":
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * weight
            else:
                for solution in valid_solutions:
                    placement2score[tuple(solution[:3])] += self.constraint_bouns * weight

        # normalize the scores
        for placement in placement2score:
            placement2score[placement] /= max(len(constraints), 1)

        sorted_placements = sorted(placement2score, key=placement2score.get, reverse=True)
        sorted_solutions = [list(placement) + [placement2score[placement]] for placement in sorted_placements]

        return sorted_solutions

    def create_grids(self, room_poly):
        # get the min and max bounds of the room
        min_x, min_z, max_x, max_z = room_poly.bounds

        # create grid points
        grid_points = []

        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_z), int(max_z), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))

        return grid_points

    def _remove_points(self, grid_points, objects_dict):
        # Create an r-tree index
        idx = index.Index()

        # Populate the index with bounding boxes of the objects
        for i, (_, _, obj, _) in enumerate(objects_dict.values()):
            idx.insert(i, Polygon(obj).bounds)

        # Create Shapely Polygon objects only once
        polygons = [Polygon(obj) for _, _, obj, _ in objects_dict.values()]

        valid_points = []

        for point in grid_points:
            p = Point(point)
            # Get a list of potential candidates
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            # Check if point is in any of the candidate polygons
            if not any(candidate.contains(p) for candidate in candidates):
                valid_points.append(point)

        return valid_points

    def get_all_solutions(self, room_poly, grid_points, object_dim):
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: (
                (-obj_half_width, -obj_half_length),
                (obj_half_width, obj_half_length),
            ),
            180: (
                (-obj_half_length, obj_half_width),
                (obj_half_length, -obj_half_width),
            ),
            270: (
                (obj_half_width, -obj_half_length),
                (-obj_half_width, obj_half_length),
            ),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[rotation]
                lower_left = (
                    center_x + lower_left_adjustment[0],
                    center_y + lower_left_adjustment[1],
                )
                upper_right = (
                    center_x + upper_right_adjustment[0],
                    center_y + upper_right_adjustment[1],
                )
                obj_box = box(*lower_left, *upper_right)

                if room_poly.contains(obj_box):
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1])

        return solutions

    def filter_collision(self, objects_dict, solutions):
        valid_solutions = []
        object_polygons = [Polygon(obj_coords) for _, _, obj_coords, _ in list(objects_dict.values())]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.intersects(obj) for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        front_center_adjustments = {
            0: (0, obj_half_width),
            90: (obj_half_width, 0),
            180: (0, -obj_half_width),
            270: (-obj_half_width, 0),
        }

        valid_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            front_center_adjustment = front_center_adjustments[rotation]
            front_center_x, front_center_y = (
                center_x + front_center_adjustment[0],
                center_y + front_center_adjustment[1],
            )

            front_center_distance = room_poly.boundary.distance(Point(front_center_x, front_center_y))

            if front_center_distance >= 30:  # better make this a parameter
                valid_solutions.append(solution)

        return valid_solutions

    def place_edge(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (-obj_half_width, 0),
            180: (0, obj_half_width),
            270: (obj_half_width, 0),
        }

        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            back_center_adjustment = back_center_adjustments[rotation]
            back_center_x, back_center_y = (
                center_x + back_center_adjustment[0],
                center_y + back_center_adjustment[1],
            )

            back_center_distance = room_poly.boundary.distance(Point(back_center_x, back_center_y))
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))

            if back_center_distance <= self.grid_size and back_center_distance < center_distance:
                solution[-1] += self.constraint_bouns
                # valid_solutions.append(solution) # those are still valid solutions, but we need to move the object to the edge

                # move the object to the edge
                center2back_vector = np.array([back_center_x - center_x, back_center_y - center_y])
                center2back_vector /= np.linalg.norm(center2back_vector)
                offset = center2back_vector * (
                    back_center_distance + 4.5
                )  # add a small distance to avoid the object cross the wall
                solution[0] = (center_x + offset[0], center_y + offset[1])
                solution[2] = (
                    (solution[2][0][0] + offset[0], solution[2][0][1] + offset[1]),
                    (solution[2][1][0] + offset[0], solution[2][1][1] + offset[1]),
                    (solution[2][2][0] + offset[0], solution[2][2][1] + offset[1]),
                    (solution[2][3][0] + offset[0], solution[2][3][1] + offset[1]),
                )
                valid_solutions.append(solution)

        return valid_solutions

    def place_corner(self, room_poly, solutions, obj_dim):
        obj_length, obj_width = obj_dim
        obj_half_length, _ = obj_length / 2, obj_width / 2

        rotation_center_adjustments = {
            0: ((-obj_half_length, 0), (obj_half_length, 0)),
            90: ((0, obj_half_length), (0, -obj_half_length)),
            180: ((obj_half_length, 0), (-obj_half_length, 0)),
            270: ((0, -obj_half_length), (0, obj_half_length)),
        }

        edge_solutions = self.place_edge(room_poly, solutions, obj_dim)

        valid_solutions = []

        for solution in edge_solutions:
            (center_x, center_y), rotation = solution[:2]
            (dx_left, dy_left), (dx_right, dy_right) = rotation_center_adjustments[rotation]

            left_center_x, left_center_y = center_x + dx_left, center_y + dy_left
            right_center_x, right_center_y = center_x + dx_right, center_y + dy_right

            left_center_distance = room_poly.boundary.distance(Point(left_center_x, left_center_y))
            right_center_distance = room_poly.boundary.distance(Point(right_center_x, right_center_y))

            if min(left_center_distance, right_center_distance) < self.grid_size:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_relative(self, place_type, target_object, solutions):
        valid_solutions = []
        _, target_rotation, target_coords, _ = target_object
        target_polygon = Polygon(target_coords)

        min_x, min_y, max_x, max_y = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        comparison_dict = {
            "left of": {
                0: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
            },
            "right of": {
                0: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
            },
            "in front of": {
                0: lambda sol_center: sol_center[1] > max_y
                and mean_x - self.grid_size < sol_center[0] < mean_x + self.grid_size,  # in front of and centered
                90: lambda sol_center: sol_center[0] > max_x
                and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
                180: lambda sol_center: sol_center[1] < min_y
                and mean_x - self.grid_size < sol_center[0] < mean_x + self.grid_size,
                270: lambda sol_center: sol_center[0] < min_x
                and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
            },
            "behind": {
                0: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                90: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                180: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                270: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
            },
            "side of": {
                0: lambda sol_center: min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: min_x <= sol_center[0] <= max_x,
            },
        }

        compare_func = comparison_dict.get(place_type).get(target_rotation)

        for solution in solutions:
            sol_center = solution[0]

            if compare_func(sol_center):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_distance(self, distance_type, target_object, solutions):
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        valid_solutions = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)

            solution[-1] = distance
            valid_solutions.append(solution)

        min_distance = min(distances)
        max_distance = max(distances)

        if distance_type == "near":
            if min_distance < 80:
                points = [(min_distance, 1), (80, 0), (max_distance, 0)]
            else:
                points = [(min_distance, 0), (max_distance, 0)]

        elif distance_type == "far":
            points = [(min_distance, 0), (max_distance, 1)]

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        f = interp1d(x, y, kind="linear", fill_value="extrapolate")

        for solution in valid_solutions:
            distance = solution[-1]
            solution[-1] = float(f(distance))

        return valid_solutions

    def place_face(self, face_type, target_object, solutions):
        if face_type == "face to":
            return self.place_face_to(target_object, solutions)

        elif face_type == "face same as":
            return self.place_face_same(target_object, solutions)

        elif face_type == "face opposite to":
            return self.place_face_opposite(target_object, solutions)

    def place_face_to(self, target_object, solutions):
        # Define unit vectors for each rotation
        unit_vectors = {
            0: np.array([0.0, 1.0]),  # Facing up
            90: np.array([1.0, 0.0]),  # Facing right
            180: np.array([0.0, -1.0]),  # Facing down
            270: np.array([-1.0, 0.0]),  # Facing left
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)

        valid_solutions = []

        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]

            # Define an arbitrarily large point in the direction of the solution's rotation
            far_point = sol_center + 1e6 * unit_vectors[sol_rotation]

            # Create a half-line from the solution's center to the far point
            half_line = LineString([sol_center, far_point])

            # Check if the half-line intersects with the target polygon
            if half_line.intersects(target_poly):
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_face_opposite(self, target_object, solutions):
        target_rotation = (target_object[1] + 180) % 360
        valid_solutions = []

        for solution in solutions:
            sol_rotation = solution[1]
            if sol_rotation == target_rotation:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)

        return valid_solutions

    def place_alignment_center(self, alignment_type, target_object, solutions):
        target_center = target_object[0]
        valid_solutions = []
        eps = 5
        for solution in solutions:
            sol_center = solution[0]
            if abs(sol_center[0] - target_center[0]) < eps or abs(sol_center[1] - target_center[1]) < eps:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions
